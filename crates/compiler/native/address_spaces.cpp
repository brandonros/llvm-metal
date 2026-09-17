// LLVM's C API cannot select the flat address space for this existing pass.
// AIR has no upstream TargetMachine, so explicitly set generic address space 0.
#include <llvm-c/Core.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Scalar/InferAddressSpaces.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

static bool privatePointer(llvm::Value *pointer) {
    return llvm::isa<llvm::AllocaInst>(llvm::getUnderlyingObject(pointer));
}

extern "C" bool LLVMMetalPrivateMemory(LLVMValueRef value) {
    auto *instruction = llvm::unwrap(value);
    if (auto *load = llvm::dyn_cast<llvm::LoadInst>(instruction))
        return privatePointer(load->getPointerOperand());
    if (auto *store = llvm::dyn_cast<llvm::StoreInst>(instruction))
        return privatePointer(store->getPointerOperand());
    return false;
}

// Preserve zeroize's volatile aggregate writes as individual volatile accesses.
// Only bounded copies entirely between private allocations are admitted.
extern "C" void LLVMMetalExpandPrivateVolatileCopies(LLVMModuleRef module) {
    llvm::SmallVector<llvm::MemCpyInst *, 8> copies;
    for (auto &function : *llvm::unwrap(module))
        for (auto &instruction : llvm::instructions(function))
            if (auto *copy = llvm::dyn_cast<llvm::MemCpyInst>(&instruction))
                if (copy->isVolatile()) copies.push_back(copy);
    for (auto *copy : copies) {
        auto *length = llvm::dyn_cast<llvm::ConstantInt>(copy->getLength());
        if (!length || length->getValue().ugt(256) ||
            !privatePointer(copy->getSource()) || !privatePointer(copy->getDest())) continue;
        llvm::IRBuilder<> builder(copy);
        for (uint64_t i = 0; i < length->getZExtValue(); ++i) {
            auto *source = builder.CreateGEP(builder.getInt8Ty(), copy->getSource(), builder.getInt64(i));
            auto *destination = builder.CreateGEP(builder.getInt8Ty(), copy->getDest(), builder.getInt64(i));
            auto *value = builder.CreateAlignedLoad(builder.getInt8Ty(), source, llvm::Align(1), true);
            builder.CreateAlignedStore(value, destination, llvm::Align(1), true);
        }
        copy->eraseFromParent();
    }
}

extern "C" void LLVMMetalInferAddressSpaces(LLVMModuleRef module) {
    llvm::LoopAnalysisManager loops;
    llvm::FunctionAnalysisManager functions;
    llvm::CGSCCAnalysisManager cgscc;
    llvm::ModuleAnalysisManager modules;
    llvm::PassBuilder builder;
    builder.registerModuleAnalyses(modules);
    builder.registerCGSCCAnalyses(cgscc);
    builder.registerFunctionAnalyses(functions);
    builder.registerLoopAnalyses(loops);
    builder.crossRegisterProxies(loops, functions, cgscc, modules);
    llvm::FunctionPassManager passes;
    passes.addPass(llvm::InferAddressSpacesPass(0));
    for (auto &function : *llvm::unwrap(module))
        if (!function.isDeclaration())
            passes.run(function, functions);
}

extern "C" void LLVMMetalRemoveCodegenFlags(LLVMModuleRef module) {
    auto *flags = llvm::unwrap(module)->getNamedMetadata("llvm.module.flags");
    if (!flags) return;
    llvm::SmallVector<llvm::MDNode *, 8> retained;
    for (auto *node : flags->operands()) {
        auto *name = llvm::dyn_cast<llvm::MDString>(node->getOperand(1));
        if (!name || (name->getString() != "PIC Level" && name->getString() != "PIE Level"))
            retained.push_back(node);
    }
    flags->clearOperands();
    for (auto *node : retained) flags->addOperand(node);
}
