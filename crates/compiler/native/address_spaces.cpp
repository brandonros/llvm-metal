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

// LLVM 21's inference handles nullable selects but joins a PHI's flat null/
// undef inputs with its device inputs as "flat". Give such PHIs an explicit
// address space only when every concrete underlying object proves the same
// device/constant space. Private or mixed-space pointers remain unsupported.
static void typeNullablePhis(llvm::Function &function) {
    llvm::SmallVector<llvm::PHINode *, 8> phis;
    for (auto &instruction : llvm::instructions(function))
        if (auto *phi = llvm::dyn_cast<llvm::PHINode>(&instruction))
            if (phi->getType()->isPointerTy() && phi->getType()->getPointerAddressSpace() == 0)
                phis.push_back(phi);
    for (auto *phi : phis) {
        bool nullable = false;
        for (auto &incoming : phi->incoming_values())
            nullable |= llvm::isa<llvm::ConstantPointerNull>(incoming) || llvm::isa<llvm::UndefValue>(incoming);
        if (!nullable) continue;
        llvm::SmallVector<const llvm::Value *, 8> objects;
        llvm::getUnderlyingObjects(phi, objects);
        unsigned space = 0;
        bool valid = true;
        for (auto *object : objects) {
            if (llvm::isa<llvm::ConstantPointerNull>(object) || llvm::isa<llvm::UndefValue>(object)) continue;
            unsigned candidate = object->getType()->getPointerAddressSpace();
            if ((candidate != 1 && candidate != 2) || (space && space != candidate)) {
                valid = false;
                break;
            }
            space = candidate;
        }
        if (!valid || !space) continue;
        auto *type = llvm::PointerType::get(function.getContext(), space);
        auto *typed = llvm::PHINode::Create(type, phi->getNumIncomingValues(), "nullable", phi->getIterator());
        for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
            auto *value = phi->getIncomingValue(i);
            auto *block = phi->getIncomingBlock(i);
            llvm::IRBuilder<> builder(block->getTerminator());
            llvm::Value *incoming;
            if (llvm::isa<llvm::ConstantPointerNull>(value)) incoming = llvm::ConstantPointerNull::get(type);
            else if (llvm::isa<llvm::PoisonValue>(value)) incoming = llvm::PoisonValue::get(type);
            else if (llvm::isa<llvm::UndefValue>(value)) incoming = llvm::UndefValue::get(type);
            else incoming = builder.CreateAddrSpaceCast(value, type);
            typed->addIncoming(incoming, block);
        }
        llvm::IRBuilder<> builder(&*phi->getParent()->getFirstNonPHIIt());
        auto *generic = builder.CreateAddrSpaceCast(typed, phi->getType());
        phi->replaceAllUsesWith(generic);
        phi->eraseFromParent();
    }
}

// AIR's supported pointer spaces use a zero null representation. LLVM has no
// AIR target model to fold the constant casts introduced by its inference pass.
static void foldNullCasts(llvm::Function &function) {
    for (auto &instruction : llvm::instructions(function))
        for (auto &operand : instruction.operands())
            if (auto *cast = llvm::dyn_cast<llvm::ConstantExpr>(operand.get()))
                if (cast->getOpcode() == llvm::Instruction::AddrSpaceCast &&
                    llvm::isa<llvm::ConstantPointerNull>(cast->getOperand(0)) &&
                    cast->getType()->getPointerAddressSpace() <= 2 &&
                    cast->getOperand(0)->getType()->getPointerAddressSpace() <= 2)
                    operand.set(llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(cast->getType())));
}

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
        if (!function.isDeclaration()) {
            typeNullablePhis(function);
            passes.run(function, functions);
            foldNullCasts(function);
        }
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
