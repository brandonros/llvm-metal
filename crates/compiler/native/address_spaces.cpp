// LLVM's C API cannot select the flat address space for this existing pass.
// AIR has no upstream TargetMachine, so explicitly set generic address space 0.
#include <llvm-c/Core.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Scalar/InferAddressSpaces.h>

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
