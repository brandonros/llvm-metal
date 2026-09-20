// Generic bridges to LLVM C++ utilities that LLVM's C API does not export.
// No Metal policy belongs here. Each entry states its gap and removal condition.
#include <llvm-c/Core.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Scalar/InferAddressSpaces.h>
#include <llvm/Transforms/Utils/Cloning.h>

// Gap: the textual pipeline's infer-address-spaces takes no flat-address-space
// parameter and AIR has no TargetMachine to supply one. Remove when LLVM's C API
// or pass parser accepts it, or when a bounded Rust inference replaces the pass.
extern "C" void LLVMExtRunInferAddressSpaces(LLVMValueRef value, unsigned flat) {
    auto &function = *llvm::cast<llvm::Function>(llvm::unwrap(value));
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
    passes.addPass(llvm::InferAddressSpacesPass(flat));
    passes.run(function, functions);
}

// Gap: no C API clones a function body into a function with another signature.
// Temporary: remove when the Rust cloner in src/calls.rs replaces it.
extern "C" LLVMBasicBlockRef LLVMExtCloneFunctionInto(LLVMValueRef to, LLVMValueRef from,
                                                     LLVMValueRef *keys, LLVMValueRef *mapped,
                                                     unsigned count) {
    auto *source = llvm::cast<llvm::Function>(llvm::unwrap(from));
    llvm::ValueToValueMapTy values;
    for (unsigned i = 0; i < count; ++i) values[llvm::unwrap(keys[i])] = llvm::unwrap(mapped[i]);
    llvm::SmallVector<llvm::ReturnInst *, 8> returns;
    llvm::CloneFunctionInto(llvm::cast<llvm::Function>(llvm::unwrap(to)), source, values,
                            llvm::CloneFunctionChangeType::LocalChangesOnly, returns);
    return llvm::wrap(llvm::cast<llvm::BasicBlock>(values[&source->getEntryBlock()]));
}
