// Remove only host code-generation flags that do not apply to AIR.
#include <llvm-c/Core.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>

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
