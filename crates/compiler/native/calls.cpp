// Structural check for retained helpers. Specialization itself is in Rust.
#include <llvm-c/Core.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>

// Retaining a nullable pointer iterator miscompiles P-256's r-only matcher on
// Apple M5, while inlining that boundary passes the complete workload. Keep this
// structural class on the proven inlining path until retained execution has a
// sufficient target-level proof. Straight-line pointer joins remain eligible.
extern "C" unsigned LLVMMetalHasNullablePointerLoop(LLVMValueRef ref) {
    auto *function = llvm::cast<llvm::Function>(llvm::unwrap(ref));
    if (function->isDeclaration()) return 0;
    llvm::DominatorTree dominators(*function);
    llvm::LoopInfo loops(dominators);
    for (auto &instruction : llvm::instructions(function)) {
        auto *loop = loops.getLoopFor(instruction.getParent());
        if (!loop) continue;
        bool nullOperation = false;
        if (llvm::isa<llvm::SelectInst>(instruction) ||
            llvm::isa<llvm::ICmpInst>(instruction) ||
            llvm::isa<llvm::PHINode>(instruction))
            for (auto &operand : instruction.operands())
                nullOperation |= llvm::isa<llvm::ConstantPointerNull>(operand);
        if (!nullOperation) continue;
        for (auto *current = loop; current; current = current->getParentLoop())
            for (auto &phi : current->getHeader()->phis())
                if (phi.getType()->isPointerTy()) return 1;
    }
    return 0;
}
