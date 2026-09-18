// Prepare PHI operands for the legacy AIR bitcode writer.
#include <llvm-c/Core.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ReplaceConstant.h>

// The pinned legacy writer demotes ConstantExpr operands immediately before
// their user, which is not legal for PHIs. Ask LLVM's existing utility to place
// equivalent instructions on incoming edges, preserving address spaces and
// pointer provenance. Run after the final optimizer so these do not refold.
extern "C" void LLVMMetalPreparePhiConstants(LLVMModuleRef module) {
    for (auto &function : *llvm::unwrap(module)) {
        llvm::SmallVector<llvm::Constant *, 8> constants;
        for (auto &instruction : llvm::instructions(function))
            if (auto *phi = llvm::dyn_cast<llvm::PHINode>(&instruction))
                for (auto &incoming : phi->incoming_values())
                    if (auto *expression = llvm::dyn_cast<llvm::ConstantExpr>(incoming))
                        constants.push_back(expression);
        if (!constants.empty())
            llvm::convertUsersOfConstantsToInstructions(constants, &function, true, true);
    }
}
