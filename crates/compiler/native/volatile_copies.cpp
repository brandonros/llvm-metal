// Expand bounded volatile memcpy operations using shared pointer provenance.
#include "pointer_provenance.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>

// Preserve zeroize's volatile aggregate writes as individual volatile accesses.
// Only bounded copies into private storage from private or defined constant storage are admitted.
extern "C" void LLVMMetalExpandPrivateVolatileCopies(LLVMModuleRef module) {
    llvm::SmallVector<llvm::MemCpyInst *, 8> copies;
    for (auto &function : *llvm::unwrap(module))
        for (auto &instruction : llvm::instructions(function))
            if (auto *copy = llvm::dyn_cast<llvm::MemCpyInst>(&instruction))
                if (copy->isVolatile()) copies.push_back(copy);
    for (auto *copy : copies) {
        auto *length = llvm::dyn_cast<llvm::ConstantInt>(copy->getLength());
        if (!length || length->getValue().ugt(4096) ||
            !(llvm_metal::privatePointer(copy->getSource()) || llvm_metal::constantPointer(copy->getSource())) || !llvm_metal::privatePointer(copy->getDest())) continue;
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
