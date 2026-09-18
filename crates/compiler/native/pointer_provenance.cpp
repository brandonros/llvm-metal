// Prove private storage or defined immutable global storage.
#include "pointer_provenance.h"
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace llvm_metal {
static bool privatePointer(llvm::Value *pointer, llvm::SmallPtrSetImpl<llvm::Value *> &visiting) {
    pointer = llvm::getUnderlyingObject(pointer);
    if (llvm::isa<llvm::AllocaInst>(pointer)) return true;
    auto *argument = llvm::dyn_cast<llvm::Argument>(pointer);
    if (!argument || !argument->getParent()->hasLocalLinkage() ||
        !visiting.insert(argument).second) return false;
    // A helper's pointer is private only when every use is a direct call and
    // every caller supplies private storage. Unknown callers/recursion fail closed.
    auto *function = argument->getParent();
    bool called = false, valid = true;
    for (auto &use : function->uses()) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(use.getUser());
        if (!call || !call->isCallee(&use) || call->getCalledFunction() != function ||
            argument->getArgNo() >= call->arg_size() ||
            !privatePointer(call->getArgOperand(argument->getArgNo()), visiting)) {
            valid = false;
            break;
        }
        called = true;
    }
    visiting.erase(argument);
    return called && valid;
}

bool privatePointer(llvm::Value *pointer) {
    llvm::SmallPtrSet<llvm::Value *, 8> visiting;
    return privatePointer(pointer, visiting);
}

bool constantPointer(llvm::Value *pointer) {
    auto *global = llvm::dyn_cast<llvm::GlobalVariable>(llvm::getUnderlyingObject(pointer));
    return global && global->isConstant() && global->hasInitializer() && !global->isExternallyInitialized();
}
} // namespace llvm_metal

extern "C" bool LLVMMetalPrivateMemory(LLVMValueRef value) {
    auto *instruction = llvm::unwrap(value);
    if (auto *load = llvm::dyn_cast<llvm::LoadInst>(instruction))
        return llvm_metal::privatePointer(load->getPointerOperand()) || llvm_metal::constantPointer(load->getPointerOperand());
    if (auto *store = llvm::dyn_cast<llvm::StoreInst>(instruction))
        return llvm_metal::privatePointer(store->getPointerOperand());
    return false;
}
