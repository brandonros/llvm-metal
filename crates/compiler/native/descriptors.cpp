#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

// LLVM's utility preserves unrelated llvm.used/compiler.used operands, including
// constant-expression casts. Dead constant users must be pruned before erasure.
extern "C" bool LLVMMetalRemoveDescriptors(LLVMModuleRef ref) {
  auto &module = *llvm::unwrap(ref);
  auto isDescriptor = [](llvm::Constant *constant) {
    auto *value = llvm::dyn_cast<llvm::GlobalValue>(constant->stripPointerCasts());
    return value && value->getName().starts_with("__llvm_metal_descriptor_");
  };
  llvm::removeFromUsedLists(module, isDescriptor);
  for (auto &global : module.globals()) {
    if (!isDescriptor(&global)) continue;
    global.removeDeadConstantUsers();
    if (!global.use_empty()) return false;
  }
  for (auto it = module.global_begin(); it != module.global_end();) {
    auto &global = *it++;
    if (isDescriptor(&global)) global.eraseFromParent();
  }
  return true;
}
