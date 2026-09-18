// Specialize internal pointer-parameter helpers by the caller's concrete AIR
// address spaces. No generic pointer is allowed to cross a retained call.
#include "pointer_provenance.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <map>
#include <vector>

extern "C" void LLVMMetalInferFunctionAddressSpaces(LLVMValueRef function);

static llvm::Value *concrete(llvm::Value *value, llvm::IRBuilder<> &builder) {
    if (auto *cast = llvm::dyn_cast<llvm::AddrSpaceCastInst>(value)) value = cast->getOperand(0);
    else if (auto *expr = llvm::dyn_cast<llvm::ConstantExpr>(value))
        if (expr->getOpcode() == llvm::Instruction::AddrSpaceCast) value = expr->getOperand(0);
    if (auto *gep = llvm::dyn_cast<llvm::GEPOperator>(value)) {
        auto *base = concrete(gep->getPointerOperand(), builder);
        if (!base) return nullptr;
        if (base->getType()->getPointerAddressSpace() != value->getType()->getPointerAddressSpace()) {
            llvm::SmallVector<llvm::Value *, 8> indices(gep->indices());
            // Drop optional no-wrap/inbounds facts rather than inventing target facts.
            return builder.CreateGEP(gep->getSourceElementType(), base, indices, "typed.gep");
        }
    }
    unsigned space = value->getType()->getPointerAddressSpace();
    if (space == 1 || space == 2) return value;
    if (space == 0) {
        llvm::SmallVector<const llvm::Value *, 8> objects;
        llvm::getUnderlyingObjects(value, objects);
        bool privateStorage = !objects.empty();
        for (auto *object : objects)
            privateStorage &= llvm::isa<llvm::ConstantPointerNull>(object) ||
                llvm_metal::privatePointer(const_cast<llvm::Value *>(object));
        if (privateStorage) return value;
    }
    return nullptr;
}

extern "C" char *LLVMMetalSpecializeCalls(LLVMModuleRef ref, const char *entry) {
    auto &module = *llvm::unwrap(ref);
    auto *root = module.getFunction(entry);
    if (!root) return LLVMCreateMessage("missing specialization entry");
    using Key = std::pair<llvm::Function *, std::vector<unsigned>>;
    std::map<Key, llvm::Function *> cache;
    llvm::SmallVector<llvm::Function *, 16> pending{root};
    llvm::SmallPtrSet<llvm::Function *, 32> visited;
    while (!pending.empty()) {
        auto *caller = pending.pop_back_val();
        if (!visited.insert(caller).second) continue;
        // Make casts/GEPs concrete before classifying this caller's nested calls.
        LLVMMetalInferFunctionAddressSpaces(llvm::wrap(caller));
        llvm::SmallVector<llvm::CallInst *, 16> calls;
        for (auto &inst : llvm::instructions(caller))
            if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) calls.push_back(call);
        for (auto *call : calls) {
            auto *callee = call->getCalledFunction();
            if (!callee || callee->isDeclaration()) continue;
            if (!callee->hasFnAttribute("llvm-metal.retained"))
                return LLVMCreateMessage("unsupported helper survived required inlining");
            llvm::IRBuilder<> callBuilder(call);
            llvm::SmallVector<llvm::Value *, 8> args;
            llvm::SmallVector<llvm::Type *, 8> types;
            std::vector<unsigned> spaces;
            bool pointers = false;
            for (auto &arg : call->args()) {
                llvm::Value *value = arg.get();
                if (value->getType()->isPointerTy()) {
                    pointers = true;
                    value = concrete(value, callBuilder);
                    if (!value) {
                        std::string message;
                        llvm::raw_string_ostream out(message);
                        out << "unresolved pointer flow at retained helper call in " << caller->getName() << ": " << *call;
                        return LLVMCreateMessage(message.c_str());
                    }
                    spaces.push_back(value->getType()->getPointerAddressSpace());
                }
                args.push_back(value);
                types.push_back(value->getType());
            }
            if (!pointers) { pending.push_back(callee); continue; }
            Key key{callee, spaces};
            auto found = cache.find(key);
            llvm::Function *specialized;
            if (found != cache.end()) specialized = found->second;
            else {
                if (cache.size() >= 4096) return LLVMCreateMessage("retained helper specialization limit exceeded");
                std::string suffix = ".metal";
                for (unsigned space : spaces) suffix += "." + std::to_string(space);
                auto *type = llvm::FunctionType::get(callee->getReturnType(), types, false);
                specialized = llvm::Function::Create(type, llvm::GlobalValue::InternalLinkage,
                    callee->getName() + suffix, module);
                specialized->setCallingConv(callee->getCallingConv());
                llvm::ValueToValueMapTy values;
                auto *prelude = llvm::BasicBlock::Create(module.getContext(), "typed.args", specialized);
                llvm::IRBuilder<> builder(prelude);
                for (auto &old : callee->args()) {
                    auto *arg = specialized->getArg(old.getArgNo());
                    arg->setName(old.getName());
                    values[&old] = arg->getType() == old.getType() ? arg :
                        builder.CreateAddrSpaceCast(arg, old.getType(), "generic.arg");
                }
                llvm::SmallVector<llvm::ReturnInst *, 8> returns;
                llvm::CloneFunctionInto(specialized, callee, values,
                    llvm::CloneFunctionChangeType::LocalChangesOnly, returns);
                builder.CreateBr(llvm::cast<llvm::BasicBlock>(values[&callee->getEntryBlock()]));
                cache.emplace(std::move(key), specialized);
            }
            llvm::IRBuilder<> builder(call);
            auto *replacement = builder.CreateCall(specialized, args);
            replacement->setCallingConv(call->getCallingConv());
            replacement->setAttributes(call->getAttributes());
            replacement->setDebugLoc(call->getDebugLoc());
            call->replaceAllUsesWith(replacement);
            call->eraseFromParent();
            pending.push_back(specialized);
        }
    }
    return nullptr;
}
