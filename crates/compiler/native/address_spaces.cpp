// Address-space inference and AIR pointer fixups, in their required order.
// LLVM's C API cannot select the flat address space for this existing pass.
// AIR has no upstream TargetMachine, so explicitly set generic address space 0.
#include "pointer_provenance.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Scalar/InferAddressSpaces.h>

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

// LLVM's generic inference deliberately leaves volatile loads untouched. These
// loads already refer to a reviewed, defined immutable global in Metal constant
// storage; retain that pointer's address space instead of casting it to private.
static void typeConstantVolatileLoads(llvm::Function &function) {
    for (auto &instruction : llvm::instructions(function)) {
        auto *load = llvm::dyn_cast<llvm::LoadInst>(&instruction);
        if (!load || !load->isVolatile() || load->isAtomic()) continue;
        auto *cast = llvm::dyn_cast<llvm::ConstantExpr>(load->getPointerOperand());
        if (!cast || cast->getOpcode() != llvm::Instruction::AddrSpaceCast ||
            cast->getType()->getPointerAddressSpace() != 0) continue;
        auto *source = cast->getOperand(0);
        if (source->getType()->getPointerAddressSpace() == 2 && llvm_metal::constantPointer(source))
            load->setOperand(llvm::LoadInst::getPointerOperandIndex(), source);
    }
}

static void infer(llvm::Module &module, llvm::Function *selected = nullptr) {
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
    for (auto &function : module)
        if (!function.isDeclaration() && (!selected || selected == &function)) {
            typeNullablePhis(function);
            passes.run(function, functions);
            typeConstantVolatileLoads(function);
            foldNullCasts(function);
        }
}

extern "C" void LLVMMetalInferAddressSpaces(LLVMModuleRef module) {
    infer(*llvm::unwrap(module));
}
extern "C" void LLVMMetalInferFunctionAddressSpaces(LLVMValueRef value) {
    auto *function = llvm::cast<llvm::Function>(llvm::unwrap(value));
    infer(*function->getParent(), function);
}
