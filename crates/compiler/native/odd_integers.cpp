// Promote non-machine scalar integers without changing memory spans.
// i2..i7 are register-only and promote to i32; their storage is refused.
// i24 -> i32; i40/i48/i56 -> i64. Every promoted value is zero-extended from
// its original width. Signed operations explicitly sign-extend that width.
#include <llvm-c/Core.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;
namespace {
static bool odd(Type *type) {
    auto *integer = dyn_cast<IntegerType>(type);
    if (!integer) return false;
    unsigned width = integer->getBitWidth();
    return (width >= 2 && width <= 7) || width == 24 || width == 40 || width == 48 || width == 56;
}
static bool containsOdd(Type *type) {
    if (odd(type)) return true;
    if (auto *array = dyn_cast<ArrayType>(type)) return containsOdd(array->getElementType());
    if (auto *vector = dyn_cast<VectorType>(type)) return containsOdd(vector->getElementType());
    if (auto *structure = dyn_cast<StructType>(type))
        if (!structure->isOpaque())
            for (auto *element : structure->elements()) if (containsOdd(element)) return true;
    return false;
}
static bool ordinaryInteger(Type *type) {
    auto *integer = dyn_cast<IntegerType>(type);
    if (!integer) return false;
    switch (integer->getBitWidth()) {
    case 1: case 8: case 16: case 32: case 64: return true;
    default: return false;
    }
}
struct Lower {
    DenseMap<Value *, Value *> values;
    SmallVector<Instruction *, 64> erased;
    std::string error;

    Value *fail(Value *value, const char *reason = "unsupported operation") {
        if (error.empty()) {
            raw_string_ostream out(error);
            out << "odd integer promotion: " << reason << ": ";
            value->print(out);
        }
        return nullptr;
    }
    IntegerType *promoted(Type *type) {
        unsigned width = type->getIntegerBitWidth();
        return IntegerType::get(type->getContext(), width <= 24 ? 32 : 64);
    }
    Value *mask(IRBuilder<> &builder, Value *value, unsigned width) {
        auto bits = value->getType()->getIntegerBitWidth();
        return builder.CreateAnd(value, ConstantInt::get(value->getType(), APInt::getLowBitsSet(bits, width)), "odd.mask");
    }
    Value *signedValue(IRBuilder<> &builder, Value *value, unsigned width) {
        unsigned shift = value->getType()->getIntegerBitWidth() - width;
        if (!shift) return value;
        return builder.CreateAShr(builder.CreateShl(value, shift), shift, "odd.signed");
    }
    // Convert a scalar operand to a chosen native width. Odd operands are
    // normalized by get; ordinary operands retain their actual signedness.
    Value *convert(IRBuilder<> &builder, Value *value, Type *destination, bool sign) {
        unsigned width = value->getType()->isIntegerTy() ? value->getType()->getIntegerBitWidth() : 0;
        if (odd(value->getType())) {
            value = get(value);
            if (!value) return nullptr;
            if (sign) value = signedValue(builder, value, width);
        } else if (!ordinaryInteger(value->getType())) return fail(value, "unsupported cast source");
        return sign ? builder.CreateSExtOrTrunc(value, destination) : builder.CreateZExtOrTrunc(value, destination);
    }
    Value *get(Value *value) {
        auto cached = values.find(value);
        if (cached != values.end()) return cached->second;
        if (!odd(value->getType())) return fail(value, "expected supported scalar width");
        unsigned width = value->getType()->getIntegerBitWidth();
        auto *type = promoted(value->getType());
        if (auto *constant = dyn_cast<ConstantInt>(value))
            return ConstantInt::get(type, constant->getValue().zext(type->getBitWidth()));
        if (isa<PoisonValue>(value)) return PoisonValue::get(type);
        // A concrete zero is a permitted refinement of undef, with no unknown
        // high bits leaking into comparisons or a zero extension.
        if (isa<UndefValue>(value)) return ConstantInt::get(type, 0);
        auto *instruction = dyn_cast<Instruction>(value);
        if (!instruction) return fail(value, "odd constant expression or argument");
        IRBuilder<> builder(instruction);
        if (auto *phi = dyn_cast<PHINode>(instruction)) {
            auto *replacement = builder.CreatePHI(type, phi->getNumIncomingValues(), "odd.phi");
            values[value] = replacement;
            erased.push_back(phi);
            for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
                auto *incoming = get(phi->getIncomingValue(i));
                if (!incoming) return nullptr;
                replacement->addIncoming(incoming, phi->getIncomingBlock(i));
            }
            return replacement;
        }
        Value *result = nullptr;
        if (auto *load = dyn_cast<LoadInst>(instruction)) {
            if (width < 8) return fail(load, "sub-byte load storage");
            if (load->isAtomic() || load->isVolatile()) return fail(load, "atomic or volatile access");
            result = ConstantInt::get(type, 0);
            for (unsigned byte = 0; byte < width / 8; ++byte) {
                auto *pointer = builder.CreateGEP(builder.getInt8Ty(), load->getPointerOperand(), builder.getInt64(byte));
                auto *part = builder.CreateAlignedLoad(builder.getInt8Ty(), pointer, Align(1));
                result = builder.CreateOr(result, builder.CreateShl(builder.CreateZExt(part, type), byte * 8));
            }
        } else if (auto *cast = dyn_cast<CastInst>(instruction)) {
            unsigned opcode = cast->getOpcode();
            if (opcode != Instruction::Trunc && opcode != Instruction::ZExt && opcode != Instruction::SExt)
                return fail(cast, "unsupported cast");
            result = convert(builder, cast->getOperand(0), type, opcode == Instruction::SExt);
            if (!result) return nullptr;
            result = mask(builder, result, width);
        } else if (auto *select = dyn_cast<SelectInst>(instruction)) {
            auto *yes = get(select->getTrueValue()), *no = get(select->getFalseValue());
            if (!yes || !no) return nullptr;
            result = builder.CreateSelect(select->getCondition(), yes, no);
        } else if (auto *freeze = dyn_cast<FreezeInst>(instruction)) {
            auto *input = get(freeze->getOperand(0));
            if (!input) return nullptr;
            result = mask(builder, builder.CreateFreeze(input), width);
        } else if (auto *binary = dyn_cast<BinaryOperator>(instruction)) {
            auto *left = get(binary->getOperand(0)), *right = get(binary->getOperand(1));
            if (!left || !right) return nullptr;
            unsigned opcode = binary->getOpcode();
            switch (opcode) {
            case Instruction::Add: result = builder.CreateAdd(left, right); break;
            case Instruction::Sub: result = builder.CreateSub(left, right); break;
            case Instruction::Mul: result = builder.CreateMul(left, right); break;
            case Instruction::And: result = builder.CreateAnd(left, right); break;
            case Instruction::Or: result = builder.CreateOr(left, right); break;
            case Instruction::Xor: result = builder.CreateXor(left, right); break;
            case Instruction::UDiv: result = builder.CreateUDiv(left, right); break;
            case Instruction::URem: result = builder.CreateURem(left, right); break;
            case Instruction::SDiv:
            case Instruction::SRem: {
                auto *a = signedValue(builder, left, width), *b = signedValue(builder, right, width);
                result = opcode == Instruction::SDiv ? builder.CreateSDiv(a, b) : builder.CreateSRem(a, b);
                // Narrow signed-division overflow remains poison, even though
                // the wider hardware quotient would fit its native integer.
                if (opcode == Instruction::SDiv) {
                    auto *minimum = ConstantInt::get(type, uint64_t(1) << (width - 1));
                    auto *negativeOne = ConstantInt::get(type, (uint64_t(1) << width) - 1);
                    auto *overflow = builder.CreateAnd(builder.CreateICmpEQ(left, minimum), builder.CreateICmpEQ(right, negativeOne));
                    result = builder.CreateSelect(overflow, PoisonValue::get(type), result);
                }
                break;
            }
            case Instruction::Shl:
            case Instruction::LShr:
            case Instruction::AShr: {
                auto *safe = builder.CreateAnd(right, ConstantInt::get(type, type->getBitWidth() - 1));
                if (opcode == Instruction::Shl) result = builder.CreateShl(left, safe);
                else if (opcode == Instruction::LShr) result = builder.CreateLShr(left, safe);
                else result = builder.CreateAShr(signedValue(builder, left, width), safe);
                result = builder.CreateSelect(builder.CreateICmpULT(right, ConstantInt::get(type, width)), result, PoisonValue::get(type));
                break;
            }
            default: return fail(binary);
            }
            // Do not propagate nuw/nsw/exact onto a changed width. Masking keeps
            // every defined original result, without strengthening its contract.
            result = mask(builder, result, width);
        } else return fail(instruction);
        values[value] = result;
        erased.push_back(instruction);
        return result;
    }

    bool run(Module &module) {
        auto &layout = module.getDataLayout();
        SmallVector<Instruction *, 64> original;
        for (auto &global : module.globals())
            if (containsOdd(global.getValueType())) { fail(&global, "odd global storage"); return false; }
        for (auto &function : module) {
            if (containsOdd(function.getReturnType())) { fail(&function, "odd function ABI"); return false; }
            for (auto &argument : function.args())
                if (containsOdd(argument.getType())) { fail(&argument, "odd function ABI"); return false; }
            for (auto &instruction : instructions(function)) {
                if (containsOdd(instruction.getType()) && !odd(instruction.getType())) {
                    fail(&instruction, "odd vector or aggregate"); return false;
                }
                if (auto *allocation = dyn_cast<AllocaInst>(&instruction))
                    if (containsOdd(allocation->getAllocatedType())) { fail(allocation, "odd stack storage"); return false; }
                if (auto *gep = dyn_cast<GetElementPtrInst>(&instruction))
                    if (containsOdd(gep->getSourceElementType())) { fail(gep, "odd GEP element storage"); return false; }
                if (auto *load = dyn_cast<LoadInst>(&instruction))
                    if (odd(load->getType()) && load->getType()->getIntegerBitWidth() < 8) {
                        fail(load, "sub-byte load storage"); return false;
                    }
                if (auto *store = dyn_cast<StoreInst>(&instruction))
                    if (odd(store->getValueOperand()->getType()) && store->getValueOperand()->getType()->getIntegerBitWidth() < 8) {
                        fail(store, "sub-byte store storage"); return false;
                    }
                auto checkType = [&](Type *type) {
                    if (containsOdd(type) && !odd(type)) {
                        fail(&instruction, "odd vector or aggregate"); return false;
                    }
                    if (!odd(type)) return true;
                    unsigned width = type->getIntegerBitWidth();
                    if (width < 8) return true; // Register-only: no storage layout is changed.
                    unsigned allocation = width == 24 ? 4 : 8;
                    if (!layout.isLittleEndian() || layout.getTypeAllocSize(type) != allocation || layout.getABITypeAlign(type) != Align(allocation)) {
                        fail(&instruction, "NVPTX/AIR layout mismatch"); return false;
                    }
                    return true;
                };
                if (!checkType(instruction.getType())) return false;
                for (auto &operand : instruction.operands()) if (!checkType(operand->getType())) return false;
                original.push_back(&instruction);
            }
        }
        for (auto *instruction : original) {
            if (odd(instruction->getType())) {
                if (!get(instruction)) return false;
            } else if (auto *compare = dyn_cast<ICmpInst>(instruction)) {
                if (!odd(compare->getOperand(0)->getType())) continue;
                auto *a = get(compare->getOperand(0)), *b = get(compare->getOperand(1));
                if (!a || !b) return false;
                IRBuilder<> builder(compare);
                if (compare->isSigned()) {
                    unsigned width = compare->getOperand(0)->getType()->getIntegerBitWidth();
                    a = signedValue(builder, a, width); b = signedValue(builder, b, width);
                }
                compare->replaceAllUsesWith(builder.CreateICmp(compare->getPredicate(), a, b));
                erased.push_back(compare);
            } else if (auto *cast = dyn_cast<CastInst>(instruction)) {
                if (!odd(cast->getOperand(0)->getType())) continue;
                if (!ordinaryInteger(cast->getType()) || (cast->getOpcode() != Instruction::Trunc && cast->getOpcode() != Instruction::ZExt && cast->getOpcode() != Instruction::SExt)) {
                    fail(cast, "unsupported cast consumer"); return false;
                }
                IRBuilder<> builder(cast);
                auto *replacement = convert(builder, cast->getOperand(0), cast->getType(), cast->getOpcode() == Instruction::SExt);
                if (!replacement) return false;
                cast->replaceAllUsesWith(replacement);
                erased.push_back(cast);
            } else if (auto *store = dyn_cast<StoreInst>(instruction)) {
                if (!odd(store->getValueOperand()->getType())) continue;
                if (store->isAtomic() || store->isVolatile()) { fail(store, "atomic or volatile access"); return false; }
                auto *value = get(store->getValueOperand());
                if (!value) return false;
                IRBuilder<> builder(store);
                unsigned width = store->getValueOperand()->getType()->getIntegerBitWidth();
                for (unsigned byte = 0; byte < width / 8; ++byte) {
                    auto *pointer = builder.CreateGEP(builder.getInt8Ty(), store->getPointerOperand(), builder.getInt64(byte));
                    builder.CreateAlignedStore(builder.CreateTrunc(builder.CreateLShr(value, byte * 8), builder.getInt8Ty()), pointer, Align(1));
                }
                erased.push_back(store);
            }
        }
        SmallPtrSet<Instruction *, 32> removing(erased.begin(), erased.end());
        for (auto *instruction : erased) {
            if (!odd(instruction->getType())) continue;
            for (auto *user : instruction->users()) {
                auto *consumer = dyn_cast<Instruction>(user);
                if (!consumer || !removing.contains(consumer)) { fail(user, "unsupported consumer"); return false; }
            }
        }
        for (auto *instruction : erased) instruction->dropAllReferences();
        for (auto *instruction : erased) instruction->eraseFromParent();
        return true;
    }
};
}
extern "C" char *LLVMMetalLowerOddIntegers(LLVMModuleRef module) {
    Lower lower;
    if (lower.run(*unwrap(module))) return nullptr;
    return LLVMCreateMessage(lower.error.c_str());
}
