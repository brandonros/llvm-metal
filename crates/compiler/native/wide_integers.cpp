// Scalar i128 arithmetic used by stock Rust/k256, expressed as (low, high) i64.
// Deliberately not a general arbitrary-width legalizer: wide loads, PHIs, calls,
// division, dynamic shifts, vectors and ABI changes remain unsupported.
#include <llvm-c/Core.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;
namespace {
using Pair = std::pair<Value *, Value *>;
struct Lower {
    DenseMap<Value *, Pair> values;
    SmallVector<Instruction *, 64> erased;
    std::string error;

    Pair fail(Value *v) {
        if (error.empty()) {
            raw_string_ostream out(error);
            out << "unsupported i128 operation: ";
            v->print(out);
        }
        return {nullptr, nullptr};
    }

    Pair get(Value *v) {
        auto found = values.find(v);
        if (found != values.end()) return found->second;
        auto *ty = Type::getInt64Ty(v->getContext());
        if (auto *c = dyn_cast<ConstantInt>(v)) {
            auto bits = c->getValue();
            return {ConstantInt::get(ty, bits.trunc(64)),
                    ConstantInt::get(ty, bits.lshr(64).trunc(64))};
        }
        auto *i = dyn_cast<Instruction>(v);
        if (!i || !i->getType()->isIntegerTy(128)) return fail(v);
        IRBuilder<> b(i);
        auto *zero = b.getInt64(0);
        Pair result;
        unsigned op = i->getOpcode();
        if (op == Instruction::ZExt || op == Instruction::SExt) {
            Value *x = i->getOperand(0);
            if (!x->getType()->isIntegerTy() || x->getType()->getIntegerBitWidth() > 64)
                return fail(i);
            auto *lo = op == Instruction::ZExt ? b.CreateZExt(x, ty) : b.CreateSExt(x, ty);
            result = {lo, op == Instruction::ZExt ? zero : b.CreateAShr(lo, 63)};
        } else if (op == Instruction::Select) {
            auto a = get(i->getOperand(1)), c = get(i->getOperand(2));
            if (!a.first || !c.first) return {nullptr, nullptr};
            result = {b.CreateSelect(i->getOperand(0), a.first, c.first),
                      b.CreateSelect(i->getOperand(0), a.second, c.second)};
        } else if (op == Instruction::Add || op == Instruction::Sub || op == Instruction::Mul || op == Instruction::And || op == Instruction::Xor) {
            auto a = get(i->getOperand(0)), c = get(i->getOperand(1));
            if (!a.first || !c.first) return {nullptr, nullptr};
            if (op == Instruction::Add) {
                auto *lo = b.CreateAdd(a.first, c.first);
                auto *carry = b.CreateZExt(b.CreateICmpULT(lo, a.first), ty);
                result = {lo, b.CreateAdd(b.CreateAdd(a.second, c.second), carry)};
            } else if (op == Instruction::Sub) {
                auto *lo = b.CreateSub(a.first, c.first);
                auto *borrow = b.CreateZExt(b.CreateICmpULT(a.first, c.first), ty);
                result = {lo, b.CreateSub(b.CreateSub(a.second, c.second), borrow)};
            } else if (op == Instruction::Xor) {
                result = {b.CreateXor(a.first, c.first), b.CreateXor(a.second, c.second)};
            } else if (op == Instruction::And) {
                result = {b.CreateAnd(a.first, c.first), b.CreateAnd(a.second, c.second)};
            } else {
                // Exact 64x64 -> 128 using 32-bit digits. Each partial sum fits
                // u64; the two cross terms from the high limbs wrap modulo 2^128.
                auto *mask = b.getInt64(0xffffffff);
                auto *a0 = b.CreateAnd(a.first, mask), *a1 = b.CreateLShr(a.first, 32);
                auto *b0 = b.CreateAnd(c.first, mask), *b1 = b.CreateLShr(c.first, 32);
                auto *w0 = b.CreateMul(a0, b0);
                auto *t = b.CreateAdd(b.CreateMul(a1, b0), b.CreateLShr(w0, 32));
                auto *w1 = b.CreateAdd(b.CreateAnd(t, mask), b.CreateMul(a0, b1));
                auto *lo = b.CreateOr(b.CreateShl(w1, 32), b.CreateAnd(w0, mask));
                auto *hi = b.CreateAdd(b.CreateAdd(b.CreateMul(a1, b1), b.CreateLShr(t, 32)), b.CreateLShr(w1, 32));
                hi = b.CreateAdd(hi, b.CreateAdd(b.CreateMul(a.first, c.second), b.CreateMul(a.second, c.first)));
                result = {lo, hi};
            }
        } else if (op == Instruction::LShr || op == Instruction::AShr || op == Instruction::Shl) {
            auto *count = dyn_cast<ConstantInt>(i->getOperand(1));
            if (!count || count->getValue().uge(128)) return fail(i);
            auto a = get(i->getOperand(0));
            if (!a.first) return a;
            unsigned n = count->getZExtValue();
            auto shiftHigh = [&](unsigned s) -> Value * {
                return op == Instruction::AShr ? b.CreateAShr(a.second, s) : b.CreateLShr(a.second, s);
            };
            if (n == 0) result = a;
            else if (op == Instruction::Shl) {
                if (n < 64)
                    result = {b.CreateShl(a.first, n), b.CreateOr(b.CreateShl(a.second, n), b.CreateLShr(a.first, 64-n))};
                else result = {zero, b.CreateShl(a.first, n-64)};
            }
            else if (n < 64)
                result = {b.CreateOr(b.CreateLShr(a.first, n), b.CreateShl(a.second, 64-n)), shiftHigh(n)};
            else
                result = {shiftHigh(n-64), op == Instruction::AShr ? b.CreateAShr(a.second, 63) : zero};
        } else return fail(i);
        // Dropping no-wrap/exact flags is conservative: all defined inputs retain
        // their result; we do not invent a stronger poison/overflow contract.
        values[v] = result;
        erased.push_back(i);
        return result;
    }

    bool run(Module &module) {
        if (!module.getDataLayout().isLittleEndian()) {
            error = "i128 lowering requires little-endian memory";
            return false;
        }
        SmallVector<Instruction *, 64> original;
        for (auto &f : module)
            for (auto &i : instructions(f)) original.push_back(&i);
        for (auto *i : original) {
            if (i->getType()->isIntegerTy(128)) {
                if (!get(i).first) return false;
            } else if (auto *cmp = dyn_cast<ICmpInst>(i)) {
                if (!cmp->getOperand(0)->getType()->isIntegerTy(128)) continue;
                auto a = get(cmp->getOperand(0)), c = get(cmp->getOperand(1));
                if (!a.first || !c.first) return false;
                IRBuilder<> b(cmp);
                auto pred = cmp->getPredicate();
                auto *sameHigh = b.CreateICmpEQ(a.second, c.second);
                auto *low = b.CreateICmp(ICmpInst::getUnsignedPredicate(pred), a.first, c.first);
                auto *high = b.CreateICmp(pred, a.second, c.second);
                cmp->replaceAllUsesWith(b.CreateSelect(sameHigh, low, high));
                erased.push_back(cmp);
            } else if (auto *t = dyn_cast<TruncInst>(i)) {
                if (!t->getSrcTy()->isIntegerTy(128)) continue;
                if (!t->getDestTy()->isIntegerTy() || t->getDestTy()->getIntegerBitWidth() > 64) {
                    fail(i); return false;
                }
                auto a = get(t->getOperand(0));
                if (!a.first) return false;
                IRBuilder<> b(t);
                t->replaceAllUsesWith(b.CreateTrunc(a.first, t->getDestTy()));
                erased.push_back(t);
            } else if (auto *s = dyn_cast<StoreInst>(i)) {
                if (!s->getValueOperand()->getType()->isIntegerTy(128)) continue;
                if (s->isVolatile() || s->isAtomic()) { fail(i); return false; }
                auto a = get(s->getValueOperand());
                if (!a.first) return false;
                IRBuilder<> b(s);
                auto *p = s->getPointerOperand();
                b.CreateAlignedStore(a.first, p, s->getAlign());
                b.CreateAlignedStore(a.second, b.CreateGEP(b.getInt8Ty(), p, b.getInt64(8)), commonAlignment(s->getAlign(), 8));
                erased.push_back(s);
            }
        }
        // Refuse unsupported consumers instead of leaving a partially lowered IR.
        SmallPtrSet<Instruction *, 32> removing(erased.begin(), erased.end());
        for (auto *i : erased) {
            if (!i->getType()->isIntegerTy(128)) continue;
            for (auto *user : i->users()) {
                auto *u = dyn_cast<Instruction>(user);
                if (!u || !removing.contains(u)) { fail(user); return false; }
            }
        }
        for (auto *i : erased) i->dropAllReferences();
        for (auto *i : erased) i->eraseFromParent();
        return true;
    }
};
}

extern "C" char *LLVMMetalLowerWideIntegers(LLVMModuleRef module) {
    Lower lower;
    if (lower.run(*unwrap(module))) return nullptr;
    return LLVMCreateMessage(lower.error.c_str());
}
