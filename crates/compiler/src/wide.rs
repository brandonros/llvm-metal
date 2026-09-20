//! Scalar i128 arithmetic used by stock Rust/k256, expressed as (low, high) i64.
//! Deliberately not a general arbitrary-width legalizer: device wide loads,
//! general calls, division, vectors and ABI changes remain unsupported.
//! Call only on a disposable verified module clone.
use crate::pointer_provenance::{private_memory, private_pointer};
use inkwell::{
    llvm_sys::{
        LLVMAtomicOrdering, LLVMIntPredicate, LLVMOpcode, LLVMTypeKind,
        core::*,
        debuginfo::{LLVMInstructionGetDebugLoc, LLVMInstructionSetDebugLoc},
        prelude::*,
        target::{LLVMABISizeOfType, LLVMByteOrder, LLVMByteOrdering, LLVMGetModuleDataLayout},
    },
    module::Module,
};
use std::{
    collections::{HashMap, HashSet},
    ffi::CStr,
};

pub fn lower(module: &Module<'_>) -> Result<(), String> {
    crate::parity::check("wide integers", module, reference, rust)?;
    module.verify().map_err(|e| e.to_string())
}

fn reference(module: &Module<'_>) -> Result<(), String> {
    unsafe extern "C" {
        fn LLVMMetalLowerWideIntegers(module: LLVMModuleRef) -> *mut std::ffi::c_char;
    }
    // SAFETY: caller owns the verified module; dispose the native diagnostic.
    unsafe {
        let error = LLVMMetalLowerWideIntegers(module.as_mut_ptr());
        if !error.is_null() {
            let text = CStr::from_ptr(error).to_string_lossy().into_owned();
            LLVMDisposeMessage(error);
            return Err(text);
        }
    }
    Ok(())
}

fn rust(module: &Module<'_>) -> Result<(), String> {
    // SAFETY: verified disposable module. Every value reached below is owned by
    // it and stays live until the final erase loop.
    unsafe {
        let raw = module.as_mut_ptr();
        let context = LLVMGetModuleContext(raw);
        let mut lower = Lower {
            module: raw,
            context,
            int64: LLVMInt64TypeInContext(context),
            values: HashMap::new(),
            erased: Vec::new(),
            error: None,
        };
        if lower.run() {
            Ok(())
        } else {
            Err(lower.error.unwrap_or_default())
        }
    }
}

type Pair = (LLVMValueRef, LLVMValueRef);

// Each lowering frame owns its insertion point, since operands are lowered
// recursively at their own positions while the user's frame is still building.
struct Builder(LLVMBuilderRef);
impl Builder {
    unsafe fn before(context: LLVMContextRef, instruction: LLVMValueRef) -> Self {
        unsafe {
            let builder = LLVMCreateBuilderInContext(context);
            LLVMPositionBuilderBefore(builder, instruction);
            Self(builder)
        }
    }
}
impl Drop for Builder {
    fn drop(&mut self) {
        // SAFETY: created above and disposed exactly once.
        unsafe { LLVMDisposeBuilder(self.0) }
    }
}

const E: *const std::ffi::c_char = c"".as_ptr();

unsafe fn wide(ty: LLVMTypeRef) -> bool {
    unsafe {
        LLVMGetTypeKind(ty) == LLVMTypeKind::LLVMIntegerTypeKind && LLVMGetIntTypeWidth(ty) == 128
    }
}
// Only one-dimensional, bounded arrays are part of the private snapshot
// profile. Nested aggregates, aggregate calls/PHIs/stores and global ABI
// types remain unsupported.
unsafe fn wide_array(ty: LLVMTypeRef) -> Option<u64> {
    unsafe {
        if LLVMGetTypeKind(ty) != LLVMTypeKind::LLVMArrayTypeKind || !wide(LLVMGetElementType(ty)) {
            return None;
        }
        Some(LLVMGetArrayLength2(ty)).filter(|count| (1..=256).contains(count))
    }
}
unsafe fn atomic(access: LLVMValueRef) -> bool {
    unsafe { LLVMGetOrdering(access) != LLVMAtomicOrdering::LLVMAtomicOrderingNotAtomic }
}
unsafe fn instructions(module: LLVMModuleRef) -> Vec<LLVMValueRef> {
    unsafe {
        let mut all = Vec::new();
        let mut function = LLVMGetFirstFunction(module);
        while !function.is_null() {
            let mut block = LLVMGetFirstBasicBlock(function);
            while !block.is_null() {
                let mut instruction = LLVMGetFirstInstruction(block);
                while !instruction.is_null() {
                    all.push(instruction);
                    instruction = LLVMGetNextInstruction(instruction);
                }
                block = LLVMGetNextBasicBlock(block);
            }
            function = LLVMGetNextFunction(function);
        }
        all
    }
}
unsafe fn users(value: LLVMValueRef) -> Vec<LLVMValueRef> {
    unsafe {
        let mut all = Vec::new();
        let mut use_ = LLVMGetFirstUse(value);
        while !use_.is_null() {
            all.push(LLVMGetUser(use_));
            use_ = LLVMGetNextUse(use_);
        }
        all
    }
}
unsafe fn intrinsic(call: LLVMValueRef) -> Option<u32> {
    unsafe {
        if LLVMIsACallInst(call).is_null() {
            return None;
        }
        let callee = LLVMIsAFunction(LLVMGetCalledValue(call));
        (!callee.is_null())
            .then(|| LLVMGetIntrinsicID(callee))
            .filter(|&id| id != 0)
    }
}
unsafe fn named(name: &str) -> u32 {
    unsafe { LLVMLookupIntrinsicID(name.as_ptr().cast(), name.len()) }
}
// The largest power of two dividing both an alignment and an offset.
fn common_alignment(alignment: u32, offset: u64) -> u32 {
    let both = u64::from(alignment) | offset;
    (both & both.wrapping_neg()) as u32
}
unsafe fn take_name(new: LLVMValueRef, old: LLVMValueRef) {
    unsafe {
        let mut length = 0;
        let text = LLVMGetValueName2(old, &mut length);
        let name = std::slice::from_raw_parts(text.cast::<u8>(), length).to_vec();
        LLVMSetValueName2(old, E, 0);
        LLVMSetValueName2(new, name.as_ptr().cast(), name.len());
    }
}

struct Lower {
    module: LLVMModuleRef,
    context: LLVMContextRef,
    int64: LLVMTypeRef,
    values: HashMap<LLVMValueRef, Pair>,
    erased: Vec<LLVMValueRef>,
    error: Option<String>,
}

impl Lower {
    unsafe fn fail(&mut self, value: LLVMValueRef) -> Option<Pair> {
        if self.error.is_none() {
            unsafe {
                let text = LLVMPrintValueToString(value);
                let printed = CStr::from_ptr(text).to_string_lossy().into_owned();
                LLVMDisposeMessage(text);
                self.error = Some(format!("unsupported i128 operation: {printed}"));
            }
        }
        None
    }
    unsafe fn int(&self, value: u64) -> LLVMValueRef {
        unsafe { LLVMConstInt(self.int64, value, 0) }
    }
    unsafe fn storage_type(&self, ty: LLVMTypeRef) -> LLVMTypeRef {
        unsafe {
            let pair = LLVMArrayType2(self.int64, 2);
            if wide(ty) {
                pair
            } else if let Some(count) = wide_array(ty) {
                LLVMArrayType2(pair, count)
            } else {
                ty
            }
        }
    }
    // Both halves of a wide constant, through LLVM's own constant folding. The
    // C API's integer getter only reads 64 bits.
    unsafe fn halves(&self, constant: LLVMValueRef) -> Option<(u64, u64)> {
        unsafe {
            let builder = LLVMCreateBuilderInContext(self.context);
            let shift = LLVMConstIntOfArbitraryPrecision(LLVMTypeOf(constant), 2, [64, 0].as_ptr());
            let high = LLVMBuildLShr(builder, constant, shift, E);
            LLVMDisposeBuilder(builder);
            let low = LLVMConstTrunc(constant, self.int64);
            let high = LLVMConstTrunc(high, self.int64);
            (!LLVMIsAConstantInt(low).is_null() && !LLVMIsAConstantInt(high).is_null()).then(|| {
                (
                    LLVMConstIntGetZExtValue(low),
                    LLVMConstIntGetZExtValue(high),
                )
            })
        }
    }

    unsafe fn lower_array_snapshots(&mut self) -> bool {
        unsafe {
            let loads: Vec<_> = instructions(self.module)
                .into_iter()
                .filter(|&i| !LLVMIsALoadInst(i).is_null() && wide_array(LLVMTypeOf(i)).is_some())
                .collect();
            for load in loads {
                let source = LLVMGetOperand(load, 0);
                if atomic(load) || !private_pointer(source) {
                    self.fail(load);
                    return false;
                }
                let mut extracts = Vec::new();
                for user in users(load) {
                    if LLVMIsAExtractValueInst(user).is_null()
                        || LLVMGetNumIndices(user) != 1
                        || !wide(LLVMTypeOf(user))
                    {
                        self.fail(user);
                        return false;
                    }
                    extracts.push(user);
                }
                let count = wide_array(LLVMTypeOf(load)).expect("filtered above");
                let builder = Builder::before(self.context, load);
                let int8 = LLVMInt8TypeInContext(self.context);
                let (alignment, volatile) = (LLVMGetAlignment(load), LLVMGetVolatile(load));
                // Read the complete snapshot at the original load, before any
                // intervening writes. Even unextracted volatile elements must read.
                let half = |offset: u64| {
                    let mut index = self.int(offset);
                    let pointer = LLVMBuildGEP2(builder.0, int8, source, &mut index, 1, E);
                    let part = LLVMBuildLoad2(builder.0, self.int64, pointer, E);
                    LLVMSetAlignment(part, common_alignment(alignment, offset));
                    LLVMSetVolatile(part, volatile);
                    part
                };
                let elements: Vec<Pair> = (0..count)
                    .map(|index| (half(index * 16), half(index * 16 + 8)))
                    .collect();
                for extract in extracts {
                    let index = *LLVMGetIndices(extract);
                    self.values.insert(extract, elements[index as usize]);
                    self.erased.push(extract);
                }
                self.erased.push(load);
            }
            true
        }
    }

    unsafe fn normalize_private_storage(&mut self) -> bool {
        unsafe {
            let layout = LLVMGetModuleDataLayout(self.module);
            let allocations: Vec<_> = instructions(self.module)
                .into_iter()
                .filter(|&i| {
                    !LLVMIsAAllocaInst(i).is_null() && {
                        let ty = LLVMGetAllocatedType(i);
                        wide(ty) || wide_array(ty).is_some()
                    }
                })
                .collect();
            let allowed = [
                "llvm.lifetime.start",
                "llvm.lifetime.end",
                "llvm.memcpy",
                "llvm.memset",
            ]
            .map(|name| named(name));
            for allocation in allocations {
                let allocated = LLVMGetAllocatedType(allocation);
                let storage = self.storage_type(allocated);
                let size = LLVMGetOperand(allocation, 0);
                // The C API can neither read nor write inalloca/swifterror, so a
                // rebuilt allocation could not keep them. Neither is AIR storage.
                let text = LLVMPrintValueToString(allocation);
                let flagged = {
                    let printed = CStr::from_ptr(text).to_string_lossy();
                    printed.contains("alloca inalloca ") || printed.contains(" swifterror ")
                };
                LLVMDisposeMessage(text);
                if flagged
                    || LLVMGetPointerAddressSpace(LLVMTypeOf(allocation)) != 0
                    || LLVMIsAConstantInt(size).is_null()
                    || LLVMGetIntTypeWidth(LLVMTypeOf(size)) > 64
                    || LLVMABISizeOfType(layout, allocated) != LLVMABISizeOfType(layout, storage)
                {
                    self.fail(allocation);
                    return false;
                }
                // Opaque pointers do not describe the allocation's element type.
                // Check the complete use chain before changing its storage type;
                // unknown calls, returned/stored pointers and mixed pointer merges
                // are not an extension of the supported private-memory contract.
                let mut pending = vec![allocation];
                let mut seen = HashSet::new();
                let mut wide_geps = Vec::new();
                while let Some(pointer) = pending.pop() {
                    if !seen.insert(pointer) {
                        continue;
                    }
                    for user in users(pointer) {
                        let supported = if !LLVMIsALoadInst(user).is_null() {
                            LLVMGetOperand(user, 0) == pointer
                        } else if !LLVMIsAStoreInst(user).is_null() {
                            LLVMGetOperand(user, 1) == pointer && LLVMGetOperand(user, 0) != pointer
                        } else if !LLVMIsAGetElementPtrInst(user).is_null() {
                            LLVMGetOperand(user, 0) == pointer && {
                                let element = LLVMGetGEPSourceElementType(user);
                                if (wide(element) || wide_array(element).is_some())
                                    && !wide_geps.contains(&user)
                                {
                                    wide_geps.push(user);
                                }
                                pending.push(user);
                                true
                            }
                        } else if !LLVMIsABitCastInst(user).is_null() {
                            LLVMGetTypeKind(LLVMTypeOf(user)) == LLVMTypeKind::LLVMPointerTypeKind
                                && {
                                    pending.push(user);
                                    true
                                }
                        } else {
                            intrinsic(user).is_some_and(|id| allowed.contains(&id))
                        };
                        if !supported {
                            self.fail(user);
                            return false;
                        }
                    }
                }
                // Keep array count, address space, alignment and name. Each
                // source i128 and replacement [2 x i64] has the same 16-byte stride;
                // one-dimensional arrays keep their original element offsets too.
                // The C API cannot retype an instruction, so rebuild it in place.
                let builder = Builder::before(self.context, allocation);
                let replacement = LLVMBuildArrayAlloca(builder.0, storage, size, E);
                LLVMSetAlignment(replacement, LLVMGetAlignment(allocation));
                self.replace(allocation, replacement);
                for gep in wide_geps {
                    let builder = Builder::before(self.context, gep);
                    let mut indices: Vec<_> = (1..LLVMGetNumOperands(gep) as u32)
                        .map(|i| LLVMGetOperand(gep, i))
                        .collect();
                    let replacement = LLVMBuildGEP2(
                        builder.0,
                        self.storage_type(LLVMGetGEPSourceElementType(gep)),
                        LLVMGetOperand(gep, 0),
                        indices.as_mut_ptr(),
                        indices.len() as u32,
                        E,
                    );
                    LLVMGEPSetNoWrapFlags(replacement, LLVMGEPGetNoWrapFlags(gep));
                    self.replace(gep, replacement);
                }
            }
            true
        }
    }
    unsafe fn replace(&self, old: LLVMValueRef, new: LLVMValueRef) {
        unsafe {
            LLVMInstructionSetDebugLoc(new, LLVMInstructionGetDebugLoc(old));
            take_name(new, old);
            LLVMReplaceAllUsesWith(old, new);
            LLVMInstructionEraseFromParent(old);
        }
    }

    unsafe fn get(&mut self, value: LLVMValueRef) -> Option<Pair> {
        unsafe {
            if let Some(&cached) = self.values.get(&value) {
                return Some(cached);
            }
            if !LLVMIsAConstantInt(value).is_null() {
                let Some((low, high)) = self.halves(value) else {
                    return self.fail(value);
                };
                return Some((self.int(low), self.int(high)));
            }
            if LLVMIsAInstruction(value).is_null() || !wide(LLVMTypeOf(value)) {
                return self.fail(value);
            }
            let ty = self.int64;
            let builder = Builder::before(self.context, value);
            let b = builder.0;
            let zero = self.int(0);
            let opcode = LLVMGetInstructionOpcode(value);
            let operand = |i| LLVMGetOperand(value, i);
            use LLVMIntPredicate::{LLVMIntEQ, LLVMIntNE, LLVMIntULT};
            let result = match opcode {
                LLVMOpcode::LLVMPHI => {
                    let low = LLVMBuildPhi(b, ty, c"wide.low".as_ptr());
                    let high = LLVMBuildPhi(b, ty, c"wide.high".as_ptr());
                    // Publish placeholders before following backedges in cyclic SSA.
                    self.values.insert(value, (low, high));
                    self.erased.push(value);
                    for n in 0..LLVMCountIncoming(value) {
                        let mut incoming = self.get(LLVMGetIncomingValue(value, n))?;
                        let mut block = LLVMGetIncomingBlock(value, n);
                        LLVMAddIncoming(low, &mut incoming.0, &mut block, 1);
                        LLVMAddIncoming(high, &mut incoming.1, &mut block, 1);
                    }
                    return Some((low, high));
                }
                LLVMOpcode::LLVMLoad => {
                    if atomic(value) || !private_memory(value) {
                        return self.fail(value);
                    }
                    let pointer = operand(0);
                    let (alignment, volatile) = (LLVMGetAlignment(value), LLVMGetVolatile(value));
                    let low = LLVMBuildLoad2(b, ty, pointer, E);
                    LLVMSetAlignment(low, alignment);
                    LLVMSetVolatile(low, volatile);
                    let mut eight = self.int(8);
                    let upper = LLVMBuildGEP2(
                        b,
                        LLVMInt8TypeInContext(self.context),
                        pointer,
                        &mut eight,
                        1,
                        E,
                    );
                    let high = LLVMBuildLoad2(b, ty, upper, E);
                    LLVMSetAlignment(high, common_alignment(alignment, 8));
                    LLVMSetVolatile(high, volatile);
                    (low, high)
                }
                LLVMOpcode::LLVMZExt | LLVMOpcode::LLVMSExt => {
                    let source = operand(0);
                    let from = LLVMTypeOf(source);
                    if LLVMGetTypeKind(from) != LLVMTypeKind::LLVMIntegerTypeKind
                        || LLVMGetIntTypeWidth(from) > 64
                    {
                        return self.fail(value);
                    }
                    if opcode == LLVMOpcode::LLVMZExt {
                        (LLVMBuildZExt(b, source, ty, E), zero)
                    } else {
                        let low = LLVMBuildSExt(b, source, ty, E);
                        (low, LLVMBuildAShr(b, low, self.int(63), E))
                    }
                }
                LLVMOpcode::LLVMCall if intrinsic(value).is_some() => {
                    if intrinsic(value) != Some(named("llvm.bswap")) {
                        return self.fail(value);
                    }
                    let a = self.get(operand(0))?;
                    let mut overload = ty;
                    let swap = LLVMGetIntrinsicDeclaration(
                        self.module,
                        named("llvm.bswap"),
                        &mut overload,
                        1,
                    );
                    let signature = LLVMGlobalGetValueType(swap);
                    let call = |mut half: LLVMValueRef| {
                        LLVMBuildCall2(b, signature, swap, &mut half, 1, E)
                    };
                    let low = call(a.1);
                    (low, call(a.0))
                }
                LLVMOpcode::LLVMSelect => {
                    let a = self.get(operand(1))?;
                    let c = self.get(operand(2))?;
                    let low = LLVMBuildSelect(b, operand(0), a.0, c.0, E);
                    (low, LLVMBuildSelect(b, operand(0), a.1, c.1, E))
                }
                LLVMOpcode::LLVMAdd
                | LLVMOpcode::LLVMSub
                | LLVMOpcode::LLVMMul
                | LLVMOpcode::LLVMAnd
                | LLVMOpcode::LLVMOr
                | LLVMOpcode::LLVMXor => {
                    let a = self.get(operand(0))?;
                    let c = self.get(operand(1))?;
                    match opcode {
                        LLVMOpcode::LLVMAdd => {
                            let low = LLVMBuildAdd(b, a.0, c.0, E);
                            let carry =
                                LLVMBuildZExt(b, LLVMBuildICmp(b, LLVMIntULT, low, a.0, E), ty, E);
                            (low, LLVMBuildAdd(b, LLVMBuildAdd(b, a.1, c.1, E), carry, E))
                        }
                        LLVMOpcode::LLVMSub => {
                            let low = LLVMBuildSub(b, a.0, c.0, E);
                            let borrow =
                                LLVMBuildZExt(b, LLVMBuildICmp(b, LLVMIntULT, a.0, c.0, E), ty, E);
                            (
                                low,
                                LLVMBuildSub(b, LLVMBuildSub(b, a.1, c.1, E), borrow, E),
                            )
                        }
                        LLVMOpcode::LLVMXor => {
                            let low = LLVMBuildXor(b, a.0, c.0, E);
                            (low, LLVMBuildXor(b, a.1, c.1, E))
                        }
                        LLVMOpcode::LLVMAnd => {
                            let low = LLVMBuildAnd(b, a.0, c.0, E);
                            (low, LLVMBuildAnd(b, a.1, c.1, E))
                        }
                        LLVMOpcode::LLVMOr => {
                            let low = LLVMBuildOr(b, a.0, c.0, E);
                            (low, LLVMBuildOr(b, a.1, c.1, E))
                        }
                        _ => {
                            // Exact 64x64 -> 128 using 32-bit digits. Each partial sum fits
                            // u64; the two cross terms from the high limbs wrap modulo 2^128.
                            let mask = self.int(0xffff_ffff);
                            let half = self.int(32);
                            let a0 = LLVMBuildAnd(b, a.0, mask, E);
                            let a1 = LLVMBuildLShr(b, a.0, half, E);
                            let b0 = LLVMBuildAnd(b, c.0, mask, E);
                            let b1 = LLVMBuildLShr(b, c.0, half, E);
                            let w0 = LLVMBuildMul(b, a0, b0, E);
                            let t = LLVMBuildMul(b, a1, b0, E);
                            let t = LLVMBuildAdd(b, t, LLVMBuildLShr(b, w0, half, E), E);
                            let w1 = LLVMBuildAnd(b, t, mask, E);
                            let w1 = LLVMBuildAdd(b, w1, LLVMBuildMul(b, a0, b1, E), E);
                            let low = LLVMBuildShl(b, w1, half, E);
                            let low = LLVMBuildOr(b, low, LLVMBuildAnd(b, w0, mask, E), E);
                            let high = LLVMBuildMul(b, a1, b1, E);
                            let high = LLVMBuildAdd(b, high, LLVMBuildLShr(b, t, half, E), E);
                            let high = LLVMBuildAdd(b, high, LLVMBuildLShr(b, w1, half, E), E);
                            let cross = LLVMBuildMul(b, a.0, c.1, E);
                            let cross = LLVMBuildAdd(b, cross, LLVMBuildMul(b, a.1, c.0, E), E);
                            (low, LLVMBuildAdd(b, high, cross, E))
                        }
                    }
                }
                LLVMOpcode::LLVMLShr | LLVMOpcode::LLVMAShr | LLVMOpcode::LLVMShl => {
                    let arithmetic = opcode == LLVMOpcode::LLVMAShr;
                    let left = opcode == LLVMOpcode::LLVMShl;
                    let count = operand(1);
                    let a = self.get(operand(0))?;
                    if LLVMIsAConstantInt(count).is_null() {
                        let n = self.get(count)?;
                        let valid = LLVMBuildICmp(b, LLVMIntEQ, n.1, zero, E);
                        let valid = LLVMBuildAnd(
                            b,
                            valid,
                            LLVMBuildICmp(b, LLVMIntULT, n.0, self.int(128), E),
                            E,
                        );
                        let small = LLVMBuildICmp(b, LLVMIntULT, n.0, self.int(64), E);
                        let k = LLVMBuildAnd(b, n.0, self.int(63), E);
                        let inverse = LLVMBuildAnd(b, LLVMBuildSub(b, zero, k, E), self.int(63), E);
                        let nonzero = LLVMBuildICmp(b, LLVMIntNE, k, zero, E);
                        let (low, high) = if left {
                            let cross = LLVMBuildLShr(b, a.0, inverse, E);
                            let cross = LLVMBuildSelect(b, nonzero, cross, zero, E);
                            let low =
                                LLVMBuildSelect(b, small, LLVMBuildShl(b, a.0, k, E), zero, E);
                            let joined = LLVMBuildOr(b, LLVMBuildShl(b, a.1, k, E), cross, E);
                            (
                                low,
                                LLVMBuildSelect(b, small, joined, LLVMBuildShl(b, a.0, k, E), E),
                            )
                        } else {
                            let upper = if arithmetic {
                                LLVMBuildAShr(b, a.1, k, E)
                            } else {
                                LLVMBuildLShr(b, a.1, k, E)
                            };
                            let fill = if arithmetic {
                                LLVMBuildAShr(b, a.1, self.int(63), E)
                            } else {
                                zero
                            };
                            let cross = LLVMBuildShl(b, a.1, inverse, E);
                            let cross = LLVMBuildSelect(b, nonzero, cross, zero, E);
                            let joined = LLVMBuildOr(b, LLVMBuildLShr(b, a.0, k, E), cross, E);
                            let low = LLVMBuildSelect(b, small, joined, upper, E);
                            (low, LLVMBuildSelect(b, small, upper, fill, E))
                        };
                        // LLVM shifts by >= width are poison. Do not wrap an invalid
                        // 128-bit count or create a shift-by-64 in an otherwise valid path.
                        let poison = LLVMGetPoison(ty);
                        let low = LLVMBuildSelect(b, valid, low, poison, E);
                        (low, LLVMBuildSelect(b, valid, high, poison, E))
                    } else {
                        let Some((n, 0)) = self.halves(count) else {
                            return self.fail(value);
                        };
                        if n >= 128 {
                            return self.fail(value);
                        }
                        let shift_high = |s: u64| {
                            if arithmetic {
                                LLVMBuildAShr(b, a.1, self.int(s), E)
                            } else {
                                LLVMBuildLShr(b, a.1, self.int(s), E)
                            }
                        };
                        if n == 0 {
                            a
                        } else if left && n < 64 {
                            let low = LLVMBuildShl(b, a.0, self.int(n), E);
                            let high = LLVMBuildShl(b, a.1, self.int(n), E);
                            let carried = LLVMBuildLShr(b, a.0, self.int(64 - n), E);
                            (low, LLVMBuildOr(b, high, carried, E))
                        } else if left {
                            (zero, LLVMBuildShl(b, a.0, self.int(n - 64), E))
                        } else if n < 64 {
                            let low = LLVMBuildLShr(b, a.0, self.int(n), E);
                            let carried = LLVMBuildShl(b, a.1, self.int(64 - n), E);
                            (LLVMBuildOr(b, low, carried, E), shift_high(n))
                        } else {
                            let low = shift_high(n - 64);
                            (low, if arithmetic { shift_high(63) } else { zero })
                        }
                    }
                }
                _ => return self.fail(value),
            };
            // Dropping no-wrap/exact flags is conservative: all defined inputs retain
            // their result; we do not invent a stronger poison/overflow contract.
            self.values.insert(value, result);
            self.erased.push(value);
            Some(result)
        }
    }

    unsafe fn run(&mut self) -> bool {
        unsafe {
            if LLVMByteOrder(LLVMGetModuleDataLayout(self.module))
                != LLVMByteOrdering::LLVMLittleEndian
            {
                self.error = Some("i128 lowering requires little-endian memory".into());
                return false;
            }
            if !self.normalize_private_storage() || !self.lower_array_snapshots() {
                return false;
            }
            for instruction in instructions(self.module) {
                if !self.rewrite(instruction) {
                    return false;
                }
            }
            // Refuse unsupported consumers instead of leaving a partially lowered IR.
            let removing: HashSet<_> = self.erased.iter().copied().collect();
            for &instruction in &self.erased.clone() {
                if !wide(LLVMTypeOf(instruction)) {
                    continue;
                }
                for user in users(instruction) {
                    if LLVMIsAInstruction(user).is_null() || !removing.contains(&user) {
                        self.fail(user);
                        return false;
                    }
                }
            }
            // Every remaining user is itself erased. The C API cannot drop
            // operand references, so detach each definition before erasing.
            for &instruction in &self.erased {
                let ty = LLVMTypeOf(instruction);
                if LLVMGetTypeKind(ty) != LLVMTypeKind::LLVMVoidTypeKind {
                    LLVMReplaceAllUsesWith(instruction, LLVMGetPoison(ty));
                }
            }
            for &instruction in &self.erased {
                LLVMInstructionEraseFromParent(instruction);
            }
            true
        }
    }

    unsafe fn rewrite(&mut self, instruction: LLVMValueRef) -> bool {
        unsafe {
            if wide(LLVMTypeOf(instruction)) {
                return self.get(instruction).is_some();
            }
            let opcode = LLVMGetInstructionOpcode(instruction);
            if !matches!(
                opcode,
                LLVMOpcode::LLVMICmp | LLVMOpcode::LLVMTrunc | LLVMOpcode::LLVMStore
            ) || !wide(LLVMTypeOf(LLVMGetOperand(instruction, 0)))
            {
                return true;
            }
            match opcode {
                LLVMOpcode::LLVMICmp => {
                    let Some(a) = self.get(LLVMGetOperand(instruction, 0)) else {
                        return false;
                    };
                    let Some(c) = self.get(LLVMGetOperand(instruction, 1)) else {
                        return false;
                    };
                    let builder = Builder::before(self.context, instruction);
                    let predicate = LLVMGetICmpPredicate(instruction);
                    use LLVMIntPredicate::*;
                    let unsigned = match predicate {
                        LLVMIntSGT => LLVMIntUGT,
                        LLVMIntSGE => LLVMIntUGE,
                        LLVMIntSLT => LLVMIntULT,
                        LLVMIntSLE => LLVMIntULE,
                        other => other,
                    };
                    let same_high = LLVMBuildICmp(builder.0, LLVMIntEQ, a.1, c.1, E);
                    let low = LLVMBuildICmp(builder.0, unsigned, a.0, c.0, E);
                    let high = LLVMBuildICmp(builder.0, predicate, a.1, c.1, E);
                    let replacement = LLVMBuildSelect(builder.0, same_high, low, high, E);
                    LLVMReplaceAllUsesWith(instruction, replacement);
                }
                LLVMOpcode::LLVMTrunc => {
                    let destination = LLVMTypeOf(instruction);
                    if LLVMGetTypeKind(destination) != LLVMTypeKind::LLVMIntegerTypeKind
                        || LLVMGetIntTypeWidth(destination) > 64
                    {
                        self.fail(instruction);
                        return false;
                    }
                    let Some(a) = self.get(LLVMGetOperand(instruction, 0)) else {
                        return false;
                    };
                    let builder = Builder::before(self.context, instruction);
                    let replacement = LLVMBuildTrunc(builder.0, a.0, destination, E);
                    LLVMReplaceAllUsesWith(instruction, replacement);
                }
                _ => {
                    let volatile = LLVMGetVolatile(instruction);
                    if atomic(instruction) || (volatile != 0 && !private_memory(instruction)) {
                        self.fail(instruction);
                        return false;
                    }
                    let Some(a) = self.get(LLVMGetOperand(instruction, 0)) else {
                        return false;
                    };
                    let builder = Builder::before(self.context, instruction);
                    let pointer = LLVMGetOperand(instruction, 1);
                    let alignment = LLVMGetAlignment(instruction);
                    let low = LLVMBuildStore(builder.0, a.0, pointer);
                    LLVMSetAlignment(low, alignment);
                    LLVMSetVolatile(low, volatile);
                    let mut eight = self.int(8);
                    let upper = LLVMBuildGEP2(
                        builder.0,
                        LLVMInt8TypeInContext(self.context),
                        pointer,
                        &mut eight,
                        1,
                        E,
                    );
                    let high = LLVMBuildStore(builder.0, a.1, upper);
                    LLVMSetAlignment(high, common_alignment(alignment, 8));
                    LLVMSetVolatile(high, volatile);
                }
            }
            self.erased.push(instruction);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::{
        OptimizationLevel,
        context::Context,
        targets::{InitializationConfig, Target},
    };

    #[test]
    fn lowered_arithmetic_matches_rust_u128_and_i128() {
        Target::initialize_native(&InitializationConfig::default()).unwrap();
        let edges = [0u64, 1, u32::MAX as u64, 1 << 32, 1 << 63, u64::MAX];
        // These constants exercise nonzero upper limbs and wrapping overflow,
        // in addition to the zero-extended 64x64 multiply fixture on Metal.

        let mut operations = [
            "add", "sub", "mul", "and", "xor", "eq", "ne", "ult", "ule", "ugt", "uge", "slt",
            "sle", "sgt", "sge",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        for shift in [0, 1, 31, 52, 63, 64, 65, 127] {
            operations.extend([
                format!("lshr:{shift}"),
                format!("ashr:{shift}"),
                format!("shl:{shift}"),
            ]);
        }
        for ca in [
            0u128,
            0x12345678ffffffff0000000000000001,
            0x92345678ffffffff0000000000000001,
            u128::MAX,
        ] {
            for cb in [ca, 0xfedcba9876543210ffffffffffffffffu128] {
                for extension in ["zext", "sext"] {
                    for operation in &operations {
                        let (opcode, shift) = operation
                            .split_once(':')
                            .map_or((operation.as_str(), None), |(a, b)| {
                                (a, Some(b.parse::<u32>().unwrap()))
                            });
                        let rhs = shift.map_or("%b".to_owned(), |s| s.to_string());
                        let comparison = [
                            "eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge",
                        ]
                        .contains(&opcode);
                        let expression = if comparison {
                            format!(
                                "%condition = icmp {opcode} i128 %a, %b\n%r = select i1 %condition, i128 %a, i128 %b"
                            )
                        } else {
                            format!("%r = {opcode} i128 %a, {rhs}")
                        };
                        let source = format!(
                            "define void @probe(i64 %x, i64 %y, ptr %out) {{
                    %a0 = {extension} i64 %x to i128
                    %b0 = {extension} i64 %y to i128
                    %a = add i128 %a0, {ca}
                    %b = add i128 %b0, {cb}
                    {expression}
                    store i128 %r, ptr %out, align 1
                    ret void
                }}"
                        );
                        let context = Context::create();
                        let module =
                            crate::parse_ir(&context, source.as_bytes(), "wide-test").unwrap();
                        lower(&module).unwrap();
                        // Native execution tests this pass, not AIR or GPU support.
                        let engine = module
                            .create_jit_execution_engine(OptimizationLevel::None)
                            .unwrap();
                        let function = unsafe {
                            engine
                                .get_function::<unsafe extern "C" fn(u64, u64, *mut u8)>("probe")
                                .unwrap()
                        };
                        for x in edges {
                            for y in edges {
                                let extend = |x: u64| {
                                    if extension == "sext" {
                                        (x as i64 as i128) as u128
                                    } else {
                                        x as u128
                                    }
                                };
                                let a = extend(x).wrapping_add(ca);
                                let b = extend(y).wrapping_add(cb);
                                let expected = match opcode {
                                    "add" => a.wrapping_add(b),
                                    "sub" => a.wrapping_sub(b),
                                    "xor" => a ^ b,
                                    "eq" => {
                                        if a == b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "ne" => {
                                        if a != b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "ult" => {
                                        if a < b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "ule" => {
                                        if a <= b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "ugt" => {
                                        if a > b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "uge" => {
                                        if a >= b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "slt" => {
                                        if (a as i128) < (b as i128) {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "sle" => {
                                        if (a as i128) <= (b as i128) {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "sgt" => {
                                        if (a as i128) > (b as i128) {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "sge" => {
                                        if (a as i128) >= (b as i128) {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "mul" => a.wrapping_mul(b),
                                    "and" => a & b,
                                    "lshr" => a >> shift.unwrap(),
                                    "shl" => a << shift.unwrap(),
                                    "ashr" => ((a as i128) >> shift.unwrap()) as u128,
                                    _ => unreachable!(),
                                };
                                let mut output = [0xa5u8; 18];
                                // SAFETY: fixed reviewed IR writes exactly 16 bytes, alignment 1.
                                unsafe {
                                    function.call(x, y, output.as_mut_ptr().add(1));
                                }
                                assert_eq!(
                                    u128::from_le_bytes(output[1..17].try_into().unwrap()),
                                    expected,
                                    "{extension} {operation} {x:x} {y:x}"
                                );
                                assert_eq!((output[0], output[17]), (0xa5, 0xa5));
                            }
                        }
                    }
                }
            }
        }
    }
}
