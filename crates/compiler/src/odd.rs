//! Exact-span promotion of byte-sized scalar integers to i32/i64.
//! Call only on a disposable, verified module clone. No ABI changes are allowed.
//!
//! i2..i7 are register-only and promote to i32; their storage is refused.
//! i24 -> i32; i40/i48/i56 -> i64. Every promoted value is zero-extended from
//! its original width. Signed operations explicitly sign-extend that width.
use inkwell::{
    llvm_sys::{
        LLVMAtomicOrdering, LLVMIntPredicate, LLVMOpcode, LLVMTypeKind,
        core::*,
        prelude::*,
        target::{
            LLVMABIAlignmentOfType, LLVMABISizeOfType, LLVMByteOrder, LLVMByteOrdering,
            LLVMGetModuleDataLayout, LLVMTargetDataRef,
        },
    },
    module::Module,
};
use std::{
    collections::{HashMap, HashSet},
    ffi::CStr,
};

pub fn lower(module: &Module<'_>) -> Result<(), String> {
    // SAFETY: verified disposable module. Every value reached below is owned by
    // it and stays live until the final erase loop.
    unsafe {
        let mut lower = Lower {
            context: LLVMGetModuleContext(module.as_mut_ptr()),
            values: HashMap::new(),
            erased: Vec::new(),
            error: None,
        };
        if !lower.run(module.as_mut_ptr()) {
            return Err(lower.error.unwrap_or_default());
        }
    }
    module.verify().map_err(|error| error.to_string())
}

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

unsafe fn width(ty: LLVMTypeRef) -> Option<u32> {
    unsafe {
        (LLVMGetTypeKind(ty) == LLVMTypeKind::LLVMIntegerTypeKind).then(|| LLVMGetIntTypeWidth(ty))
    }
}
unsafe fn odd(ty: LLVMTypeRef) -> bool {
    unsafe { width(ty) }.is_some_and(|w| (2..=7).contains(&w) || matches!(w, 24 | 40 | 48 | 56))
}
unsafe fn contains_odd(ty: LLVMTypeRef) -> bool {
    unsafe {
        if odd(ty) {
            return true;
        }
        match LLVMGetTypeKind(ty) {
            LLVMTypeKind::LLVMArrayTypeKind
            | LLVMTypeKind::LLVMVectorTypeKind
            | LLVMTypeKind::LLVMScalableVectorTypeKind => contains_odd(LLVMGetElementType(ty)),
            LLVMTypeKind::LLVMStructTypeKind if LLVMIsOpaqueStruct(ty) == 0 => (0
                ..LLVMCountStructElementTypes(ty))
                .any(|i| contains_odd(LLVMStructGetTypeAtIndex(ty, i))),
            _ => false,
        }
    }
}
unsafe fn ordinary_integer(ty: LLVMTypeRef) -> bool {
    unsafe { width(ty) }.is_some_and(|w| matches!(w, 1 | 8 | 16 | 32 | 64))
}
unsafe fn atomic(access: LLVMValueRef) -> bool {
    unsafe { LLVMGetOrdering(access) != LLVMAtomicOrdering::LLVMAtomicOrderingNotAtomic }
}

struct Lower {
    context: LLVMContextRef,
    values: HashMap<LLVMValueRef, LLVMValueRef>,
    erased: Vec<LLVMValueRef>,
    error: Option<String>,
}

const UNSUPPORTED: &str = "unsupported operation";

impl Lower {
    unsafe fn fail(&mut self, value: LLVMValueRef, reason: &str) -> Option<LLVMValueRef> {
        if self.error.is_none() {
            unsafe {
                let text = LLVMPrintValueToString(value);
                let printed = CStr::from_ptr(text).to_string_lossy().into_owned();
                LLVMDisposeMessage(text);
                self.error = Some(format!("odd integer promotion: {reason}: {printed}"));
            }
        }
        None
    }
    unsafe fn promoted(&self, ty: LLVMTypeRef) -> LLVMTypeRef {
        unsafe {
            if LLVMGetIntTypeWidth(ty) <= 24 {
                LLVMInt32TypeInContext(self.context)
            } else {
                LLVMInt64TypeInContext(self.context)
            }
        }
    }
    unsafe fn mask(&self, builder: &Builder, value: LLVMValueRef, width: u32) -> LLVMValueRef {
        unsafe {
            let low = LLVMConstInt(LLVMTypeOf(value), (1u64 << width) - 1, 0);
            LLVMBuildAnd(builder.0, value, low, c"odd.mask".as_ptr())
        }
    }
    unsafe fn signed(&self, builder: &Builder, value: LLVMValueRef, width: u32) -> LLVMValueRef {
        unsafe {
            let ty = LLVMTypeOf(value);
            let shift = LLVMGetIntTypeWidth(ty) - width;
            if shift == 0 {
                return value;
            }
            let count = LLVMConstInt(ty, shift.into(), 0);
            let shifted = LLVMBuildShl(builder.0, value, count, c"".as_ptr());
            LLVMBuildAShr(builder.0, shifted, count, c"odd.signed".as_ptr())
        }
    }
    // Convert a scalar operand to a chosen native width. Odd operands are
    // normalized by get; ordinary operands retain their actual signedness.
    unsafe fn convert(
        &mut self,
        builder: &Builder,
        mut value: LLVMValueRef,
        destination: LLVMTypeRef,
        sign: bool,
    ) -> Option<LLVMValueRef> {
        unsafe {
            let ty = LLVMTypeOf(value);
            if odd(ty) {
                let original = LLVMGetIntTypeWidth(ty);
                value = self.get(value)?;
                if sign {
                    value = self.signed(builder, value, original);
                }
            } else if !ordinary_integer(ty) {
                return self.fail(value, "unsupported cast source");
            }
            let from = LLVMGetIntTypeWidth(LLVMTypeOf(value));
            let to = LLVMGetIntTypeWidth(destination);
            Some(if from > to {
                LLVMBuildTrunc(builder.0, value, destination, c"".as_ptr())
            } else if from == to {
                value
            } else if sign {
                LLVMBuildSExt(builder.0, value, destination, c"".as_ptr())
            } else {
                LLVMBuildZExt(builder.0, value, destination, c"".as_ptr())
            })
        }
    }
    unsafe fn get(&mut self, value: LLVMValueRef) -> Option<LLVMValueRef> {
        unsafe {
            if let Some(&cached) = self.values.get(&value) {
                return Some(cached);
            }
            let original = LLVMTypeOf(value);
            if !odd(original) {
                return self.fail(value, "expected supported scalar width");
            }
            let width = LLVMGetIntTypeWidth(original);
            let ty = self.promoted(original);
            if !LLVMIsAConstantInt(value).is_null() {
                return Some(LLVMConstInt(ty, LLVMConstIntGetZExtValue(value), 0));
            }
            if !LLVMIsAPoisonValue(value).is_null() {
                return Some(LLVMGetPoison(ty));
            }
            // A concrete zero is a permitted refinement of undef, with no unknown
            // high bits leaking into comparisons or a zero extension.
            if !LLVMIsAUndefValue(value).is_null() {
                return Some(LLVMConstInt(ty, 0, 0));
            }
            if LLVMIsAInstruction(value).is_null() {
                return self.fail(value, "odd constant expression or argument");
            }
            let builder = Builder::before(self.context, value);
            let b = builder.0;
            let opcode = LLVMGetInstructionOpcode(value);
            if opcode == LLVMOpcode::LLVMPHI {
                let replacement = LLVMBuildPhi(b, ty, c"odd.phi".as_ptr());
                self.values.insert(value, replacement);
                self.erased.push(value);
                for i in 0..LLVMCountIncoming(value) {
                    let mut incoming = self.get(LLVMGetIncomingValue(value, i))?;
                    let mut block = LLVMGetIncomingBlock(value, i);
                    LLVMAddIncoming(replacement, &mut incoming, &mut block, 1);
                }
                return Some(replacement);
            }
            let int8 = LLVMInt8TypeInContext(self.context);
            let int64 = LLVMInt64TypeInContext(self.context);
            let mut result;
            if opcode == LLVMOpcode::LLVMLoad {
                if width < 8 {
                    return self.fail(value, "sub-byte load storage");
                }
                if atomic(value) || LLVMGetVolatile(value) != 0 {
                    return self.fail(value, "atomic or volatile access");
                }
                let base = LLVMGetOperand(value, 0);
                result = LLVMConstInt(ty, 0, 0);
                for byte in 0..width / 8 {
                    let mut index = LLVMConstInt(int64, byte.into(), 0);
                    let pointer = LLVMBuildGEP2(b, int8, base, &mut index, 1, c"".as_ptr());
                    let part = LLVMBuildLoad2(b, int8, pointer, c"".as_ptr());
                    LLVMSetAlignment(part, 1);
                    let wide = LLVMBuildZExt(b, part, ty, c"".as_ptr());
                    let count = LLVMConstInt(ty, (byte * 8).into(), 0);
                    let shifted = LLVMBuildShl(b, wide, count, c"".as_ptr());
                    result = LLVMBuildOr(b, result, shifted, c"".as_ptr());
                }
            } else if !LLVMIsACastInst(value).is_null() {
                if !matches!(
                    opcode,
                    LLVMOpcode::LLVMTrunc | LLVMOpcode::LLVMZExt | LLVMOpcode::LLVMSExt
                ) {
                    return self.fail(value, "unsupported cast");
                }
                let source = LLVMGetOperand(value, 0);
                result = self.convert(&builder, source, ty, opcode == LLVMOpcode::LLVMSExt)?;
                result = self.mask(&builder, result, width);
            } else if opcode == LLVMOpcode::LLVMSelect {
                let yes = self.get(LLVMGetOperand(value, 1))?;
                let no = self.get(LLVMGetOperand(value, 2))?;
                result = LLVMBuildSelect(b, LLVMGetOperand(value, 0), yes, no, c"".as_ptr());
            } else if opcode == LLVMOpcode::LLVMFreeze {
                let input = self.get(LLVMGetOperand(value, 0))?;
                result = self.mask(&builder, LLVMBuildFreeze(b, input, c"".as_ptr()), width);
            } else if !LLVMIsABinaryOperator(value).is_null() {
                let left = self.get(LLVMGetOperand(value, 0))?;
                let right = self.get(LLVMGetOperand(value, 1))?;
                result = match opcode {
                    LLVMOpcode::LLVMAdd => LLVMBuildAdd(b, left, right, c"".as_ptr()),
                    LLVMOpcode::LLVMSub => LLVMBuildSub(b, left, right, c"".as_ptr()),
                    LLVMOpcode::LLVMMul => LLVMBuildMul(b, left, right, c"".as_ptr()),
                    LLVMOpcode::LLVMAnd => LLVMBuildAnd(b, left, right, c"".as_ptr()),
                    LLVMOpcode::LLVMOr => LLVMBuildOr(b, left, right, c"".as_ptr()),
                    LLVMOpcode::LLVMXor => LLVMBuildXor(b, left, right, c"".as_ptr()),
                    LLVMOpcode::LLVMUDiv => LLVMBuildUDiv(b, left, right, c"".as_ptr()),
                    LLVMOpcode::LLVMURem => LLVMBuildURem(b, left, right, c"".as_ptr()),
                    LLVMOpcode::LLVMSDiv | LLVMOpcode::LLVMSRem => {
                        let x = self.signed(&builder, left, width);
                        let y = self.signed(&builder, right, width);
                        if opcode == LLVMOpcode::LLVMSRem {
                            LLVMBuildSRem(b, x, y, c"".as_ptr())
                        } else {
                            // Narrow signed-division overflow remains poison, even though
                            // the wider hardware quotient would fit its native integer.
                            let quotient = LLVMBuildSDiv(b, x, y, c"".as_ptr());
                            let minimum = LLVMConstInt(ty, 1u64 << (width - 1), 0);
                            let negative_one = LLVMConstInt(ty, (1u64 << width) - 1, 0);
                            let is_minimum = LLVMBuildICmp(
                                b,
                                LLVMIntPredicate::LLVMIntEQ,
                                left,
                                minimum,
                                c"".as_ptr(),
                            );
                            let is_negative_one = LLVMBuildICmp(
                                b,
                                LLVMIntPredicate::LLVMIntEQ,
                                right,
                                negative_one,
                                c"".as_ptr(),
                            );
                            let overflow =
                                LLVMBuildAnd(b, is_minimum, is_negative_one, c"".as_ptr());
                            LLVMBuildSelect(b, overflow, LLVMGetPoison(ty), quotient, c"".as_ptr())
                        }
                    }
                    LLVMOpcode::LLVMShl | LLVMOpcode::LLVMLShr | LLVMOpcode::LLVMAShr => {
                        let limit = LLVMConstInt(ty, (LLVMGetIntTypeWidth(ty) - 1).into(), 0);
                        let safe = LLVMBuildAnd(b, right, limit, c"".as_ptr());
                        let shifted = match opcode {
                            LLVMOpcode::LLVMShl => LLVMBuildShl(b, left, safe, c"".as_ptr()),
                            LLVMOpcode::LLVMLShr => LLVMBuildLShr(b, left, safe, c"".as_ptr()),
                            _ => {
                                let x = self.signed(&builder, left, width);
                                LLVMBuildAShr(b, x, safe, c"".as_ptr())
                            }
                        };
                        let valid = LLVMBuildICmp(
                            b,
                            LLVMIntPredicate::LLVMIntULT,
                            right,
                            LLVMConstInt(ty, width.into(), 0),
                            c"".as_ptr(),
                        );
                        LLVMBuildSelect(b, valid, shifted, LLVMGetPoison(ty), c"".as_ptr())
                    }
                    _ => return self.fail(value, UNSUPPORTED),
                };
                // Do not propagate nuw/nsw/exact onto a changed width. Masking keeps
                // every defined original result, without strengthening its contract.
                result = self.mask(&builder, result, width);
            } else {
                return self.fail(value, UNSUPPORTED);
            }
            self.values.insert(value, result);
            self.erased.push(value);
            Some(result)
        }
    }

    unsafe fn check_type(
        &mut self,
        layout: LLVMTargetDataRef,
        instruction: LLVMValueRef,
        ty: LLVMTypeRef,
    ) -> bool {
        unsafe {
            if !odd(ty) {
                if contains_odd(ty) {
                    self.fail(instruction, "odd vector or aggregate");
                    return false;
                }
                return true;
            }
            let width = LLVMGetIntTypeWidth(ty);
            if width < 8 {
                return true; // Register-only: no storage layout is changed.
            }
            let allocation = if width == 24 { 4 } else { 8 };
            if LLVMByteOrder(layout) != LLVMByteOrdering::LLVMLittleEndian
                || LLVMABISizeOfType(layout, ty) != allocation
                || u64::from(LLVMABIAlignmentOfType(layout, ty)) != allocation
            {
                self.fail(instruction, "NVPTX/AIR layout mismatch");
                return false;
            }
            true
        }
    }

    unsafe fn run(&mut self, module: LLVMModuleRef) -> bool {
        unsafe {
            let layout = LLVMGetModuleDataLayout(module);
            let mut original = Vec::new();
            let mut global = LLVMGetFirstGlobal(module);
            while !global.is_null() {
                if contains_odd(LLVMGlobalGetValueType(global)) {
                    self.fail(global, "odd global storage");
                    return false;
                }
                global = LLVMGetNextGlobal(global);
            }
            let mut function = LLVMGetFirstFunction(module);
            while !function.is_null() {
                if contains_odd(LLVMGetReturnType(LLVMGlobalGetValueType(function))) {
                    self.fail(function, "odd function ABI");
                    return false;
                }
                for i in 0..LLVMCountParams(function) {
                    let argument = LLVMGetParam(function, i);
                    if contains_odd(LLVMTypeOf(argument)) {
                        self.fail(argument, "odd function ABI");
                        return false;
                    }
                }
                let mut block = LLVMGetFirstBasicBlock(function);
                while !block.is_null() {
                    let mut instruction = LLVMGetFirstInstruction(block);
                    while !instruction.is_null() {
                        if !self.admit(layout, instruction) {
                            return false;
                        }
                        original.push(instruction);
                        instruction = LLVMGetNextInstruction(instruction);
                    }
                    block = LLVMGetNextBasicBlock(block);
                }
                function = LLVMGetNextFunction(function);
            }
            for instruction in original {
                if !self.rewrite(instruction) {
                    return false;
                }
            }
            let removing: HashSet<LLVMValueRef> = self.erased.iter().copied().collect();
            for &instruction in &self.erased.clone() {
                if !odd(LLVMTypeOf(instruction)) {
                    continue;
                }
                let mut use_ = LLVMGetFirstUse(instruction);
                while !use_.is_null() {
                    let user = LLVMGetUser(use_);
                    if LLVMIsAInstruction(user).is_null() || !removing.contains(&user) {
                        self.fail(user, "unsupported consumer");
                        return false;
                    }
                    use_ = LLVMGetNextUse(use_);
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

    // Refuse every storage, ABI and aggregate shape before any rewriting.
    unsafe fn admit(&mut self, layout: LLVMTargetDataRef, instruction: LLVMValueRef) -> bool {
        unsafe {
            let ty = LLVMTypeOf(instruction);
            if contains_odd(ty) && !odd(ty) {
                self.fail(instruction, "odd vector or aggregate");
                return false;
            }
            let subbyte = |ty| odd(ty) && LLVMGetIntTypeWidth(ty) < 8;
            match LLVMGetInstructionOpcode(instruction) {
                LLVMOpcode::LLVMAlloca if contains_odd(LLVMGetAllocatedType(instruction)) => {
                    self.fail(instruction, "odd stack storage");
                    return false;
                }
                LLVMOpcode::LLVMGetElementPtr
                    if contains_odd(LLVMGetGEPSourceElementType(instruction)) =>
                {
                    self.fail(instruction, "odd GEP element storage");
                    return false;
                }
                LLVMOpcode::LLVMLoad if subbyte(ty) => {
                    self.fail(instruction, "sub-byte load storage");
                    return false;
                }
                LLVMOpcode::LLVMStore if subbyte(LLVMTypeOf(LLVMGetOperand(instruction, 0))) => {
                    self.fail(instruction, "sub-byte store storage");
                    return false;
                }
                _ => {}
            }
            if !self.check_type(layout, instruction, ty) {
                return false;
            }
            for i in 0..LLVMGetNumOperands(instruction) {
                let operand = LLVMTypeOf(LLVMGetOperand(instruction, i as u32));
                if !self.check_type(layout, instruction, operand) {
                    return false;
                }
            }
            true
        }
    }

    unsafe fn rewrite(&mut self, instruction: LLVMValueRef) -> bool {
        unsafe {
            if odd(LLVMTypeOf(instruction)) {
                return self.get(instruction).is_some();
            }
            let opcode = LLVMGetInstructionOpcode(instruction);
            if opcode != LLVMOpcode::LLVMICmp
                && opcode != LLVMOpcode::LLVMStore
                && LLVMIsACastInst(instruction).is_null()
            {
                return true;
            }
            let operand = LLVMGetOperand(instruction, 0);
            let source = LLVMTypeOf(operand);
            if !odd(source) {
                return true;
            }
            let width = LLVMGetIntTypeWidth(source);
            if opcode == LLVMOpcode::LLVMICmp {
                let Some(mut a) = self.get(operand) else {
                    return false;
                };
                let Some(mut b) = self.get(LLVMGetOperand(instruction, 1)) else {
                    return false;
                };
                let builder = Builder::before(self.context, instruction);
                let predicate = LLVMGetICmpPredicate(instruction);
                if matches!(
                    predicate,
                    LLVMIntPredicate::LLVMIntSGT
                        | LLVMIntPredicate::LLVMIntSGE
                        | LLVMIntPredicate::LLVMIntSLT
                        | LLVMIntPredicate::LLVMIntSLE
                ) {
                    a = self.signed(&builder, a, width);
                    b = self.signed(&builder, b, width);
                }
                let replacement = LLVMBuildICmp(builder.0, predicate, a, b, c"".as_ptr());
                LLVMReplaceAllUsesWith(instruction, replacement);
            } else if opcode == LLVMOpcode::LLVMStore {
                if atomic(instruction) || LLVMGetVolatile(instruction) != 0 {
                    self.fail(instruction, "atomic or volatile access");
                    return false;
                }
                let Some(value) = self.get(operand) else {
                    return false;
                };
                let builder = Builder::before(self.context, instruction);
                let b = builder.0;
                let int8 = LLVMInt8TypeInContext(self.context);
                let int64 = LLVMInt64TypeInContext(self.context);
                let base = LLVMGetOperand(instruction, 1);
                for byte in 0..width / 8 {
                    let mut index = LLVMConstInt(int64, byte.into(), 0);
                    let pointer = LLVMBuildGEP2(b, int8, base, &mut index, 1, c"".as_ptr());
                    let count = LLVMConstInt(LLVMTypeOf(value), (byte * 8).into(), 0);
                    let shifted = LLVMBuildLShr(b, value, count, c"".as_ptr());
                    let part = LLVMBuildTrunc(b, shifted, int8, c"".as_ptr());
                    LLVMSetAlignment(LLVMBuildStore(b, part, pointer), 1);
                }
            } else {
                let destination = LLVMTypeOf(instruction);
                if !ordinary_integer(destination)
                    || !matches!(
                        opcode,
                        LLVMOpcode::LLVMTrunc | LLVMOpcode::LLVMZExt | LLVMOpcode::LLVMSExt
                    )
                {
                    self.fail(instruction, "unsupported cast consumer");
                    return false;
                }
                let builder = Builder::before(self.context, instruction);
                let sign = opcode == LLVMOpcode::LLVMSExt;
                let Some(replacement) = self.convert(&builder, operand, destination, sign) else {
                    return false;
                };
                LLVMReplaceAllUsesWith(instruction, replacement);
            }
            self.erased.push(instruction);
            true
        }
    }
}
