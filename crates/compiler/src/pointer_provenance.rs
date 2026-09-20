//! Prove private storage or defined immutable global storage.
//! Unknown provenance, escaping helpers and unresolved recursion fail closed.
use inkwell::llvm_sys::{LLVMLinkage, LLVMOpcode, LLVMTypeKind, core::*, prelude::*};
use std::collections::HashSet;

// LLVM's own underlying-object walk gives up after this many steps. Keeping the
// bound keeps the set of admitted programs the same as the reviewed profile.
const MAX_LOOKUP: usize = 6;

unsafe fn pointer(value: LLVMValueRef) -> bool {
    unsafe { LLVMGetTypeKind(LLVMTypeOf(value)) == LLVMTypeKind::LLVMPointerTypeKind }
}

// Opcode of an instruction or constant expression, as llvm::Operator sees it.
unsafe fn operator(value: LLVMValueRef) -> Option<LLVMOpcode> {
    unsafe {
        if !LLVMIsAInstruction(value).is_null() {
            Some(LLVMGetInstructionOpcode(value))
        } else if !LLVMIsAConstantExpr(value).is_null() {
            Some(LLVMGetConstOpcode(value))
        } else {
            None
        }
    }
}

/// Strip GEPs, pointer casts and single-entry PHIs. Aliases and calls are not
/// followed: a call result or alias is never treated as its argument or aliasee.
pub(crate) unsafe fn underlying_object(mut value: LLVMValueRef) -> LLVMValueRef {
    unsafe {
        for _ in 0..MAX_LOOKUP {
            let base = match operator(value) {
                Some(
                    LLVMOpcode::LLVMGetElementPtr
                    | LLVMOpcode::LLVMBitCast
                    | LLVMOpcode::LLVMAddrSpaceCast,
                ) => LLVMGetOperand(value, 0),
                Some(LLVMOpcode::LLVMPHI) if LLVMCountIncoming(value) == 1 => {
                    value = LLVMGetIncomingValue(value, 0);
                    continue;
                }
                _ => return value,
            };
            // Only a scalar pointer base is followed.
            if !pointer(base) {
                return value;
            }
            value = base;
        }
        value
    }
}

/// Every object a pointer may be based on, looking through selects and PHIs.
pub(crate) unsafe fn underlying_objects(value: LLVMValueRef) -> Vec<LLVMValueRef> {
    unsafe {
        let mut objects = Vec::new();
        let mut visited = HashSet::new();
        let mut pending = vec![value];
        while let Some(next) = pending.pop() {
            let object = underlying_object(next);
            if !visited.insert(object) {
                continue;
            }
            match operator(object) {
                Some(LLVMOpcode::LLVMSelect) if !LLVMIsAInstruction(object).is_null() => {
                    pending.push(LLVMGetOperand(object, 1));
                    pending.push(LLVMGetOperand(object, 2));
                }
                Some(LLVMOpcode::LLVMPHI) => {
                    pending.extend(
                        (0..LLVMCountIncoming(object)).map(|i| LLVMGetIncomingValue(object, i)),
                    );
                }
                _ => objects.push(object),
            }
        }
        objects
    }
}

unsafe fn call(value: LLVMValueRef) -> bool {
    unsafe {
        !LLVMIsAInstruction(value).is_null()
            && matches!(
                LLVMGetInstructionOpcode(value),
                LLVMOpcode::LLVMCall | LLVMOpcode::LLVMInvoke | LLVMOpcode::LLVMCallBr
            )
    }
}

unsafe fn private(pointer: LLVMValueRef, visiting: &mut HashSet<LLVMValueRef>) -> bool {
    unsafe {
        let pointer = underlying_object(pointer);
        if !LLVMIsAAllocaInst(pointer).is_null() {
            return true;
        }
        if !LLVMIsAPHINode(pointer).is_null() || !LLVMIsASelectInst(pointer).is_null() {
            if !visiting.insert(pointer) {
                return false;
            }
            let objects = underlying_objects(pointer);
            let valid = !objects.is_empty() && objects.iter().all(|&o| private(o, visiting));
            visiting.remove(&pointer);
            return valid;
        }
        if LLVMIsAArgument(pointer).is_null() {
            return false;
        }
        let function = LLVMGetParamParent(pointer);
        if !matches!(
            LLVMGetLinkage(function),
            LLVMLinkage::LLVMInternalLinkage | LLVMLinkage::LLVMPrivateLinkage
        ) || !visiting.insert(pointer)
        {
            return false;
        }
        // A helper's pointer is private only when every use is a direct call and
        // every caller supplies private storage. Unknown callers/recursion fail closed.
        let Some(index) =
            (0..LLVMCountParams(function)).find(|&i| LLVMGetParam(function, i) == pointer)
        else {
            return false;
        };
        let (mut called, mut valid) = (false, true);
        let mut use_ = LLVMGetFirstUse(function);
        while !use_.is_null() {
            let user = LLVMGetUser(use_);
            if !call(user)
                || LLVMGetOperandUse(user, LLVMGetNumOperands(user) as u32 - 1) != use_
                || LLVMGetCalledFunctionType(user) != LLVMGlobalGetValueType(function)
                || index >= LLVMGetNumArgOperands(user)
                || !private(LLVMGetOperand(user, index), visiting)
            {
                valid = false;
                break;
            }
            called = true;
            use_ = LLVMGetNextUse(use_);
        }
        visiting.remove(&pointer);
        called && valid
    }
}

pub(crate) unsafe fn private_pointer(pointer: LLVMValueRef) -> bool {
    unsafe { private(pointer, &mut HashSet::new()) }
}

pub(crate) unsafe fn constant_pointer(pointer: LLVMValueRef) -> bool {
    unsafe {
        let global = LLVMIsAGlobalVariable(underlying_object(pointer));
        !global.is_null()
            && LLVMIsGlobalConstant(global) != 0
            && !LLVMGetInitializer(global).is_null()
            && LLVMIsExternallyInitialized(global) == 0
    }
}

/// Instruction-level predicate: a load from, or store to, proven storage.
pub(crate) unsafe fn private_memory(instruction: LLVMValueRef) -> bool {
    unsafe {
        if !LLVMIsALoadInst(instruction).is_null() {
            let source = LLVMGetOperand(instruction, 0);
            private_pointer(source) || constant_pointer(source)
        } else if !LLVMIsAStoreInst(instruction).is_null() {
            private_pointer(LLVMGetOperand(instruction, 1))
        } else {
            false
        }
    }
}
