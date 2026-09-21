//! Reading LLVM IR through the C API: the walks and questions the stages share.
//!
//! Every function here is `unsafe` for one reason: its arguments must be live
//! values of the module being compiled. The stages uphold that by collecting
//! what they will change before they change it.
use inkwell::llvm_sys::{LLVMOpcode, LLVMTypeKind, core::*, prelude::*};
use std::ffi::CStr;

pub type Value = LLVMValueRef;

pub unsafe fn functions(module: LLVMModuleRef) -> Vec<Value> {
    unsafe { chain(LLVMGetFirstFunction(module), |f| LLVMGetNextFunction(f)) }
}

pub unsafe fn globals(module: LLVMModuleRef) -> Vec<Value> {
    unsafe { chain(LLVMGetFirstGlobal(module), |g| LLVMGetNextGlobal(g)) }
}

pub unsafe fn blocks(function: Value) -> Vec<LLVMBasicBlockRef> {
    unsafe {
        chain(LLVMGetFirstBasicBlock(function), |b| {
            LLVMGetNextBasicBlock(b)
        })
    }
}

pub unsafe fn instructions(function: Value) -> Vec<Value> {
    unsafe {
        blocks(function)
            .into_iter()
            .flat_map(|block| {
                chain(LLVMGetFirstInstruction(block), |i| {
                    LLVMGetNextInstruction(i)
                })
            })
            .collect()
    }
}

pub unsafe fn operands(value: Value) -> Vec<Value> {
    unsafe {
        (0..LLVMGetNumOperands(value) as u32)
            .map(|i| LLVMGetOperand(value, i))
            .collect()
    }
}

pub unsafe fn users(value: Value) -> Vec<Value> {
    unsafe {
        chain(LLVMGetFirstUse(value), |u| LLVMGetNextUse(u))
            .into_iter()
            .map(|u| LLVMGetUser(u))
            .collect()
    }
}

/// Follow a C-API linked list from its first element to null.
unsafe fn chain<T>(first: *mut T, next: impl Fn(*mut T) -> *mut T) -> Vec<*mut T> {
    let mut all = Vec::new();
    let mut item = first;
    while !item.is_null() {
        all.push(item);
        item = next(item);
    }
    all
}

pub unsafe fn name(value: Value) -> String {
    let mut length = 0;
    unsafe {
        let start = LLVMGetValueName2(value, &mut length);
        String::from_utf8_lossy(std::slice::from_raw_parts(start.cast(), length)).into_owned()
    }
}

pub unsafe fn defined(function: Value) -> bool {
    unsafe { LLVMIsDeclaration(function) == 0 }
}

pub unsafe fn opcode(instruction: Value) -> LLVMOpcode {
    unsafe { LLVMGetInstructionOpcode(instruction) }
}

pub unsafe fn is_call(instruction: Value) -> bool {
    unsafe { !LLVMIsACallInst(instruction).is_null() }
}

/// The function a call names directly, if it names one.
pub unsafe fn callee(call: Value) -> Option<Value> {
    unsafe {
        let target = LLVMIsAFunction(LLVMGetCalledValue(call));
        (!target.is_null()).then_some(target)
    }
}

pub unsafe fn has_attribute(function: Value, attribute: &CStr) -> bool {
    unsafe {
        let kind = LLVMGetEnumAttributeKindForName(attribute.as_ptr(), attribute.count_bytes());
        !LLVMGetEnumAttributeAtIndex(
            function,
            inkwell::llvm_sys::LLVMAttributeFunctionIndex,
            kind,
        )
        .is_null()
    }
}

pub unsafe fn is_pointer(ty: LLVMTypeRef) -> bool {
    unsafe { LLVMGetTypeKind(ty) == LLVMTypeKind::LLVMPointerTypeKind }
}

/// `file:line` of the Rust source an instruction came from, when the kernel
/// was built with line tables.
pub unsafe fn location(instruction: Value) -> Option<String> {
    unsafe {
        let mut length = 0;
        let file = LLVMGetDebugLocFilename(instruction, &mut length);
        if file.is_null() || length == 0 {
            return None;
        }
        let file =
            String::from_utf8_lossy(std::slice::from_raw_parts(file.cast(), length as usize));
        Some(format!("{file}:{}", LLVMGetDebugLocLine(instruction)))
    }
}

/// Erase an instruction whose result, if any, is only used by code that is
/// itself about to be erased.
pub unsafe fn erase(instruction: Value) {
    unsafe {
        let ty = LLVMTypeOf(instruction);
        if LLVMGetTypeKind(ty) != LLVMTypeKind::LLVMVoidTypeKind {
            LLVMReplaceAllUsesWith(instruction, LLVMGetPoison(ty));
        }
        LLVMInstructionEraseFromParent(instruction);
    }
}
