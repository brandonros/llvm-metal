//! Specialize internal pointer-parameter helpers by the caller's concrete AIR
//! address spaces. No generic pointer is allowed to cross a retained call.
use crate::{
    address_spaces::infer_function,
    pointer_provenance::{private_pointer, underlying_objects},
};
use inkwell::{
    llvm_sys::{
        LLVMAttributeFunctionIndex, LLVMLinkage, LLVMOpcode, LLVMTypeKind,
        core::*,
        debuginfo::{LLVMInstructionGetDebugLoc, LLVMInstructionSetDebugLoc},
        prelude::*,
    },
    module::Module,
};
use std::collections::{HashMap, HashSet};

const RETAINED: &str = "llvm-metal.retained";
const LIMIT: usize = 4096;

unsafe extern "C" {
    fn LLVMExtCloneFunctionInto(
        to: LLVMValueRef,
        from: LLVMValueRef,
        keys: *mut LLVMValueRef,
        mapped: *mut LLVMValueRef,
        count: u32,
    ) -> LLVMBasicBlockRef;
}

unsafe fn pointer(value: LLVMValueRef) -> bool {
    unsafe { LLVMGetTypeKind(LLVMTypeOf(value)) == LLVMTypeKind::LLVMPointerTypeKind }
}
unsafe fn space(value: LLVMValueRef) -> u32 {
    unsafe { LLVMGetPointerAddressSpace(LLVMTypeOf(value)) }
}
unsafe fn opcode(value: LLVMValueRef) -> Option<LLVMOpcode> {
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
unsafe fn name(value: LLVMValueRef) -> String {
    unsafe {
        let mut length = 0;
        let text = LLVMGetValueName2(value, &mut length);
        String::from_utf8_lossy(std::slice::from_raw_parts(text.cast(), length)).into_owned()
    }
}
unsafe fn printed(value: LLVMValueRef) -> String {
    unsafe {
        let text = LLVMPrintValueToString(value);
        let printed = std::ffi::CStr::from_ptr(text)
            .to_string_lossy()
            .into_owned();
        LLVMDisposeMessage(text);
        printed
    }
}

// The argument as the caller can prove it: a device or constant pointer, or
// private storage. Anything else has no concrete Metal address space.
unsafe fn concrete(mut value: LLVMValueRef, builder: LLVMBuilderRef) -> Option<LLVMValueRef> {
    unsafe {
        if opcode(value) == Some(LLVMOpcode::LLVMAddrSpaceCast) {
            value = LLVMGetOperand(value, 0);
        }
        if opcode(value) == Some(LLVMOpcode::LLVMGetElementPtr) {
            let base = concrete(LLVMGetOperand(value, 0), builder)?;
            if space(base) != space(value) {
                let mut indices: Vec<_> = (1..LLVMGetNumOperands(value) as u32)
                    .map(|i| LLVMGetOperand(value, i))
                    .collect();
                // Drop optional no-wrap/inbounds facts rather than inventing target facts.
                return Some(LLVMBuildGEP2(
                    builder,
                    LLVMGetGEPSourceElementType(value),
                    base,
                    indices.as_mut_ptr(),
                    indices.len() as u32,
                    c"typed.gep".as_ptr(),
                ));
            }
        }
        match space(value) {
            1 | 2 => Some(value),
            0 => {
                let objects = underlying_objects(value);
                (!objects.is_empty()
                    && objects.iter().all(|&object| {
                        !LLVMIsAConstantPointerNull(object).is_null() || private_pointer(object)
                    }))
                .then_some(value)
            }
            _ => None,
        }
    }
}

// A direct call whose callee type matches, as CallBase::getCalledFunction.
unsafe fn called_function(call: LLVMValueRef) -> Option<LLVMValueRef> {
    unsafe {
        let callee = LLVMIsAFunction(LLVMGetCalledValue(call));
        (!callee.is_null() && LLVMGetCalledFunctionType(call) == LLVMGlobalGetValueType(callee))
            .then_some(callee)
    }
}

unsafe fn copy_call_attributes(from: LLVMValueRef, to: LLVMValueRef) {
    unsafe {
        let indices =
            std::iter::once(LLVMAttributeFunctionIndex).chain(0..=LLVMGetNumArgOperands(from));
        for index in indices {
            let count = LLVMGetCallSiteAttributeCount(from, index);
            let mut attributes = vec![std::ptr::null_mut(); count as usize];
            LLVMGetCallSiteAttributes(from, index, attributes.as_mut_ptr());
            for attribute in attributes {
                LLVMAddCallSiteAttribute(to, index, attribute);
            }
        }
    }
}

unsafe fn specialized(
    module: LLVMModuleRef,
    builder: LLVMBuilderRef,
    callee: LLVMValueRef,
    arguments: &[LLVMValueRef],
    spaces: &[u32],
    shim: bool,
) -> Result<LLVMValueRef, String> {
    unsafe {
        let mut types: Vec<_> = arguments.iter().map(|&a| LLVMTypeOf(a)).collect();
        let original = LLVMGlobalGetValueType(callee);
        let ty = LLVMFunctionType(
            LLVMGetReturnType(original),
            types.as_mut_ptr(),
            types.len() as u32,
            0,
        );
        let suffix: String = spaces.iter().map(|space| format!(".{space}")).collect();
        let symbol = std::ffi::CString::new(format!("{}.metal{suffix}", name(callee)))
            .expect("LLVM names contain no NUL");
        let function = LLVMAddFunction(module, symbol.as_ptr(), ty);
        LLVMSetLinkage(function, LLVMLinkage::LLVMInternalLinkage);
        LLVMSetFunctionCallConv(function, LLVMGetFunctionCallConv(callee));
        let context = LLVMGetModuleContext(module);
        let prelude = LLVMAppendBasicBlockInContext(context, function, c"typed.args".as_ptr());
        LLVMPositionBuilderAtEnd(builder, prelude);
        let (mut keys, mut mapped) = (Vec::new(), Vec::new());
        for i in 0..LLVMCountParams(callee) {
            let (old, new) = (LLVMGetParam(callee, i), LLVMGetParam(function, i));
            let mut length = 0;
            let text = LLVMGetValueName2(old, &mut length);
            LLVMSetValueName2(new, text, length);
            keys.push(old);
            mapped.push(if LLVMTypeOf(new) == LLVMTypeOf(old) {
                new
            } else {
                LLVMBuildAddrSpaceCast(builder, new, LLVMTypeOf(old), c"generic.arg".as_ptr())
            });
        }
        let entry = if shim {
            LLVMExtCloneFunctionInto(
                function,
                callee,
                keys.as_mut_ptr(),
                mapped.as_mut_ptr(),
                keys.len() as u32,
            )
        } else {
            crate::clone::function_into(function, callee, keys.into_iter().zip(mapped).collect())?
        };
        LLVMPositionBuilderAtEnd(builder, prelude);
        LLVMBuildBr(builder, entry);
        Ok(function)
    }
}

pub(crate) fn run(module: &Module<'_>, entry: &str, shim: bool) -> Result<(), String> {
    let entry = std::ffi::CString::new(entry).map_err(|e| e.to_string())?;
    // SAFETY: a verified, exclusively owned module. Calls are collected per
    // caller before any is replaced, and a replaced call is erased only after
    // its uses moved to the specialized call.
    unsafe {
        let raw = module.as_mut_ptr();
        let root = LLVMGetNamedFunction(raw, entry.as_ptr());
        if root.is_null() {
            return Err("missing specialization entry".into());
        }
        let builder = LLVMCreateBuilderInContext(LLVMGetModuleContext(raw));
        let result = specialize(raw, builder, root, shim);
        LLVMDisposeBuilder(builder);
        result
    }
}

unsafe fn specialize(
    module: LLVMModuleRef,
    builder: LLVMBuilderRef,
    root: LLVMValueRef,
    shim: bool,
) -> Result<(), String> {
    unsafe {
        let mut cache: HashMap<(LLVMValueRef, Vec<u32>), LLVMValueRef> = HashMap::new();
        let mut pending = vec![root];
        let mut visited = HashSet::new();
        while let Some(caller) = pending.pop() {
            if !visited.insert(caller) {
                continue;
            }
            // Make casts/GEPs concrete before classifying this caller's nested calls.
            infer_function(caller);
            let mut calls = Vec::new();
            let mut block = LLVMGetFirstBasicBlock(caller);
            while !block.is_null() {
                let mut instruction = LLVMGetFirstInstruction(block);
                while !instruction.is_null() {
                    if !LLVMIsACallInst(instruction).is_null() {
                        calls.push(instruction);
                    }
                    instruction = LLVMGetNextInstruction(instruction);
                }
                block = LLVMGetNextBasicBlock(block);
            }
            for call in calls {
                let Some(callee) = called_function(call) else {
                    continue;
                };
                if LLVMCountBasicBlocks(callee) == 0 {
                    continue;
                }
                let retained = LLVMGetStringAttributeAtIndex(
                    callee,
                    LLVMAttributeFunctionIndex,
                    RETAINED.as_ptr().cast(),
                    RETAINED.len() as u32,
                );
                if retained.is_null() {
                    return Err("unsupported helper survived required inlining".into());
                }
                LLVMPositionBuilderBefore(builder, call);
                let mut arguments = Vec::new();
                let mut spaces = Vec::new();
                for i in 0..LLVMGetNumArgOperands(call) {
                    let mut value = LLVMGetOperand(call, i);
                    if pointer(value) {
                        value = concrete(value, builder).ok_or_else(|| {
                            format!(
                                "unresolved pointer flow at retained helper call in {}: {}",
                                name(caller),
                                printed(call)
                            )
                        })?;
                        spaces.push(space(value));
                    }
                    arguments.push(value);
                }
                if spaces.is_empty() {
                    pending.push(callee);
                    continue;
                }
                let key = (callee, spaces);
                let target = match cache.get(&key) {
                    Some(&existing) => existing,
                    None => {
                        if cache.len() >= LIMIT {
                            return Err("retained helper specialization limit exceeded".into());
                        }
                        let new = specialized(module, builder, callee, &arguments, &key.1, shim)?;
                        cache.insert(key, new);
                        new
                    }
                };
                LLVMPositionBuilderBefore(builder, call);
                let replacement = LLVMBuildCall2(
                    builder,
                    LLVMGlobalGetValueType(target),
                    target,
                    arguments.as_mut_ptr(),
                    arguments.len() as u32,
                    c"".as_ptr(),
                );
                LLVMSetInstructionCallConv(replacement, LLVMGetInstructionCallConv(call));
                copy_call_attributes(call, replacement);
                LLVMInstructionSetDebugLoc(replacement, LLVMInstructionGetDebugLoc(call));
                LLVMReplaceAllUsesWith(call, replacement);
                LLVMInstructionEraseFromParent(call);
                pending.push(target);
            }
        }
        Ok(())
    }
}
