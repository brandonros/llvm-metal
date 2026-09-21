//! Reduce the linked crates to the program one kernel runs: give panics their
//! device meaning, then keep only what the entry reaches. Neither step is an
//! optimization: both are exact, and neither looks at what the code computes.
use crate::ir::{self, Value};
use inkwell::llvm_sys::{core::*, prelude::*};
use std::collections::HashSet;

/// The operation a panic becomes. `lower` implements it.
pub const PANIC: &std::ffi::CStr = c"llvm_metal.panic";

/// Rust reports a panic by calling a function that never returns: every
/// undefined `noreturn` function is one, since libcore is not linked. A GPU has
/// nowhere to unwind to, so the call becomes `llvm_metal.panic()` and the
/// function returns zero to its caller.
///
/// OPEN (#30): the caller carries on with that zero. Device memory stays safe,
/// because every access through a handle is checked, but a zeroed reference
/// dereferenced in thread memory is not. Closing this needs the panic to
/// propagate: each call that can panic followed by a check and a return.
pub unsafe fn rewrite_panics(module: LLVMModuleRef) {
    unsafe {
        let context = LLVMGetModuleContext(module);
        let signature = LLVMFunctionType(LLVMVoidTypeInContext(context), [].as_mut_ptr(), 0, 0);
        let panic = LLVMAddFunction(module, PANIC.as_ptr(), signature);
        let builder = LLVMCreateBuilderInContext(context);
        for function in ir::functions(module) {
            if ir::defined(function)
                || function == panic
                || !ir::has_attribute(function, c"noreturn")
            {
                continue;
            }
            let calls = ir::users(function)
                .into_iter()
                .filter(|&user| ir::is_call(user) && ir::callee(user) == Some(function));
            for call in calls.collect::<Vec<_>>() {
                LLVMPositionBuilderBefore(builder, call);
                LLVMBuildCall2(builder, signature, panic, [].as_mut_ptr(), 0, c"".as_ptr());
                let caller = LLVMGetBasicBlockParent(LLVMGetInstructionParent(call));
                let returned = LLVMGetReturnType(LLVMGlobalGetValueType(caller));
                if returned == LLVMVoidTypeInContext(context) {
                    LLVMBuildRetVoid(builder);
                } else {
                    LLVMBuildRet(builder, LLVMConstNull(returned));
                }
                // The call and the `unreachable` after it are now dead code
                // behind a terminator.
                let mut dead = call;
                while !dead.is_null() {
                    let next = LLVMGetNextInstruction(dead);
                    ir::erase(dead);
                    dead = next;
                }
            }
        }
        LLVMDisposeBuilder(builder);
    }
}

/// Delete every function and global the entry does not reach.
pub unsafe fn keep_reachable(module: LLVMModuleRef, entry: Value) {
    unsafe {
        let mut reached = HashSet::new();
        let mut pending = vec![entry];
        while let Some(value) = pending.pop() {
            if !reached.insert(value) {
                continue;
            }
            if !LLVMIsAFunction(value).is_null() {
                for instruction in ir::instructions(value) {
                    pending.extend(
                        ir::operands(instruction)
                            .into_iter()
                            .filter(|&o| constant(o)),
                    );
                }
            } else if !LLVMIsAGlobalVariable(value).is_null() {
                let initializer = LLVMGetInitializer(value);
                if !initializer.is_null() {
                    pending.push(initializer);
                }
            } else {
                pending.extend(ir::operands(value).into_iter().filter(|&o| constant(o)));
            }
        }
        let dead_functions: Vec<_> = ir::functions(module)
            .into_iter()
            .filter(|f| !reached.contains(f))
            .collect();
        let dead_globals: Vec<_> = ir::globals(module)
            .into_iter()
            .filter(|g| !reached.contains(g))
            .collect();
        // Dead code may still refer to other dead code. Empty everything first,
        // then delete what is left without uses.
        for &function in &dead_functions {
            for instruction in ir::instructions(function).into_iter().rev() {
                ir::erase(instruction);
            }
            for block in ir::blocks(function) {
                LLVMDeleteBasicBlock(block);
            }
        }
        for &global in &dead_globals {
            LLVMSetInitializer(global, std::ptr::null_mut());
        }
        for function in dead_functions {
            LLVMDeleteFunction(function);
        }
        for global in dead_globals {
            LLVMDeleteGlobal(global);
        }
    }
}

/// Constants are where references to functions and globals hide.
unsafe fn constant(value: Value) -> bool {
    unsafe { !LLVMIsAConstant(value).is_null() }
}
