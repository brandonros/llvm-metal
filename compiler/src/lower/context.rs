//! Every function receives the launch. A kernel names its buffers, its thread
//! and its constants as if they were ambient; on the GPU they are arguments of
//! the entry point. So every function gains the same trailing parameters, every
//! call passes them on, and the device operations become ordinary instructions
//! on them. Giving all functions all of it needs no analysis of who uses what.
//!
//! Pre: calls are direct and slots are constants (`Rule::DirectCall`,
//! `Rule::ConstantSlot`). Post: no call to a device operation remains.
use crate::ir::{self, Value};
use inkwell::llvm_sys::{core::*, prelude::*};
use std::collections::HashMap;

/// The values a function body may use: parameters, in `types` order.
pub struct Launch {
    pub buffers: Vec<Value>,
    pub lengths: Value,
    pub status: Value,
    pub thread: Value,
    pub constants: Value,
}

/// `buffers` device pointers, the buffer lengths, the status word, the thread
/// index, and the thread's copy of the constants.
pub unsafe fn types(context: LLVMContextRef, buffers: usize) -> Vec<LLVMTypeRef> {
    unsafe {
        let device = LLVMPointerTypeInContext(context, 1);
        let mut types = vec![device; buffers + 2];
        types.push(LLVMInt32TypeInContext(context));
        types.push(LLVMPointerTypeInContext(context, 0));
        types
    }
}

pub unsafe fn launch(parameters: &[Value], buffers: usize) -> Launch {
    Launch {
        buffers: parameters[..buffers].to_vec(),
        lengths: parameters[buffers],
        status: parameters[buffers + 1],
        thread: parameters[buffers + 2],
        constants: parameters[buffers + 3],
    }
}

/// Replace each defined function by one that also takes the launch, and return
/// the replacements with the launch each sees. Attributes are not carried over:
/// both sides of every call are ours, so none is needed for correctness, and
/// the ones rustc chose for another target mean nothing here.
pub unsafe fn extend(module: LLVMModuleRef, buffers: usize) -> HashMap<Value, (Value, Launch)> {
    unsafe {
        let context = LLVMGetModuleContext(module);
        let mut replaced = HashMap::new();
        for old in ir::functions(module)
            .into_iter()
            .filter(|&f| ir::defined(f))
        {
            let signature = LLVMGlobalGetValueType(old);
            let mut parameters =
                vec![std::ptr::null_mut(); LLVMCountParamTypes(signature) as usize];
            LLVMGetParamTypes(signature, parameters.as_mut_ptr());
            let own = parameters.len();
            parameters.extend(types(context, buffers));
            let extended = LLVMFunctionType(
                LLVMGetReturnType(signature),
                parameters.as_mut_ptr(),
                parameters.len() as u32,
                0,
            );
            let name = ir::name(old);
            LLVMSetValueName2(old, c"".as_ptr(), 0);
            let new = LLVMAddFunction(module, c"".as_ptr(), extended);
            LLVMSetValueName2(new, name.as_ptr().cast(), name.len());
            LLVMSetLinkage(new, inkwell::llvm_sys::LLVMLinkage::LLVMInternalLinkage);
            for block in ir::blocks(old) {
                LLVMRemoveBasicBlockFromParent(block);
                LLVMAppendExistingBasicBlock(new, block);
            }
            let values: Vec<_> = (0..parameters.len() as u32)
                .map(|i| LLVMGetParam(new, i))
                .collect();
            for (index, &value) in values[..own].iter().enumerate() {
                LLVMReplaceAllUsesWith(LLVMGetParam(old, index as u32), value);
            }
            replaced.insert(old, (new, launch(&values[own..], buffers)));
        }
        replaced
    }
}

/// Point every call at the extended function, passing the caller's launch on,
/// and turn every device operation into instructions on it.
pub unsafe fn rewrite_calls(module: LLVMModuleRef, replaced: &HashMap<Value, (Value, Launch)>) {
    unsafe {
        let context = LLVMGetModuleContext(module);
        let builder = LLVMCreateBuilderInContext(context);
        for (caller, launch) in replaced.values() {
            for call in ir::instructions(*caller)
                .into_iter()
                .filter(|&i| ir::is_call(i))
            {
                let target = ir::callee(call).expect("Rule::DirectCall");
                LLVMPositionBuilderBefore(builder, call);
                let arguments: Vec<_> = (0..LLVMGetNumArgOperands(call))
                    .map(|i| LLVMGetOperand(call, i))
                    .collect();
                let outcome = match replaced.get(&target) {
                    Some((extended, _)) => {
                        let mut all = arguments;
                        all.extend(&launch.buffers);
                        all.extend([
                            launch.lengths,
                            launch.status,
                            launch.thread,
                            launch.constants,
                        ]);
                        let signature = LLVMGlobalGetValueType(*extended);
                        let count = all.len() as u32;
                        Outcome::Replace(LLVMBuildCall2(
                            builder,
                            signature,
                            *extended,
                            all.as_mut_ptr(),
                            count,
                            c"".as_ptr(),
                        ))
                    }
                    None => operation(context, builder, &ir::name(target), &arguments, launch),
                };
                match outcome {
                    Outcome::Keep => continue,
                    Outcome::Erase => {}
                    Outcome::Replace(value) => LLVMReplaceAllUsesWith(call, value),
                }
                LLVMInstructionEraseFromParent(call);
            }
        }
        LLVMDisposeBuilder(builder);
    }
}

enum Outcome {
    /// An LLVM intrinsic Metal implements.
    Keep,
    /// The call had no result and its effect is now in place, or it had none.
    Erase,
    Replace(Value),
}

/// What a call to the undefined function `name` becomes.
unsafe fn operation(
    context: LLVMContextRef,
    builder: LLVMBuilderRef,
    name: &str,
    arguments: &[Value],
    launch: &Launch,
) -> Outcome {
    unsafe {
        let byte = LLVMInt8TypeInContext(context);
        let word = LLVMInt64TypeInContext(context);
        let slot = |argument: Value| LLVMConstIntGetZExtValue(argument) as usize;
        let element = |base: Value, ty: LLVMTypeRef, mut index: Value| {
            LLVMBuildGEP2(builder, ty, base, &mut index, 1, c"".as_ptr())
        };
        Outcome::Replace(match name {
            "llvm_metal.thread_index" => launch.thread,
            "llvm_metal.length" => {
                let place = element(launch.lengths, word, arguments[0]);
                LLVMBuildLoad2(builder, word, place, c"".as_ptr())
            }
            "llvm_metal.load" => {
                let place = element(launch.buffers[slot(arguments[0])], byte, arguments[1]);
                LLVMBuildLoad2(builder, byte, place, c"".as_ptr())
            }
            "llvm_metal.store" => {
                let place = element(launch.buffers[slot(arguments[0])], byte, arguments[1]);
                LLVMBuildStore(builder, arguments[2], place);
                return Outcome::Erase;
            }
            "llvm_metal.panic" => {
                let raised = LLVMConstInt(LLVMInt32TypeInContext(context), 1, 0);
                LLVMBuildStore(builder, raised, launch.status);
                return Outcome::Erase;
            }
            // Hints mean nothing to Metal.
            hint if hint.starts_with("llvm.assume")
                || hint.starts_with("llvm.lifetime.")
                || hint.starts_with("llvm.experimental.noalias.scope.decl") =>
            {
                return Outcome::Erase;
            }
            _ => return Outcome::Keep,
        })
    }
}
