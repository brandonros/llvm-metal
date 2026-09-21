//! From a verified program to a module Metal accepts. A fixed list of rewrites,
//! each run once; none can fail, because `verify` has already refused whatever
//! they cannot express.
mod constants;
mod context;

use crate::ir::{self, Value};
use inkwell::llvm_sys::{core::*, debuginfo::LLVMStripModuleDebugInfo, prelude::*};

/// What the runtime must bind: the kernel's own buffers at `0..buffers`, then
/// their lengths in bytes (`u64` each), then the status word (`u32`, zero before
/// the launch, nonzero after it if any thread panicked).
pub struct Bindings {
    pub buffers: usize,
}

pub unsafe fn lower(module: LLVMModuleRef, kernel: Value, name: &str) -> Bindings {
    unsafe {
        let buffers = slots(module);
        let (image, relocations) = constants::image(module);
        let replaced = context::extend(module, buffers);
        context::rewrite_calls(module, &replaced);
        let builder = LLVMCreateBuilderInContext(LLVMGetModuleContext(module));
        for (function, launch) in replaced.values() {
            address_constants(builder, &image, *function, launch.constants);
        }
        entry(
            module,
            builder,
            name,
            buffers,
            &image,
            &relocations,
            replaced[&kernel].0,
        );
        LLVMDisposeBuilder(builder);
        // Everything the old functions and globals were is now elsewhere.
        for function in ir::functions(module) {
            if LLVMGetFirstUse(function).is_null() && LLVMCountBasicBlocks(function) == 0 {
                LLVMDeleteFunction(function);
            }
        }
        for &global in image.offsets.keys() {
            LLVMSetInitializer(global, std::ptr::null_mut());
        }
        for &global in image.offsets.keys() {
            LLVMDeleteGlobal(global);
        }
        LLVMStripModuleDebugInfo(module);
        Bindings { buffers }
    }
}

/// One more than the highest slot any device operation names.
unsafe fn slots(module: LLVMModuleRef) -> usize {
    unsafe {
        let calls = ir::functions(module)
            .into_iter()
            .flat_map(|f| ir::instructions(f));
        calls
            .filter(|&call| ir::is_call(call))
            .filter_map(|call| {
                let name = ir::name(ir::callee(call)?);
                ["llvm_metal.length", "llvm_metal.load", "llvm_metal.store"]
                    .contains(&name.as_str())
                    .then(|| LLVMConstIntGetZExtValue(LLVMGetOperand(call, 0)) as usize + 1)
            })
            .max()
            .unwrap_or(0)
    }
}

/// Compute every global's address from the thread's copy. A phi's operand is
/// computed at the end of the block it arrives from.
unsafe fn address_constants(
    builder: LLVMBuilderRef,
    image: &constants::Image,
    function: Value,
    copy: Value,
) {
    unsafe {
        for instruction in ir::instructions(function) {
            for (index, operand) in ir::operands(instruction).into_iter().enumerate() {
                if !constants::refers(operand) {
                    continue;
                }
                if LLVMIsAPHINode(instruction).is_null() {
                    LLVMPositionBuilderBefore(builder, instruction);
                } else {
                    let from = LLVMGetIncomingBlock(instruction, index as u32);
                    LLVMPositionBuilderBefore(builder, LLVMGetBasicBlockTerminator(from));
                }
                let value = constants::materialize(builder, image, copy, operand);
                LLVMSetOperand(instruction, index as u32, value);
            }
        }
    }
}

/// The function Metal calls: copy the constants, fix the addresses they hold,
/// and run the kernel.
unsafe fn entry(
    module: LLVMModuleRef,
    builder: LLVMBuilderRef,
    name: &str,
    buffers: usize,
    image: &constants::Image,
    relocations: &[constants::Relocation],
    kernel: Value,
) {
    unsafe {
        let context = LLVMGetModuleContext(module);
        let mut parameters = context::types(context, buffers);
        parameters.pop(); // The constants are made here, not received.
        let void = LLVMVoidTypeInContext(context);
        let signature = LLVMFunctionType(void, parameters.as_mut_ptr(), parameters.len() as u32, 0);
        let symbol = std::ffi::CString::new(name).expect("a Rust identifier");
        let function = LLVMAddFunction(module, symbol.as_ptr(), signature);
        let block = LLVMAppendBasicBlockInContext(context, function, c"".as_ptr());
        LLVMPositionBuilderAtEnd(builder, block);

        let byte = LLVMInt8TypeInContext(context);
        let word = LLVMInt64TypeInContext(context);
        let copy = LLVMBuildAlloca(
            builder,
            LLVMArrayType2(byte, image.size.max(1)),
            c"constants".as_ptr(),
        );
        LLVMSetAlignment(copy, image.alignment);
        let mut types = [LLVMTypeOf(copy), LLVMTypeOf(image.global), word];
        let memcpy = LLVMGetIntrinsicDeclaration(
            module,
            LLVMLookupIntrinsicID(c"llvm.memcpy".as_ptr(), 11),
            types.as_mut_ptr(),
            types.len(),
        );
        let mut arguments = [
            copy,
            image.global,
            LLVMConstInt(word, image.size, 0),
            LLVMConstInt(LLVMInt1TypeInContext(context), 0, 0),
        ];
        LLVMBuildCall2(
            builder,
            LLVMGlobalGetValueType(memcpy),
            memcpy,
            arguments.as_mut_ptr(),
            arguments.len() as u32,
            c"".as_ptr(),
        );
        for relocation in relocations {
            let address = constants::materialize(builder, image, copy, relocation.target);
            let mut offset = LLVMConstInt(word, relocation.offset, 0);
            let place = LLVMBuildGEP2(builder, byte, copy, &mut offset, 1, c"".as_ptr());
            LLVMSetAlignment(LLVMBuildStore(builder, address, place), 1);
        }

        let mut launch: Vec<_> = (0..parameters.len() as u32)
            .map(|i| LLVMGetParam(function, i))
            .collect();
        launch.push(copy);
        LLVMBuildCall2(
            builder,
            LLVMGlobalGetValueType(kernel),
            kernel,
            launch.as_mut_ptr(),
            launch.len() as u32,
            c"".as_ptr(),
        );
        LLVMBuildRetVoid(builder);
    }
}
