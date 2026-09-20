//! Expand bounded volatile memcpy operations using shared pointer provenance.
use crate::pointer_provenance::{constant_pointer, private_pointer};
use inkwell::{
    llvm_sys::{core::*, prelude::*},
    module::Module,
};

const LIMIT: u64 = 4096;

unsafe fn volatile_copy(instruction: LLVMValueRef) -> bool {
    unsafe {
        if LLVMIsACallInst(instruction).is_null() {
            return false;
        }
        let callee = LLVMGetCalledValue(instruction);
        if LLVMIsAFunction(callee).is_null() {
            return false;
        }
        let id = LLVMGetIntrinsicID(callee);
        let named = |name: &str| LLVMLookupIntrinsicID(name.as_ptr().cast(), name.len());
        if id == 0 || (id != named("llvm.memcpy") && id != named("llvm.memcpy.inline")) {
            return false;
        }
        let flag = LLVMGetOperand(instruction, 3);
        !LLVMIsAConstantInt(flag).is_null() && LLVMConstIntGetZExtValue(flag) != 0
    }
}

/// Preserve zeroize's volatile aggregate writes as individual volatile accesses.
/// Only bounded copies into private storage from private or defined constant
/// storage are admitted; every other volatile copy is left for later refusal.
pub(crate) fn expand(module: &Module<'_>) {
    // SAFETY: verified disposable clone. Copies are collected before any edit,
    // and each is erased only after its replacement accesses are built.
    unsafe {
        let mut copies = Vec::new();
        let mut function = LLVMGetFirstFunction(module.as_mut_ptr());
        while !function.is_null() {
            let mut block = LLVMGetFirstBasicBlock(function);
            while !block.is_null() {
                let mut instruction = LLVMGetFirstInstruction(block);
                while !instruction.is_null() {
                    if volatile_copy(instruction) {
                        copies.push(instruction);
                    }
                    instruction = LLVMGetNextInstruction(instruction);
                }
                block = LLVMGetNextBasicBlock(block);
            }
            function = LLVMGetNextFunction(function);
        }
        let context = LLVMGetModuleContext(module.as_mut_ptr());
        let int8 = LLVMInt8TypeInContext(context);
        let int64 = LLVMInt64TypeInContext(context);
        let builder = LLVMCreateBuilderInContext(context);
        for copy in copies {
            let destination = LLVMGetOperand(copy, 0);
            let source = LLVMGetOperand(copy, 1);
            let length = LLVMGetOperand(copy, 2);
            if LLVMIsAConstantInt(length).is_null()
                || LLVMGetIntTypeWidth(LLVMTypeOf(length)) > 64
                || LLVMConstIntGetZExtValue(length) > LIMIT
                || !(private_pointer(source) || constant_pointer(source))
                || !private_pointer(destination)
            {
                continue;
            }
            LLVMPositionBuilderBefore(builder, copy);
            for i in 0..LLVMConstIntGetZExtValue(length) {
                let mut index = LLVMConstInt(int64, i, 0);
                let from = LLVMBuildGEP2(builder, int8, source, &mut index, 1, c"".as_ptr());
                let to = LLVMBuildGEP2(builder, int8, destination, &mut index, 1, c"".as_ptr());
                let value = LLVMBuildLoad2(builder, int8, from, c"".as_ptr());
                LLVMSetAlignment(value, 1);
                LLVMSetVolatile(value, 1);
                let store = LLVMBuildStore(builder, value, to);
                LLVMSetAlignment(store, 1);
                LLVMSetVolatile(store, 1);
            }
            LLVMInstructionEraseFromParent(copy);
        }
        LLVMDisposeBuilder(builder);
    }
}
