//! Prepare PHI operands for the legacy AIR bitcode writer.
//!
//! The pinned legacy writer demotes constant-expression operands immediately
//! before their user, which is not legal for PHIs. Place equivalent
//! instructions on incoming edges instead, preserving address spaces and
//! pointer provenance. Run after the final optimizer so these do not refold.
use inkwell::{
    llvm_sys::{
        LLVMOpcode,
        core::*,
        debuginfo::{LLVMInstructionGetDebugLoc, LLVMInstructionSetDebugLoc},
        prelude::*,
    },
    module::Module,
};
use std::collections::{HashMap, HashSet};

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

// The builder folds an operation whose operands are all constant. Build each
// instruction around a non-constant stand-in, then install the real operand.
unsafe fn materialize(
    builder: LLVMBuilderRef,
    expression: LLVMValueRef,
) -> Result<LLVMValueRef, String> {
    unsafe {
        let first = LLVMGetOperand(expression, 0);
        let stand_in = LLVMBuildFreeze(builder, LLVMGetPoison(LLVMTypeOf(first)), c"".as_ptr());
        let opcode = LLVMGetConstOpcode(expression);
        let instruction = match opcode {
            LLVMOpcode::LLVMGetElementPtr => {
                let mut indices: Vec<_> = (1..LLVMGetNumOperands(expression))
                    .map(|i| LLVMGetOperand(expression, i as u32))
                    .collect();
                let instruction = LLVMBuildGEP2(
                    builder,
                    LLVMGetGEPSourceElementType(expression),
                    stand_in,
                    indices.as_mut_ptr(),
                    indices.len() as u32,
                    c"".as_ptr(),
                );
                LLVMGEPSetNoWrapFlags(instruction, LLVMGEPGetNoWrapFlags(expression));
                instruction
            }
            LLVMOpcode::LLVMTrunc
            | LLVMOpcode::LLVMPtrToInt
            | LLVMOpcode::LLVMIntToPtr
            | LLVMOpcode::LLVMBitCast
            | LLVMOpcode::LLVMAddrSpaceCast => LLVMBuildCast(
                builder,
                opcode,
                stand_in,
                LLVMTypeOf(expression),
                c"".as_ptr(),
            ),
            // No-wrap flags of a constant expression are not readable through
            // the C API. Omitting them only weakens the poison contract.
            LLVMOpcode::LLVMAdd | LLVMOpcode::LLVMSub | LLVMOpcode::LLVMXor => LLVMBuildBinOp(
                builder,
                opcode,
                stand_in,
                LLVMGetOperand(expression, 1),
                c"".as_ptr(),
            ),
            _ => {
                LLVMInstructionEraseFromParent(stand_in);
                return Err(format!(
                    "unsupported constant expression in PHI operand: {}",
                    printed(expression)
                ));
            }
        };
        LLVMSetOperand(instruction, 0, first);
        LLVMInstructionEraseFromParent(stand_in);
        Ok(instruction)
    }
}

pub(crate) fn prepare(module: &Module<'_>) -> Result<(), String> {
    // SAFETY: final verified-shape module with exclusive mutation. Instructions
    // are only added, never erased, so collected references stay live.
    unsafe {
        let builder = LLVMCreateBuilderInContext(LLVMGetModuleContext(module.as_mut_ptr()));
        let mut function = LLVMGetFirstFunction(module.as_mut_ptr());
        let mut result = Ok(());
        while !function.is_null() && result.is_ok() {
            result = prepare_function(builder, function);
            function = LLVMGetNextFunction(function);
        }
        LLVMDisposeBuilder(builder);
        result
    }
}

unsafe fn prepare_function(builder: LLVMBuilderRef, function: LLVMValueRef) -> Result<(), String> {
    unsafe {
        let mut pending = Vec::new();
        let mut block = LLVMGetFirstBasicBlock(function);
        while !block.is_null() {
            let mut instruction = LLVMGetFirstInstruction(block);
            while !instruction.is_null() && !LLVMIsAPHINode(instruction).is_null() {
                pending.extend(
                    (0..LLVMCountIncoming(instruction))
                        .map(|i| LLVMGetIncomingValue(instruction, i))
                        .filter(|&v| !LLVMIsAConstantExpr(v).is_null()),
                );
                instruction = LLVMGetNextInstruction(instruction);
            }
            block = LLVMGetNextBasicBlock(block);
        }
        // Include constant expressions built on those operands: once an operand
        // is an instruction, no constant expression can use it any more.
        let mut expandable = HashSet::new();
        let mut order = Vec::new();
        while let Some(constant) = pending.pop() {
            if !expandable.insert(constant) {
                continue;
            }
            order.push(constant);
            let mut use_ = LLVMGetFirstUse(constant);
            while !use_.is_null() {
                let user = LLVMGetUser(use_);
                if !LLVMIsAConstantExpr(user).is_null() {
                    pending.push(user);
                }
                use_ = LLVMGetNextUse(use_);
            }
        }
        let mut worklist = Vec::new();
        let mut queued = HashSet::new();
        for &constant in &order {
            let mut use_ = LLVMGetFirstUse(constant);
            while !use_.is_null() {
                let user = LLVMGetUser(use_);
                if !LLVMIsAInstruction(user).is_null()
                    && LLVMGetBasicBlockParent(LLVMGetInstructionParent(user)) == function
                    && queued.insert(user)
                {
                    worklist.push(user);
                }
                use_ = LLVMGetNextUse(use_);
            }
        }
        // One instruction per incoming block and constant: a PHI must see the
        // same value on every edge from the same predecessor.
        let mut placed: HashMap<(LLVMBasicBlockRef, LLVMValueRef), LLVMValueRef> = HashMap::new();
        while let Some(user) = worklist.pop() {
            let phi = !LLVMIsAPHINode(user).is_null();
            for i in 0..LLVMGetNumOperands(user) as u32 {
                let operand = LLVMGetOperand(user, i);
                if !expandable.contains(&operand) {
                    continue;
                }
                let replacement = if phi {
                    let incoming = LLVMGetIncomingBlock(user, i);
                    match placed.get(&(incoming, operand)) {
                        Some(&existing) => existing,
                        None => {
                            // Like LLVM's utility: the first non-PHI position.
                            let mut first = LLVMGetFirstInstruction(incoming);
                            while !LLVMIsAPHINode(first).is_null() {
                                first = LLVMGetNextInstruction(first);
                            }
                            LLVMPositionBuilderBefore(builder, first);
                            let new = materialize(builder, operand)?;
                            LLVMInstructionSetDebugLoc(new, LLVMInstructionGetDebugLoc(user));
                            placed.insert((incoming, operand), new);
                            worklist.push(new);
                            new
                        }
                    }
                } else {
                    LLVMPositionBuilderBefore(builder, user);
                    let new = materialize(builder, operand)?;
                    LLVMInstructionSetDebugLoc(new, LLVMInstructionGetDebugLoc(user));
                    worklist.push(new);
                    new
                };
                LLVMSetOperand(user, i, replacement);
            }
        }
        Ok(())
    }
}
