//! Clone a function body into a function with another signature. LLVM's C API
//! has no CloneFunctionInto, so this covers the shapes retained helpers use and
//! refuses the rest.
use inkwell::llvm_sys::{
    LLVMAttributeFunctionIndex, LLVMAttributeReturnIndex, LLVMOpcode, LLVMValueKind,
    core::*,
    debuginfo::{LLVMGetMetadataKind, LLVMInstructionSetDebugLoc, LLVMMetadataKind},
    prelude::*,
};
use std::collections::HashMap;

unsafe fn copy_name(new: LLVMValueRef, old: LLVMValueRef) {
    unsafe {
        let mut length = 0;
        let text = LLVMGetValueName2(old, &mut length);
        LLVMSetValueName2(new, text, length);
    }
}

unsafe fn copy_attributes(to: LLVMValueRef, to_index: u32, from: LLVMValueRef, from_index: u32) {
    unsafe {
        let count = LLVMGetAttributeCountAtIndex(from, from_index);
        let mut attributes = vec![std::ptr::null_mut(); count as usize];
        LLVMGetAttributesAtIndex(from, from_index, attributes.as_mut_ptr());
        for attribute in attributes {
            LLVMAddAttributeAtIndex(to, to_index, attribute);
        }
    }
}

// Function, return and unchanged-parameter attributes, plus the global
// properties Function::copyAttributesFrom carries. A parameter whose type
// changed keeps none of its attributes; that only weakens what is assumed.
unsafe fn copy_function_properties(
    to: LLVMValueRef,
    from: LLVMValueRef,
    values: &HashMap<LLVMValueRef, LLVMValueRef>,
) -> Result<(), String> {
    unsafe {
        if LLVMHasPersonalityFn(from) != 0
            || LLVMHasPrefixData(from) != 0
            || LLVMHasPrologueData(from) != 0
        {
            return Err("retained helper has personality, prefix or prologue data".into());
        }
        copy_attributes(
            to,
            LLVMAttributeFunctionIndex,
            from,
            LLVMAttributeFunctionIndex,
        );
        copy_attributes(to, LLVMAttributeReturnIndex, from, LLVMAttributeReturnIndex);
        for i in 0..LLVMCountParams(from) {
            let mapped = values[&LLVMGetParam(from, i)];
            if let Some(n) = (0..LLVMCountParams(to)).find(|&n| LLVMGetParam(to, n) == mapped) {
                copy_attributes(to, n + 1, from, i + 1);
            }
        }
        LLVMSetVisibility(to, LLVMGetVisibility(from));
        LLVMSetUnnamedAddress(to, LLVMGetUnnamedAddress(from));
        LLVMSetDLLStorageClass(to, LLVMGetDLLStorageClass(from));
        LLVMSetAlignment(to, LLVMGetAlignment(from));
        let section = LLVMGetSection(from);
        if !section.is_null() && *section != 0 {
            LLVMSetSection(to, section);
        }
        let gc = LLVMGetGC(from);
        if !gc.is_null() {
            LLVMSetGC(to, gc);
        }
        // Instruction metadata is shared with the original, including distinct
        // nodes such as the inliner's alias scopes: the C API cannot create a
        // distinct node. Those scopes only relate accesses within one function
        // and AIR legalization strips them after specialization.
        // A subprogram may describe one function only and the C API cannot clone
        // one, so the copy carries no debug info. Other attachments are shared.
        let context = LLVMGetTypeContext(LLVMTypeOf(to));
        let debug = LLVMGetMDKindIDInContext(context, c"dbg".as_ptr(), 3);
        let mut count = 0;
        let entries = LLVMGlobalCopyAllMetadata(from, &mut count);
        for i in 0..count as u32 {
            let kind = LLVMValueMetadataEntriesGetKind(entries, i);
            if kind != debug {
                LLVMGlobalSetMetadata(to, kind, LLVMValueMetadataEntriesGetMetadata(entries, i));
            }
        }
        LLVMDisposeValueMetadataEntries(entries);
        Ok(())
    }
}

/// Append a copy of `from`'s blocks to `to`, replacing each key of `values`
/// with its mapped value, and return the copy of the entry block. `values`
/// must map every parameter of `from`.
pub(crate) unsafe fn function_into(
    to: LLVMValueRef,
    from: LLVMValueRef,
    mut values: HashMap<LLVMValueRef, LLVMValueRef>,
) -> Result<LLVMBasicBlockRef, String> {
    unsafe {
        copy_function_properties(to, from, &values)?;
        let context = LLVMGetTypeContext(LLVMTypeOf(to));
        let builder = LLVMCreateBuilderInContext(context);
        let mut copies = Vec::new();
        let mut phis = Vec::new();
        let mut block = LLVMGetFirstBasicBlock(from);
        while !block.is_null() {
            let copy = LLVMAppendBasicBlockInContext(context, to, c"".as_ptr());
            copy_name(LLVMBasicBlockAsValue(copy), LLVMBasicBlockAsValue(block));
            values.insert(LLVMBasicBlockAsValue(block), LLVMBasicBlockAsValue(copy));
            LLVMPositionBuilderAtEnd(builder, copy);
            let mut instruction = LLVMGetFirstInstruction(block);
            while !instruction.is_null() {
                // The incoming blocks of a copied PHI cannot be changed through
                // the C API, so a PHI is rebuilt once every block has a copy.
                let new = if LLVMGetInstructionOpcode(instruction) == LLVMOpcode::LLVMPHI {
                    let new = LLVMBuildPhi(builder, LLVMTypeOf(instruction), c"".as_ptr());
                    if LLVMCanValueUseFastMathFlags(instruction) != 0 {
                        LLVMSetFastMathFlags(new, LLVMGetFastMathFlags(instruction));
                    }
                    phis.push((instruction, new));
                    new
                } else {
                    let new = LLVMInstructionClone(instruction);
                    LLVMInsertIntoBuilderWithName(builder, new, c"".as_ptr());
                    copies.push(new);
                    new
                };
                LLVMInstructionSetDebugLoc(new, std::ptr::null_mut());
                copy_name(new, instruction);
                values.insert(instruction, new);
                instruction = LLVMGetNextInstruction(instruction);
            }
            block = LLVMGetNextBasicBlock(block);
        }
        LLVMDisposeBuilder(builder);
        // A value may be defined in a later block than its user, so operands
        // are replaced only after every instruction has its copy.
        for copy in copies {
            for i in 0..LLVMGetNumOperands(copy) as u32 {
                let operand = LLVMGetOperand(copy, i);
                if let Some(&mapped) = values.get(&operand) {
                    LLVMSetOperand(copy, i, mapped);
                } else if !LLVMIsABlockAddress(operand).is_null()
                    || (LLVMGetValueKind(operand) == LLVMValueKind::LLVMMetadataAsValueValueKind
                        && matches!(
                            LLVMGetMetadataKind(LLVMValueAsMetadata(operand)),
                            LLVMMetadataKind::LLVMLocalAsMetadataMetadataKind
                        ))
                {
                    return Err("retained helper uses a block address or local metadata".into());
                }
            }
        }
        for (phi, new) in phis {
            for i in 0..LLVMCountIncoming(phi) {
                let incoming = LLVMGetIncomingValue(phi, i);
                let mut incoming = values.get(&incoming).copied().unwrap_or(incoming);
                let mut block = LLVMValueAsBasicBlock(
                    values[&LLVMBasicBlockAsValue(LLVMGetIncomingBlock(phi, i))],
                );
                LLVMAddIncoming(new, &mut incoming, &mut block, 1);
            }
        }
        Ok(LLVMValueAsBasicBlock(
            values[&LLVMBasicBlockAsValue(LLVMGetEntryBasicBlock(from))],
        ))
    }
}
