//! Eliminate local scalar-i128 helper ABIs before splitting wide operations.
//! Operates only on the disposable module clone owned by AIR legalization.
use inkwell::{
    attributes::{Attribute, AttributeLoc},
    llvm_sys::{core::*, prelude::LLVMValueRef},
    module::{Linkage, Module},
    values::{AsValueRef, CallSiteValue, FunctionValue},
};
use std::collections::HashSet;

fn wide_signature(function: FunctionValue<'_>) -> bool {
    let wide = |ty: inkwell::types::BasicTypeEnum<'_>| {
        ty.is_int_type() && ty.into_int_type().get_bit_width() == 128
    };
    function.get_type().get_return_type().is_some_and(wide)
        || function.get_param_iter().any(|p| wide(p.get_type()))
}

pub(crate) fn direct_uses(function: FunctionValue<'_>) -> Result<(), String> {
    // SAFETY: the caller provides a verified module. Each use and its user are
    // live; inspect instruction-only APIs only after checking the value kind.
    unsafe {
        let value = function.as_value_ref();
        let mut use_ = LLVMGetFirstUse(value);
        while !use_.is_null() {
            let user = LLVMGetUser(use_);
            if LLVMIsACallInst(user).is_null() || LLVMGetCalledValue(user) != value {
                return Err(
                    "i128 helper must have only direct call uses; its address escapes".into(),
                );
            }
            // A direct call may also pass the callee itself as an argument or
            // operand-bundle value. That is an escaping use, too.
            let occurrences = (0..LLVMGetNumOperands(user))
                .filter(|&i| LLVMGetOperand(user, i as u32) == value)
                .count();
            if occurrences != 1 {
                return Err("i128 helper address escapes through a call operand".into());
            }
            use_ = LLVMGetNextUse(use_);
        }
    }
    Ok(())
}

pub(crate) fn recursive(function: FunctionValue<'_>) -> bool {
    // Reachability back to the helper also catches mutual recursion through
    // functions whose own ABI has no i128. Use a worklist, not host recursion.
    let origin = function.as_value_ref();
    let mut pending: Vec<LLVMValueRef> = vec![origin];
    let mut seen = HashSet::new();
    while let Some(current) = pending.pop() {
        if !seen.insert(current as usize) {
            continue;
        }
        // SAFETY: the worklist contains only functions in the verified module;
        // non-call instructions and non-function callees are checked explicitly.
        unsafe {
            let mut block = LLVMGetFirstBasicBlock(current);
            while !block.is_null() {
                let mut instruction = LLVMGetFirstInstruction(block);
                while !instruction.is_null() {
                    if !LLVMIsACallInst(instruction).is_null() {
                        let callee = LLVMGetCalledValue(instruction);
                        if callee == origin {
                            return true;
                        }
                        if !LLVMIsAFunction(callee).is_null() {
                            pending.push(callee);
                        }
                    }
                    instruction = LLVMGetNextInstruction(instruction);
                }
                block = LLVMGetNextBasicBlock(block);
            }
        }
    }
    false
}

pub(crate) fn lower(module: &Module<'_>) -> Result<(), String> {
    let mut targets = Vec::new();
    for function in module
        .get_functions()
        .filter(|f| f.get_intrinsic_id() == 0 && wide_signature(*f))
    {
        // Intrinsic semantics (e.g. bswap.i128) belong to the wide-operation pass.
        if function.get_intrinsic_id() != 0 {
            continue;
        }
        if function.count_basic_blocks() == 0
            || !matches!(function.get_linkage(), Linkage::Internal | Linkage::Private)
        {
            return Err("i128 helper ABI requires an internal definition; external interfaces remain unsupported".into());
        }
        direct_uses(function)?;
        if recursive(function) {
            return Err("recursive i128 helper is unsupported".into());
        }
        targets.push(
            function
                .get_name()
                .to_str()
                .map_err(|e| e.to_string())?
                .to_owned(),
        );
    }
    if targets.is_empty() {
        return Ok(());
    }
    let always = Attribute::get_named_enum_kind_id("alwaysinline");
    let never = Attribute::get_named_enum_kind_id("noinline");
    let optnone = Attribute::get_named_enum_kind_id("optnone");
    let context = module.get_context();
    let mut restore = Vec::new();
    for function in module.get_functions() {
        let name = function.get_name().to_str().map_err(|e| e.to_string())?;
        if targets.iter().any(|t| t == name) {
            function.remove_enum_attribute(AttributeLoc::Function, never);
            // optnone implies noinline; both are optimization requests, not
            // arithmetic semantics. These local functions must be eliminated.
            function.remove_enum_attribute(AttributeLoc::Function, optnone);
            function.add_attribute(
                AttributeLoc::Function,
                context.create_enum_attribute(always, 0),
            );
        } else if function
            .get_enum_attribute(AttributeLoc::Function, always)
            .is_some()
        {
            // Keep this early pass limited to the selected wide-ABI helpers.
            function.remove_enum_attribute(AttributeLoc::Function, always);
            restore.push(name.to_owned());
        }
        for block in function.get_basic_blocks() {
            for instruction in block.get_instructions() {
                if let Ok(call) = CallSiteValue::try_from(instruction)
                    && let Some(callee) = call.get_called_fn_value()
                    && targets
                        .iter()
                        .any(|t| callee.get_name().to_bytes() == t.as_bytes())
                {
                    call.remove_enum_attribute(AttributeLoc::Function, never);
                }
            }
        }
    }
    crate::air::passes(module, "always-inline")?;
    for name in restore {
        if let Some(function) = module.get_function(&name) {
            function.add_attribute(
                AttributeLoc::Function,
                context.create_enum_attribute(always, 0),
            );
        }
    }
    for name in targets {
        if let Some(function) = module.get_function(&name) {
            // LLVM may retain an unused definition. Its ABI cannot reach AIR;
            // remove only the selected definition, after proving it has no users.
            unsafe {
                if !LLVMGetFirstUse(function.as_value_ref()).is_null() {
                    return Err(format!("LLVM could not inline i128 helper {name}"));
                }
                function.delete();
            }
        }
    }
    module.verify().map_err(|e| e.to_string())
}
