//! Deliberately narrow retained-call profile. Unsupported interfaces still inline.
use crate::air::InliningPolicy;
use inkwell::{
    module::{Linkage, Module},
    values::{AsValueRef, CallSiteValue, FunctionValue},
};
use std::collections::HashSet;

pub(crate) fn validate(module: &Module<'_>) -> Result<(), String> {
    validate_calls(module, true)
}
fn validate_calls(module: &Module<'_>, reject_recursion: bool) -> Result<(), String> {
    for f in module
        .get_functions()
        .filter(|f| f.count_basic_blocks() != 0)
    {
        f.get_name()
            .to_str()
            .map_err(|_| "helper names must be UTF-8")?;
        crate::wide_helpers::direct_uses(f).map_err(|_| {
            format!(
                "function address escapes: {}",
                f.get_name().to_string_lossy()
            )
        })?;
        if reject_recursion && crate::wide_helpers::recursive(f) {
            return Err(format!(
                "recursive helper is unsupported: {}",
                f.get_name().to_string_lossy()
            ));
        }
        for i in f
            .get_basic_blocks()
            .iter()
            .flat_map(|b| b.get_instructions())
        {
            if let Ok(call) = CallSiteValue::try_from(i) {
                let callee = call
                    .get_called_fn_value()
                    .ok_or("indirect calls are unsupported")?;
                // Operand bundles can encode deoptimization or target semantics;
                // rebuilding a call must never silently drop them.
                if unsafe { inkwell::llvm_sys::core::LLVMGetNumOperandBundles(call.as_value_ref()) }
                    != 0
                {
                    return Err("call operand bundles are unsupported".into());
                }
                if unsafe { inkwell::llvm_sys::core::LLVMGetTailCallKind(call.as_value_ref()) }
                    == inkwell::llvm_sys::LLVMTailCallKind::LLVMTailCallKindMustTail
                {
                    return Err("musttail helper calls are unsupported".into());
                }
                // Opaque-pointer IR can express a direct call with a different
                // function type. Such type-punned interfaces are not retained.
                if unsafe {
                    inkwell::llvm_sys::core::LLVMGetCalledFunctionType(call.as_value_ref())
                        != inkwell::llvm_sys::core::LLVMGlobalGetValueType(callee.as_value_ref())
                } {
                    return Err("helper call-site signature mismatch".into());
                }
                if call.get_call_convention() != callee.get_call_conventions() {
                    return Err("helper call-site calling convention mismatch".into());
                }
            }
        }
    }
    Ok(())
}
fn supported_signature(f: FunctionValue<'_>, policy: InliningPolicy) -> bool {
    let scalar = |t: inkwell::types::BasicTypeEnum<'_>| {
        t.is_int_type() && matches!(t.into_int_type().get_bit_width(), 8 | 16 | 32 | 64)
    };
    // Copy/stack/register ABI parameters need their own retained-call proof.
    // Let LLVM's existing inliner remove these interfaces for now; sret is
    // covered by the writer regression and real SHA candidate kernels.
    let unsupported_abi = [
        "byval",
        "byref",
        "inalloca",
        "preallocated",
        "nest",
        "swiftself",
        "swifterror",
    ];
    let has_unsupported_abi = (0..f.count_params()).any(|i| {
        unsupported_abi.iter().any(|name| {
            f.get_enum_attribute(
                inkwell::attributes::AttributeLoc::Param(i),
                inkwell::attributes::Attribute::get_named_enum_kind_id(name),
            )
            .is_some()
        })
    });
    !has_unsupported_abi
        && !f.get_type().is_var_arg()
        && matches!(f.get_call_conventions(), 0 | 8)
        && f.get_type().get_return_type().is_none_or(scalar)
        && f.get_param_iter().all(|p| {
            scalar(p.get_type()) || (policy == InliningPolicy::Selective && p.is_pointer_value())
        })
}
pub(crate) fn retained(
    module: &Module<'_>,
    entry: &str,
    policy: InliningPolicy,
) -> Result<HashSet<String>, String> {
    select(module, entry, policy, true)
}
fn select(
    module: &Module<'_>,
    entry: &str,
    policy: InliningPolicy,
    final_input: bool,
) -> Result<HashSet<String>, String> {
    if policy == InliningPolicy::All {
        return Ok(HashSet::new());
    }
    validate_calls(module, final_input)?;
    // Propagate the thread-index requirement to every caller, so none can retain
    // a dependency on the entry-only builtin. Atomic operations have explicit args.
    let mut indexed = HashSet::from(["llvm_metal.linear_thread_index".to_owned()]);
    loop {
        let before = indexed.len();
        for f in module.get_functions() {
            if f.get_basic_blocks()
                .iter()
                .flat_map(|b| b.get_instructions())
                .any(|i| {
                    CallSiteValue::try_from(i)
                        .ok()
                        .and_then(|c| c.get_called_fn_value())
                        .is_some_and(|c| {
                            indexed.contains(&c.get_name().to_string_lossy().into_owned())
                        })
                })
            {
                indexed.insert(f.get_name().to_string_lossy().into_owned());
            }
        }
        if indexed.len() == before {
            break;
        }
    }
    Ok(module
        .get_functions()
        .filter(|f| {
            f.count_basic_blocks() != 0
                && supported_signature(*f, policy)
                && matches!(f.get_linkage(), Linkage::Internal | Linkage::Private)
                && !crate::wide_helpers::recursive(*f)
                && (policy != InliningPolicy::Selective
                    || f.get_enum_attribute(
                        inkwell::attributes::AttributeLoc::Function,
                        inkwell::attributes::Attribute::get_named_enum_kind_id("noinline"),
                    )
                    .is_some()
                    || f.get_basic_blocks()
                        .iter()
                        .map(|b| b.get_instructions().count())
                        .sum::<usize>()
                        >= 32)
        })
        .map(|f| f.get_name().to_string_lossy().into_owned())
        .filter(|name| name != entry && !indexed.contains(name))
        .collect())
}

pub(crate) fn specialize(module: &Module<'_>, entry: &str) -> Result<(), String> {
    unsafe extern "C" {
        fn LLVMMetalSpecializeCalls(
            module: inkwell::llvm_sys::prelude::LLVMModuleRef,
            entry: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_char;
    }
    let entry = std::ffi::CString::new(entry).map_err(|e| e.to_string())?;
    // SAFETY: a verified, exclusively owned module; native diagnostics use LLVM's allocator.
    unsafe {
        let error = LLVMMetalSpecializeCalls(module.as_mut_ptr(), entry.as_ptr());
        if !error.is_null() {
            let text = std::ffi::CStr::from_ptr(error)
                .to_string_lossy()
                .into_owned();
            inkwell::llvm_sys::core::LLVMDisposeMessage(error);
            return Err(text);
        }
    }
    module.verify().map_err(|e| e.to_string())
}

/// Preserve eligible boundaries before the consumer runs its LLVM optimizations.
/// Operates on a clone, leaving source bitcode unchanged.
pub fn prepare<'ctx>(
    input: &Module<'ctx>,
    entry: &str,
    policy: InliningPolicy,
) -> Result<Module<'ctx>, String> {
    use inkwell::attributes::{Attribute, AttributeLoc};
    input.verify().map_err(|e| e.to_string())?;
    crate::require_entry(input, entry).map_err(|e| e.to_string())?;
    let module = input.clone();
    // The producer may contain recursive fallback paths which ordinary LLVM
    // inlining/constant propagation removes. Never retain these boundaries;
    // final legalization still rejects any recursion left after optimization.
    let names = select(&module, entry, policy, false)?;
    let context = module.get_context();
    for function in module
        .get_functions()
        .filter(|f| f.count_basic_blocks() != 0)
    {
        let keep = names.contains(&function.get_name().to_string_lossy().into_owned());
        for name in ["alwaysinline", "noinline", "optnone"] {
            function.remove_enum_attribute(
                AttributeLoc::Function,
                Attribute::get_named_enum_kind_id(name),
            );
        }
        // The selected entry remains externally visible to the upstream optimizer.
        if function.get_name().to_bytes() != entry.as_bytes() {
            function.add_attribute(
                AttributeLoc::Function,
                context.create_enum_attribute(
                    Attribute::get_named_enum_kind_id(if keep {
                        "noinline"
                    } else {
                        "alwaysinline"
                    }),
                    0,
                ),
            );
        }
        for i in function
            .get_basic_blocks()
            .iter()
            .flat_map(|b| b.get_instructions())
        {
            if let Ok(call) = CallSiteValue::try_from(i) {
                if call
                    .get_called_fn_value()
                    .is_some_and(|f| f.count_basic_blocks() != 0)
                {
                    for name in ["alwaysinline", "noinline"] {
                        call.remove_enum_attribute(
                            AttributeLoc::Function,
                            Attribute::get_named_enum_kind_id(name),
                        );
                    }
                }
            }
        }
    }
    module.verify().map_err(|e| e.to_string())?;
    Ok(module)
}

/// LLVM 21 optimization facts absent from the AIR writer's legacy attribute
/// encoding. Dropping these facts weakens optimization assumptions, not the ABI.
/// Keep ABI attributes (sret/byval/signext/zeroext) intact.
pub(crate) fn strip_modern_parameter_facts(module: &Module<'_>) {
    use inkwell::attributes::{Attribute, AttributeLoc};
    let kinds: Vec<_> = [
        "captures",
        "dead_on_unwind",
        "dead_on_return",
        "writable",
        "initializes",
        "range",
    ]
    .into_iter()
    .map(Attribute::get_named_enum_kind_id)
    .filter(|id| *id != 0)
    .collect();
    for f in module.get_functions() {
        for loc in std::iter::once(AttributeLoc::Return)
            .chain((0..f.count_params()).map(AttributeLoc::Param))
        {
            for &kind in &kinds {
                f.remove_enum_attribute(loc, kind);
            }
        }
        for i in f
            .get_basic_blocks()
            .iter()
            .flat_map(|b| b.get_instructions())
        {
            if let Ok(call) = CallSiteValue::try_from(i) {
                for loc in std::iter::once(AttributeLoc::Return)
                    .chain((0..call.count_arguments()).map(AttributeLoc::Param))
                {
                    for &kind in &kinds {
                        call.remove_enum_attribute(loc, kind);
                    }
                }
            }
        }
    }
}
