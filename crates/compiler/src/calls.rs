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
fn validate_calls(module: &Module<'_>, final_input: bool) -> Result<(), String> {
    for f in module
        .get_functions()
        .filter(|f| f.count_basic_blocks() != 0)
    {
        f.get_name()
            .to_str()
            .map_err(|_| "helper names must be UTF-8")?;
        if final_input {
            crate::wide_helpers::direct_uses(f).map_err(|_| {
                format!(
                    "function address escapes: {}",
                    f.get_name().to_string_lossy()
                )
            })?;
        }
        if final_input && crate::wide_helpers::recursive(f) {
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
                let Some(callee) = call.get_called_fn_value() else {
                    if final_input {
                        return Err("indirect calls are unsupported".into());
                    }
                    continue;
                };
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
            scalar(p.get_type())
                || (matches!(policy, InliningPolicy::Selective | InliningPolicy::Llvm)
                    && p.is_pointer_value())
        })
}
// Constant initializers expose state fields (lengths, domains, tags) which
// callers need to prove bounds/panic branches unreachable. Their many stores
// should not make them expensive retained helpers under the instruction heuristic.
fn constant_initializer(f: FunctionValue<'_>) -> bool {
    use inkwell::values::InstructionOpcode;
    f.get_type().get_return_type().is_none()
        && f.count_basic_blocks() == 1
        && f.get_first_basic_block()
            .unwrap()
            .get_instructions()
            .any(|i| i.get_opcode() == InstructionOpcode::Store)
        && f.get_first_basic_block()
            .unwrap()
            .get_instructions()
            .all(|i| match i.get_opcode() {
                InstructionOpcode::GetElementPtr | InstructionOpcode::Return => true,
                InstructionOpcode::Store => unsafe {
                    inkwell::llvm_sys::core::LLVMGetVolatile(i.as_value_ref()) == 0
                        && inkwell::llvm_sys::core::LLVMIsConstant(
                            inkwell::llvm_sys::core::LLVMGetOperand(i.as_value_ref(), 0),
                        ) != 0
                },
                _ => false,
            })
}

// Retaining a nullable pointer iterator miscompiles P-256's r-only matcher on
// Apple M5, while inlining that boundary passes the complete workload. Keep this
// structural class on the proven inlining path until retained execution has a
// sufficient target-level proof. Straight-line pointer joins remain eligible.
fn nullable_pointer_loop(f: FunctionValue<'_>) -> bool {
    use inkwell::llvm_sys::{LLVMTypeKind, core::*};
    if f.count_basic_blocks() == 0 {
        return false;
    }
    // SAFETY: read-only analysis of a verified live function.
    unsafe {
        let loops = crate::loops::Loops::of(f.as_value_ref());
        let pointer_phi = |header| {
            let mut phi = LLVMGetFirstInstruction(header);
            while !phi.is_null() && !LLVMIsAPHINode(phi).is_null() {
                if LLVMGetTypeKind(LLVMTypeOf(phi)) == LLVMTypeKind::LLVMPointerTypeKind {
                    return true;
                }
                phi = LLVMGetNextInstruction(phi);
            }
            false
        };
        f.get_basic_blocks().into_iter().any(|block| {
            let null_operation = block.get_instructions().any(|i| {
                use inkwell::values::InstructionOpcode::{ICmp, Phi, Select};
                let value = i.as_value_ref();
                matches!(i.get_opcode(), ICmp | Phi | Select)
                    && (0..LLVMGetNumOperands(value) as u32)
                        .any(|n| !LLVMIsAConstantPointerNull(LLVMGetOperand(value, n)).is_null())
            });
            null_operation
                && loops
                    .headers_containing(block.as_mut_ptr())
                    .any(pointer_phi)
        })
    }
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
    // Propagate required inlining to callers: thread-index use needs the entry,
    // and unsupported runtime paths need caller facts before LLVM can eliminate
    // them. Never assume a panic guard is false or erase a runtime call ourselves.
    let mut required = HashSet::from(["llvm_metal.linear_thread_index".to_owned()]);
    if !final_input {
        for f in module.get_functions() {
            let name = f.get_name().to_string_lossy().into_owned();
            let unsupported_external = f.count_basic_blocks() == 0
                && f.get_intrinsic_id() == 0
                && !matches!(
                    name.as_str(),
                    "llvm_metal.atomic_add_device_u32" | "memcmp" | "bcmp"
                );
            let indirect = f
                .get_basic_blocks()
                .iter()
                .flat_map(|b| b.get_instructions())
                .filter_map(|i| CallSiteValue::try_from(i).ok())
                .any(|call| call.get_called_fn_value().is_none());
            // Generic pointers carried through memory require inlining/SROA.
            // Pointer-parameter specialization alone cannot retag a struct's
            // stored pointer fields without changing its memory contract.
            let pointer_memory = f
                .get_basic_blocks()
                .iter()
                .flat_map(|b| b.get_instructions())
                .any(|i| {
                    use inkwell::values::InstructionOpcode;
                    match i.get_opcode() {
                        InstructionOpcode::Load => i.get_type().is_pointer_type(),
                        InstructionOpcode::Store => unsafe {
                            use inkwell::llvm_sys::{LLVMTypeKind, core::*};
                            LLVMGetTypeKind(LLVMTypeOf(LLVMGetOperand(i.as_value_ref(), 0)))
                                == LLVMTypeKind::LLVMPointerTypeKind
                        },
                        _ => false,
                    }
                });
            if unsupported_external || indirect || pointer_memory {
                required.insert(name);
            }
        }
    }
    loop {
        let before = required.len();
        for f in module.get_functions() {
            if f.get_basic_blocks()
                .iter()
                .flat_map(|b| b.get_instructions())
                .any(|i| {
                    CallSiteValue::try_from(i)
                        .ok()
                        .and_then(|c| c.get_called_fn_value())
                        .is_some_and(|c| {
                            required.contains(&c.get_name().to_string_lossy().into_owned())
                        })
                })
            {
                required.insert(f.get_name().to_string_lossy().into_owned());
            }
        }
        if required.len() == before {
            break;
        }
    }
    Ok(module
        .get_functions()
        .filter(|f| {
            f.count_basic_blocks() != 0
                && supported_signature(*f, policy)
                && !constant_initializer(*f)
                // Keep producer optimization boundaries until final legalization;
                // early expansion can create new unsupported storage widths.
                && (!final_input || !nullable_pointer_loop(*f))
                && matches!(f.get_linkage(), Linkage::Internal | Linkage::Private)
                && !crate::wide_helpers::recursive(*f)
                && crate::wide_helpers::direct_uses(*f).is_ok()
                && (policy == InliningPolicy::RetainScalar
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
        .filter(|name| name != entry && !required.contains(name))
        .collect())
}

pub(crate) fn specialize(module: &Module<'_>, entry: &str) -> Result<(), String> {
    crate::specialize::run(module, entry)?;
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
    // Producer bitcode can contain recursive fallbacks and escaping formatting
    // callbacks on panic paths. Let LLVM remove unreachable paths; never retain
    // recursive or escaping boundaries. Final legalization still rejects any
    // surviving recursion, function addresses, or indirect calls.
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
        if function.get_name().to_bytes() != entry.as_bytes()
            && (keep || policy != InliningPolicy::Llvm)
        {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_eliminates_pointer_fields_before_retaining_arithmetic() {
        let context = inkwell::context::Context::create();
        let input = crate::parse_ir(
            &context,
            br#"
            define internal i64 @compute(i64 %x) noinline {
                %r = mul i64 %x, 17
                ret i64 %r
            }
            define internal i64 @read_record(ptr %record) noinline {
                %p = load ptr, ptr %record
                %x = load i64, ptr %p
                %r = call i64 @compute(i64 %x)
                ret i64 %r
            }
            define internal i64 @wrap(ptr %record) noinline {
                %r = call i64 @read_record(ptr %record)
                ret i64 %r
            }
            define void @kernel(ptr %p) {
                %record = alloca ptr
                store ptr %p, ptr %record
                %r = call i64 @wrap(ptr %record)
                store i64 %r, ptr %p
                ret void
            }
        "#,
            "pointer-fields",
        )
        .unwrap();
        let prepared = prepare(&input, "kernel", InliningPolicy::Selective).unwrap();
        crate::air::passes(&prepared, "always-inline,default<O3>,globaldce").unwrap();
        validate(&prepared).unwrap();
        assert!(prepared.get_function("compute").is_some());
        assert!(prepared.get_function("read_record").is_none());
        assert!(prepared.get_function("wrap").is_none());
        assert!(!prepared.print_to_string().to_string().contains("load ptr"));
    }

    #[test]
    fn preparation_exposes_constant_state_and_dead_runtime_callbacks() {
        let context = inkwell::context::Context::create();
        let stores = (0..24)
            .map(|i| {
                format!("%p{i} = getelementptr i64, ptr %p, i64 {i}\nstore i64 7, ptr %p{i}\n")
            })
            .collect::<String>();
        let source = format!(
            r#"
            declare void @report(ptr)
            define internal void @callback(ptr %p) {{ ret void }}
            define internal void @initialize(ptr %p) {{ {stores} ret void }}
            define internal i64 @compute(i64 %x) noinline {{
                %r = mul i64 %x, 17
                ret i64 %r
            }}
            define internal i64 @checked(ptr %state, i64 %x) noinline {{
                %n = load i64, ptr %state
                %ok = icmp eq i64 %n, 7
                br i1 %ok, label %valid, label %invalid
            valid:
                %r = call i64 @compute(i64 %x)
                ret i64 %r
            invalid:
                %slot = alloca ptr
                store ptr @callback, ptr %slot
                call void @report(ptr %slot)
                ret i64 0
            }}
            define void @kernel(ptr %p) {{
                %state = alloca [24 x i64]
                call void @initialize(ptr %state)
                %x = load i64, ptr %p
                %r = call i64 @checked(ptr %state, i64 %x)
                store i64 %r, ptr %p
                ret void
            }}
        "#
        );
        let input = crate::parse_ir(&context, source.as_bytes(), "runtime-paths").unwrap();
        let original = input.print_to_string().to_string();
        let prepared = prepare(&input, "kernel", InliningPolicy::Selective).unwrap();
        crate::air::passes(&prepared, "always-inline,default<O3>,globaldce").unwrap();
        validate(&prepared).unwrap();
        assert!(prepared.get_function("compute").is_some());
        for name in ["report", "callback", "initialize", "checked"] {
            assert!(prepared.get_function(name).is_none(), "surviving {name}");
        }
        assert_eq!(input.print_to_string().to_string(), original);

        // If initialization is absent, the callback/runtime path remains live.
        // Preparation must not erase it or treat its guard as an assumption.
        let live = source.replace(
            "call void @initialize(ptr %state)",
            "store i64 0, ptr %state",
        );
        let input = crate::parse_ir(&context, live.as_bytes(), "live-runtime").unwrap();
        let prepared = prepare(&input, "kernel", InliningPolicy::Selective).unwrap();
        crate::air::passes(&prepared, "always-inline,default<O3>,globaldce").unwrap();
        assert!(prepared.get_function("report").is_some());
        assert!(validate(&prepared).unwrap_err().contains("escapes"));
    }
}
