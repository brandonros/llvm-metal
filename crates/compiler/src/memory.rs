//! Expand dynamic nonvolatile memcpy/memset after optimization for AIR.
//!
//! A runtime-zero private memcpy stalls on the tested Apple GPU, including when
//! the intrinsic is placed behind a length != 0 branch. Scalar loops preserve
//! LLVM's zero-length no-access contract without submitting that intrinsic.
//! Dynamic memset uses the same lowering; a memset driver failure is not established.
use inkwell::{
    IntPredicate,
    attributes::{Attribute, AttributeLoc},
    module::{Linkage, Module},
    types::AsTypeRef,
    values::{BasicValue, CallSiteValue, InstructionOpcode},
};

pub(crate) fn lower_dynamic_memory(module: &Module<'_>) -> Result<(), String> {
    let context = module.get_context();
    let copies: Vec<_> = module
        .get_functions()
        .flat_map(|f| f.get_basic_blocks())
        .flat_map(|b| b.get_instructions())
        .filter(|i| i.get_opcode() == InstructionOpcode::Call)
        .filter(|i| {
            CallSiteValue::try_from(*i)
                .ok()
                .and_then(|c| c.get_called_fn_value())
                .is_some_and(|f| {
                    let name = f.get_name().to_bytes();
                    name.starts_with(b"llvm.memcpy.") || name.starts_with(b"llvm.memset.")
                })
                && i.get_operand(2)
                    .and_then(|o| o.value())
                    .is_some_and(|v| !v.into_int_value().is_const())
        })
        .collect();
    if copies.is_empty() {
        return Ok(());
    }
    let mut helpers = std::collections::HashMap::new();
    for copy in copies {
        let argument = |n| {
            copy.get_operand(n)
                .and_then(|o| o.value())
                .ok_or_else(|| "memory intrinsic argument missing".to_string())
        };
        let destination = argument(0)?.into_pointer_value();
        let source = argument(1)?;
        let is_set = source.is_int_value();
        let length = argument(2)?.into_int_value();
        if argument(3)?.into_int_value().get_zero_extended_constant() != Some(0) {
            return Err("dynamic volatile memory intrinsic requires explicit legalization".into());
        }
        // SAFETY: destination and any pointer source have verified LLVM pointer types.
        let key = unsafe {
            (
                is_set,
                inkwell::llvm_sys::core::LLVMGetPointerAddressSpace(
                    destination.get_type().as_type_ref(),
                ),
                if is_set {
                    0
                } else {
                    inkwell::llvm_sys::core::LLVMGetPointerAddressSpace(
                        source.get_type().as_type_ref(),
                    )
                },
                length.get_type().get_bit_width(),
            )
        };
        let helper = if let Some(helper) = helpers.get(&key) {
            *helper
        } else {
            let helper = module.add_function(
                "__llvm_metal_memory_loop",
                context.void_type().fn_type(
                    &[
                        destination.get_type().into(),
                        source.get_type().into(),
                        length.get_type().into(),
                    ],
                    false,
                ),
                Some(Linkage::Internal),
            );
            helper.add_attribute(
                AttributeLoc::Function,
                context.create_enum_attribute(Attribute::get_named_enum_kind_id("alwaysinline"), 0),
            );
            let entry = context.append_basic_block(helper, "copy.entry");
            let even_head = context.append_basic_block(helper, "copy.even.head");
            let even_body = context.append_basic_block(helper, "copy.even.body");
            let odd_head = context.append_basic_block(helper, "copy.odd.head");
            let odd_body = context.append_basic_block(helper, "copy.odd.body");
            let tail_check = context.append_basic_block(helper, "copy.tail.check");
            let tail_body = context.append_basic_block(helper, "copy.tail.body");
            let exit = context.append_basic_block(helper, "copy.exit");
            let builder = context.create_builder();
            let from = helper.get_nth_param(1).unwrap();
            let to = helper.get_nth_param(0).unwrap().into_pointer_value();
            let original_count = helper.get_nth_param(2).unwrap().into_int_value();
            builder.position_at_end(entry);
            // Intrinsic lengths are unsigned, whereas GEP sign-extends narrow
            // indices. Widen before arithmetic so i8 offsets >=128 stay positive
            // and an i1 count never executes a shift by its own bit width.
            let count = if original_count.get_type().get_bit_width() < 64 {
                builder
                    .build_int_z_extend(original_count, context.i64_type(), "copy.count")
                    .map_err(|e| e.to_string())?
            } else {
                original_count
            };
            let ty = count.get_type();
            let pairs = builder
                .build_right_shift(count, ty.const_int(1, false), false, "copy.pairs")
                .map_err(|e| e.to_string())?;
            builder
                .build_unconditional_branch(even_head)
                .map_err(|e| e.to_string())?;
            // A contiguous byte loop can become memcpy/memset during later
            // optimization. Two stride-2 loops followed by an odd-byte tail
            // express the same nonoverlapping copy or fill without that loop idiom.
            // Pair counts keep all indices in range even for n = UINT64_MAX.
            let emit_byte = |offset| -> Result<(), String> {
                // SAFETY: indices are below count, and the intrinsic provides
                // valid byte ranges. The zero path performs no memory access.
                let destination_byte =
                    unsafe { builder.build_gep(context.i8_type(), to, &[offset], "copy.to") }
                        .map_err(|e| e.to_string())?;
                let byte = if is_set {
                    from
                } else {
                    // SAFETY: a nonvolatile memcpy source is a pointer to at
                    // least count readable bytes; offset is strictly below count.
                    let source_byte = unsafe {
                        builder.build_gep(
                            context.i8_type(),
                            from.into_pointer_value(),
                            &[offset],
                            "copy.from",
                        )
                    }
                    .map_err(|e| e.to_string())?;
                    let byte = builder
                        .build_load(context.i8_type(), source_byte, "copy.byte")
                        .map_err(|e| e.to_string())?;
                    byte.as_instruction_value()
                        .unwrap()
                        .set_alignment(1)
                        .map_err(|e| e.to_string())?;
                    byte
                };
                builder
                    .build_store(destination_byte, byte)
                    .map_err(|e| e.to_string())?
                    .set_alignment(1)
                    .map_err(|e| e.to_string())?;
                Ok(())
            };
            for (parity, head, body, predecessor, next_block) in [
                (0, even_head, even_body, entry, odd_head),
                (1, odd_head, odd_body, even_head, tail_check),
            ] {
                builder.position_at_end(head);
                let index = builder
                    .build_phi(ty, "copy.index")
                    .map_err(|e| e.to_string())?;
                index.add_incoming(&[(&ty.const_zero(), predecessor)]);
                let more = builder
                    .build_int_compare(
                        IntPredicate::ULT,
                        index.as_basic_value().into_int_value(),
                        pairs,
                        "copy.more",
                    )
                    .map_err(|e| e.to_string())?;
                builder
                    .build_conditional_branch(more, body, next_block)
                    .map_err(|e| e.to_string())?;
                builder.position_at_end(body);
                let doubled = builder
                    .build_left_shift(
                        index.as_basic_value().into_int_value(),
                        ty.const_int(1, false),
                        "copy.pair.offset",
                    )
                    .map_err(|e| e.to_string())?;
                let offset = builder
                    .build_int_add(doubled, ty.const_int(parity, false), "copy.offset")
                    .map_err(|e| e.to_string())?;
                emit_byte(offset)?;
                let next = builder
                    .build_int_add(
                        index.as_basic_value().into_int_value(),
                        ty.const_int(1, false),
                        "copy.next",
                    )
                    .map_err(|e| e.to_string())?;
                index.add_incoming(&[(&next, body)]);
                builder
                    .build_unconditional_branch(head)
                    .map_err(|e| e.to_string())?;
            }
            builder.position_at_end(tail_check);
            let odd = builder
                .build_and(count, ty.const_int(1, false), "copy.odd")
                .map_err(|e| e.to_string())?;
            let has_tail = builder
                .build_int_compare(IntPredicate::NE, odd, ty.const_zero(), "copy.has.tail")
                .map_err(|e| e.to_string())?;
            builder
                .build_conditional_branch(has_tail, tail_body, exit)
                .map_err(|e| e.to_string())?;
            builder.position_at_end(tail_body);
            let last = builder
                .build_int_sub(count, ty.const_int(1, false), "copy.last")
                .map_err(|e| e.to_string())?;
            emit_byte(last)?;
            builder
                .build_unconditional_branch(exit)
                .map_err(|e| e.to_string())?;
            builder.position_at_end(exit);
            builder.build_return(None).map_err(|e| e.to_string())?;
            helpers.insert(key, helper);
            helper
        };
        // LLVM's TargetLibraryInfo honors this attribute; the stride-2 form
        // is also tested without it because Apple's optimizer is independent.
        copy.get_parent()
            .and_then(|b| b.get_parent())
            .ok_or("memory intrinsic has no parent function")?
            .add_attribute(
                AttributeLoc::Function,
                context.create_string_attribute(
                    if is_set {
                        "no-builtin-memset"
                    } else {
                        "no-builtin-memcpy"
                    },
                    "",
                ),
            );
        let builder = context.create_builder();
        builder.position_before(&copy);
        builder
            .build_call(
                helper,
                &[destination.into(), source.into(), length.into()],
                "",
            )
            .map_err(|e| e.to_string())?;
        copy.erase_from_basic_block();
    }
    // Inlining fixes successor PHIs and repeated calls in the same source block.
    // No loop-idiom/InstCombine pass may run afterward and recreate intrinsics.
    crate::air::passes(module, "always-inline,globaldce")
}
