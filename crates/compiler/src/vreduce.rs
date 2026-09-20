//! Scalar expansion of vector reduction intrinsics, which AIR cannot lower.
use inkwell::{
    IntPredicate,
    module::Module,
    types::IntType,
    values::{AsValueRef, BasicValueEnum, CallSiteValue},
};

fn integer_seed<'ctx>(ty: IntType<'ctx>, op: &str) -> Option<BasicValueEnum<'ctx>> {
    let words = ty.get_bit_width().div_ceil(64) as usize;
    let mut seed = vec![u64::MAX; words];
    match op {
        "add" | "or" | "xor" | "umax" => Some(ty.const_int(0, false).into()),
        "mul" | "umin" | "and" => Some(ty.const_all_ones().into()),
        // Signed extremes: the seeds fold toward the opposite signed bound.
        "smin" => {
            seed[words - 1] = 0x7fff_ffff_ffff_ffff;
            Some(ty.const_int_arbitrary_precision(&seed).into())
        }
        "smax" => {
            seed.fill(0);
            seed[words - 1] = 0x8000_0000_0000_0000;
            Some(ty.const_int_arbitrary_precision(&seed).into())
        }
        _ => None,
    }
}

/// `llvm.vector.reduce.<op>.vN` folds a vector down to one scalar. Rebuild it
/// as an extract-and-fold chain so surviving reductions keep scalar types.
pub(crate) fn lower(module: &Module<'_>) -> Result<(), String> {
    let context = module.get_context();
    let builder = context.create_builder();
    for function in module.get_functions() {
        if function.count_basic_blocks() == 0 {
            continue;
        }
        for block in function.get_basic_blocks() {
            let instructions: Vec<_> = block.get_instructions().collect();
            for instruction in instructions {
                let Ok(call) = CallSiteValue::try_from(instruction) else {
                    continue;
                };
                let Some(callee) = call.get_called_fn_value() else {
                    continue;
                };
                let name = callee.get_name().to_string_lossy().into_owned();
                let Some(spec) = name.strip_prefix("llvm.vector.reduce.") else {
                    continue;
                };
                let op = spec.split('.').next().unwrap_or_default();
                if !matches!(
                    op,
                    "add" | "mul" | "and" | "or" | "xor" | "umin" | "umax" | "smin" | "smax"
                        | "fadd" | "fmul"
                ) {
                    return Err(format!("unsupported vector reduction: {name}"));
                }
                let call_instruction = instruction;
                let operands: Vec<BasicValueEnum> = (0..call.count_arguments())
                    .filter_map(|i| {
                        call_instruction.get_operand(i).and_then(|o| o.value())
                    })
                    .collect();
                let (vector, mut folded) = match operands.as_slice() {
                    [vector] if vector.is_vector_value() => (*vector, None),
                    [seed, vector]
                        if vector.is_vector_value()
                            && (seed.is_int_value() || seed.is_float_value()) =>
                    {
                        (*vector, Some(*seed))
                    }
                    _ => return Err(format!("unsupported vector reduction: {name}")),
                };
                let lanes = vector.get_type().into_vector_type().get_size();
                builder.position_before(&instruction);
                for lane in 0..lanes {
                    let index = context.i32_type().const_int(lane as u64, false);
                    let element = builder
                        .build_extract_element(
                            vector.into_vector_value(),
                            index,
                            "reduce.lane",
                        )
                        .map_err(|e| e.to_string())?;
                    folded = Some(match folded {
                        None => element,
                        Some(accumulator) if element.is_int_value() => {
                            let acc = accumulator.into_int_value();
                            let elem = element.into_int_value();
                            let folded = match op {
                                "add" => builder.build_int_add(acc, elem, "reduce.add"),
                                "mul" => builder.build_int_mul(acc, elem, "reduce.mul"),
                                "and" => builder.build_and(acc, elem, "reduce.and"),
                                "or" => builder.build_or(acc, elem, "reduce.or"),
                                "xor" => builder.build_xor(acc, elem, "reduce.xor"),
                                "umin" | "umax" | "smin" | "smax" => {
                                    let predicate = match op {
                                        "umin" => IntPredicate::ULT,
                                        "umax" => IntPredicate::UGT,
                                        "smin" => IntPredicate::SLT,
                                        _ => IntPredicate::SGT,
                                    };
                                    let take_acc = builder
                                        .build_int_compare(predicate, acc, elem, "reduce.cmp")
                                        .map_err(|e| e.to_string())?;
                                    builder
                                        .build_select(take_acc, acc, elem, "reduce.cmp")
                                        .map(BasicValueEnum::into_int_value)
                                }
                                _ => unreachable!("checked above"),
                            };
                            folded.map_err(|e| e.to_string())?.into()
                        }
                        Some(accumulator) if element.is_float_value() => {
                            let acc = accumulator.into_float_value();
                            let elem = element.into_float_value();
                            let folded = match op {
                                "fadd" => builder.build_float_add(acc, elem, "reduce.fadd"),
                                "fmul" => builder.build_float_mul(acc, elem, "reduce.fmul"),
                                _ => unreachable!("checked above"),
                            };
                            folded.map_err(|e| e.to_string())?.into()
                        }
                        Some(_) => return Err(format!("unsupported vector reduction: {name}")),
                    });
                }
                let folded = match folded {
                    Some(value) => value,
                    None => {
                        // A start-free reduction folds from the op's identity.
                        let element = vector
                            .get_type()
                            .into_vector_type()
                            .get_element_type();
                        match element {
                            t if t.is_int_type() => integer_seed(t.into_int_type(), op)
                                .ok_or_else(|| format!("unsupported vector reduction: {name}"))?,
                            t if t.is_float_type() => match op {
                                "fadd" => t.into_float_type().const_float(0.0).into(),
                                _ => t.into_float_type().const_float(1.0).into(),
                            },
                            _ => {
                                return Err(format!("unsupported vector reduction: {name}"));
                            }
                        }
                    }
                };
                // SAFETY: verified module; the call result is replaced wholly.
                unsafe {
                    inkwell::llvm_sys::core::LLVMReplaceAllUsesWith(
                        call.as_value_ref(),
                        folded.as_value_ref(),
                    );
                }
                call_instruction.erase_from_basic_block();
            }
        }
    }
    Ok(())
}
