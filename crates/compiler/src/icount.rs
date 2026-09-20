//! Normalize integer intrinsics at widths the AIR profile cannot lower.
//! `llvm.ctpop` has no AIR form at any width; counting, absolute-value, and
//! min/max intrinsics are only lowered for scalar i8/i16/i32/i64.
use inkwell::{
    IntPredicate,
    module::Module,
    types::IntType,
    values::{AsValueRef, BasicValueEnum, CallSiteValue, IntValue},
};

fn supported(width: u32) -> bool {
    matches!(width, 8 | 16 | 32 | 64)
}

fn next_supported(width: u32) -> Option<u32> {
    [8, 16, 32, 64].into_iter().find(|w| *w > width)
}

/// Repeating low-byte pattern `0x01`, `0x55`, `0x33`, `0x0f` at width `width`.
fn mask(ty: IntType<'_>, byte: u64) -> IntValue<'_> {
    let words = ty.get_bit_width().div_ceil(64) as usize;
    let mut pattern = vec![0u64; words];
    for (i, word) in pattern.iter_mut().enumerate() {
        let mut value = 0u64;
        for byte_index in 0..8 {
            let bit = i * 64 + byte_index * 8;
            if bit + 8 <= ty.get_bit_width() as usize {
                value |= byte << (byte_index * 8);
            } else if bit < ty.get_bit_width() as usize {
                value |= (byte & ((1u64 << (ty.get_bit_width() as usize - bit)) - 1))
                    << (byte_index * 8);
            }
        }
        *word = value;
    }
    ty.const_int_arbitrary_precision(&pattern)
}

/// SWAR popcount at any integer width up to i128.
fn popcount<'ctx>(
    builder: &inkwell::builder::Builder<'ctx>,
    value: IntValue<'ctx>,
) -> Result<IntValue<'ctx>, String> {
    let ty = value.get_type();
    let width = ty.get_bit_width();
    let one = ty.const_int(1, false);
    let mut x = builder
        .build_int_sub(
            value,
            builder
                .build_and(
                    builder
                        .build_right_shift(value, one, false, "pop.shr1")
                        .map_err(|e| e.to_string())?,
                    mask(ty, 0x55),
                    "pop.m1",
                )
                .map_err(|e| e.to_string())?,
            "pop.d1",
        )
        .map_err(|e| e.to_string())?;
    let m2 = mask(ty, 0x33);
    x = builder
        .build_int_add(
            builder.build_and(x, m2, "pop.lo").map_err(|e| e.to_string())?,
            builder
                .build_and(
                    builder
                        .build_right_shift(x, ty.const_int(2, false), false, "pop.shr2")
                        .map_err(|e| e.to_string())?,
                    m2,
                    "pop.hi",
                )
                .map_err(|e| e.to_string())?,
            "pop.d2",
        )
        .map_err(|e| e.to_string())?;
    if width > 4 {
        x = builder
            .build_and(
                builder
                    .build_int_add(
                        x,
                        builder
                            .build_right_shift(x, ty.const_int(4, false), false, "pop.shr4")
                            .map_err(|e| e.to_string())?,
                        "pop.d4",
                    )
                    .map_err(|e| e.to_string())?,
                mask(ty, 0x0f),
                "pop.m4",
            )
            .map_err(|e| e.to_string())?;
    }
    if width > 8 {
        // Sum the per-byte counts in the high byte via a multiply carry.
        x = builder
            .build_right_shift(
                builder
                    .build_int_mul(x, mask(ty, 0x01), "pop.mul")
                    .map_err(|e| e.to_string())?,
                ty.const_int((width - 8) as u64, false),
                false,
                "pop.carry",
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(x)
}

pub(crate) fn lower(module: &Module<'_>) -> Result<(), String> {
    let context = module.get_context();
    let builder = context.create_builder();
    let mut rewritten: std::collections::HashSet<inkwell::values::FunctionValue<'_>> =
        std::collections::HashSet::new();
    for function in module.get_functions() {
        if function.count_basic_blocks() == 0 {
            continue;
        }
        for block in function.get_basic_blocks() {
            let instructions: Vec<_> = block.get_instructions().collect();
            for instruction in instructions {
                if instruction.get_opcode() == inkwell::values::InstructionOpcode::BitCast {
                    // A <N x i1> to iN bitcast packs comparison lanes; rebuild
                    // it as extract/zext/shift so scalarizer can dissolve the
                    // vector entirely.
                    let Some(result) = instruction
                        .get_type()
                        .is_int_type()
                        .then(|| instruction.get_type().into_int_type())
                    else {
                        continue;
                    };
                    let Some(vector) = instruction
                        .get_operand(0)
                        .and_then(|o| o.value())
                        .and_then(|v| v.is_vector_value().then(|| v.into_vector_value()))
                    else {
                        continue;
                    };
                    let vector_type = vector.get_type();
                    if !vector_type.get_element_type().is_int_type()
                        || vector_type
                            .get_element_type()
                            .into_int_type()
                            .get_bit_width()
                            != 1
                        || vector_type.get_size() != result.get_bit_width()
                    {
                        continue;
                    }
                    builder.position_before(&instruction);
                    let mut packed = result.const_zero();
                    for lane in 0..vector_type.get_size() {
                        let index = context.i32_type().const_int(lane as u64, false);
                        let element = builder
                            .build_extract_element(vector, index, "pack.lane")
                            .map_err(|e| e.to_string())?;
                        let bit = builder
                            .build_int_z_extend(
                                element.into_int_value(),
                                result,
                                "pack.bit",
                            )
                            .map_err(|e| e.to_string())?;
                        let shifted = builder
                            .build_left_shift(bit, result.const_int(lane as u64, false), "pack.shift")
                            .map_err(|e| e.to_string())?;
                        packed = builder
                            .build_or(packed, shifted, "pack.or")
                            .map_err(|e| e.to_string())?;
                    }
                    // SAFETY: verified module; the bitcast result is replaced wholly.
                    unsafe {
                        inkwell::llvm_sys::core::LLVMReplaceAllUsesWith(
                            instruction.as_value_ref(),
                            packed.as_value_ref(),
                        );
                    }
                    instruction.erase_from_basic_block();
                    continue;
                }
                let Ok(call) = CallSiteValue::try_from(instruction) else {
                    continue;
                };
                let Some(callee) = call.get_called_fn_value() else {
                    continue;
                };
                let name = callee.get_name().to_string_lossy().into_owned();
                if std::env::var_os("LLVM_METAL_TRACE").is_some() && name.starts_with("llvm.") {
                    eprintln!("[trace] icount sees {name}");
                }
                let narrow = call
                    .try_as_basic_value()
                    .basic()
                    .and_then(|v| v.is_int_value().then(|| v.into_int_value().get_type()));
                let replacement: BasicValueEnum = match name.split('.').nth(1) {
                    Some("ctpop") => {
                        let ty = narrow.ok_or_else(|| {
                            format!("unsupported population count: {name}")
                        })?;
                        let value = instruction
                            .get_operand(0)
                            .and_then(|o| o.value())
                            .and_then(|v| v.is_int_value().then(|| v.into_int_value()))
                            .ok_or_else(|| format!("unsupported population count: {name}"))?;
                        builder.position_before(&instruction);
                        popcount(&builder, value)?.into()
                    }
                    Some("ctlz") | Some("cttz") => {
                        let ty = narrow.ok_or_else(|| {
                            format!("unsupported zero count: {name}")
                        })?;
                        if supported(ty.get_bit_width()) {
                            continue;
                        }
                        let Some(width) = next_supported(ty.get_bit_width()) else {
                            return Err(format!("unsupported zero count: {name}"));
                        };
                        let value = instruction
                            .get_operand(0)
                            .and_then(|o| o.value())
                            .and_then(|v| v.is_int_value().then(|| v.into_int_value()))
                            .ok_or_else(|| format!("unsupported zero count: {name}"))?;
                        let flag = instruction
                            .get_operand(1)
                            .and_then(|o| o.value())
                            .ok_or_else(|| format!("unsupported zero count: {name}"))?;
                        builder.position_before(&instruction);
                        let wide = context
                            .custom_width_int_type(std::num::NonZero::new(width).unwrap())
                            .map_err(|e| e.to_string())?;
                        let stem = if name.contains("ctlz") { "llvm.ctlz" } else { "llvm.cttz" };
                        let wide_name = format!("{stem}.i{width}");
                        let signature = wide.fn_type(&[wide.into(), context.bool_type().into()], false);
                        let wide_callee = module
                            .get_function(&wide_name)
                            .unwrap_or_else(|| module.add_function(&wide_name, signature, None));
                        let extended = builder
                            .build_int_z_extend(value, wide, "count.zext")
                            .map_err(|e| e.to_string())?;
                        let counted = builder
                            .build_call(wide_callee, &[extended.into(), flag.into()], "count.wide")
                            .map_err(|e| e.to_string())?
                            .try_as_basic_value()
                            .basic()
                            .ok_or("zero count must produce a value")?
                            .into_int_value();
                        let adjusted = if name.contains("ctlz") {
                            // zext adds width-narrow leading zeros to the count.
                            builder
                                .build_int_sub(
                                    counted,
                                    wide.const_int((width - ty.get_bit_width()) as u64, false),
                                    "count.adjust",
                                )
                                .map_err(|e| e.to_string())?
                        } else {
                            // zext keeps trailing zeros; an all-zero narrow input
                            // reports the narrow width, not the wide width.
                            let cap = wide.const_int(ty.get_bit_width() as u64, false);
                            let under = builder
                                .build_int_compare(IntPredicate::ULT, counted, cap, "count.cmp")
                                .map_err(|e| e.to_string())?;
                            builder
                                .build_select(under, counted, cap, "count.cap")
                                .map_err(|e| e.to_string())?
                                .into_int_value()
                        };
                        builder
                            .build_int_truncate(adjusted, ty, "count.trunc")
                            .map_err(|e| e.to_string())?
                            .into()
                    }
                    _ => continue,
                };
                // SAFETY: verified module; the call result is replaced wholly.
                unsafe {
                    inkwell::llvm_sys::core::LLVMReplaceAllUsesWith(
                        call.as_value_ref(),
                        replacement.as_value_ref(),
                    );
                }
                instruction.erase_from_basic_block();
                rewritten.insert(callee);
            }
        }
    }
    if std::env::var_os("LLVM_METAL_TRACE").is_some() {
        eprintln!("[trace] icount cleanup: {} callees", rewritten.len());
    }
    for callee in rewritten {
        // A fully expanded intrinsic declaration is still an odd-width ABI
        // to validation; drop it once no use remains.
        let unused = unsafe { inkwell::llvm_sys::core::LLVMGetFirstUse(callee.as_value_ref()) }
            .is_null();
        if std::env::var_os("LLVM_METAL_SKIP_DELETE").is_none() && callee.count_basic_blocks() == 0 && unused {
            unsafe { inkwell::llvm_sys::core::LLVMDeleteFunction(callee.as_value_ref()) };
        }
    }
    Ok(())
}
