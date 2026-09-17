//! Narrow i128 legalization. Call only on a disposable verified module clone.
use inkwell::module::Module;
use std::ffi::CStr;

pub(crate) fn lower(module: &Module<'_>) -> Result<(), String> {
    unsafe extern "C" {
        fn LLVMMetalLowerWideIntegers(
            module: inkwell::llvm_sys::prelude::LLVMModuleRef,
        ) -> *mut std::ffi::c_char;
    }
    // SAFETY: caller owns the verified module; dispose the native diagnostic.
    unsafe {
        let error = LLVMMetalLowerWideIntegers(module.as_mut_ptr());
        if !error.is_null() {
            let text = CStr::from_ptr(error).to_string_lossy().into_owned();
            inkwell::llvm_sys::core::LLVMDisposeMessage(error);
            return Err(text);
        }
    }
    module.verify().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::{
        OptimizationLevel,
        context::Context,
        targets::{InitializationConfig, Target},
    };

    #[test]
    fn lowered_arithmetic_matches_rust_u128_and_i128() {
        Target::initialize_native(&InitializationConfig::default()).unwrap();
        let edges = [0u64, 1, u32::MAX as u64, 1 << 32, 1 << 63, u64::MAX];
        // These constants exercise nonzero upper limbs and wrapping overflow,
        // in addition to the zero-extended 64x64 multiply fixture on Metal.

        let mut operations = [
            "add", "sub", "mul", "and", "xor", "eq", "ne", "ult", "ule", "ugt", "uge", "slt",
            "sle", "sgt", "sge",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        for shift in [0, 1, 31, 52, 63, 64, 65, 127] {
            operations.extend([
                format!("lshr:{shift}"),
                format!("ashr:{shift}"),
                format!("shl:{shift}"),
            ]);
        }
        for ca in [
            0u128,
            0x12345678ffffffff0000000000000001,
            0x92345678ffffffff0000000000000001,
            u128::MAX,
        ] {
            for cb in [ca, 0xfedcba9876543210ffffffffffffffffu128] {
                for extension in ["zext", "sext"] {
                    for operation in &operations {
                        let (opcode, shift) = operation
                            .split_once(':')
                            .map_or((operation.as_str(), None), |(a, b)| {
                                (a, Some(b.parse::<u32>().unwrap()))
                            });
                        let rhs = shift.map_or("%b".to_owned(), |s| s.to_string());
                        let comparison = [
                            "eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge",
                        ]
                        .contains(&opcode);
                        let expression = if comparison {
                            format!(
                                "%condition = icmp {opcode} i128 %a, %b\n%r = select i1 %condition, i128 %a, i128 %b"
                            )
                        } else {
                            format!("%r = {opcode} i128 %a, {rhs}")
                        };
                        let source = format!(
                            "define void @probe(i64 %x, i64 %y, ptr %out) {{
                    %a0 = {extension} i64 %x to i128
                    %b0 = {extension} i64 %y to i128
                    %a = add i128 %a0, {ca}
                    %b = add i128 %b0, {cb}
                    {expression}
                    store i128 %r, ptr %out, align 1
                    ret void
                }}"
                        );
                        let context = Context::create();
                        let module =
                            crate::parse_ir(&context, source.as_bytes(), "wide-test").unwrap();
                        lower(&module).unwrap();
                        // Native execution tests this pass, not AIR or GPU support.
                        let engine = module
                            .create_jit_execution_engine(OptimizationLevel::None)
                            .unwrap();
                        let function = unsafe {
                            engine
                                .get_function::<unsafe extern "C" fn(u64, u64, *mut u8)>("probe")
                                .unwrap()
                        };
                        for x in edges {
                            for y in edges {
                                let extend = |x: u64| {
                                    if extension == "sext" {
                                        (x as i64 as i128) as u128
                                    } else {
                                        x as u128
                                    }
                                };
                                let a = extend(x).wrapping_add(ca);
                                let b = extend(y).wrapping_add(cb);
                                let expected = match opcode {
                                    "add" => a.wrapping_add(b),
                                    "sub" => a.wrapping_sub(b),
                                    "xor" => a ^ b,
                                    "eq" => {
                                        if a == b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "ne" => {
                                        if a != b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "ult" => {
                                        if a < b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "ule" => {
                                        if a <= b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "ugt" => {
                                        if a > b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "uge" => {
                                        if a >= b {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "slt" => {
                                        if (a as i128) < (b as i128) {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "sle" => {
                                        if (a as i128) <= (b as i128) {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "sgt" => {
                                        if (a as i128) > (b as i128) {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "sge" => {
                                        if (a as i128) >= (b as i128) {
                                            a
                                        } else {
                                            b
                                        }
                                    }
                                    "mul" => a.wrapping_mul(b),
                                    "and" => a & b,
                                    "lshr" => a >> shift.unwrap(),
                                    "shl" => a << shift.unwrap(),
                                    "ashr" => ((a as i128) >> shift.unwrap()) as u128,
                                    _ => unreachable!(),
                                };
                                let mut output = [0xa5u8; 18];
                                // SAFETY: fixed reviewed IR writes exactly 16 bytes, alignment 1.
                                unsafe {
                                    function.call(x, y, output.as_mut_ptr().add(1));
                                }
                                assert_eq!(
                                    u128::from_le_bytes(output[1..17].try_into().unwrap()),
                                    expected,
                                    "{extension} {operation} {x:x} {y:x}"
                                );
                                assert_eq!((output[0], output[17]), (0xa5, 0xa5));
                            }
                        }
                    }
                }
            }
        }
    }
}
