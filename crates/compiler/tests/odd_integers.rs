//! SROA emits byte-sized scalar integers which are not machine-word widths.
//! Check exact memory spans as well as arithmetic: i24 stores three bytes even
//! though its allocation size and ABI alignment are four bytes.
use inkwell::{context::Context, targets::TargetData};
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{
    air::{AIR_LAYOUT, legalize},
    parse_ir,
};

const NVPTX_LAYOUT: &str = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64";
const RECORD_BYTES: usize = 224;
const WIDTHS: [u32; 4] = [24, 40, 48, 56];
const OUTPUTS: [&str; 17] = [
    "x",
    "sum",
    "difference",
    "product",
    "bits_and",
    "bits_or",
    "bits_xor",
    "left",
    "right",
    "signed_right",
    "accumulated",
    "selected",
    "quotient",
    "remainder",
    "signed_quotient",
    "signed_remainder",
    "narrow",
];

fn interface() -> KernelInterface {
    KernelInterface {
        schema: 1,
        entry: "kernel".into(),
        calling_convention: "C".into(),
        invocations: Some(1),
        dispatch: Dispatch::Single,
        aliasing: "all buffers disjoint".into(),
        arguments: vec![BufferArgument {
            name: "data".into(),
            kind: "buffer".into(),
            access: Access::ReadWrite,
            bytes: RECORD_BYTES,
            alignment: 8,
        }],
    }
}

fn source(width: u32) -> String {
    let minimum = 1u64 << (width - 1);
    let mut source = format!(
        r#"target triple = "nvptx64-nvidia-cuda"
target datalayout = "{NVPTX_LAYOUT}"
define void @kernel(ptr %p) {{
entry:
 %xp = getelementptr i8, ptr %p, i64 1
 %x = load i{width}, ptr %xp, align 1
 %yp = getelementptr i8, ptr %p, i64 11
 %y = load i{width}, ptr %yp, align 1
 %shiftp = getelementptr i8, ptr %p, i64 20
 %shift_raw = load i8, ptr %shiftp, align 1
 %shift_mod = urem i8 %shift_raw, {width}
 %shift = zext i8 %shift_mod to i{width}
 %np = getelementptr i8, ptr %p, i64 21
 %n8 = load i8, ptr %np, align 1
 %n = zext i8 %n8 to i32
 %wp = getelementptr i8, ptr %p, i64 24
 %wide = load i64, ptr %wp, align 8
 %narrow = trunc i64 %wide to i{width}
 %sum = add i{width} %x, %y
 %difference = sub i{width} %x, %y
 %product = mul i{width} %x, %y
 %bits_and = and i{width} %x, %y
 %bits_or = or i{width} %x, %y
 %bits_xor = xor i{width} %x, %y
 %left = shl i{width} %x, %shift
 %right = lshr i{width} %x, %shift
 %signed_right = ashr i{width} %x, %shift
 %less = icmp slt i{width} %x, %y
 %selected = select i1 %less, i{width} %x, i{width} %y
 %zero_divisor = icmp eq i{width} %y, 0
 %divisor = select i1 %zero_divisor, i{width} 1, i{width} %y
 %quotient = udiv i{width} %x, %divisor
 %remainder = urem i{width} %x, %divisor
 %minimum = icmp eq i{width} %x, {minimum}
 %negative_one = icmp eq i{width} %divisor, -1
 %overflow = and i1 %minimum, %negative_one
 %signed_divisor = select i1 %overflow, i{width} 1, i{width} %divisor
 %signed_quotient = sdiv i{width} %x, %signed_divisor
 %signed_remainder = srem i{width} %x, %signed_divisor
 %empty = icmp eq i32 %n, 0
 br i1 %empty, label %exit, label %loop
loop:
 %i = phi i32 [0, %entry], [%j, %loop]
 %acc = phi i{width} [%x, %entry], [%next, %loop]
 %next = add i{width} %acc, %y
 %j = add i32 %i, 1
 %done = icmp eq i32 %j, %n
 br i1 %done, label %exit, label %loop
exit:
 %accumulated = phi i{width} [%x, %entry], [%next, %loop]
"#
    );
    for (i, output) in OUTPUTS.iter().enumerate() {
        source.push_str(&format!(" %o{i} = getelementptr i8, ptr %p, i64 {}\n store i{width} %{output}, ptr %o{i}, align 1\n", 33 + i * 10));
    }
    source.push_str(&format!(
        r#" %unsigned = zext i{width} %x to i64
 %signed = sext i{width} %x to i64
 %uout = getelementptr i8, ptr %p, i64 200
 %sout = getelementptr i8, ptr %p, i64 208
 store i64 %unsigned, ptr %uout, align 8
 store i64 %signed, ptr %sout, align 8
 %unsigned_less = icmp ult i{width} %x, %y
 %ult = zext i1 %unsigned_less to i8
 %slt = zext i1 %less to i8
 %ucmp = getelementptr i8, ptr %p, i64 216
 %scmp = getelementptr i8, ptr %p, i64 217
 store i8 %ult, ptr %ucmp, align 1
 store i8 %slt, ptr %scmp, align 1
 ret void
}}
"#
    ));
    source
}

#[test]
fn scalar_and_aggregate_layouts_match_without_rounding_store_spans() {
    let context = Context::create();
    let source = TargetData::create(NVPTX_LAYOUT);
    let destination = TargetData::create(AIR_LAYOUT);
    for width in WIDTHS {
        let scalar = context
            .custom_width_int_type(std::num::NonZeroU32::new(width).unwrap())
            .unwrap();
        let allocation = if width == 24 { 4 } else { 8 };
        for data in [&source, &destination] {
            assert_eq!(data.get_bit_size(&scalar), u64::from(width));
            assert_eq!(data.get_store_size(&scalar), u64::from(width / 8));
            assert_eq!(data.get_abi_size(&scalar), allocation);
            assert_eq!(u64::from(data.get_abi_alignment(&scalar)), allocation);
        }
        let array = scalar.array_type(3);
        assert_eq!(
            source.get_abi_size(&array),
            destination.get_abi_size(&array)
        );
        let aggregate = context.struct_type(
            &[
                context.i8_type().into(),
                scalar.into(),
                array.into(),
                context.i8_type().into(),
            ],
            false,
        );
        assert_eq!(
            source.get_abi_size(&aggregate),
            destination.get_abi_size(&aggregate)
        );
        assert_eq!(
            source.get_abi_alignment(&aggregate),
            destination.get_abi_alignment(&aggregate)
        );
        for field in 0..4 {
            assert_eq!(
                source.offset_of_element(&aggregate, field),
                destination.offset_of_element(&aggregate, field)
            );
        }
    }
}

#[test]
fn byte_width_arithmetic_legalizes_and_preserves_input_module() {
    let context = Context::create();
    for width in WIDTHS {
        let module = parse_ir(&context, source(width).as_bytes(), "odd-integer").unwrap();
        let original = module.print_to_string().to_string();
        let (air, _) = legalize(&module, &interface()).unwrap();
        air.verify().unwrap();
        let text = air.print_to_string().to_string();
        for odd in WIDTHS {
            assert!(
                !text
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .any(|token| token == format!("i{odd}")),
                "odd integer survived final AIR promotion: i{odd}"
            );
        }
        assert_eq!(module.print_to_string().to_string(), original);
    }
}

#[test]
fn unsupported_widths_and_changed_abi_remain_refused() {
    let context = Context::create();
    for width in [2, 7, 9, 15, 17, 23, 25, 31, 33, 63, 65, 72, 96] {
        let ir = format!(
            r#"target triple = "nvptx64-nvidia-cuda"
target datalayout = "{NVPTX_LAYOUT}"
define void @kernel(ptr %p) {{ %x = load i{width}, ptr %p, align 1
 store i{width} %x, ptr %p, align 1
 ret void }}"#
        );
        let module = parse_ir(&context, ir.as_bytes(), "unsupported-width").unwrap();
        assert!(legalize(&module, &interface()).is_err(), "i{width}");
    }
    for width in WIDTHS {
        let vector = format!(
            r#"target triple = "nvptx64-nvidia-cuda"
target datalayout = "{NVPTX_LAYOUT}"
define void @kernel(ptr %p) {{ %x = load <2 x i{width}>, ptr %p, align 1
 store <2 x i{width}> %x, ptr %p, align 1
 ret void }}"#
        );
        let module = parse_ir(&context, vector.as_bytes(), "odd-vector").unwrap();
        let before = module.print_to_string().to_string();
        assert!(legalize(&module, &interface()).is_err(), "vector i{width}");
        assert_eq!(module.print_to_string().to_string(), before);
    }
    for width in WIDTHS {
        let changed =
            source(width).replace(NVPTX_LAYOUT, &format!("{NVPTX_LAYOUT}-i{width}:128:128"));
        let module = parse_ir(&context, changed.as_bytes(), "wrong-abi").unwrap();
        let before = module.print_to_string().to_string();
        let error = legalize(&module, &interface()).err().unwrap();
        assert!(error.contains("layout mismatch"), "{error}");
        assert_eq!(module.print_to_string().to_string(), before);
    }
}

#[test]
fn unsupported_odd_storage_abi_and_consumers_fail_closed() {
    let context = Context::create();
    for body in [
        "@data = constant i24 7\ndefine void @kernel(ptr %p) { %x = load i24, ptr @data\n store i24 %x, ptr %p\n ret void }",
        "define void @kernel(ptr %p) { %a = alloca i24\n store i24 7, ptr %a\n ret void }",
        "define void @kernel(ptr %p) { %a = getelementptr i24, ptr %p, i64 1\n store i8 7, ptr %a\n ret void }",
        "define void @kernel(ptr %p) { %x = load volatile i24, ptr %p, align 1\n store i24 %x, ptr %p, align 1\n ret void }",
        "define void @kernel(ptr %p) { store volatile i24 7, ptr %p, align 1\n ret void }",
        "define i24 @helper(i24 %x) { ret i24 %x }\ndefine void @kernel(ptr %p) { ret void }",
        "define void @kernel(ptr %p) { %x = load i24, ptr %p, align 1\n %v = bitcast i24 %x to <3 x i8>\n store <3 x i8> %v, ptr %p, align 1\n ret void }",
        "define void @kernel(ptr %p) { %x = load i24, ptr %p, align 1\n %q = getelementptr i8, ptr %p, i24 %x\n store i8 7, ptr %q, align 1\n ret void }",
    ] {
        let ir = format!(
            "target triple = \"nvptx64-nvidia-cuda\"\ntarget datalayout = \"{NVPTX_LAYOUT}\"\n{body}"
        );
        let module = parse_ir(&context, ir.as_bytes(), "unsupported-odd-shape").unwrap();
        let before = module.print_to_string().to_string();
        assert!(legalize(&module, &interface()).is_err(), "{body}");
        assert_eq!(module.print_to_string().to_string(), before);
    }
}

fn check_cases(width: u32, mut run: impl FnMut(&mut [u8])) {
    let mask = (1u64 << width) - 1;
    let sign = |value: u64| ((value << (64 - width)) as i64) >> (64 - width);
    let mut values = vec![
        0,
        1,
        mask,
        mask >> 1,
        1 << (width - 1),
        0x55aa_1234_5678_9abc & mask,
    ];
    values.extend((0..width).flat_map(|bit| [1 << bit, mask ^ (1 << bit)]));
    let shifts = [0, 1, width / 2, width - 1, width, width + 1, 255];
    for (case, &x) in values.iter().enumerate() {
        for y in [0, mask, values[(case + 17) % values.len()]] {
            let shift_raw = shifts[case % shifts.len()] as u8;
            let shift = u32::from(shift_raw) % width;
            let iterations = [0u8, 1, 2, 17][case % 4];
            let wide = 0xfedc_ba98_7654_3210u64 ^ (x << 8) ^ y;
            let divisor = if y == 0 { 1 } else { y };
            let signed_divisor = if sign(x) == -(1i64 << (width - 1)) && sign(divisor) == -1 {
                1
            } else {
                sign(divisor)
            };
            let expected_values = [
                x,
                x.wrapping_add(y),
                x.wrapping_sub(y),
                x.wrapping_mul(y),
                x & y,
                x | y,
                x ^ y,
                x << shift,
                x >> shift,
                (sign(x) >> shift) as u64,
                x.wrapping_add(y.wrapping_mul(u64::from(iterations))),
                if sign(x) < sign(y) { x } else { y },
                x / divisor,
                x % divisor,
                (sign(x) / signed_divisor) as u64,
                (sign(x) % signed_divisor) as u64,
                wide,
            ];
            let n = (width / 8) as usize;
            let mut bytes = vec![0xa5; RECORD_BYTES + 512];
            bytes[257..257 + n].copy_from_slice(&x.to_le_bytes()[..n]);
            bytes[267..267 + n].copy_from_slice(&y.to_le_bytes()[..n]);
            bytes[276] = shift_raw;
            bytes[277] = iterations;
            bytes[280..288].copy_from_slice(&wide.to_le_bytes());
            let mut expected = bytes.clone();
            for (i, value) in expected_values.into_iter().enumerate() {
                let offset = 256 + 33 + i * 10;
                expected[offset..offset + n].copy_from_slice(&(value & mask).to_le_bytes()[..n]);
            }
            expected[456..464].copy_from_slice(&x.to_le_bytes());
            expected[464..472].copy_from_slice(&sign(x).to_le_bytes());
            expected[472] = u8::from(x < y);
            expected[473] = u8::from(sign(x) < sign(y));
            run(&mut bytes);
            assert_eq!(
                bytes, expected,
                "i{width} x={x:x} y={y:x} shift={shift_raw} iterations={iterations}"
            );
        }
    }
}

// A loop body listed before its header is reached before the header's PHI has
// a placeholder, so promotion re-enters the body instruction through the PHI.
#[test]
fn body_listed_before_its_phi_is_promoted_and_erased_once() {
    let source = format!(
        r#"target triple = "nvptx64-nvidia-cuda"
target datalayout = "{NVPTX_LAYOUT}"
define void @kernel(ptr %p) {{
entry:
 %initial = load i24, ptr %p, align 1
 br label %head
body:
 %next = add i24 %acc, 65793
 %j = add i32 %i, 1
 br label %head
head:
 %i = phi i32 [0, %entry], [%j, %body]
 %acc = phi i24 [%initial, %entry], [%next, %body]
 %done = icmp eq i32 %i, 3
 br i1 %done, label %exit, label %body
exit:
 store i24 %acc, ptr %p, align 1
 ret void
}}
"#
    );
    let context = Context::create();
    let module = parse_ir(&context, source.as_bytes(), "odd-phi-order").unwrap();
    promote(&module);
    assert!(!module.print_to_string().to_string().contains("i24"));
}

fn promote(module: &inkwell::module::Module<'_>) {
    llvm_metal_compiler::odd::lower(module).unwrap_or_else(|message| panic!("{message}"));
}

#[test]
fn promoted_values_match_original_llvm_and_rust_on_native_cpu() {
    use inkwell::{
        OptimizationLevel,
        targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
    };
    Target::initialize_native(&InitializationConfig::default()).unwrap();
    let triple = TargetMachine::get_default_triple();
    let machine = Target::from_triple(&triple)
        .unwrap()
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::None,
            RelocMode::Default,
            CodeModel::Default,
        )
        .unwrap();
    for width in WIDTHS {
        let context = Context::create();
        let original = parse_ir(&context, source(width).as_bytes(), "native-original").unwrap();
        let lowered = original.clone();
        promote(&lowered);
        let once = lowered.print_to_string().to_string();
        promote(&lowered);
        assert_eq!(
            lowered.print_to_string().to_string(),
            once,
            "promotion must be idempotent"
        );
        for module in [&original, &lowered] {
            module.set_triple(&triple);
            module.set_data_layout(&machine.get_target_data().get_data_layout());
        }
        let before = original
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let after = lowered
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        // SAFETY: reviewed pointer-only entry and 224-byte span in guarded memory.
        unsafe {
            let reference = before
                .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                .unwrap();
            let promoted = after
                .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                .unwrap();
            check_cases(width, |bytes| {
                let mut native = bytes.to_vec();
                reference.call(native.as_mut_ptr().add(256));
                promoted.call(bytes.as_mut_ptr().add(256));
                assert_eq!(
                    bytes, native,
                    "i{width}: promoted/original native execution"
                );
            });
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn odd_integer_arithmetic_and_exact_unaligned_memory_spans_match_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for width in WIDTHS {
        let context = Context::create();
        let module = parse_ir(&context, source(width).as_bytes(), "odd-integer").unwrap();
        let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("kernel.metallib");
        std::fs::write(&path, artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        check_cases(width, |bytes| {
            let mut buffers = [Buffer {
                bytes: bytes.to_vec(),
                offset: 256,
            }];
            // SAFETY: one invocation, valid shifts/divisors, bounded loop,
            // initialized inputs and exact-span outputs in a guarded record.
            unsafe {
                kernel.run(&mut buffers, 1, 1).unwrap();
            }
            bytes.copy_from_slice(&buffers[0].bytes);
        });
    }
}
