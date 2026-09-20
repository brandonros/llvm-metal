//! Register-only i2..i7 promotion: preserve each narrow value and reject storage.
use inkwell::{context::Context, module::Module};
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{air::legalize, parse_ir};

const LAYOUT: &str = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64";
const BYTES: usize = 96;
const OUTPUTS: [&str; 20] = [
    "x", "add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr", "udiv", "urem", "sdiv",
    "srem", "selected", "frozen", "looped", "signed", "ult", "slt",
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
            bytes: BYTES,
            alignment: 4,
        }],
    }
}
fn source(n: u32) -> String {
    let mut ir = format!(
        r#"target triple = "nvptx64-nvidia-cuda"
target datalayout = "{LAYOUT}"
define void @kernel(ptr %p) {{
entry:
 %rawx = load i8, ptr %p
 %yp = getelementptr i8, ptr %p, i64 1
 %rawy = load i8, ptr %yp
 %x = trunc i8 %rawx to i{n}
 %y = trunc i8 %rawy to i{n}
 %amount = urem i{n} %y, {n}
 %add = add i{n} %x, %y
 %sub = sub i{n} %x, %y
 %mul = mul i{n} %x, %y
 %and = and i{n} %x, %y
 %or = or i{n} %x, %y
 %xor = xor i{n} %x, %y
 %shl = shl i{n} %x, %amount
 %lshr = lshr i{n} %x, %amount
 %ashr = ashr i{n} %x, %amount
 %zero = icmp eq i{n} %y, 0
 %divisor = select i1 %zero, i{n} 1, i{n} %y
 %udiv = udiv i{n} %x, %divisor
 %urem = urem i{n} %x, %divisor
 %min = icmp eq i{n} %x, {minimum}
 %minus1 = icmp eq i{n} %divisor, -1
 %overflow = and i1 %min, %minus1
 %sdivisor = select i1 %overflow, i{n} 1, i{n} %divisor
 %sdiv = sdiv i{n} %x, %sdivisor
 %srem = srem i{n} %x, %sdivisor
 %lt = icmp slt i{n} %x, %y
 %ultbit = icmp ult i{n} %x, %y
 %selected = select i1 %lt, i{n} %x, i{n} %y
 %frozen = freeze i{n} %x
 %signed = sext i{n} %x to i32
 %ult = zext i1 %ultbit to i32
 %slt = zext i1 %lt to i32
 br label %loop
loop:
 %iteration = phi i32 [0, %entry], [%nexti, %loop]
 %acc = phi i{n} [%x, %entry], [%looped, %loop]
 %looped = add i{n} %acc, %y
 %nexti = add i32 %iteration, 1
 %done = icmp eq i32 %nexti, 3
 br i1 %done, label %exit, label %loop
exit:
"#,
        minimum = 1 << (n - 1)
    );
    for (i, name) in OUTPUTS.iter().enumerate() {
        let value = if i < 17 {
            ir.push_str(&format!(" %out{i} = zext i{n} %{name} to i32\n"));
            format!("%out{i}")
        } else {
            format!("%{name}")
        };
        ir.push_str(&format!(" %ptr{i} = getelementptr i8, ptr %p, i64 {}\n store i32 {value}, ptr %ptr{i}, align 4\n", 8 + i*4));
    }
    // The optimizer-generated SEC1 bitset shape which originally exposed i6.
    ir.push_str(" %validtag = icmp ult i8 %rawx, 5\n br i1 %validtag, label %tag, label %end\ntag:\n %smalltag = trunc nuw nsw i8 %rawx to i6\n %bit = shl nuw nsw i6 1, %smalltag\n %masked = and i6 %bit, 12\n %absent = icmp eq i6 %masked, 0\n %tagvalue = zext i1 %absent to i32\n br label %end\nend:\n %tagout = phi i32 [2, %exit], [%tagvalue, %tag]\n %tagptr = getelementptr i8, ptr %p, i64 88\n store i32 %tagout, ptr %tagptr, align 4\n ret void\n}\n");
    ir
}
fn promote(module: &Module<'_>) -> Result<(), String> {
    llvm_metal_compiler::odd::lower(module)
}
fn cases(n: u32, exhaustive: bool, mut run: impl FnMut(&mut [u8])) {
    let mask = (1u32 << n) - 1;
    let sign = |x: u32| ((x << (32 - n)) as i32) >> (32 - n);
    for x in 0..=mask {
        for y in 0..=mask {
            if !exhaustive && ![0, 1, mask, mask >> 1].contains(&y) {
                continue;
            }
            let amount = y % n;
            let d = if y == 0 { 1 } else { y };
            let sd = if sign(x) == -(1 << (n - 1)) && sign(d) == -1 {
                1
            } else {
                sign(d)
            };
            let values = [
                x,
                x.wrapping_add(y) & mask,
                x.wrapping_sub(y) & mask,
                x.wrapping_mul(y) & mask,
                x & y,
                x | y,
                x ^ y,
                (x << amount) & mask,
                x >> amount,
                (sign(x) >> amount) as u32 & mask,
                x / d,
                x % d,
                (sign(x) / sd) as u32 & mask,
                (sign(x) % sd) as u32 & mask,
                if sign(x) < sign(y) { x } else { y },
                x,
                x.wrapping_add(3 * y) & mask,
                sign(x) as u32,
                u32::from(x < y),
                u32::from(sign(x) < sign(y)),
            ];
            let mut data = vec![0xa5; BYTES + 512];
            // High input bits prove truncation masks inputs before arithmetic.
            data[256] = x as u8 | (!mask as u8);
            data[257] = y as u8 | (!mask as u8);
            // Keep the exact bitset case reachable as well as high-bit truncation.
            if y == 0 {
                data[256] = x as u8;
            }
            let mut expected = data.clone();
            for (i, value) in values.into_iter().enumerate() {
                expected[264 + i * 4..268 + i * 4].copy_from_slice(&value.to_le_bytes());
            }
            let tag = if data[256] < 5 {
                u32::from((1u32 << data[256]) & 12 == 0)
            } else {
                2
            };
            expected[344..348].copy_from_slice(&tag.to_le_bytes());
            run(&mut data);
            assert_eq!(data, expected, "i{n} x={x} y={y}");
        }
    }
}
#[test]
fn all_subbyte_values_match_original_llvm_and_rust() {
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
    for n in 2..=7 {
        let context = Context::create();
        let original = parse_ir(&context, source(n).as_bytes(), "original").unwrap();
        let lowered = original.clone();
        promote(&lowered).unwrap();
        let once = lowered.print_to_string();
        promote(&lowered).unwrap();
        assert_eq!(once, lowered.print_to_string());
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
        // SAFETY: one initialized guarded96-byte record; valid shifts/divisors and a bounded3-iteration loop.
        unsafe {
            let reference = before
                .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                .unwrap();
            let promoted = after
                .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                .unwrap();
            cases(n, true, |data| {
                let mut expected = data.to_vec();
                reference.call(expected.as_mut_ptr().add(256));
                promoted.call(data.as_mut_ptr().add(256));
                assert_eq!(data, expected, "i{n} original/lowered");
            });
        }
    }
}
#[test]
fn subbyte_registers_legalize_and_storage_is_refused() {
    let context = Context::create();
    for n in 2..=7 {
        let module = parse_ir(&context, source(n).as_bytes(), "registers").unwrap();
        let original = module.print_to_string();
        let (air, _) = legalize(&module, &interface()).unwrap();
        air.verify().unwrap();
        assert_eq!(module.print_to_string(), original);
        let text = air.print_to_string().to_string();
        for width in 2..=7 {
            assert!(
                !text
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .any(|token| token == format!("i{width}"))
            );
        }
        for body in [
            format!("define void @kernel(ptr %p) {{ %v=load i{n}, ptr %p\n ret void }}"),
            format!("define void @kernel(ptr %p) {{ store i{n} 1, ptr %p\n ret void }}"),
            format!("@g=global i{n} 0\ndefine void @kernel(ptr %p) {{ ret void }}"),
            format!("define void @kernel(ptr %p) {{ %a=alloca i{n}\n ret void }}"),
            format!(
                "define void @kernel(ptr %p) {{ %a=getelementptr i{n}, ptr %p, i64 1\n ret void }}"
            ),
            format!(
                "define i{n} @helper(i{n} %v) {{ ret i{n} %v }}\ndefine void @kernel(ptr %p) {{ ret void }}"
            ),
            format!("define void @kernel(ptr %p) {{ %v=load <2 x i{n}>, ptr %p\n ret void }}"),
        ] {
            let ir = format!(
                "target triple = \"nvptx64-nvidia-cuda\"\ntarget datalayout = \"{LAYOUT}\"\n{body}"
            );
            let module = parse_ir(&context, ir.as_bytes(), "refusal").unwrap();
            let before = module.print_to_string();
            assert!(promote(&module).is_err(), "{body}");
            assert_eq!(module.print_to_string(), before);
        }
    }
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires explicit Apple GPU execution"]
fn subbyte_arithmetic_matches_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for n in 2..=7 {
        let context = Context::create();
        let module = parse_ir(&context, source(n).as_bytes(), "subbyte").unwrap();
        let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("kernel.metallib");
        std::fs::write(&path, artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        cases(n, false, |data| {
            let mut buffers = [Buffer {
                bytes: data.to_vec(),
                offset: 256,
            }];
            // SAFETY: single invocation, guarded record, defined arithmetic, bounded loop.
            unsafe {
                kernel.run(&mut buffers, 1, 1).unwrap();
            }
            data.copy_from_slice(&buffers[0].bytes);
        });
    }
}
