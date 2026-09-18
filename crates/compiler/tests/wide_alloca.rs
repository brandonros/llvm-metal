use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{air::legalize, parse_ir};

const HEADER: &str =
    "target triple = \"nvptx64-nvidia-cuda\"\ntarget datalayout = \"e-p:64:64-i64:64-i128:128\"\n";
const LEFT: u128 = 0x00112233445566778899aabbccddeeff;
const RIGHT: u128 = 0xfedcba98765432100123456789abcdef;
fn interface() -> KernelInterface {
    KernelInterface {
        schema: 1,
        entry: "kernel".into(),
        calling_convention: "C".into(),
        invocations: Some(1),
        dispatch: Dispatch::Single,
        aliasing: "disjoint".into(),
        arguments: vec![BufferArgument {
            name: "data".into(),
            kind: "buffer".into(),
            access: Access::ReadWrite,
            bytes: 64,
            alignment: 8,
        }],
    }
}
fn source(alignment: u32) -> String {
    format!(
        r#"{HEADER}
define void @kernel(ptr %p) {{
 %storage = alloca i128, i64 3, align {alignment}
 %middle = getelementptr inbounds i128, ptr %storage, i64 1
 %last = getelementptr inbounds i128, ptr %storage, i64 2
 store volatile i128 {LEFT}, ptr %storage, align 1
 store volatile i128 {RIGHT}, ptr %last, align 1
 %lo = load i64, ptr %p, align 8
 %hi_input = getelementptr i8, ptr %p, i64 8
 %hi = load i64, ptr %hi_input, align 8
 %lo_wide = zext i64 %lo to i128
 %hi_wide = zext i64 %hi to i128
 %upper = shl i128 %hi_wide, 64
 %value = or i128 %lo_wide, %upper
 store i128 %value, ptr %middle, align 1
 %read = load volatile i128, ptr %middle, align 1
 %sum = add i128 %read, 18446744073709551617
 store volatile i128 %sum, ptr %middle, align 1
 %left = load volatile i128, ptr %storage, align 1
 %result = load volatile i128, ptr %middle, align 1
 %right = load volatile i128, ptr %last, align 1
 %left_out = getelementptr i8, ptr %p, i64 16
 %out = getelementptr i8, ptr %p, i64 32
 %right_out = getelementptr i8, ptr %p, i64 48
 store i128 %left, ptr %left_out, align 8
 store i128 %result, ptr %out, align 8
 store i128 %right, ptr %right_out, align 8
 ret void
}}
"#
    )
}
#[test]
fn scalar_alloca_counts_alignment_and_strided_accesses_are_legalized() {
    for alignment in [1, 8, 16, 32] {
        let context = Context::create();
        let module = parse_ir(&context, source(alignment).as_bytes(), "wide-alloca").unwrap();
        let original = module.print_to_string().to_string();
        let (air, _) = legalize(&module, &interface()).unwrap();
        air.verify().unwrap();
        let ir = air.print_to_string().to_string();
        assert!(!ir.contains("i128"));
        assert!(ir.contains("store volatile i64"));
        assert!(ir.contains("load volatile i64"));
        assert_eq!(module.print_to_string().to_string(), original);
    }
}
#[test]
fn single_scalar_allocation_with_lifetime_markers_is_supported() {
    // Reduced from the original Solana arith_u128_mul self-test's black_box.
    let text = format!(
        r#"{HEADER}
        declare void @llvm.lifetime.start.p0(i64 immarg, ptr)
        declare void @llvm.lifetime.end.p0(i64 immarg, ptr)
        define void @kernel(ptr %p) {{
          %slot = alloca i128, align 16
          call void @llvm.lifetime.start.p0(i64 16, ptr %slot)
          store i128 {RIGHT}, ptr %slot, align 16
          %value = load volatile i128, ptr %slot, align 16
          call void @llvm.lifetime.end.p0(i64 16, ptr %slot)
          %product = mul i128 %value, {LEFT}
          store i128 %product, ptr %p, align 8
          ret void
        }}
    "#
    );
    let context = Context::create();
    let module = parse_ir(&context, text.as_bytes(), "wide-alloca-lifetime").unwrap();
    legalize(&module, &interface()).unwrap().0.verify().unwrap();
}
#[test]
fn escaping_dynamic_and_nested_aggregate_wide_allocations_are_refused() {
    let cases = [
        ("%slot = alloca i128\nstore ptr %slot, ptr %p", ""),
        (
            "%slot = alloca i128\ncall void @unknown(ptr %slot)",
            "declare void @unknown(ptr)",
        ),
        (
            "%count = load i32, ptr %p\n%slot = alloca i128, i32 %count",
            "",
        ),
        (
            "%slot = alloca [2 x [2 x i128]], align 16\nstore i128 1, ptr %slot",
            "",
        ),
        (
            "%slot = alloca i128\n%address = ptrtoint ptr %slot to i64\nstore i64 %address, ptr %p",
            "",
        ),
        (
            "%slot = alloca i128\n%condition = load i1, ptr %p\n%mixed = select i1 %condition, ptr %slot, ptr %p\nstore i128 1, ptr %mixed",
            "",
        ),
    ];
    for (body, declarations) in cases {
        let context = Context::create();
        let text = format!(
            "{HEADER}\n{declarations}\ndefine void @kernel(ptr %p) {{\n{body}\nret void\n}}"
        );
        let module = parse_ir(&context, text.as_bytes(), "wide-alloca-refusal").unwrap();
        let original = module.print_to_string().to_string();
        assert!(legalize(&module, &interface()).is_err(), "{text}");
        assert_eq!(module.print_to_string().to_string(), original);
    }
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn private_scalar_allocations_preserve_array_stride_carries_and_guards_on_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for alignment in [1, 16, 32] {
        let context = Context::create();
        let module = parse_ir(&context, source(alignment).as_bytes(), "wide-alloca").unwrap();
        let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("kernel.metallib");
        std::fs::write(&path, artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        for input in [
            0u128,
            1,
            u64::MAX as u128,
            1 << 64,
            1 << 127,
            u128::MAX,
            RIGHT,
        ] {
            let mut bytes = vec![0xa5; 576];
            bytes[256..272].copy_from_slice(&input.to_le_bytes());
            let mut expected = bytes.clone();
            expected[272..288].copy_from_slice(&LEFT.to_le_bytes());
            expected[288..304]
                .copy_from_slice(&input.wrapping_add(18446744073709551617).to_le_bytes());
            expected[304..320].copy_from_slice(&RIGHT.to_le_bytes());
            let mut buffers = [Buffer { bytes, offset: 256 }];
            // SAFETY: all three private scalar elements are initialized; the
            // input and three output values fit the guarded 64-byte ABI record.
            unsafe {
                kernel.run(&mut buffers, 1, 1).unwrap();
            }
            assert_eq!(buffers[0].bytes, expected, "alignment {alignment}");
        }
    }
}
