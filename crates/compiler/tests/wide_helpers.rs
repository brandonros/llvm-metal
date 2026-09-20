use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{air::legalize, parse_ir};

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
            bytes: 32,
            alignment: 8,
        }],
    }
}
const HEADER: &str =
    "target triple = \"nvptx64-nvidia-cuda\"\ntarget datalayout = \"e-p:64:64-i64:64-i128:128\"\n";
const KERNEL: &str = r#"
define void @kernel(ptr %p) {
 %lo = load i64, ptr %p, align 8
 %q = getelementptr i8, ptr %p, i64 8
 %hi = load i64, ptr %q, align 8
 %l = zext i64 %lo to i128
 %h = zext i64 %hi to i128
 %upper = shl i128 %h, 64
 %value = or i128 %l, %upper
 %result = call fastcc i128 @wide(i128 %value) noinline
 %out = getelementptr i8, ptr %p, i64 16
 store i128 %result, ptr %out, align 8
 ret void
}
"#;
const HELPERS: &str = r#"
declare i128 @llvm.bswap.i128(i128)
define internal fastcc i128 @inner(i128 %value) noinline {
 %sum = add i128 %value, 18446744073709551617
 ret i128 %sum
}
define internal fastcc i128 @wide(i128 %value) noinline {
 %sum = call fastcc i128 @inner(i128 %value) noinline
 %result = call i128 @llvm.bswap.i128(i128 %sum)
 ret i128 %result
}
"#;

#[test]
fn local_wide_arguments_returns_and_nested_helpers_inline_before_lowering() {
    let context = Context::create();
    let source = format!("{HEADER}{KERNEL}{HELPERS}");
    let module = parse_ir(&context, source.as_bytes(), "wide-helpers").unwrap();
    let original = module.print_to_string().to_string();
    let (air, _) = legalize(&module, &interface()).unwrap();
    air.verify().unwrap();
    let ir = air.print_to_string().to_string();
    assert!(!ir.contains("@wide("));
    assert!(!ir.contains("@inner("));
    assert!(!ir.contains("i128"));
    assert_eq!(module.print_to_string().to_string(), original);
}

#[test]
fn metadata_intrinsic_signatures_do_not_enter_wide_abi_inspection() {
    let context = Context::create();
    let kernel = KERNEL.replace(
        " %lo = load",
        " call void @llvm.experimental.noalias.scope.decl(metadata !0)\n %lo = load",
    );
    let source = format!(
        r#"{HEADER}{kernel}{HELPERS}
        declare void @llvm.experimental.noalias.scope.decl(metadata)
        !0 = !{{!1}}
        !1 = distinct !{{!1, !2, !"scope"}}
        !2 = distinct !{{!2, !"domain"}}
    "#
    );
    let module = parse_ir(&context, source.as_bytes(), "wide-metadata-intrinsic").unwrap();
    let original = module.print_to_string().to_string();
    let (air, _) = legalize(&module, &interface()).unwrap();
    air.verify().unwrap();
    assert_eq!(module.print_to_string().to_string(), original);
}

#[test]
fn escaping_recursive_and_external_wide_abis_remain_rejected() {
    let cases = [
        ("declare fastcc i128 @wide(i128)", "internal definition"),
        (
            "define fastcc i128 @wide(i128 %x) { ret i128 %x }",
            "internal definition",
        ),
        (
            "define internal fastcc i128 @wide(i128 %x) { %r = call fastcc i128 @wide(i128 %x) ret i128 %r }",
            "recursive",
        ),
        (
            "define internal fastcc i128 @wide(i128 %x) { %v = call i64 @bridge() ret i128 %x } define internal i64 @bridge() { %v = call fastcc i128 @wide(i128 1) %r = trunc i128 %v to i64 ret i64 %r }",
            "recursive",
        ),
        (
            "@escaped = internal constant ptr @wide\ndefine internal fastcc i128 @wide(i128 %x) { ret i128 %x }",
            "address escapes",
        ),
        (
            "define internal fastcc i128 @wide(i128 %x) { call void @consume(ptr @wide) ret i128 %x } declare void @consume(ptr)",
            "address escapes",
        ),
    ];
    for (helper, diagnostic) in cases {
        let context = Context::create();
        let module = parse_ir(
            &context,
            format!("{HEADER}{KERNEL}{helper}").as_bytes(),
            "wide-refusal",
        )
        .unwrap();
        let original = module.print_to_string().to_string();
        let error = legalize(&module, &interface()).unwrap_err();
        assert!(error.contains(diagnostic), "{error}");
        assert_eq!(module.print_to_string().to_string(), original);
    }
}

#[test]
fn inlining_does_not_admit_unsupported_wide_operations() {
    let context = Context::create();
    let helper = "define internal fastcc i128 @wide(i128 %x) { %r = lshr i128 %x, 128 ret i128 %r }";
    let module = parse_ir(
        &context,
        format!("{HEADER}{KERNEL}{helper}").as_bytes(),
        "wide-div",
    )
    .unwrap();
    let original = module.print_to_string().to_string();
    let error = legalize(&module, &interface()).unwrap_err();
    assert!(error.contains("unsupported i128"), "{error}");
    assert_eq!(module.print_to_string().to_string(), original);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn inlined_wide_helpers_preserve_carry_and_upper_limbs_on_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    let context = Context::create();
    let source = format!("{HEADER}{KERNEL}{HELPERS}");
    let module = parse_ir(&context, source.as_bytes(), "wide-helpers").unwrap();
    let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kernel.metallib");
    std::fs::write(&path, artifact.metallib).unwrap();
    let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
    for value in [
        0u128,
        1,
        u64::MAX as u128,
        1 << 64,
        1 << 127,
        u128::MAX,
        0x123456789abcdef0fedcba9876543210,
    ] {
        let mut bytes = vec![0xa5; 544];
        bytes[256..272].copy_from_slice(&value.to_le_bytes());
        let mut expected = bytes.clone();
        expected[272..288].copy_from_slice(
            &value
                .wrapping_add(18446744073709551617)
                .swap_bytes()
                .to_le_bytes(),
        );
        let mut buffers = [Buffer { bytes, offset: 256 }];
        // SAFETY: reviewed input ABI, one aligned guarded 32-byte record.
        unsafe {
            kernel.run(&mut buffers, 1, 1).unwrap();
        }
        assert_eq!(buffers[0].bytes, expected);
    }
}
