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
            bytes: 24,
            alignment: 8,
        }],
    }
}

// Freeze must survive serialization; replacing it by its operand would turn
// defined uses of frozen poison/undef into undefined behavior.
const SOURCE: &str = r#"target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
define void @kernel(ptr %p) {
    %x = load i64, ptr %p, align 8
    %v = freeze i64 %x
    %q = getelementptr i8, ptr %p, i64 8
    store i64 %v, ptr %q, align 8
    %f = freeze i64 poison
    %same = icmp eq i64 %f, %f
    %ok = zext i1 %same to i64
    %r = getelementptr i8, ptr %p, i64 16
    store i64 %ok, ptr %r, align 8
    ret void
}"#;
#[test]
fn freeze_preserves_defined_value_semantics_and_input() {
    let context = Context::create();
    let module = parse_ir(&context, SOURCE.as_bytes(), "freeze").unwrap();
    let original = module.print_to_string().to_string();
    let (air, _) = legalize(&module, &interface()).unwrap();
    assert!(air.print_to_string().to_string().contains("freeze i64"));
    assert_eq!(module.print_to_string().to_string(), original);
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn freeze_roundtrips_through_writer_and_apple_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    let context = Context::create();
    let module = parse_ir(&context, SOURCE.as_bytes(), "freeze").unwrap();
    let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
    let decoded =
        llvm_metal_compiler::parse_bitcode(&context, &artifact.air_bitcode, "AIR").unwrap();
    assert!(decoded.print_to_string().to_string().contains("freeze i64"));
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("kernel.metallib");
    std::fs::write(&path, artifact.metallib).unwrap();
    let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
    for x in [0u64, 1, u64::MAX, 1 << 63, 0xabcdef0123456789] {
        let mut bytes = vec![0xa5; 536];
        bytes[256..264].copy_from_slice(&x.to_le_bytes());
        let mut expected = bytes.clone();
        expected[264..272].copy_from_slice(&x.to_le_bytes());
        expected[272..280].copy_from_slice(&1u64.to_le_bytes());
        let mut buffers = [Buffer { bytes, offset: 256 }];
        // SAFETY: one invocation, initialized 24-byte record with guards.
        unsafe {
            kernel.run(&mut buffers, 1, 1).unwrap();
        }
        assert_eq!(buffers[0].bytes, expected);
    }
}

#[test]
fn assume_is_an_optimization_hint_not_a_device_call() {
    let context = Context::create();
    let source=SOURCE.replace("define void @kernel", "declare void @llvm.assume(i1)\ndefine void @kernel")
        .replace("%v = freeze i64 %x", "%test = icmp ne i64 %x, 0\n    call void @llvm.assume(i1 %test)\n    %v = freeze i64 %x");
    let module = parse_ir(&context, source.as_bytes(), "assume").unwrap();
    let original = module.print_to_string().to_string();
    let (air, _) = legalize(&module, &interface()).unwrap();
    assert!(!air.print_to_string().to_string().contains("llvm.assume"));
    assert_eq!(module.print_to_string().to_string(), original);
}
