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
            bytes: 40,
            alignment: 8,
        }],
    }
}
const SOURCE: &str = r#"target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
define void @kernel(ptr %p) {
entry:
 %lo = load i64, ptr %p, align 8
 %q = getelementptr i8, ptr %p, i64 8
 %hi = load i64, ptr %q, align 8
 %r = getelementptr i8, ptr %p, i64 16
 %n = load i32, ptr %r, align 4
 %l = zext i64 %lo to i128
 %h = zext i64 %hi to i128
 %upper = shl i128 %h, 64
 %initial = or i128 %l, %upper
 %empty = icmp eq i32 %n, 0
 br i1 %empty, label %exit, label %loop
loop:
 %i = phi i32 [0, %entry], [%j, %loop]
 %acc = phi i128 [%initial, %entry], [%next, %loop]
 %next = add i128 %acc, 18446744073709551617
 %j = add i32 %i, 1
 %done = icmp eq i32 %j, %n
 br i1 %done, label %exit, label %loop
exit:
 %result = phi i128 [%initial, %entry], [%next, %loop]
 %out = getelementptr i8, ptr %p, i64 24
 store i128 %result, ptr %out, align 8
 ret void
}"#;
#[test]
fn wide_loop_and_join_phis_are_split_and_verified() {
    let context = Context::create();
    let module = parse_ir(&context, SOURCE.as_bytes(), "wide-phi").unwrap();
    let original = module.print_to_string().to_string();
    let (air, _) = legalize(&module, &interface()).unwrap();
    air.verify().unwrap();
    assert!(!air.print_to_string().to_string().contains("phi i128"));
    assert_eq!(module.print_to_string().to_string(), original);
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn wide_loop_carries_match_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    let context = Context::create();
    let module = parse_ir(&context, SOURCE.as_bytes(), "wide-phi").unwrap();
    let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kernel.metallib");
    std::fs::write(&path, artifact.metallib).unwrap();
    let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
    for value in [
        0u128,
        1,
        u64::MAX as u128,
        u128::MAX,
        1 << 127,
        0x123456789abcdef0fedcba9876543210,
    ] {
        for n in [0u32, 1, 2, 3, 17, 129] {
            let mut bytes = vec![0xa5; 552];
            bytes[256..272].copy_from_slice(&value.to_le_bytes());
            bytes[272..276].copy_from_slice(&n.to_le_bytes());
            let mut expected = bytes.clone();
            expected[280..296].copy_from_slice(
                &value
                    .wrapping_add(u128::from(n) * 18446744073709551617)
                    .to_le_bytes(),
            );
            let mut buffers = [Buffer { bytes, offset: 256 }];
            // SAFETY: bounded known loop and one initialized guarded 40-byte record.
            unsafe {
                kernel.run(&mut buffers, 1, 1).unwrap();
            }
            assert_eq!(buffers[0].bytes, expected);
        }
    }
}
