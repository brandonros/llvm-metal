use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};

// Reduced from Rust's checked pair of slices: either length can fail before
// the nullable pointer is consumed. All three words are initialized by the host.
const SOURCE: &str = r#"
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
define void @kernel(ptr %p) {
entry:
  %value = getelementptr i32, ptr %p, i64 2
  %a = load i32, ptr %p, align 4
  %ok_a = icmp ult i32 %a, 65
  br i1 %ok_a, label %second, label %merge
second:
  %b_ptr = getelementptr i32, ptr %p, i64 1
  %b = load i32, ptr %b_ptr, align 4
  %ok_b = icmp ult i32 %b, 65
  br i1 %ok_b, label %valid, label %merge
valid:
  br label %merge
merge:
  %optional = phi ptr [ %value, %valid ], [ null, %entry ], [ null, %second ]
  %missing = icmp eq ptr %optional, null
  br i1 %missing, label %end, label %write
write:
  store i32 42, ptr %optional, align 4
  br label %end
end:
  ret void
}
"#;

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
            bytes: 12,
            alignment: 4,
        }],
    }
}

fn constant_source() -> String {
    SOURCE
        .replace(
            "define void",
            "@table = private constant i32 42\ndefine void",
        )
        .replace(
            "%value = getelementptr i32, ptr %p, i64 2",
            "%value = getelementptr i32, ptr @table, i64 0",
        )
        .replace(
            "store i32 42, ptr %optional, align 4",
            "%x = load i32, ptr %optional, align 4\n  store i32 %x, ptr %p, align 4",
        )
}

#[test]
fn checked_nullable_pointer_phi_is_legalized_without_changing_input() {
    let context = Context::create();
    let module = llvm_metal_compiler::parse_ir(&context, SOURCE.as_bytes(), "nullable").unwrap();
    let original = module.print_to_string().to_string();
    let (air, _) = llvm_metal_compiler::air::legalize(&module, &interface()).unwrap();
    air.verify().unwrap();
    assert!(!air.print_to_string().to_string().contains("addrspacecast"));
    assert_eq!(module.print_to_string().to_string(), original);
}

#[test]
fn nullable_constant_pointer_is_legalized_but_mixed_private_device_is_rejected() {
    let context = Context::create();
    let constant = constant_source();
    let module = llvm_metal_compiler::parse_ir(&context, constant.as_bytes(), "constant").unwrap();
    let (air, _) = llvm_metal_compiler::air::legalize(&module, &interface()).unwrap();
    assert!(!air.print_to_string().to_string().contains("addrspacecast"));
    let mixed = SOURCE
        .replace(
            "entry:\n",
            "entry:\n  %private = alloca i32\n  store i32 7, ptr %private, align 4\n",
        )
        .replace("[ null, %entry ]", "[ %private, %entry ]")
        .replace(
            "store i32 42, ptr %optional, align 4",
            "%x = load i32, ptr %optional, align 4\n  store i32 %x, ptr %p, align 4",
        );
    let module = llvm_metal_compiler::parse_ir(&context, mixed.as_bytes(), "mixed").unwrap();
    let original = module.print_to_string().to_string();
    assert!(
        llvm_metal_compiler::air::legalize(&module, &interface())
            .unwrap_err()
            .contains("address-space inference")
    );
    assert_eq!(module.print_to_string().to_string(), original);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn nullable_pointer_branches_preserve_data_and_guards_on_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for (source, constant) in [(SOURCE.to_owned(), false), (constant_source(), true)] {
        let context = Context::create();
        let module =
            llvm_metal_compiler::parse_ir(&context, source.as_bytes(), "nullable").unwrap();
        let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("kernel.metallib");
        std::fs::write(&path, artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        for a in [0u32, 1, 64, 65, u32::MAX] {
            for b in [0u32, 1, 64, 65, u32::MAX] {
                let mut bytes = vec![0xa5; 524];
                bytes[256..260].copy_from_slice(&a.to_le_bytes());
                bytes[260..264].copy_from_slice(&b.to_le_bytes());
                let mut expected = bytes.clone();
                if a < 65 && b < 65 {
                    let offset = if constant { 256 } else { 264 };
                    expected[offset..offset + 4].copy_from_slice(&42u32.to_le_bytes());
                }
                let mut buffers = [Buffer { bytes, offset: 256 }];
                // SAFETY: one invocation, initialized 12-byte record with guards.
                unsafe {
                    kernel.run(&mut buffers, 1, 1).unwrap();
                }
                assert_eq!(buffers[0].bytes, expected, "a={a}, b={b}");
            }
        }
    }
}
