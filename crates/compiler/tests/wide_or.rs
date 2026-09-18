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
// Both low and high halves vary independently; the sign extension also exercises
// negative wide operands. Input uses supported scalar loads, not an i128 ABI.
const SOURCE: &str = r#"target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
define void @kernel(ptr %p) {
    %x = load i64, ptr %p, align 8
    %q = getelementptr i8, ptr %p, i64 8
    %y = load i64, ptr %q, align 8
    %a = sext i64 %x to i128
    %b = zext i64 %y to i128
    %c = shl i128 %b, 32
    %v = or i128 %a, %c
    %r = getelementptr i8, ptr %p, i64 16
    store i128 %v, ptr %r, align 8
    ret void
}"#;
fn source(bswap: bool) -> String {
    if bswap {
        SOURCE.replace(
            "store i128 %v",
            "%swapped = call i128 @llvm.bswap.i128(i128 %v)\n    store i128 %swapped",
        ) + "\ndeclare i128 @llvm.bswap.i128(i128)\n"
    } else {
        SOURCE.into()
    }
}
#[test]
fn wide_or_lowers_without_mutating_input() {
    let context = Context::create();
    for bswap in [false, true] {
        let module = parse_ir(&context, source(bswap).as_bytes(), "wide-or").unwrap();
        let before = module.print_to_string().to_string();
        let (air, _) = legalize(&module, &interface()).unwrap();
        assert!(!air.print_to_string().to_string().contains("or i128"));
        assert!(!air.print_to_string().to_string().contains("call i128"));
        assert_eq!(module.print_to_string().to_string(), before);
    }
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn wide_or_cpu_gpu_edges_and_walking_bits() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for bswap in [false, true] {
        let context = Context::create();
        let module = parse_ir(&context, source(bswap).as_bytes(), "wide-or").unwrap();
        let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("kernel.metallib");
        std::fs::write(&path, artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        let mut values = vec![0u64, 1, u64::MAX, 0xabcdef0123456789];
        values.extend((0..64).map(|bit| 1 << bit));
        for (i, x) in values.iter().copied().enumerate() {
            for y in [0, u64::MAX, values[(i + 17) % values.len()]] {
                let mut bytes = vec![0xa5; 544];
                bytes[256..264].copy_from_slice(&x.to_le_bytes());
                bytes[264..272].copy_from_slice(&y.to_le_bytes());
                let mut expected = bytes.clone();
                let result = (x as i64 as i128) | ((y as i128) << 32);
                let result = if bswap { result.swap_bytes() } else { result };
                expected[272..288].copy_from_slice(&result.to_le_bytes());
                let mut buffers = [Buffer { bytes, offset: 256 }];
                // SAFETY: one invocation and a disjoint initialized 32-byte record.
                unsafe {
                    kernel.run(&mut buffers, 1, 1).unwrap();
                }
                assert_eq!(buffers[0].bytes, expected, "x={x:x}, y={y:x}");
            }
        }
    }
}
