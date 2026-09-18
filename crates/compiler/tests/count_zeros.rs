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
            bytes: 16,
            alignment: 8,
        }],
    }
}
fn source(width: u32, op: &str, poison: bool) -> String {
    format!(
        r#"target triple = "nvptx64-nvidia-cuda"
        target datalayout = "e-p:64:64-i64:64-i128:128"
        declare i{width} @llvm.{op}.i{width}(i{width}, i1 immarg)
        define void @kernel(ptr %p) {{
            %x = load i{width}, ptr %p, align 1
            %r = call i{width} @llvm.{op}.i{width}(i{width} %x, i1 {poison})
            %q = getelementptr i8, ptr %p, i64 8
            store i{width} %r, ptr %q, align 1
            ret void
        }}"#
    )
}
#[test]
fn zero_count_types_and_poison_contract() {
    let context = Context::create();
    for op in ["ctlz", "cttz"] {
        for width in [8, 16, 32, 64, 128] {
            for poison in [false, true] {
                let module = parse_ir(&context, source(width, op, poison).as_bytes(), op).unwrap();
                let before = module.print_to_string().to_string();
                let result = legalize(&module, &interface());
                assert_eq!(module.print_to_string().to_string(), before);
                if width == 128 {
                    assert!(result.is_err());
                    continue;
                }
                let ir = result.unwrap().0.print_to_string().to_string();
                assert!(!ir.contains(&format!("llvm.{op}")));
                assert_eq!(ir.contains("poison"), poison);
            }
        }
        let vector = source(32, op, false)
            .replace(&format!("llvm.{op}.i32"), &format!("llvm.{op}.v2i32"))
            .replace("i32", "<2 x i32>")
            .replace("v2<2 x i32>", "v2i32");
        assert!(
            legalize(
                &parse_ir(&context, vector.as_bytes(), op).unwrap(),
                &interface()
            )
            .is_err()
        );
    }
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn zero_counts_edges_and_every_bit_match_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for width in [8, 16, 32, 64] {
        let mask = u64::MAX >> (64 - width);
        let mut inputs = vec![0, 1, mask, mask >> 1, 0xaaaaaaaa55555555 & mask];
        inputs.extend((0..width).flat_map(|i| [1 << i, mask ^ (1 << i)]));
        for op in ["ctlz", "cttz"] {
            for poison in [false, true] {
                let context = Context::create();
                let module = parse_ir(&context, source(width, op, poison).as_bytes(), op).unwrap();
                let artifact =
                    llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("kernel.metallib");
                std::fs::write(&path, artifact.metallib).unwrap();
                let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
                for &x in &inputs {
                    if poison && x == 0 {
                        continue;
                    }
                    let expected_value = if op == "ctlz" {
                        x.leading_zeros() - (64 - width)
                    } else {
                        x.trailing_zeros().min(width)
                    };
                    let n = (width / 8) as usize;
                    let mut bytes = vec![0xa5; 528];
                    bytes[256..256 + n].copy_from_slice(&x.to_le_bytes()[..n]);
                    let mut expected = bytes.clone();
                    expected[264..264 + n]
                        .copy_from_slice(&u64::from(expected_value).to_le_bytes()[..n]);
                    let mut buffers = [Buffer { bytes, offset: 256 }];
                    // SAFETY: one invocation with an initialized guarded 16-byte record.
                    unsafe {
                        kernel.run(&mut buffers, 1, 1).unwrap();
                    }
                    assert_eq!(
                        buffers[0].bytes, expected,
                        "{op} i{width}, poison={poison}, x={x:x}"
                    );
                }
            }
        }
    }
}
