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

fn source(width: u32, poison: bool) -> String {
    format!(
        "target triple = \"nvptx64-nvidia-cuda\"\n\
         target datalayout = \"e-p:64:64-i64:64-i128:128\"\n\
         declare i{width} @llvm.abs.i{width}(i{width}, i1 immarg)\n\
         define void @kernel(ptr %p) {{\n\
           %x = load i{width}, ptr %p, align 1\n\
           %r = call i{width} @llvm.abs.i{width}(i{width} %x, i1 {poison})\n\
           store i{width} %r, ptr %p, align 1\n\
           ret void\n}}"
    )
}

#[test]
fn absolute_value_preserves_minimum_integer_semantics_and_input() {
    let context = Context::create();
    for width in [8, 16, 32, 64] {
        for poison in [false, true] {
            let module = parse_ir(&context, source(width, poison).as_bytes(), "abs").unwrap();
            let original = module.print_to_string().to_string();
            let (air, _) = legalize(&module, &interface()).unwrap();
            let ir = air.print_to_string().to_string();
            assert!(!ir.contains("llvm.abs"));
            assert!(ir.contains("icmp slt"));
            assert!(ir.contains("select i1"));
            assert_eq!(ir.contains("sub nsw"), poison, "{ir}");
            assert_eq!(module.print_to_string().to_string(), original);
        }
    }
}

#[test]
fn unsupported_absolute_value_types_are_rejected_without_mutating_input() {
    let context = Context::create();
    for text in [
        source(128, false),
        source(32, false)
            .replace("llvm.abs.i32", "llvm.abs.v2i32")
            .replace("i32", "<2 x i32>")
            .replace("llvm.abs.v2<2 x i32>", "llvm.abs.v2i32"),
    ] {
        let module = parse_ir(&context, text.as_bytes(), "unsupported abs").unwrap();
        let original = module.print_to_string().to_string();
        assert!(legalize(&module, &interface()).is_err());
        assert_eq!(module.print_to_string().to_string(), original);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn absolute_value_edges_and_every_byte_match_on_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for width in [8, 16, 32, 64] {
        for poison in [false, true] {
            let context = Context::create();
            let module = parse_ir(&context, source(width, poison).as_bytes(), "abs").unwrap();
            let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("kernel.metallib");
            std::fs::write(&path, artifact.metallib).unwrap();
            let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
            let mask = u64::MAX >> (64 - width);
            let sign = 1u64 << (width - 1);
            let mut cases: Vec<u64> = (0..=255).collect();
            cases.extend([mask, mask - 1, sign - 1, sign, sign + 1]);
            let mut state = 0x123456789abcdef0u64;
            for _ in 0..32 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                cases.push(state & mask);
            }
            for x in cases {
                let x = x & mask;
                if poison && x == sign {
                    continue; // No observable result is defined for this combination.
                }
                let n = (width / 8) as usize;
                let mut bytes = vec![0xa5; 536];
                bytes[256..256 + n].copy_from_slice(&x.to_le_bytes()[..n]);
                let mut expected = bytes.clone();
                let absolute = if x & sign == 0 {
                    x
                } else {
                    x.wrapping_neg() & mask
                };
                expected[256..256 + n].copy_from_slice(&absolute.to_le_bytes()[..n]);
                let mut buffers = [Buffer { bytes, offset: 256 }];
                // SAFETY: one invocation, initialized 24-byte record with guards.
                unsafe { kernel.run(&mut buffers, 1, 1).unwrap() };
                assert_eq!(
                    buffers[0].bytes, expected,
                    "i{width} {x:#x}, poison={poison}"
                );
            }
        }
    }
}
