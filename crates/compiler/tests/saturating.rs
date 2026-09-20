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

fn source(width: u32, operation: &str) -> String {
    format!(
        r#"target triple = "nvptx64-nvidia-cuda"
        target datalayout = "e-p:64:64-i64:64-i128:128"
        declare i{width} @llvm.{operation}.i{width}(i{width}, i{width})
        define void @kernel(ptr %p) {{
            %q = getelementptr i8, ptr %p, i64 8
            %o = getelementptr i8, ptr %p, i64 16
            %x = load i{width}, ptr %p, align 1
            %y = load i{width}, ptr %q, align 1
            %r = call i{width} @llvm.{operation}.i{width}(i{width} %x, i{width} %y)
            store i{width} %r, ptr %o, align 1
            ret void
        }}"#
    )
}

#[test]
fn unsigned_saturating_arithmetic_becomes_compare_and_select() {
    let context = Context::create();
    for width in [8, 16, 32, 64] {
        for (op, predicate) in [("uadd.sat", "ult"), ("usub.sat", "ugt")] {
            let module = parse_ir(&context, source(width, op).as_bytes(), op).unwrap();
            let original = module.print_to_string().to_string();
            let (air, _) = legalize(&module, &interface()).unwrap();
            let ir = air.print_to_string().to_string();
            assert!(!ir.contains(&format!("llvm.{op}")));
            assert!(ir.contains(&format!("icmp {predicate}")));
            assert!(ir.contains("select i1"));
            assert_eq!(module.print_to_string().to_string(), original);
        }
    }
}
#[test]
fn unsupported_saturating_types_and_signed_forms_are_rejected_without_mutation() {
    let context = Context::create();
    for op in ["uadd.sat", "usub.sat"] {
        for source in [
            source(128, op),
            source(32, op).replace(op, &op.replace('u', "s")),
            source(32, op)
                .replace(&format!("llvm.{op}.i32"), &format!("llvm.{op}.v2i32"))
                .replace("i32", "<2 x i32>")
                .replace("v2<2 x i32>", "v2i32"),
        ] {
            let module = parse_ir(&context, source.as_bytes(), op).unwrap();
            let original = module.print_to_string().to_string();
            assert!(legalize(&module, &interface()).is_err());
            assert_eq!(module.print_to_string().to_string(), original);
        }
    }
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn unsigned_saturating_edges_match_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for width in [8, 16, 32, 64] {
        let mask = u64::MAX >> (64 - width);
        let sign = 1u64 << (width - 1);
        let values = [
            0,
            1,
            mask,
            mask - 1,
            sign,
            sign - 1,
            sign + 1,
            0xaaaaaaaa55555555 & mask,
        ];
        for op in ["uadd.sat", "usub.sat"] {
            let context = Context::create();
            let module = parse_ir(&context, source(width, op).as_bytes(), op).unwrap();
            let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("kernel.metallib");
            std::fs::write(&path, artifact.metallib).unwrap();
            let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
            for x in values {
                for y in values {
                    let result = if op == "uadd.sat" {
                        x.checked_add(y).filter(|sum| *sum <= mask).unwrap_or(mask)
                    } else {
                        x.saturating_sub(y)
                    };
                    let n = (width / 8) as usize;
                    let mut bytes = vec![0xa5; 536];
                    bytes[256..256 + n].copy_from_slice(&x.to_le_bytes()[..n]);
                    bytes[264..264 + n].copy_from_slice(&y.to_le_bytes()[..n]);
                    let mut expected = bytes.clone();
                    expected[272..272 + n].copy_from_slice(&result.to_le_bytes()[..n]);
                    let mut buffers = [Buffer { bytes, offset: 256 }];
                    // SAFETY: one invocation, initialized 24-byte record with guards.
                    unsafe {
                        kernel.run(&mut buffers, 1, 1).unwrap();
                    }
                    assert_eq!(buffers[0].bytes, expected, "{op} i{width} {x:#x} {y:#x}");
                }
            }
        }
    }
}
