use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{
    air::{InliningPolicy, legalize_with_policy},
    parse_ir,
};
const SOURCE: &str = include_str!("../../../tests/fixtures/retained-calls/byte-iterator.ll");
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
fn inputs() -> [[u64; 4]; 4] {
    [
        [0; 4],
        [1, 2, 3, 4],
        [u64::MAX; 4],
        [
            0x123456789abcdef0,
            0xfedcba9876543210,
            0x80ff00cc55779911,
            0x789abcde00001234,
        ],
    ]
}
fn expected(words: [u64; 4]) -> Vec<u8> {
    words.into_iter().rev().flat_map(u64::to_be_bytes).collect()
}
#[test]
fn nullable_pointer_iterator_uses_required_inlining() {
    let context = Context::create();
    let source = parse_ir(&context, SOURCE.as_bytes(), "iterator").unwrap();
    let prepared =
        llvm_metal_compiler::calls::prepare(&source, "kernel", InliningPolicy::Selective).unwrap();
    let never = inkwell::attributes::Attribute::get_named_enum_kind_id("noinline");
    assert!(
        prepared
            .get_function("encode")
            .unwrap()
            .get_enum_attribute(inkwell::attributes::AttributeLoc::Function, never)
            .is_some()
    );
    let (air, _) = legalize_with_policy(&source, &interface(), InliningPolicy::Selective).unwrap();
    assert!(air.get_function("encode.metal.0.0").is_none());
    assert_eq!(
        air.get_functions()
            .filter(|f| f.count_basic_blocks() != 0)
            .count(),
        1
    );
}
#[test]
fn iterator_encoding_matches_native_reference_before_and_after_lowering() {
    use inkwell::{
        OptimizationLevel,
        targets::{InitializationConfig, Target, TargetMachine},
    };
    Target::initialize_native(&InitializationConfig::default()).unwrap();
    let context = Context::create();
    for lowered in [false, true] {
        let source = parse_ir(&context, SOURCE.as_bytes(), "iterator").unwrap();
        let module = if lowered {
            legalize_with_policy(&source, &interface(), InliningPolicy::Selective)
                .unwrap()
                .0
        } else {
            source
        };
        let triple = TargetMachine::get_default_triple();
        let machine = Target::from_triple(&triple)
            .unwrap()
            .create_target_machine(
                &triple,
                "generic",
                "",
                OptimizationLevel::None,
                inkwell::targets::RelocMode::Default,
                inkwell::targets::CodeModel::Default,
            )
            .unwrap();
        module.set_triple(&triple);
        module.set_data_layout(&machine.get_target_data().get_data_layout());
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        // SAFETY: this reviewed fixture exports exactly one pointer argument.
        let kernel = unsafe {
            engine
                .get_function::<unsafe extern "C" fn(*mut u64)>("kernel")
                .unwrap_or_else(|e| panic!("lowered={lowered}: {e:?}"))
        };
        for words in inputs() {
            let mut data = [0xa5a5a5a5a5a5a5a5u64; 8];
            data[..4].copy_from_slice(&words);
            // SAFETY: the kernel reads the first 32 bytes and writes the next
            // 32 of this initialized, aligned 64-byte allocation.
            unsafe {
                kernel.call(data.as_mut_ptr());
            }
            assert_eq!(&data[..4], &words);
            let bytes: Vec<_> = data[4..].iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(bytes, expected(words), "lowered={lowered}");
        }
    }
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn iterator_encoding_matches_on_gpu_with_default_and_full_inlining() {
    use llvm_metal_runtime::{Buffer, Kernel};
    let context = Context::create();
    for policy in [InliningPolicy::All, InliningPolicy::Selective] {
        let source = parse_ir(&context, SOURCE.as_bytes(), "iterator").unwrap();
        let artifact =
            llvm_metal_compiler::compile::compile_with_policy(&source, &interface(), policy)
                .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kernel.metallib");
        std::fs::write(&path, &artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        for words in inputs() {
            let mut buffers = [Buffer {
                bytes: vec![0xa5; 576],
                offset: 256,
            }];
            for (i, w) in words.iter().enumerate() {
                buffers[0].bytes[256 + i * 8..264 + i * 8].copy_from_slice(&w.to_le_bytes());
            }
            let before = buffers[0].bytes.clone();
            // SAFETY: one invocation, disjoint input/output spans and guards.
            unsafe {
                kernel.run(&mut buffers, 1, 1).unwrap();
            }
            assert_eq!(
                &buffers[0].bytes[288..320],
                expected(words),
                "policy={policy:?}"
            );
            assert_eq!(&buffers[0].bytes[..288], &before[..288]);
            assert_eq!(&buffers[0].bytes[320..], &before[320..]);
        }
    }
}
