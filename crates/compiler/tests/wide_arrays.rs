//! Private wide-array snapshots emitted by Rust's opaque Dalek test inputs.
use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{air::legalize, parse_ir};

const HEADER: &str =
    "target triple = \"nvptx64-nvidia-cuda\"\ntarget datalayout = \"e-p:64:64-i64:64-i128:128\"\n";
const BASE: u128 = 0xfedcba98765432100123456789abcdef;

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
            bytes: 160,
            alignment: 8,
        }],
    }
}

fn source(alignment: u32) -> String {
    let mut text = format!(
        "{HEADER}\ndefine void @kernel(ptr %p) {{\n%storage = alloca [9 x i128], i64 2, align {alignment}\n%second = getelementptr [9 x i128], ptr %storage, i64 1\n%x = load i64, ptr %p, align 1\n%x128 = zext i64 %x to i128\n"
    );
    for index in 0..9 {
        text += &format!(
            "%slot{index} = getelementptr [9 x i128], ptr %second, i64 0, i64 {index}\n%value{index} = add i128 %x128, {}\nstore i128 %value{index}, ptr %slot{index}, align 1\n",
            BASE.wrapping_add(index as u128)
        );
    }
    text += "%snapshot = load volatile [9 x i128], ptr %second, align 1\n";
    // Extracts observe the snapshot, not the newer private-memory contents.
    text += "store volatile i128 0, ptr %slot2, align 1\n";
    for index in 0..9 {
        text += &format!(
            "%v{index} = extractvalue [9 x i128] %snapshot, {index}\n%out{index} = getelementptr i8, ptr %p, i64 {}\nstore i128 %v{index}, ptr %out{index}, align 1\n",
            8 + 16 * index
        );
    }
    text += "ret void\n}\n";
    text
}

fn check_cases(offset: usize, mut run: impl FnMut(&mut [u8])) {
    for input in [0, 1, u32::MAX as u64, 1 << 63, u64::MAX] {
        let mut bytes = vec![0xa5; offset + 160 + 256];
        bytes[offset..offset + 8].copy_from_slice(&input.to_le_bytes());
        let mut expected = bytes.clone();
        for index in 0..9 {
            let begin = offset + 8 + index * 16;
            expected[begin..begin + 16].copy_from_slice(
                &BASE
                    .wrapping_add(index as u128)
                    .wrapping_add(input as u128)
                    .to_le_bytes(),
            );
        }
        run(&mut bytes);
        assert_eq!(bytes, expected, "input {input:x}");
    }
}

fn lower(module: &inkwell::module::Module<'_>) {
    unsafe extern "C" {
        fn LLVMMetalLowerWideIntegers(
            module: inkwell::llvm_sys::prelude::LLVMModuleRef,
        ) -> *mut std::ffi::c_char;
    }
    // SAFETY: disposable, verified module and owned native diagnostic.
    unsafe {
        let error = LLVMMetalLowerWideIntegers(module.as_mut_ptr());
        if !error.is_null() {
            let message = std::ffi::CStr::from_ptr(error)
                .to_string_lossy()
                .into_owned();
            inkwell::llvm_sys::core::LLVMDisposeMessage(error);
            panic!("{message}");
        }
    }
    module.verify().unwrap();
}

#[test]
fn array_snapshot_preserves_storage_stride_alignment_and_all_volatile_reads() {
    for alignment in [1, 8, 16, 32] {
        let context = Context::create();
        let module = parse_ir(&context, source(alignment).as_bytes(), "wide-array").unwrap();
        lower(&module);
        let ir = module.print_to_string().to_string();
        assert!(
            ir.lines()
                .filter(|line| !line.starts_with("target datalayout"))
                .all(|line| !line.contains("i128"))
        );
        assert!(ir.contains(&format!("alloca [9 x [2 x i64]], i64 2, align {alignment}")));
        assert!(ir.contains("getelementptr [9 x [2 x i64]]"));
        assert_eq!(ir.matches("load volatile i64").count(), 18);
        let once = ir;
        lower(&module);
        assert_eq!(module.print_to_string().to_string(), once);
        let fresh = parse_ir(&context, source(alignment).as_bytes(), "wide-array-air").unwrap();
        let before = fresh.print_to_string().to_string();
        let (air, _) = legalize(&fresh, &interface()).unwrap();
        air.verify().unwrap();
        assert!(!air.print_to_string().to_string().contains("i128"));
        assert_eq!(fresh.print_to_string().to_string(), before);
    }
    for count in [1, 9, 256] {
        let mut text = format!(
            "{HEADER}\ndefine void @kernel(ptr %p) {{\n%s = alloca [{count} x i128], align 16\n"
        );
        for index in 0..count {
            text += &format!(
                "%p{index} = getelementptr i8, ptr %s, i64 {}\nstore i128 {index}, ptr %p{index}, align 16\n",
                16 * index
            );
        }
        text += &format!("%x = load volatile [{count} x i128], ptr %s, align 16\nret void\n}}\n");
        let context = Context::create();
        let module = parse_ir(&context, text.as_bytes(), "unused-wide-snapshot").unwrap();
        lower(&module);
        // Unused volatile elements still have observable loads.
        assert_eq!(
            module
                .print_to_string()
                .to_string()
                .matches("load volatile i64")
                .count(),
            count * 2
        );
        assert!(
            module
                .print_to_string()
                .to_string()
                .lines()
                .filter(|line| !line.starts_with("target datalayout"))
                .all(|line| !line.contains("i128"))
        );
    }
}

#[test]
fn array_snapshot_matches_original_llvm_and_rust_on_cpu() {
    use inkwell::{
        OptimizationLevel,
        targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
    };
    Target::initialize_native(&InitializationConfig::default()).unwrap();
    let triple = TargetMachine::get_default_triple();
    let machine = Target::from_triple(&triple)
        .unwrap()
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::None,
            RelocMode::Default,
            CodeModel::Default,
        )
        .unwrap();
    for alignment in [1, 16, 32] {
        let context = Context::create();
        let original = parse_ir(&context, source(alignment).as_bytes(), "original").unwrap();
        let lowered = original.clone();
        lower(&lowered);
        for module in [&original, &lowered] {
            module.set_triple(&triple);
            module.set_data_layout(&machine.get_target_data().get_data_layout());
        }
        let before = original
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let after = lowered
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        // SAFETY: reviewed pointer-only kernels read 8 and write 144 bytes
        // within the guarded record; stores are explicitly byte aligned.
        let (before, after) = unsafe {
            (
                before
                    .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                    .unwrap(),
                after
                    .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                    .unwrap(),
            )
        };
        check_cases(17, |bytes| {
            // SAFETY: the shared oracle supplies the complete 160-byte record
            // and guard regions; all kernel accesses are explicitly unaligned.
            unsafe {
                before.call(bytes.as_mut_ptr().add(17));
            }
        });
        check_cases(17, |bytes| {
            // SAFETY: same reviewed kernel signature and bounds after lowering.
            unsafe {
                after.call(bytes.as_mut_ptr().add(17));
            }
        });
    }
}

#[test]
fn nonprivate_nested_oversized_and_escaping_aggregate_operations_are_refused() {
    let cases = [
        "%snapshot = load [9 x i128], ptr %p\n%v = extractvalue [9 x i128] %snapshot, 0\nstore i128 %v, ptr %p",
        "%slot = alloca [2 x [2 x i128]]\nstore i128 1, ptr %slot",
        "%slot = alloca [257 x i128]\nstore i128 1, ptr %slot",
        "%slot = alloca [9 x i128]\nstore ptr %slot, ptr %p",
        "%slot = alloca [9 x i128]\n%snapshot = load [9 x i128], ptr %slot\nstore [9 x i128] %snapshot, ptr %p",
        "%slot = alloca [9 x i128]\n%snapshot = load [9 x i128], ptr %slot\n%changed = insertvalue [9 x i128] %snapshot, i128 1, 0\n%v = extractvalue [9 x i128] %changed, 1\nstore i128 %v, ptr %p",
    ];
    for body in cases {
        let context = Context::create();
        let text = format!("{HEADER}\ndefine void @kernel(ptr %p) {{\n{body}\nret void\n}}");
        let module = parse_ir(&context, text.as_bytes(), "wide-array-refusal").unwrap();
        let before = module.print_to_string().to_string();
        assert!(legalize(&module, &interface()).is_err(), "{text}");
        assert_eq!(module.print_to_string().to_string(), before);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn array_snapshot_preserves_upper_limbs_overwrite_order_and_guards_on_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for alignment in [1, 16, 32] {
        let context = Context::create();
        let module = parse_ir(&context, source(alignment).as_bytes(), "wide-array-gpu").unwrap();
        let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("kernel.metallib");
        std::fs::write(&path, artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        check_cases(256, |bytes| {
            let mut buffers = [Buffer {
                bytes: bytes.to_vec(),
                offset: 256,
            }];
            // SAFETY: one invocation reads the input word and writes nine
            // 128-bit outputs within the guarded 160-byte buffer contract.
            // All snapshot elements are initialized before their volatile read.
            unsafe {
                kernel.run(&mut buffers, 1, 1).unwrap();
            }
            bytes.copy_from_slice(&buffers[0].bytes);
        });
    }
}
