use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{air::legalize, parse_ir};

const MODES: [&str; 8] = [
    "private-private",
    "private-device",
    "device-private",
    "device-device",
    "constant-private",
    "constant-device",
    "equal-private",
    "equal-device",
];
const BYTES: usize = 224;

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
            bytes: BYTES,
            alignment: 8,
        }],
    }
}

fn source(mode: &str, offset: usize) -> String {
    let (source_kind, to) = mode.split_once('-').unwrap();
    let from = match source_kind {
        "private" => "%private_source",
        "device" => "%input",
        "constant" => "%constant_source",
        "equal" => {
            if to == "private" {
                "%private_destination"
            } else {
                "%output"
            }
        }
        _ => unreachable!(),
    };
    let to = if to == "private" {
        "%private_destination"
    } else {
        "%output"
    };
    let dump = if to == "%private_destination" {
        "call void @llvm.memcpy.p0.p0.i64(ptr %dump, ptr %b, i64 100, i1 false)"
    } else {
        ""
    };
    let constants = (0..65)
        .map(|i| format!("i8 {}", (i * 37 + 11) as u8))
        .collect::<Vec<_>>()
        .join(", ");
    let prefix = "i8 165, ".repeat(offset);
    let allocation = 65 + offset;
    let input_offset = 16 + offset;
    let output_offset = 128 + offset;
    format!(
        r#"
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
@constant = private constant [{allocation} x i8] [{prefix}{constants}], align 1
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)
define void @kernel(ptr %data) {{
entry:
  %a = alloca [100 x i8], align 8
  %b = alloca [100 x i8], align 8
  %length = load i64, ptr %data, align 8
  %valid = icmp ule i64 %length, 65
  br i1 %valid, label %checked, label %exit
checked:
  %input = getelementptr i8, ptr %data, i64 {input_offset}
  %output = getelementptr i8, ptr %data, i64 {output_offset}
  %dump = getelementptr i8, ptr %data, i64 112
  %constant_source = getelementptr i8, ptr @constant, i64 {offset}
  %private_source = getelementptr i8, ptr %a, i64 {input_offset}
  %private_destination = getelementptr i8, ptr %b, i64 {input_offset}
  call void @llvm.memset.p0.i64(ptr %a, i8 90, i64 100, i1 false)
  call void @llvm.memset.p0.i64(ptr %b, i8 150, i64 100, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %private_source, ptr %input, i64 65, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr {to}, ptr {from}, i64 %length, i1 false)
  {dump}
  br label %exit
exit:
  ret void
}}
"#
    )
}

fn check_cases(mut run: impl FnMut(&mut [u8]), mode: &str, offset: usize) {
    for length in 0usize..=65 {
        let mut bytes = vec![0xa5; BYTES + 512];
        bytes[256..264].copy_from_slice(&(length as u64).to_le_bytes());
        for i in 0..65 {
            bytes[272 + offset + i] = (i * 37 + 11) as u8;
        }
        bytes[368..468].fill(150);
        let mut expected = bytes.clone();
        if !mode.starts_with("equal-") {
            for i in 0..length {
                expected[384 + offset + i] = (i * 37 + 11) as u8;
            }
        }
        run(&mut bytes);
        assert_eq!(bytes, expected, "{mode} length={length} offset={offset}");
    }
}

#[test]
fn runtime_copies_become_byte_loops_without_mutating_input() {
    for (mode, offset) in MODES
        .into_iter()
        .flat_map(|mode| [0, 1].map(|offset| (mode, offset)))
    {
        let context = Context::create();
        let module = parse_ir(&context, source(mode, offset).as_bytes(), "dynamic-copy").unwrap();
        let before = module.print_to_string().to_string();
        let (air, _) = legalize(&module, &interface()).unwrap();
        air.verify().unwrap();
        let text = air.print_to_string().to_string();
        if !mode.starts_with("equal-") {
            assert!(
                text.contains("copy.more"),
                "missing zero-safe copy loop: {mode}"
            );
        }
        assert_no_dynamic_copies(&air, mode);
        assert_eq!(module.print_to_string().to_string(), before);
    }
}

fn assert_no_dynamic_copies(air: &inkwell::module::Module<'_>, mode: &str) {
    for f in air.get_functions() {
        for block in f.get_basic_blocks() {
            for i in block.get_instructions() {
                if let Ok(call) = inkwell::values::CallSiteValue::try_from(i) {
                    if call.get_called_fn_value().is_some_and(|f| {
                        let name = f.get_name().to_bytes();
                        name.starts_with(b"llvm.memcpy.") || name.starts_with(b"llvm.memset.")
                    }) {
                        assert!(
                            i.get_operand(2)
                                .unwrap()
                                .value()
                                .unwrap()
                                .into_int_value()
                                .is_const(),
                            "dynamic memory intrinsic remains: {mode}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn runtime_copy_counts_and_guards_match_original_on_native_cpu() {
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
    for (mode, offset) in MODES
        .into_iter()
        .flat_map(|mode| [0, 1].map(|offset| (mode, offset)))
    {
        let context = Context::create();
        let original = parse_ir(&context, source(mode, offset).as_bytes(), "dynamic-copy").unwrap();
        let (air, _) = legalize(&original, &interface()).unwrap();
        for module in [&original, &air] {
            module.set_triple(&triple);
            module.set_data_layout(&machine.get_target_data().get_data_layout());
        }
        for f in air.get_functions() {
            f.remove_string_attribute(
                inkwell::attributes::AttributeLoc::Function,
                "no-builtin-memcpy",
            );
        }
        air.run_passes(
            "default<O3>,function(loop(loop-idiom))",
            &machine,
            inkwell::passes::PassBuilderOptions::create(),
        )
        .unwrap();
        air.verify().unwrap();
        assert_no_dynamic_copies(&air, mode);
        let before = original
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let after = air
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        // SAFETY: reviewed one-pointer ABI, n<=65, valid input/output spans
        // that are disjoint or exactly identical, and 224-byte allocation surrounded by observable host guards.
        unsafe {
            let reference = before
                .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                .unwrap();
            let lowered = after
                .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                .unwrap();
            check_cases(
                |bytes| {
                    let mut original = bytes.to_vec();
                    reference.call(original.as_mut_ptr().add(256));
                    lowered.call(bytes.as_mut_ptr().add(256));
                    assert_eq!(bytes, original, "original/lowered mismatch: {mode}");
                },
                mode,
                offset,
            );
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade; zero-copy regression can stall an unfixed driver"]
fn runtime_copy_counts_and_guards_match_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for (mode, offset) in MODES
        .into_iter()
        .flat_map(|mode| [0, 1].map(|offset| (mode, offset)))
    {
        let context = Context::create();
        let module = parse_ir(&context, source(mode, offset).as_bytes(), "dynamic-copy").unwrap();
        let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("kernel.metallib");
        std::fs::write(&path, artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        check_cases(
            |bytes| {
                let mut buffers = [Buffer {
                    bytes: bytes.to_vec(),
                    offset: 256,
                }];
                // SAFETY: same reviewed bounded ranges as the native differential test.
                unsafe {
                    kernel.run(&mut buffers, 1, 1).unwrap();
                }
                bytes.copy_from_slice(&buffers[0].bytes);
            },
            mode,
            offset,
        );
    }
}

fn set_source(private: bool, offset: usize) -> String {
    let destination = if private {
        "%local_destination"
    } else {
        "%device_destination"
    };
    let dump = if private {
        "call void @llvm.memcpy.p0.p0.i64(ptr %dump, ptr %local, i64 100, i1 false)"
    } else {
        ""
    };
    let local_offset = 16 + offset;
    let device_offset = 128 + offset;
    format!(
        r#"
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)
define void @kernel(ptr %data) {{
entry:
  %local = alloca [100 x i8], align 8
  %length = load i64, ptr %data, align 8
  %fill_pointer = getelementptr i8, ptr %data, i64 8
  %fill = load i8, ptr %fill_pointer, align 1
  %valid = icmp ule i64 %length, 65
  br i1 %valid, label %checked, label %exit
checked:
  %dump = getelementptr i8, ptr %data, i64 112
  %device_destination = getelementptr i8, ptr %data, i64 {device_offset}
  %local_destination = getelementptr i8, ptr %local, i64 {local_offset}
  call void @llvm.memset.p0.i64(ptr %local, i8 150, i64 100, i1 false)
  call void @llvm.memset.p0.i64(ptr {destination}, i8 %fill, i64 %length, i1 false)
  {dump}
  br label %exit
exit:
  ret void
}}
"#
    )
}

fn check_set_cases(mut run: impl FnMut(&mut [u8]), private: bool, offset: usize) {
    for length in 0usize..=65 {
        for fill in [0u8, 1, 128, 255] {
            let mut bytes = vec![0xa5; BYTES + 512];
            bytes[256..264].copy_from_slice(&(length as u64).to_le_bytes());
            bytes[264] = fill;
            bytes[368..468].fill(150);
            let mut expected = bytes.clone();
            expected[384 + offset..384 + offset + length].fill(fill);
            run(&mut bytes);
            assert_eq!(
                bytes, expected,
                "memset private={private} offset={offset} length={length} fill={fill}"
            );
        }
    }
}

#[test]
fn runtime_set_counts_bytes_and_guards_match_original_on_native_cpu() {
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
    for (private, offset) in [false, true]
        .into_iter()
        .flat_map(|private| [0, 1].map(|offset| (private, offset)))
    {
        let context = Context::create();
        let original = parse_ir(
            &context,
            set_source(private, offset).as_bytes(),
            "dynamic-set",
        )
        .unwrap();
        let snapshot = original.print_to_string().to_string();
        let (air, _) = legalize(&original, &interface()).unwrap();
        air.verify().unwrap();
        assert_no_dynamic_copies(&air, "memset");
        assert_eq!(original.print_to_string().to_string(), snapshot);
        for module in [&original, &air] {
            module.set_triple(&triple);
            module.set_data_layout(&machine.get_target_data().get_data_layout());
        }
        // Exercise optimization without relying on Apple's support for these
        // LLVM TargetLibraryInfo attributes, just as in the memcpy regression.
        for f in air.get_functions() {
            for attr in ["no-builtin-memcpy", "no-builtin-memset"] {
                f.remove_string_attribute(inkwell::attributes::AttributeLoc::Function, attr);
            }
        }
        air.run_passes(
            "default<O3>,function(loop(loop-idiom))",
            &machine,
            inkwell::passes::PassBuilderOptions::create(),
        )
        .unwrap();
        air.verify().unwrap();
        assert_no_dynamic_copies(&air, "optimized memset");
        let before = original
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let after = air
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        // SAFETY: one-pointer ABI, n<=65; all fill stores are within the reviewed
        // private/device allocation, and complete host/private guards are compared.
        unsafe {
            let reference = before
                .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                .unwrap();
            let lowered = after
                .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                .unwrap();
            check_set_cases(
                |bytes| {
                    let mut original = bytes.to_vec();
                    reference.call(original.as_mut_ptr().add(256));
                    lowered.call(bytes.as_mut_ptr().add(256));
                    assert_eq!(bytes, original, "original/lowered memset mismatch");
                },
                private,
                offset,
            );
        }
    }
}

#[test]
fn dynamic_volatile_memset_remains_refused() {
    for private in [false, true] {
        let context = Context::create();
        let source = set_source(private, 1).replace(
            "i8 %fill, i64 %length, i1 false",
            "i8 %fill, i64 %length, i1 true",
        );
        let module = parse_ir(&context, source.as_bytes(), "volatile-set").unwrap();
        let before = module.print_to_string().to_string();
        assert!(legalize(&module, &interface()).is_err());
        assert_eq!(module.print_to_string().to_string(), before);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn runtime_set_counts_bytes_and_guards_match_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for (private, offset) in [false, true]
        .into_iter()
        .flat_map(|private| [0, 1].map(|offset| (private, offset)))
    {
        let context = Context::create();
        let module = parse_ir(
            &context,
            set_source(private, offset).as_bytes(),
            "dynamic-set",
        )
        .unwrap();
        let artifact = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("kernel.metallib");
        std::fs::write(&path, artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        check_set_cases(
            |bytes| {
                let mut buffers = [Buffer {
                    bytes: bytes.to_vec(),
                    offset: 256,
                }];
                // SAFETY: same reviewed bounded ranges as the native differential test.
                unsafe {
                    kernel.run(&mut buffers, 1, 1).unwrap();
                }
                bytes.copy_from_slice(&buffers[0].bytes);
            },
            private,
            offset,
        );
    }
}

fn narrow_count_source(width: u32, is_set: bool) -> String {
    let operation = if is_set {
        format!(
            "call void @llvm.memset.p0.i{width}(ptr %destination, i8 %fill, i{width} %count, i1 false)"
        )
    } else {
        format!(
            "call void @llvm.memcpy.p0.p0.i{width}(ptr %destination, ptr %source, i{width} %count, i1 false)"
        )
    };
    format!(
        r#"
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
declare void @llvm.memcpy.p0.p0.i{width}(ptr, ptr, i{width}, i1 immarg)
declare void @llvm.memset.p0.i{width}(ptr, i8, i{width}, i1 immarg)
define void @kernel(ptr %data) {{
  %count = load i{width}, ptr %data, align 1
  %fill_address = getelementptr i8, ptr %data, i64 8
  %fill = load i8, ptr %fill_address, align 1
  %source = getelementptr i8, ptr %data, i64 33
  %destination = getelementptr i8, ptr %data, i64 353
  {operation}
  ret void
}}
"#
    )
}

#[test]
fn narrow_unsigned_copy_and_set_counts_keep_positive_pointer_offsets() {
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
    for width in [1, 8] {
        for is_set in [false, true] {
            let context = Context::create();
            // Both overloaded signatures must pass LLVM's verifier; they are
            // legal LLVM inputs even though the Rust producer normally uses i64.
            let original = parse_ir(
                &context,
                narrow_count_source(width, is_set).as_bytes(),
                "narrow-memory-count",
            )
            .unwrap();
            let mut abi = interface();
            abi.arguments[0].bytes = 640;
            let (air, _) = legalize(&original, &abi).unwrap();
            assert_no_dynamic_copies(&air, "narrow count");
            for module in [&original, &air] {
                module.set_triple(&triple);
                module.set_data_layout(&machine.get_target_data().get_data_layout());
            }
            let before = original
                .create_jit_execution_engine(OptimizationLevel::None)
                .unwrap();
            let after = air
                .create_jit_execution_engine(OptimizationLevel::None)
                .unwrap();
            let counts: &[u8] = if width == 1 {
                &[0, 1]
            } else {
                &[0, 1, 127, 128, 129, 255]
            };
            // SAFETY: every possible i1/i8 count fits the two disjoint256-byte
            // ranges. The byte-offset1 spans deliberately exercise unaligned GEPs.
            unsafe {
                let reference = before
                    .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                    .unwrap();
                let lowered = after
                    .get_function::<unsafe extern "C" fn(*mut u8)>("kernel")
                    .unwrap();
                for &count in counts {
                    for fill in [0u8, 129, 255] {
                        let mut bytes = vec![0xa5; 640 + 512];
                        bytes[256] = count;
                        bytes[264] = fill;
                        for i in 0..256 {
                            bytes[256 + 33 + i] = (i * 37 + 11) as u8;
                        }
                        bytes[256 + 353..256 + 353 + 256].fill(150);
                        let mut expected = bytes.clone();
                        for i in 0..usize::from(count) {
                            expected[256 + 353 + i] =
                                if is_set { fill } else { bytes[256 + 33 + i] };
                        }
                        let mut native = bytes.clone();
                        reference.call(native.as_mut_ptr().add(256));
                        lowered.call(bytes.as_mut_ptr().add(256));
                        assert_eq!(
                            native, expected,
                            "native i{width} count={count} memset={is_set}"
                        );
                        assert_eq!(
                            bytes, expected,
                            "lowered i{width} count={count} memset={is_set}"
                        );
                    }
                }
            }
        }
    }
}
