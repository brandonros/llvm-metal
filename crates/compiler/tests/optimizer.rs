//! The actual producer pass sequence must compile a reduced Bech32 loop promptly
//! and preserve its writes. Real encoders are checked separately on the GPU.
use inkwell::{
    OptimizationLevel,
    context::Context,
    module::Module,
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
};
use llvm_metal_compiler::{parse_bitcode, parse_ir};
use std::{fs, path::PathBuf, process::Command, sync::OnceLock};

fn check_writes(module: Module<'_>) {
    let context = module.get_context();
    assert_eq!(
        module.get_function("bech32_counters").unwrap().get_type(),
        context.void_type().fn_type(
            &[context.ptr_type(inkwell::AddressSpace::default()).into()],
            false,
        )
    );
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| Target::initialize_native(&InitializationConfig::default()))
        .as_ref()
        .unwrap();
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
    module.set_triple(&triple);
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    let engine = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    // SAFETY: only the committed, reviewed bounded-loop fixture is executed.
    // Its signature is void(pointer), and 64 writable bytes have guards on both sides.
    let function = unsafe {
        engine
            .get_function::<unsafe extern "C" fn(*mut u8)>("bech32_counters")
            .unwrap()
    };
    let mut bytes = [0xa5; 80];
    unsafe { function.call(bytes.as_mut_ptr().add(8)) };
    let mut expected = [0xa5; 80];
    expected[9..12].fill(0);
    assert_eq!(bytes, expected);
}

fn check_producer(script: PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = root.join("tests/fixtures/optimizer/bech32-counters.ll");
    let context = Context::create();
    check_writes(parse_ir(&context, &fs::read(&source).unwrap(), "bech32").unwrap());
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("optimized.bc");
    // Call the producer's real helper, with a generous deadline instead of
    // duplicating its pipeline in the test. Python kills/reaps opt on timeout.
    let result = Command::new("python3")
        .args([
            "-B",
            "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1])['post_inline'](sys.argv[2], sys.argv[3], timeout=10)",
        ])
        .arg(&script)
        .arg(&source)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}: {}",
        script.display(),
        String::from_utf8_lossy(&result.stderr)
    );
    check_writes(parse_bitcode(&context, &fs::read(output).unwrap(), "optimized").unwrap());
}

#[test]
#[ignore = "requires Python 3 and LLVM 21.1.8 from .#rust-fixtures"]
fn fixture_producer_bounds_bech32_analysis_and_preserves_writes() {
    check_producer(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/rust-fixtures/build.py"),
    );
}

#[test]
#[ignore = "requires .#rust-fixtures and LLVM_METAL_CONSUMER_PATH"]
fn consumer_producer_bounds_bech32_analysis_and_preserves_writes() {
    let root =
        std::env::var_os("LLVM_METAL_CONSUMER_PATH").expect("set the isolated consumer path");
    check_producer(PathBuf::from(root).join("scripts/build-metal.py"));
}
