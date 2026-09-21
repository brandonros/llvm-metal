//! The builder's cleanup pipeline must compile a reduced Bech32 loop promptly
//! and preserve its writes. Real encoders are checked separately on the GPU.
use inkwell::{
    OptimizationLevel,
    context::Context,
    module::Module,
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
};
use llvm_metal_compiler::{build::unit::post_inline, parse_ir};
use std::{fs, path::PathBuf, sync::OnceLock, time::Instant};

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

#[test]
fn cleanup_bounds_bech32_analysis_and_preserves_writes() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = fs::read(root.join("tests/fixtures/optimizer/bech32-counters.ll")).unwrap();
    let (context, reread) = (Context::create(), Context::create());
    check_writes(parse_ir(&context, &source, "bech32").unwrap());
    // O3 directly on these loops spends minutes in ScalarEvolution; the
    // builder's ordering exists to avoid that, so a regression shows as time.
    let start = Instant::now();
    let optimized = post_inline(
        parse_ir(&context, &source, "bech32").unwrap(),
        &reread,
        llvm_metal_compiler::air::InliningPolicy::Selective,
    )
    .unwrap();
    assert!(start.elapsed().as_secs() < 10, "{:?}", start.elapsed());
    check_writes(optimized);
}
