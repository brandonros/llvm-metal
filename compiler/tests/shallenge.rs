//! The example kernel is accepted whatever rustc did to it.
use llvm_metal_compiler::{cargo, program};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn accepted_at_every_optimization_level() {
    for level in ["0", "s", "3"] {
        // SAFETY: this test binary has one test and no other threads.
        unsafe { std::env::set_var("CARGO_PROFILE_RELEASE_OPT_LEVEL", level) };
        let bitcode = cargo::bitcode(
            &root().join("examples/shallenge/kernel/Cargo.toml"),
            &root().join("target/kernels").join(level),
        )
        .unwrap();
        let context = inkwell::context::Context::create();
        let module = program(&context, &bitcode, "shallenge")
            .unwrap_or_else(|error| panic!("opt-level {level}: {error:#?}"));
        let functions = module
            .get_functions()
            .filter(|f| f.count_basic_blocks() != 0)
            .count();
        eprintln!("opt-level {level}: {functions} functions");
    }
}
