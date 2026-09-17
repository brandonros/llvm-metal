//! Explicit integration lane: stock Rust dependency -> complete device bitcode.
//! This checks the frontend boundary; it does not execute NVPTX or Metal code.

use inkwell::{AddressSpace, context::Context, module::Module};
use llvm_metal_compiler::{parse_bitcode, parse_ir, require_entry};
use std::{fs, path::PathBuf, process::Command};

fn check_contract(module: &Module<'_>) {
    require_entry(module, "shallenge_sha256_32").unwrap();
    assert_eq!(
        module.get_triple().as_str().to_str().unwrap(),
        "nvptx64-nvidia-cuda"
    );
    let context = module.get_context();
    let pointer = context.ptr_type(AddressSpace::default());
    let entry = module.get_function("shallenge_sha256_32").unwrap();
    assert_eq!(
        entry.get_type(),
        context
            .void_type()
            .fn_type(&[pointer.into(), pointer.into()], false)
    );
    assert_eq!(
        entry.get_call_conventions(),
        0,
        "ordinary C function, not a PTX entry"
    );
    for function in module.get_functions() {
        assert!(
            function.count_basic_blocks() > 0 || function.get_intrinsic_id() != 0,
            "unresolved non-intrinsic function: {:?}",
            function.get_name()
        );
    }
    for global in module.get_globals() {
        assert!(
            global.get_initializer().is_some(),
            "unresolved global: {:?}",
            global.get_name()
        );
    }
}

#[test]
#[ignore = "run in nix develop .#rust-fixtures; builds the pinned Git dependency"]
fn pinned_shallenge_rust_build_produces_complete_device_bitcode() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let status = Command::new("python3")
        .arg(root.join("tests/rust-fixtures/build.py"))
        .current_dir(&root)
        .status()
        .expect("Python 3 from the rust-fixtures Nix shell");
    assert!(
        status.success(),
        "fixture build/CPU known-answer tests failed"
    );
    let artifacts = root.join("target/rust-fixtures/shallenge");
    let context = Context::create();
    let bitcode = fs::read(artifacts.join("kernel.bc")).unwrap();
    let module = parse_bitcode(&context, &bitcode, "shallenge.bc").unwrap();
    check_contract(&module);
    let text = fs::read(artifacts.join("kernel.ll")).unwrap();
    let textual = parse_ir(&context, &text, "shallenge.ll").unwrap();
    check_contract(&textual);
    let round_trip = module.write_bitcode_to_memory();
    check_contract(&parse_bitcode(&context, round_trip.as_slice(), "round-trip.bc").unwrap());
}
