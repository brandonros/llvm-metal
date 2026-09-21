//! What Apple's compiler accepts, asked directly: hand-written modules that
//! bypass `verify` and `lower`. Each rule that names a limit of the target
//! points at the test here that shows it.
#![cfg(target_os = "macos")]
use inkwell::{context::Context, memory_buffer::MemoryBuffer};
use llvm_metal_compiler::emit;
use std::path::PathBuf;

/// Compile `facts/<name>.ll`, run its `probe` once, and return buffer 1.
fn probe(name: &str, input: u64) -> Result<u64, String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read(root.join("tests/facts").join(format!("{name}.ll"))).unwrap();
    let context = Context::create();
    let buffer = MemoryBuffer::create_from_memory_range_copy(&source, name);
    let module = context
        .create_module_from_ir(buffer)
        .map_err(|error| error.to_string())?;
    let library = emit::library(&module, "probe", &root.join("../target/facts").join(name))?;
    let mut buffers = [input.to_le_bytes().to_vec(), vec![0xa5; 8]];
    llvm_metal_runtime::Pipeline::load(&library, "probe")?.launch(1, &mut buffers)?;
    Ok(u64::from_le_bytes(buffers[1][..].try_into().unwrap()))
}

#[test]
#[ignore = "requires an Apple GPU"]
fn the_difference_of_two_thread_addresses_is_their_distance() {
    assert_eq!(probe("ptrtoint", 11), Ok(11));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn a_thread_address_can_be_observed_as_an_integer() {
    assert_eq!(probe("address", 11), Ok(11));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn constant_memory_copies_to_thread_memory() {
    assert_eq!(probe("constant_copy", 2), Ok(7));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn flags_announcing_stripped_debug_info_are_harmless() {
    assert_eq!(probe("debug_flags", 21), Ok(42));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn a_noalias_scope_without_its_declaration_is_harmless() {
    assert_eq!(probe("noalias_scope", 11), Ok(11));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn attributes_newer_than_the_encoding_are_harmless() {
    assert_eq!(probe("modern_attributes", 11), Ok(11));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn a_copy_of_a_length_known_only_at_run_time_works() {
    assert_eq!(probe("dynamic_copy", 8), Ok(8));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn a_large_constant_table_copies_to_thread_memory() {
    assert_eq!(probe("large_constant", 50), Ok(150));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn a_function_returns_a_small_struct_by_value() {
    assert_eq!(probe("aggregate_return", 11), Ok(42));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn a_value_without_data_has_an_address() {
    assert_eq!(probe("empty_alloca", 21), Ok(42));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn a_position_independent_module_runs_once_emit_clears_the_flag() {
    // The kernel's body is empty, so the output keeps what the test put there.
    assert_eq!(probe("pic_relocation", 0), Ok(0xa5a5_a5a5_a5a5_a5a5));
}
