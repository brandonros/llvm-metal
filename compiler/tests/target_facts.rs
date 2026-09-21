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
    llvm_metal_runtime::launch(&library, "probe", 1, &mut buffers)?;
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
