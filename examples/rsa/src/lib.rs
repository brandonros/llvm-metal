//! Run the RSA signing kernel on the host and on the GPU.
pub mod key;

use llvm_metal_compiler::Compiled;
use llvm_metal_kernel::{Handle, host};
use rsa_kernel::{FL, SignKey, sign};
use std::path::{Path, PathBuf};

pub type Number = [u32; FL];

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Stand-ins for padded message hashes: random, top limb clear so each is below n.
pub fn messages(count: usize) -> Vec<Number> {
    let mut random = 0x9E37_79B9_7F4A_7C15u64;
    let mut limb = |j| {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        if j == FL - 1 {
            0
        } else {
            (random >> 16) as u32
        }
    };
    (0..count).map(|_| std::array::from_fn(&mut limb)).collect()
}

fn buffers(key: &SignKey, messages: &[Number]) -> Vec<Vec<u8>> {
    let input = messages.iter().flat_map(host::bytes).collect();
    vec![host::bytes(key), input, vec![0; size_of_val(messages)]]
}

/// The reference: the same function, run once per thread on this machine.
pub fn on_host(key: &SignKey, messages: &[Number]) -> Vec<Number> {
    let kernel = || {
        sign(
            Handle::HANDLE,
            Handle::HANDLE,
            Handle::HANDLE,
            Handle::HANDLE,
        )
    };
    let threads = messages.len() as u32;
    host::values(&host::launch(threads, buffers(key, messages), kernel)[2])
}

/// Compile the kernel crate into `directory`.
pub fn compile(directory: &Path) -> Result<Compiled, String> {
    let manifest = root().join("kernel/Cargo.toml");
    llvm_metal_compiler::compile(&manifest, "sign", directory).map_err(|e| format!("{e:#?}"))
}

/// Sign every message on the GPU, one thread each.
pub fn on_gpu(
    compiled: &Compiled,
    key: &SignKey,
    messages: &[Number],
) -> Result<Vec<Number>, String> {
    let mut buffers = buffers(key, messages);
    llvm_metal_runtime::run(
        &compiled.library,
        "sign",
        messages.len(),
        compiled.bindings.buffers,
        &mut buffers,
    )?;
    Ok(host::values(&buffers[2]))
}
