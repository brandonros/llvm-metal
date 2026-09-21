//! Run the shallenge kernel on the host and on the GPU.
use llvm_metal_kernel::{Handle, host};
use shallenge_kernel::{Record, Request, shallenge};
use std::path::{Path, PathBuf};

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn buffers(request: &Request, threads: usize) -> Vec<Vec<u8>> {
    vec![host::bytes(request), vec![0; threads * size_of::<Record>()]]
}

/// The reference: the same function, run once per thread on this machine.
pub fn on_host(request: &Request, threads: usize) -> Vec<Record> {
    let kernel = || shallenge(Handle::HANDLE, Handle::HANDLE, Handle::HANDLE);
    host::values(&host::launch(threads as u32, buffers(request, threads), kernel)[1])
}

/// Compile the kernel crate into `directory` and run it on the GPU.
pub fn on_gpu(request: &Request, threads: usize, directory: &Path) -> Result<Vec<Record>, String> {
    let manifest = root().join("kernel/Cargo.toml");
    let compiled = llvm_metal_compiler::compile(&manifest, "shallenge", directory)
        .map_err(|error| format!("{error:#?}"))?;
    let pipeline = llvm_metal_runtime::Pipeline::load(&compiled.library, "shallenge")?;
    let mut request = pipeline.buffer_from(std::slice::from_ref(request))?;
    let mut records = pipeline.buffer::<Record>(threads)?;
    let slots = compiled.bindings.buffers;
    pipeline.run(threads, slots, &mut [&mut request, &mut records])?;
    Ok(records.read(<[Record]>::to_vec))
}
