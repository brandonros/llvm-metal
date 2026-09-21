//! Run the shallenge kernel on the host and on the GPU.
pub mod sha256;

use llvm_metal_kernel::{Handle, host};
use llvm_metal_runtime::{Buffer, Pipeline};
use shallenge_kernel::{NONCE, Record, Request, character, shallenge};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Lay out the one block every thread hashes: the prefix, room for the nonce,
/// the 0x80 marker and the bit length. The prefix is at most 32 bytes, so the
/// message always fits one block.
pub fn request(prefix: &[u8], base: u64, salt: u64, attempts: u32) -> Request {
    assert!(prefix.len() <= 32, "the prefix is at most 32 bytes");
    let length = prefix.len() + NONCE;
    let mut bytes = [0u8; 64];
    bytes[..prefix.len()].copy_from_slice(prefix);
    bytes[length] = 0x80;
    bytes[56..].copy_from_slice(&(length as u64 * 8).to_be_bytes());
    Request {
        base,
        salt,
        block: std::array::from_fn(|i| u32::from_be_bytes(bytes[4 * i..][..4].try_into().unwrap())),
        offset: prefix.len() as u32,
        attempts,
    }
}

/// The text of the nonce a record names.
pub fn nonce(request: &Request, record: &Record) -> [u8; NONCE] {
    std::array::from_fn(|i| character(record.counter, request.salt, i) as u8)
}

/// The reference: the same function, run once per thread on this machine.
pub fn on_host(request: &Request, threads: usize) -> Vec<Record> {
    let buffers = vec![host::bytes(request), vec![0; threads * size_of::<Record>()]];
    let kernel = || shallenge(Handle::HANDLE, Handle::HANDLE, Handle::HANDLE);
    host::values(&host::launch(threads as u32, buffers, kernel)[1])
}

/// The kernel, compiled and ready to launch, with its buffers on the GPU: the
/// caller writes `request` and looks at `records` in place.
pub struct Gpu {
    pipeline: Pipeline,
    slots: usize,
    pub request: Buffer<Request>,
    pub records: Buffer<Record>,
}

impl Gpu {
    /// Compile the kernel crate into `directory`, with room for `threads` records.
    pub fn compile(directory: &Path, threads: usize) -> Result<Self, String> {
        let manifest = root().join("kernel/Cargo.toml");
        let compiled = llvm_metal_compiler::compile(&manifest, "shallenge", directory)
            .map_err(|error| format!("{error:#?}"))?;
        let pipeline = Pipeline::load(&compiled.library, "shallenge")?;
        Ok(Gpu {
            request: pipeline.buffer(1)?,
            records: pipeline.buffer(threads)?,
            slots: compiled.bindings.buffers,
            pipeline,
        })
    }

    /// Run `threads` threads over the request. Returns the GPU's execution time.
    pub fn search(&mut self, threads: usize) -> Result<Duration, String> {
        let buffers = &mut [&mut self.request as _, &mut self.records as _];
        self.pipeline.run(threads, self.slots, buffers)
    }
}
