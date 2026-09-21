//! Run the P-256 signing kernel on the host and on the GPU.
pub mod curve;

use llvm_metal_kernel::{Handle, host};
use llvm_metal_runtime::{Buffer, Pipeline};
use p256_kernel::{Key, Request, Signature, point::Point, sign};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Stand-ins for hashed messages and nonces, written in place: random, top limb
/// clear so each is below n. TOY: a real nonce is derived, never drawn like this.
pub fn fill(requests: &mut [Request]) {
    let mut random = 0x9E37_79B9_7F4A_7C15u64;
    let mut limb = |j| {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        if j == 7 { 0 } else { (random >> 16) as u32 }
    };
    for request in requests {
        request.z = std::array::from_fn(&mut limb);
        request.k = std::array::from_fn(&mut limb);
    }
}

pub fn requests(count: usize) -> Vec<Request> {
    let mut requests = vec![
        Request {
            z: [0; 8],
            k: [0; 8]
        };
        count
    ];
    fill(&mut requests);
    requests
}

/// The reference: the same function, run once per thread on this machine.
pub fn on_host(key: &Key, table: &[Point], requests: &[Request]) -> Vec<Signature> {
    let buffers = vec![
        host::bytes(key),
        table.iter().flat_map(host::bytes).collect(),
        requests.iter().flat_map(host::bytes).collect(),
        vec![0; size_of_val(requests)],
    ];
    let kernel = || {
        sign(
            Handle::HANDLE,
            Handle::HANDLE,
            Handle::HANDLE,
            Handle::HANDLE,
            Handle::HANDLE,
        )
    };
    host::values(&host::launch(requests.len() as u32, buffers, kernel)[3])
}

/// The kernel, compiled and ready to launch. Every buffer lives on the GPU for
/// as long as this does: the key and the table are written once, and the caller
/// fills `requests` and looks at `signatures` in place.
pub struct Gpu {
    pipeline: Pipeline,
    slots: usize,
    key: Buffer<Key>,
    table: Buffer<Point>,
    pub requests: Buffer<Request>,
    pub signatures: Buffer<Signature>,
}

impl Gpu {
    /// Compile the kernel crate into `directory`, with room for `capacity` requests.
    pub fn compile(
        directory: &Path,
        key: &Key,
        table: &[Point],
        capacity: usize,
    ) -> Result<Self, String> {
        let manifest = root().join("kernel/Cargo.toml");
        let compiled = llvm_metal_compiler::compile(&manifest, "sign", directory)
            .map_err(|error| format!("{error:#?}"))?;
        let pipeline = Pipeline::load(&compiled.library, "sign")?;
        Ok(Gpu {
            key: pipeline.buffer_from(std::slice::from_ref(key))?,
            table: pipeline.buffer_from(table)?,
            requests: pipeline.buffer(capacity)?,
            signatures: pipeline.buffer(capacity)?,
            slots: compiled.bindings.buffers,
            pipeline,
        })
    }

    /// Sign the first `threads` requests, one thread each. Returns the GPU's
    /// execution time.
    pub fn sign(&mut self, threads: usize) -> Result<Duration, String> {
        self.pipeline.run(
            threads,
            self.slots,
            &mut [
                &mut self.key,
                &mut self.table,
                &mut self.requests,
                &mut self.signatures,
            ],
        )
    }
}
