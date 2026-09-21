//! Run the P-256 signing kernel on the host and on the GPU.
pub mod curve;

use llvm_metal_kernel::{Handle, host};
use llvm_metal_runtime::Pipeline;
use p256_kernel::{Key, Request, Signature, point::Point, sign};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Stand-ins for hashed messages and nonces: random, top limb clear so each is
/// below n. TOY: a real nonce is derived, never drawn like this.
pub fn requests(count: usize) -> Vec<Request> {
    let mut random = 0x9E37_79B9_7F4A_7C15u64;
    let mut limb = |j| {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        if j == 7 { 0 } else { (random >> 16) as u32 }
    };
    (0..count)
        .map(|_| {
            [
                std::array::from_fn(&mut limb),
                std::array::from_fn(&mut limb),
            ]
        })
        .collect()
}

fn buffers(key: &Key, table: &[Point], requests: &[Request]) -> Vec<Vec<u8>> {
    vec![
        host::bytes(key),
        table.iter().flat_map(host::bytes).collect(),
        requests.iter().flat_map(host::bytes).collect(),
        vec![0; size_of_val(requests)],
    ]
}

/// The reference: the same function, run once per thread on this machine.
pub fn on_host(key: &Key, table: &[Point], requests: &[Request]) -> Vec<Signature> {
    let kernel = || {
        sign(
            Handle::HANDLE,
            Handle::HANDLE,
            Handle::HANDLE,
            Handle::HANDLE,
            Handle::HANDLE,
        )
    };
    let threads = requests.len() as u32;
    host::values(&host::launch(threads, buffers(key, table, requests), kernel)[3])
}

/// The kernel, compiled and ready to launch.
pub struct Gpu {
    pipeline: Pipeline,
    slots: usize,
}

impl Gpu {
    /// Compile the kernel crate into `directory` and build its pipeline.
    pub fn compile(directory: &Path) -> Result<Self, String> {
        let manifest = root().join("kernel/Cargo.toml");
        let compiled = llvm_metal_compiler::compile(&manifest, "sign", directory)
            .map_err(|error| format!("{error:#?}"))?;
        Ok(Gpu {
            pipeline: Pipeline::load(&compiled.library, "sign")?,
            slots: compiled.bindings.buffers,
        })
    }

    /// Sign every request, one thread each; also the GPU's execution time.
    pub fn sign(
        &self,
        key: &Key,
        table: &[Point],
        requests: &[Request],
    ) -> Result<(Vec<Signature>, Duration), String> {
        let mut buffers = buffers(key, table, requests);
        let elapsed = self
            .pipeline
            .run(requests.len(), self.slots, &mut buffers)?;
        Ok((host::values(&buffers[3]), elapsed))
    }
}
