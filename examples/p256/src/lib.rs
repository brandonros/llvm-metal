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
        .map(|_| Request {
            z: std::array::from_fn(&mut limb),
            k: std::array::from_fn(&mut limb),
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

/// The kernel, compiled and ready to launch, with the key and the table
/// already on the GPU: they are written once, however many batches follow.
pub struct Gpu {
    pipeline: Pipeline,
    slots: usize,
    key: Buffer<Key>,
    table: Buffer<Point>,
}

impl Gpu {
    /// Compile the kernel crate into `directory` and build its pipeline.
    pub fn compile(directory: &Path, key: &Key, table: &[Point]) -> Result<Self, String> {
        let manifest = root().join("kernel/Cargo.toml");
        let compiled = llvm_metal_compiler::compile(&manifest, "sign", directory)
            .map_err(|error| format!("{error:#?}"))?;
        let pipeline = Pipeline::load(&compiled.library, "sign")?;
        Ok(Gpu {
            key: pipeline.buffer_from(std::slice::from_ref(key))?,
            table: pipeline.buffer_from(table)?,
            slots: compiled.bindings.buffers,
            pipeline,
        })
    }

    /// Sign every request, one thread each; also the GPU's execution time.
    pub fn sign(&mut self, requests: &[Request]) -> Result<(Vec<Signature>, Duration), String> {
        let mut input = self.pipeline.buffer_from(requests)?;
        let mut output = self.pipeline.buffer::<Signature>(requests.len())?;
        let elapsed = self.pipeline.run(
            requests.len(),
            self.slots,
            &mut [&mut self.key, &mut self.table, &mut input, &mut output],
        )?;
        Ok((output.read(<[Signature]>::to_vec), elapsed))
    }
}
