//! Run the RSA signing kernel on the host and on the GPU.
pub mod key;

use llvm_metal_kernel::{Handle, host};
use llvm_metal_runtime::{Buffer, Pipeline};
use rsa_kernel::{FL, SignKey, sign};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

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

/// The kernel, compiled and ready to launch, with the key already on the GPU.
pub struct Gpu {
    pipeline: Pipeline,
    slots: usize,
    key: Buffer<SignKey>,
}

impl Gpu {
    /// Compile the kernel crate into `directory` and build its pipeline.
    pub fn compile(directory: &Path, key: &SignKey) -> Result<Self, String> {
        let manifest = root().join("kernel/Cargo.toml");
        let compiled = llvm_metal_compiler::compile(&manifest, "sign", directory)
            .map_err(|error| format!("{error:#?}"))?;
        let pipeline = Pipeline::load(&compiled.library, "sign")?;
        Ok(Gpu {
            key: pipeline.buffer_from(std::slice::from_ref(key))?,
            slots: compiled.bindings.buffers,
            pipeline,
        })
    }

    /// Sign every message, one thread each; also the GPU's execution time.
    pub fn sign(&mut self, messages: &[Number]) -> Result<(Vec<Number>, Duration), String> {
        let mut input = self.pipeline.buffer_from(messages)?;
        let mut output = self.pipeline.buffer::<Number>(messages.len())?;
        let elapsed = self.pipeline.run(
            messages.len(),
            self.slots,
            &mut [&mut self.key, &mut input, &mut output],
        )?;
        Ok((output.read(<[Number]>::to_vec), elapsed))
    }
}
