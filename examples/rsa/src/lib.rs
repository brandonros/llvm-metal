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

/// Stand-ins for padded message hashes, written in place: random, top limb
/// clear so each is below n.
pub fn fill(messages: &mut [Number]) {
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
    for message in messages {
        *message = std::array::from_fn(&mut limb);
    }
}

pub fn messages(count: usize) -> Vec<Number> {
    let mut messages = vec![[0; FL]; count];
    fill(&mut messages);
    messages
}

/// The reference: the same function, run once per thread on this machine.
pub fn on_host(key: &SignKey, messages: &[Number]) -> Vec<Number> {
    let buffers = vec![
        host::bytes(key),
        messages.iter().flat_map(host::bytes).collect(),
        vec![0; size_of_val(messages)],
    ];
    let kernel = || {
        sign(
            Handle::HANDLE,
            Handle::HANDLE,
            Handle::HANDLE,
            Handle::HANDLE,
        )
    };
    host::values(&host::launch(messages.len() as u32, buffers, kernel)[2])
}

/// The kernel, compiled and ready to launch. Every buffer lives on the GPU for
/// as long as this does: the key is written once, and the caller fills
/// `messages` and looks at `signatures` in place.
pub struct Gpu {
    pipeline: Pipeline,
    slots: usize,
    key: Buffer<SignKey>,
    pub messages: Buffer<Number>,
    pub signatures: Buffer<Number>,
}

impl Gpu {
    /// Compile the kernel crate into `directory`, with room for `capacity` messages.
    pub fn compile(directory: &Path, key: &SignKey, capacity: usize) -> Result<Self, String> {
        let manifest = root().join("kernel/Cargo.toml");
        let compiled = llvm_metal_compiler::compile(&manifest, "sign", directory)
            .map_err(|error| format!("{error:#?}"))?;
        let pipeline = Pipeline::load(&compiled.library, "sign")?;
        Ok(Gpu {
            key: pipeline.buffer_from(std::slice::from_ref(key))?,
            messages: pipeline.buffer(capacity)?,
            signatures: pipeline.buffer(capacity)?,
            slots: compiled.bindings.buffers,
            pipeline,
        })
    }

    /// Sign the first `threads` messages, one thread each. Returns the GPU's
    /// execution time.
    pub fn sign(&mut self, threads: usize) -> Result<Duration, String> {
        self.pipeline.run(
            threads,
            self.slots,
            &mut [&mut self.key, &mut self.messages, &mut self.signatures],
        )
    }
}
