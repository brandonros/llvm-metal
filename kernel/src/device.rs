//! The four operations the compiler knows. On the GPU target they are
//! undefined functions, which `lower` replaces; `slot` is always a constant
//! there, because it comes from a const generic. On the host they act on the
//! buffers bound by `host::launch`.

#[cfg(not(target_arch = "nvptx64"))]
pub(crate) use crate::host::{length, load, store, thread_index};

#[cfg(target_arch = "nvptx64")]
unsafe extern "C" {
    #[link_name = "llvm_metal.thread_index"]
    pub(crate) safe fn thread_index() -> u32;

    /// The buffer's length in bytes.
    #[link_name = "llvm_metal.length"]
    pub(crate) safe fn length(slot: u32) -> u64;

    /// # Safety
    /// `offset < length(slot)`.
    #[link_name = "llvm_metal.load"]
    pub(crate) fn load(slot: u32, offset: u64) -> u8;

    /// # Safety
    /// `offset < length(slot)`, and the buffer is an `Out`.
    #[link_name = "llvm_metal.store"]
    pub(crate) fn store(slot: u32, offset: u64, byte: u8);
}
