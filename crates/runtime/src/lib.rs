//! Synchronous execution of trusted compute kernels with shared Metal buffers.

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use metal::{Buffer, Kernel};
