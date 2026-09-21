//! The same kernels, run on the host: the reference for the GPU.
use std::cell::{Cell, RefCell};

thread_local! {
    static THREAD: Cell<u32> = const { Cell::new(0) };
    static BUFFERS: RefCell<Vec<Vec<u8>>> = const { RefCell::new(Vec::new()) };
}

/// Run `kernel` once per thread index against `buffers`, indexed by slot, and
/// return them as the kernel left them.
pub fn launch(threads: u32, buffers: Vec<Vec<u8>>, kernel: impl Fn()) -> Vec<Vec<u8>> {
    BUFFERS.set(buffers);
    for thread in 0..threads {
        THREAD.set(thread);
        kernel();
    }
    BUFFERS.take()
}

pub(crate) fn thread_index() -> u32 {
    THREAD.get()
}

pub(crate) fn length(slot: u32) -> u64 {
    BUFFERS.with_borrow(|buffers| buffers[slot as usize].len() as u64)
}

pub(crate) unsafe fn load(slot: u32, offset: u64) -> u8 {
    BUFFERS.with_borrow(|buffers| buffers[slot as usize][offset as usize])
}

pub(crate) unsafe fn store(slot: u32, offset: u64, byte: u8) {
    BUFFERS.with_borrow_mut(|buffers| buffers[slot as usize][offset as usize] = byte);
}

/// The bytes of a value, as a buffer holds them.
pub fn bytes<T: crate::Plain>(value: &T) -> Vec<u8> {
    // SAFETY: `Plain` values have no padding, so every byte is initialized.
    unsafe { std::slice::from_raw_parts((&raw const *value).cast::<u8>(), size_of::<T>()) }.to_vec()
}

/// The values a buffer holds.
pub fn values<T: crate::Plain>(bytes: &[u8]) -> Vec<T> {
    let chunks = bytes.chunks_exact(size_of::<T>());
    // SAFETY: each chunk is `size_of::<T>()` bytes, and any bits are a `Plain`.
    chunks
        .map(|chunk| unsafe { chunk.as_ptr().cast::<T>().read_unaligned() })
        .collect()
}
