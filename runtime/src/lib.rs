//! Run a compute kernel from a Metal library, synchronously, over byte buffers.
#![cfg(target_os = "macos")]
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;
use std::{path::Path, ptr::NonNull};

/// Launch `entry` from `library` once per thread. `buffers[i]` is bound at
/// Metal buffer index `i` and holds what the kernel left there on return.
pub fn launch(
    library: &Path,
    entry: &str,
    threads: usize,
    buffers: &mut [Vec<u8>],
) -> Result<(), String> {
    let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
    let path = library.canonicalize().map_err(|error| error.to_string())?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    let library = device
        .newLibraryWithURL_error(&url)
        .map_err(|e| format!("library: {e}"))?;
    let function = library
        .newFunctionWithName(&NSString::from_str(entry))
        .ok_or("entry not in library")?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| format!("pipeline: {e}"))?;
    let queue = device.newCommandQueue().ok_or("no command queue")?;
    let command = queue.commandBuffer().ok_or("no command buffer")?;
    let encoder = command.computeCommandEncoder().ok_or("no encoder")?;
    encoder.setComputePipelineState(&pipeline);
    let mut shared = Vec::new();
    for (index, bytes) in buffers.iter().enumerate() {
        let start = NonNull::new(bytes.as_ptr().cast_mut().cast()).ok_or("empty buffer")?;
        // SAFETY: `start` is valid for `bytes.len()` bytes, which Metal copies.
        let buffer = unsafe {
            device.newBufferWithBytes_length_options(
                start,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or("buffer allocation failed")?;
        // SAFETY: the buffer outlives the command, and the offset is inside it.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer), 0, index) };
        shared.push(buffer);
    }
    let group = pipeline.maxTotalThreadsPerThreadgroup().min(threads.max(1));
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: group,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!("GPU command failed: {:?}", command.error()));
    }
    for (bytes, buffer) in buffers.iter_mut().zip(&shared) {
        // SAFETY: the GPU is done, and both sides are `bytes.len()` long.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.contents().as_ptr().cast::<u8>(),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
    }
    Ok(())
}

/// Launch a compiled kernel over its own `buffers`, binding after them what
/// every kernel receives: the buffers' lengths and the status word. Fails if
/// any thread panicked, and then the buffers hold nothing meaningful.
pub fn run(
    library: &Path,
    entry: &str,
    threads: usize,
    buffers: &mut Vec<Vec<u8>>,
) -> Result<(), String> {
    let own = buffers.len();
    let lengths = buffers
        .iter()
        .flat_map(|buffer| (buffer.len() as u64).to_le_bytes())
        .collect();
    buffers.extend([lengths, 0u32.to_le_bytes().to_vec()]);
    let result = launch(library, entry, threads, buffers);
    let status = buffers.split_off(own).pop();
    result?;
    match status.as_deref() {
        Some([0, 0, 0, 0]) => Ok(()),
        _ => Err("a thread panicked".into()),
    }
}
