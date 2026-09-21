//! Run a compute kernel from a Metal library, synchronously, over byte buffers.
#![cfg(target_os = "macos")]
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;
use std::{path::Path, ptr::NonNull, time::Duration};

/// A kernel ready to launch. Loading the library and building the pipeline is
/// the slow part, and happens once, here.
pub struct Pipeline {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl Pipeline {
    pub fn load(library: &Path, entry: &str) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let path = library.canonicalize().map_err(|error| error.to_string())?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        let library = device
            .newLibraryWithURL_error(&url)
            .map_err(|e| format!("library: {e}"))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or("entry not in library")?;
        let state = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| format!("pipeline: {e}"))?;
        let queue = device.newCommandQueue().ok_or("no command queue")?;
        Ok(Pipeline {
            device,
            state,
            queue,
        })
    }

    /// Launch the kernel once per thread. `buffers[i]` is bound at Metal buffer
    /// index `i` and holds what the kernel left there on return. Returns how
    /// long the GPU spent executing, by its own clock.
    pub fn launch(&self, threads: usize, buffers: &mut [Vec<u8>]) -> Result<Duration, String> {
        let command = self.queue.commandBuffer().ok_or("no command buffer")?;
        let encoder = command.computeCommandEncoder().ok_or("no encoder")?;
        encoder.setComputePipelineState(&self.state);
        let mut shared = Vec::new();
        for (index, bytes) in buffers.iter().enumerate() {
            let start = NonNull::new(bytes.as_ptr().cast_mut().cast()).ok_or("empty buffer")?;
            // SAFETY: `start` is valid for `bytes.len()` bytes, which Metal copies.
            let buffer = unsafe {
                self.device.newBufferWithBytes_length_options(
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
        // One SIMD group per threadgroup. Nothing here uses threadgroup memory, so a
        // larger group buys nothing, and the largest (1,024 on an M5) ran the RSA
        // example 2.6 times slower (issue #31).
        let group = self.state.threadExecutionWidth();
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
        Ok(Duration::from_secs_f64(
            command.GPUEndTime() - command.GPUStartTime(),
        ))
    }

    /// Launch a compiled kernel over its own `buffers`. `slots` is how many the
    /// kernel names (the compiler reports it); after those come what every
    /// kernel receives, the lengths of its buffers and the status word. Fails if
    /// any thread panicked, and then the buffers hold nothing meaningful.
    pub fn run(
        &self,
        threads: usize,
        slots: usize,
        buffers: &mut [Vec<u8>],
    ) -> Result<Duration, String> {
        if buffers.len() < slots {
            return Err(format!(
                "the kernel names {slots} buffers; {} given",
                buffers.len()
            ));
        }
        let mut bound = buffers[..slots].to_vec();
        let lengths = bound
            .iter()
            .flat_map(|buffer| (buffer.len() as u64).to_le_bytes())
            .collect();
        bound.extend([lengths, 0u32.to_le_bytes().to_vec()]);
        let elapsed = self.launch(threads, &mut bound)?;
        if bound[slots + 1] != [0; 4] {
            return Err("a thread panicked".into());
        }
        for (buffer, result) in buffers.iter_mut().zip(bound.into_iter().take(slots)) {
            *buffer = result;
        }
        Ok(elapsed)
    }
}
