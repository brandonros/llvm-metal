//! Run a compute kernel from a Metal library, synchronously, over buffers the
//! runtime owns. The host and the GPU take turns with a buffer's memory, and
//! the borrow checker keeps the turns: the host sees a buffer only inside
//! `Buffer::read` or `Buffer::write`, and a launch borrows every buffer it
//! binds mutably until the GPU is done. Nothing is copied on either side.
#![cfg(target_os = "macos")]
use llvm_metal_kernel::Plain;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;
use std::{marker::PhantomData, path::Path, time::Duration};

type Raw = ProtocolObject<dyn MTLBuffer>;

/// `len` values of `T` in memory the GPU can be given.
pub struct Buffer<T> {
    raw: Retained<Raw>,
    len: usize,
    values: PhantomData<T>,
}

impl<T: Plain> Buffer<T> {
    fn start(&self) -> *mut T {
        // Metal's memory is page-aligned, which is enough for any `T`.
        self.raw.contents().as_ptr().cast()
    }

    /// The host's turn, with the values to itself.
    pub fn write<R>(&mut self, turn: impl FnOnce(&mut [T]) -> R) -> R {
        // SAFETY: the buffer holds `len` values, any bits are a `Plain` value,
        // and `&mut self` says no launch and no other turn is using them.
        turn(unsafe { std::slice::from_raw_parts_mut(self.start(), self.len) })
    }

    /// The host's turn, only looking.
    pub fn read<R>(&self, turn: impl FnOnce(&[T]) -> R) -> R {
        // SAFETY: as in `write`; `&self` rules out a launch and a `write`.
        turn(unsafe { std::slice::from_raw_parts(self.start(), self.len) })
    }
}

mod sealed {
    pub trait Sealed {}
    impl<T> Sealed for super::Buffer<T> {}
}

/// A buffer of any element type, as a launch binds it.
pub trait Bound: sealed::Sealed {
    #[doc(hidden)]
    fn raw(&mut self) -> &Raw;
}

impl<T> Bound for Buffer<T> {
    fn raw(&mut self) -> &Raw {
        &self.raw
    }
}

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

    /// `len` zeroed values. Metal has no empty buffer, so `len` is not zero.
    pub fn buffer<T: Plain>(&self, len: usize) -> Result<Buffer<T>, String> {
        let bytes = len * size_of::<T>();
        let raw = (bytes > 0)
            .then(|| {
                self.device
                    .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            })
            .flatten()
            .ok_or(format!("cannot allocate a buffer of {bytes} bytes"))?;
        Ok(Buffer {
            raw,
            len,
            values: PhantomData,
        })
    }

    /// A buffer holding a copy of `values`.
    pub fn buffer_from<T: Plain>(&self, values: &[T]) -> Result<Buffer<T>, String> {
        let mut buffer = self.buffer(values.len())?;
        buffer.write(|slice| slice.copy_from_slice(values));
        Ok(buffer)
    }

    /// Launch the kernel once per thread, with `buffers[i]` bound at Metal
    /// buffer index `i`. Returns how long the GPU spent executing, by its own
    /// clock.
    pub fn launch(
        &self,
        threads: usize,
        buffers: &mut [&mut dyn Bound],
    ) -> Result<Duration, String> {
        let bound: Vec<&Raw> = buffers.iter_mut().map(|buffer| buffer.raw()).collect();
        self.dispatch(threads, &bound)
    }

    fn dispatch(&self, threads: usize, buffers: &[&Raw]) -> Result<Duration, String> {
        let command = self.queue.commandBuffer().ok_or("no command buffer")?;
        let encoder = command.computeCommandEncoder().ok_or("no encoder")?;
        encoder.setComputePipelineState(&self.state);
        for (index, buffer) in buffers.iter().enumerate() {
            // SAFETY: the caller's borrow outlives the command, which this
            // function waits for, and the offset is inside the buffer.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
        }
        // One SIMD group per threadgroup. Nothing here uses threadgroup memory, so a
        // larger group buys nothing, and the largest (1,024 on an M5) ran the RSA
        // example 2.6 times slower (issue #31).
        let group = self.state.threadExecutionWidth();
        let size = |width| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(size(threads), size(group));
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("GPU command failed: {:?}", command.error()));
        }
        Ok(Duration::from_secs_f64(
            command.GPUEndTime() - command.GPUStartTime(),
        ))
    }

    /// Launch a compiled kernel over the first `slots` of `buffers`: as many as the
    /// compiler reported it names, because what follows them is placed by count:
    /// the lengths of the buffers in bytes, and the status word. Fails if any
    /// thread panicked, and then the buffers hold nothing meaningful.
    pub fn run(
        &self,
        threads: usize,
        slots: usize,
        buffers: &mut [&mut dyn Bound],
    ) -> Result<Duration, String> {
        if buffers.len() < slots {
            return Err(format!(
                "the kernel names {slots} buffers; {} given",
                buffers.len()
            ));
        }
        let buffers = &mut buffers[..slots];
        let mut lengths = self.buffer::<u64>(buffers.len().max(1))?;
        lengths.write(|lengths| {
            for (length, buffer) in lengths.iter_mut().zip(buffers.iter_mut()) {
                *length = buffer.raw().length() as u64;
            }
        });
        let status = self.buffer::<u32>(1)?;
        let mut bound: Vec<&Raw> = buffers.iter_mut().map(|buffer| buffer.raw()).collect();
        bound.extend([&*lengths.raw, &*status.raw]);
        let elapsed = self.dispatch(threads, &bound)?;
        match status.read(|status| status[0]) {
            0 => Ok(elapsed),
            _ => Err("a thread panicked".into()),
        }
    }
}
