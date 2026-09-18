use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;
use std::{
    path::Path,
    time::{Duration, Instant},
};

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

pub struct Buffer {
    pub bytes: Vec<u8>,
    pub offset: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LoadTimings {
    pub library: Duration,
    pub pipeline: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchTimings {
    pub upload: Duration,
    pub download: Duration,
    pub wall: Duration,
    /// Metal command-buffer timestamps, absent if the device did not supply them.
    pub gpu: Option<Duration>,
}

pub struct PreparedKernel {
    kernel: Kernel,
    resources: Resources,
    shapes: Vec<(usize, usize)>,
}

// Shared buffers may contain secrets. A resource owner clears them before release,
// including partially completed/erroring dispatches and replacement allocations.
struct Resources(Vec<Retained<ProtocolObject<dyn MTLBuffer>>>);
impl Resources {
    fn clear(&mut self) {
        for buffer in &self.0 {
            // SAFETY: shared storage, exclusive owner, all dispatches are synchronous.
            unsafe {
                let pointer = buffer.contents().as_ptr().cast::<u8>();
                for offset in 0..buffer.length() {
                    pointer.add(offset).write_volatile(0);
                }
            }
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
}
impl std::ops::Deref for Resources {
    type Target = [Retained<ProtocolObject<dyn MTLBuffer>>];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for Resources {
    fn drop(&mut self) {
        self.clear();
    }
}

pub struct Kernel {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    bindings: llvm_metal_abi::MetalBindings,
    load_timings: LoadTimings,
}

impl Kernel {
    pub fn load(path: &Path, bindings: &llvm_metal_abi::MetalBindings) -> Result<Self, String> {
        if bindings.buffers.iter().enumerate().any(|(i, b)| {
            b.index != i
                || b.argument != i
                || b.minimum_bytes == 0
                || !b.alignment.is_power_of_two()
                || b.alignment > 16
        }) {
            return Err("invalid Metal buffer bindings".into());
        }
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        let path = path.to_str().ok_or("library path is not UTF-8")?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(path));
        let start = Instant::now();
        let library = device
            .newLibraryWithURL_error(&url)
            .map_err(|e| format!("library load: {e}"))?;
        let library_time = start.elapsed();
        let start = Instant::now();
        let function = library
            .newFunctionWithName(&NSString::from_str(&bindings.entry))
            .ok_or("entry not found")?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| format!("pipeline creation: {e}"))?;
        let pipeline_time = start.elapsed();
        let queue = device
            .newCommandQueue()
            .ok_or("command queue allocation failed")?;
        Ok(Self {
            device,
            queue,
            pipeline,
            bindings: bindings.clone(),
            load_timings: LoadTimings {
                library: library_time,
                pipeline: pipeline_time,
            },
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn load_timings(&self) -> LoadTimings {
        self.load_timings
    }

    /// Allocate device buffers once. Subsequent runs must preserve lengths/offsets.
    pub fn prepare(self, buffers: &[Buffer]) -> Result<PreparedKernel, String> {
        self.validate(buffers, 1, 1)?;
        let resources = self.allocate(buffers)?;
        let shapes = buffers.iter().map(|b| (b.bytes.len(), b.offset)).collect();
        Ok(PreparedKernel {
            kernel: self,
            resources,
            shapes,
        })
    }

    /// Dispatch a 1D grid synchronously. Buffer order is the Metal binding index.
    ///
    /// # Safety
    /// The shader must be trusted and its argument contract must match these
    /// buffers, offsets and grid. All shader accesses must be in bounds and free
    /// of data races. Buffer contents must satisfy the shader's alignment/type rules.
    pub unsafe fn run(
        &self,
        buffers: &mut [Buffer],
        threads: usize,
        group_size: usize,
    ) -> Result<(), String> {
        self.validate(buffers, threads, group_size)?;
        let resources = self.allocate(buffers)?;
        // SAFETY: forwarded caller contract, validated buffers, matching resources.
        unsafe {
            self.execute(&resources, buffers, threads, group_size)
                .map(|_| ())
        }
    }

    fn validate(
        &self,
        buffers: &[Buffer],
        threads: usize,
        group_size: usize,
    ) -> Result<(), String> {
        if threads > u32::MAX as usize
            || (self.bindings.dispatch == llvm_metal_abi::Dispatch::Single && threads != 1)
        {
            return Err("dispatch does not satisfy the kernel invocation contract".into());
        }
        if group_size == 0 || group_size > self.pipeline.maxTotalThreadsPerThreadgroup() {
            return Err("invalid threadgroup size".into());
        }
        if buffers.len() > 31
            || buffers
                .iter()
                .any(|b| b.bytes.is_empty() || b.offset >= b.bytes.len())
        {
            return Err("invalid buffer count, length or offset".into());
        }
        if buffers.len() != self.bindings.buffers.len()
            || buffers.iter().zip(&self.bindings.buffers).any(|(b, spec)| {
                b.offset % spec.alignment != 0 || b.bytes.len() - b.offset < spec.minimum_bytes
            })
        {
            return Err("buffers do not satisfy generated Metal bindings".into());
        }
        Ok(())
    }

    fn allocate(&self, buffers: &[Buffer]) -> Result<Resources, String> {
        let mut resources = Resources(Vec::with_capacity(buffers.len()));
        for buffer in buffers.iter() {
            let resource = self
                .device
                .newBufferWithLength_options(
                    buffer.bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("buffer allocation failed")?;
            resources.0.push(resource);
        }
        Ok(resources)
    }

    unsafe fn execute(
        &self,
        resources: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        buffers: &mut [Buffer],
        threads: usize,
        group_size: usize,
    ) -> Result<DispatchTimings, String> {
        let start = Instant::now();
        if threads == 0 {
            return Ok(DispatchTimings::default());
        }
        for (resource, buffer) in resources.iter().zip(buffers.iter()) {
            // SAFETY: exact-size resources and CPU buffers, no in-flight commands.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.bytes.as_ptr(),
                    resource.contents().as_ptr().cast(),
                    buffer.bytes.len(),
                );
            }
        }
        let upload = start.elapsed();
        let command = self
            .queue
            .commandBuffer()
            .ok_or("command buffer allocation failed")?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or("encoder allocation failed")?;
        encoder.setComputePipelineState(&self.pipeline);
        for (index, (resource, buffer)) in resources.iter().zip(buffers.iter()).enumerate() {
            // SAFETY: resources live through completion, offsets checked above;
            // the caller guarantees the shader's access contract.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(resource), buffer.offset, index);
            }
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group_size,
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
        let gpu_seconds = command.GPUEndTime() - command.GPUStartTime();
        let gpu = (gpu_seconds.is_finite() && gpu_seconds > 0.)
            .then(|| Duration::from_secs_f64(gpu_seconds));
        let download_start = Instant::now();
        for (resource, buffer) in resources.iter().zip(buffers.iter_mut()) {
            // SAFETY: GPU work completed; CPU exclusively owns the destination.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    resource.contents().as_ptr().cast(),
                    buffer.bytes.as_mut_ptr(),
                    buffer.bytes.len(),
                );
            }
        }
        Ok(DispatchTimings {
            upload,
            download: download_start.elapsed(),
            wall: start.elapsed(),
            gpu,
        })
    }
}

impl PreparedKernel {
    /// Clear retained shared device storage after synchronous use. Host mirrors
    /// belong to the caller and must be cleared separately when they hold secrets.
    pub fn clear(&mut self) {
        self.resources.clear();
    }

    /// Replace allocations only when lengths/offsets change, retaining the pipeline.
    /// Validation/allocation failure leaves the previous configuration usable.
    /// Returns whether new storage was allocated. Replaced storage is cleared.
    pub fn reconfigure(&mut self, buffers: &[Buffer]) -> Result<bool, String> {
        self.kernel.validate(buffers, 1, 1)?;
        if buffers
            .iter()
            .map(|b| (b.bytes.len(), b.offset))
            .eq(self.shapes.iter().copied())
        {
            return Ok(false);
        }
        let shapes: Vec<_> = buffers.iter().map(|b| (b.bytes.len(), b.offset)).collect();
        let resources = self.kernel.allocate(buffers)?;
        self.resources = resources;
        self.shapes = shapes;
        Ok(true)
    }

    pub fn device_name(&self) -> String {
        self.kernel.device_name()
    }

    /// Reuse allocated shared buffers for a synchronous dispatch.
    ///
    /// # Safety
    /// The same trusted-shader, bounds, types, aliasing and race requirements as
    /// `Kernel::run` apply. CPU storage lengths and offsets must match preparation.
    pub unsafe fn run(
        &mut self,
        buffers: &mut [Buffer],
        threads: usize,
        group_size: usize,
    ) -> Result<DispatchTimings, String> {
        self.kernel.validate(buffers, threads, group_size)?;
        if buffers
            .iter()
            .map(|b| (b.bytes.len(), b.offset))
            .ne(self.shapes.iter().copied())
        {
            return Err("prepared buffer lengths and offsets must stay fixed".into());
        }
        // SAFETY: matching retained resources, exclusive synchronous access,
        // and the caller guarantees the shader's dynamic memory contract.
        unsafe {
            self.kernel
                .execute(&self.resources, buffers, threads, group_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn shared_resources_clear_retained_storage_on_drop() {
        let device = MTLCreateSystemDefaultDevice().unwrap();
        let buffer = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();
        // Hold a second reference so we can inspect storage after owner destruction.
        unsafe {
            buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .write_bytes(0x5a, 4096);
        }
        {
            let _owner = Resources(vec![buffer.clone()]);
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), 4096) };
        assert!(bytes.iter().all(|&b| b == 0));
    }
}
