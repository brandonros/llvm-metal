#![no_std]
use llvm_metal_kernel::{kernel, record};
record! { #[derive(Clone, Copy)] pub struct Request { pub count: u32, pub add: u32 } }
kernel! {
    pub mod contract;
    entry "descriptor_add";
    /// # Safety
    /// Disjoint buffers; data/output contain at least request.count u32 elements.
    pub unsafe extern "C" fn add_with_descriptor(
        request: Read Fixed Request,
        data: Read Slice u32,
        output: Write Slice u32,
    ) {
        let r = unsafe { request.read() };
        let mut i = 0;
        while i < r.count {
            unsafe { output.add(i as usize).write(data.add(i as usize).read().wrapping_add(r.add)); }
            i += 1;
        }
    } dispatch Single;
}
