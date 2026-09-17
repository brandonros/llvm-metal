#![no_std]

unsafe extern "C" {
    #[link_name = "llvm_metal.linear_thread_index"]
    fn thread_index() -> u32;
    #[link_name = "llvm_metal.atomic_add_device_u32"]
    fn atomic_add(pointer: *mut u32, value: u32) -> u32;
}

/// # Safety
/// input: count u32; output: count u32 slots; counter: one aligned u32.
/// Buffers are disjoint. Device-scoped relaxed atomic add returns the old value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn atomic_tickets(input: *const u32, output: *mut u32, counter: *mut u32) {
    unsafe {
        let index = thread_index();
        if index < *input {
            let old = atomic_add(counter, 1);
            output.add(index as usize).write(old);
        }
    }
}
