#![no_std]

/// # Safety
/// Readable/writable disjoint, aligned one-u32 buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn buffer_add42(input: *const u32, output: *mut u32) {
    unsafe {
        *output = (*input).wrapping_add(42);
    }
}
