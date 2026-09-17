#![no_std]

/// # Safety
/// Disjoint readable/writable 8-byte buffers, byte aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rotate64(input: *const u8, output: *mut u8) {
    let x = unsafe { input.cast::<u64>().read_unaligned() };
    unsafe {
        output
            .cast::<u64>()
            .write_unaligned(x.wrapping_mul(5).rotate_left(24));
    }
}

/// # Safety
/// Disjoint readable/writable one-byte buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lookup(input: *const u8, output: *mut u8) {
    let table = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    unsafe {
        *output = table[(*input & 63) as usize];
    }
}

/// # Safety
/// Disjoint readable/writable 21-byte buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy21(input: *const u8, output: *mut u8) {
    unsafe {
        core::ptr::copy_nonoverlapping(input, output, 21);
    }
}

/// # Safety
/// Disjoint readable one-byte input and writable 21-byte output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fill21(input: *const u8, output: *mut u8) {
    unsafe {
        core::ptr::write_bytes(output, *input, 21);
    }
}
