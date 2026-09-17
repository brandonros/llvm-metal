#![no_std]

/// Hash exactly 32 runtime bytes into a separate 32-byte output buffer.
///
/// # Safety
/// `input` must point to 32 initialized readable bytes and `output` to 32
/// writable bytes. The two regions must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shallenge_sha256_32(input: *const u8, output: *mut u8) {
    // SAFETY: the caller supplies valid, disjoint byte buffers of this size.
    let input = unsafe { &*input.cast::<[u8; 32]>() };
    let digest = vanity_logic::crypto::sha256::sha256_32_from_bytes(input);
    unsafe { core::ptr::copy_nonoverlapping(digest.as_ptr(), output, digest.len()) };
}
