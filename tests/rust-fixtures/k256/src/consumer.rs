use vanity_logic::crypto::secp256k1::{
    try_secp256k1_derive_public_key, try_secp256k1_derive_public_key_uncompressed,
};

/// # Safety
/// Disjoint readable private key[32] and writable validity + SEC1[33] + SEC1[65].
/// Calls the production checked APIs; invalid inputs produce 99 zero bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_public_keys(input: *const u8, output: *mut u8) {
    let key = unsafe { super::read::<32>(input) };
    let mut result = [0u8; 99];
    if let (Some(compressed), Some(uncompressed)) = (
        try_secp256k1_derive_public_key(&key),
        try_secp256k1_derive_public_key_uncompressed(&key),
    ) {
        result[0] = 1;
        result[1..34].copy_from_slice(&compressed);
        result[34..].copy_from_slice(&uncompressed);
    }
    unsafe {
        output.cast::<[u8; 99]>().write_unaligned(result);
    }
}

#[cfg(target_arch = "nvptx64")]
unsafe extern "C" {
    #[link_name = "llvm_metal.linear_thread_index"]
    fn linear_thread_index() -> u32;
}

/// # Safety
/// Input: LE count[4], count private keys[32]. Output: count records[99].
/// Dispatch a 1D grid; each invocation exclusively writes its own record.
#[cfg(target_arch = "nvptx64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_public_keys_batch(input: *const u8, output: *mut u8) {
    let lane = unsafe { linear_thread_index() } as usize;
    let count = u32::from_le_bytes(unsafe { super::read::<4>(input) }) as usize;
    if lane < count {
        unsafe {
            consumer_public_keys(input.add(4 + 32 * lane), output.add(99 * lane));
        }
    }
}
