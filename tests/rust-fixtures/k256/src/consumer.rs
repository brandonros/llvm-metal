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

/// # Safety
/// Disjoint readable input[64] and writable Keccak-256 digest[32].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_keccak256(input: *const u8, output: *mut u8) {
    let input = unsafe { super::read::<64>(input) };
    let digest = vanity_logic::crypto::keccak256::keccak256_64bytes(&input);
    unsafe { output.cast::<[u8; 32]>().write_unaligned(digest) };
}

/// # Safety
/// Disjoint readable private key[32] and writable validity + public XY[64] +
/// Ethereum address[20]. Invalid keys produce 85 zero bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_ethereum_address(input: *const u8, output: *mut u8) {
    let key = unsafe { super::read::<32>(input) };
    let mut result = [0u8; 85];
    if let Some((public, address)) =
        vanity_logic::modes::ethereum::try_derive_ethereum_address(&key)
    {
        result[0] = 1;
        result[1..65].copy_from_slice(&public);
        result[65..].copy_from_slice(&address);
    }
    unsafe { output.cast::<[u8; 85]>().write_unaligned(result) };
}

/// # Safety
/// Input: LE count[4], count private keys[32]. Output: count records[85].
/// Dispatch a 1D grid; each invocation exclusively writes its own record.
#[cfg(target_arch = "nvptx64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_ethereum_address_batch(input: *const u8, output: *mut u8) {
    let lane = unsafe { linear_thread_index() } as usize;
    let count = u32::from_le_bytes(unsafe { super::read::<4>(input) }) as usize;
    if lane < count {
        unsafe {
            consumer_ethereum_address(input.add(4 + 32 * lane), output.add(85 * lane));
        }
    }
}
