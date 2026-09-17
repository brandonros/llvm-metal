use vanity_logic::{
    crypto::{ripemd160, sha256},
    encoding::bech32,
};

/// # Safety
/// Disjoint readable input[32] and writable RIPEMD-160 digest[20].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_bitcoin_ripemd160(input: *const u8, output: *mut u8) {
    let input = unsafe { super::read::<32>(input) };
    let digest = ripemd160::ripemd160_32bytes_from_bytes(&input);
    unsafe { output.cast::<[u8; 20]>().write_unaligned(digest) };
}

/// # Safety
/// Disjoint readable compressed public key[33] and writable SHA-256[32] + HASH160[20].
/// Hashes all 33 bytes, without validating the SEC1 encoding.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_bitcoin_hash160(input: *const u8, output: *mut u8) {
    let input = unsafe { super::read::<33>(input) };
    let sha = sha256::sha256_from_bytes(&input);
    let hash = ripemd160::ripemd160_32bytes_from_bytes(&sha);
    unsafe {
        output.cast::<[u8; 32]>().write_unaligned(sha);
        output.add(32).cast::<[u8; 20]>().write_unaligned(hash);
    }
}

/// # Safety
/// Disjoint readable HASH160[20] and writable length[1] + zero-padded address[64].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_bitcoin_bech32(input: *const u8, output: *mut u8) {
    let input = unsafe { super::read::<20>(input) };
    let mut address = [0; 64];
    let len = bech32::encode_p2wpkh_address(&input, true, &mut address);
    unsafe {
        output.write(len as u8);
        output.add(1).cast::<[u8; 64]>().write_unaligned(address);
    }
}

/// # Safety
/// Disjoint key[32] and output[119]: validity, SEC1[33], HASH160[20], length,
/// zero-padded Bech32 address[64]. Invalid private keys produce all-zero output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_bitcoin_address(input: *const u8, output: *mut u8) {
    let key = unsafe { super::read::<32>(input) };
    let mut result = [0; 119];
    if let Some(key) = vanity_logic::modes::bitcoin::try_check_bitcoin_key(key, &[], &[]) {
        result[0] = 1;
        result[1..34].copy_from_slice(&key.public_key);
        result[34..54].copy_from_slice(&key.public_key_hash);
        result[54] = key.encoded_len as u8;
        result[55..].copy_from_slice(&key.encoded_public_key);
    }
    unsafe { output.cast::<[u8; 119]>().write_unaligned(result) };
}
