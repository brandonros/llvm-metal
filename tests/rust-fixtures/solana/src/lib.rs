#![no_std]
//! Runtime-input Solana stages using the consumer's dalek release and crypto.
#[cfg(feature = "consumer")]
use curve25519_dalek::Scalar;
pub type Probe = unsafe extern "C" fn(*const u8, *mut u8);

/// # Safety
/// Disjoint readable input[32] and writable output[32], byte aligned.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_solana_clamp(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 32]>().read_unaligned() };
    let result: [u8; 32] = curve25519_dalek::scalar::clamp_integer(bytes);
    unsafe { output.cast::<[u8; 32]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[32] and writable output[32], byte aligned.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_solana_scalar_reduce(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 32]>().read_unaligned() };
    let result: [u8; 32] = Scalar::from_bytes_mod_order(bytes).to_bytes();
    unsafe { output.cast::<[u8; 32]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[64] and writable output[32], byte aligned.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_solana_scalar_wide(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 64]>().read_unaligned() };
    let result: [u8; 32] = Scalar::from_bytes_mod_order_wide(&bytes).to_bytes();
    unsafe { output.cast::<[u8; 32]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[64] and writable output[32], byte aligned.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_solana_scalar_product(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 64]>().read_unaligned() };
    let result: [u8; 32] = (Scalar::from_bytes_mod_order(bytes[..32].try_into().unwrap())
        * Scalar::from_bytes_mod_order(bytes[32..].try_into().unwrap()))
    .to_bytes();
    unsafe { output.cast::<[u8; 32]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[32] and writable output[33], byte aligned.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_solana_point_double(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 32]>().read_unaligned() };
    let result: [u8; 33] = {
        let mut result = [0; 33];
        if let Some(point) = curve25519_dalek::edwards::CompressedEdwardsY(bytes).decompress() {
            result[0] = 1;
            result[1..].copy_from_slice(&(point + point).compress().to_bytes());
        }
        result
    };
    unsafe { output.cast::<[u8; 33]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[32] and writable output[32], byte aligned.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_solana_base_mul(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 32]>().read_unaligned() };
    let result: [u8; 32] = (curve25519_dalek::constants::ED25519_BASEPOINT_TABLE
        * &Scalar::from_bytes_mod_order(bytes))
        .compress()
        .to_bytes();
    unsafe { output.cast::<[u8; 32]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[32] and writable output[64], byte aligned.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_solana_sha512(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 32]>().read_unaligned() };
    let result: [u8; 64] = vanity_logic::crypto::sha512::sha512_32bytes_from_bytes(&bytes);
    unsafe { output.cast::<[u8; 64]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[32] and writable output[32], byte aligned.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_solana_public_key(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 32]>().read_unaligned() };
    let result: [u8; 32] = vanity_logic::crypto::ed25519::ed25519_derive_public_key(
        &vanity_logic::crypto::sha512::sha512_32bytes_from_bytes(&bytes),
    );
    unsafe { output.cast::<[u8; 32]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[32] and writable output[65], byte aligned.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_solana_base58(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 32]>().read_unaligned() };
    let result: [u8; 65] = {
        let mut result = [0; 65];
        let mut encoded = [0; 64];
        result[0] = vanity_logic::encoding::base58::base58_encode_32(&bytes, &mut encoded) as u8;
        result[1..].copy_from_slice(&encoded);
        result
    };
    unsafe { output.cast::<[u8; 65]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[32] and writable output[97], byte aligned.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_solana_address(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 32]>().read_unaligned() };
    let result: [u8; 97] = {
        let public = vanity_logic::crypto::ed25519::ed25519_derive_public_key(
            &vanity_logic::crypto::sha512::sha512_32bytes_from_bytes(&bytes),
        );
        let mut encoded = [0; 64];
        let length = vanity_logic::encoding::base58::base58_encode_32(&public, &mut encoded);
        let mut result = [0; 97];
        result[..32].copy_from_slice(&public);
        result[32] = length as u8;
        result[33..].copy_from_slice(&encoded);
        result
    };
    unsafe { output.cast::<[u8; 97]>().write_unaligned(result) };
}

pub fn probe(entry: &str) -> Option<(usize, usize, Probe)> {
    match entry {
        #[cfg(feature = "consumer")]
        "consumer_solana_clamp" => Some((32, 32, consumer_solana_clamp)),
        #[cfg(feature = "consumer")]
        "consumer_solana_scalar_reduce" => Some((32, 32, consumer_solana_scalar_reduce)),
        #[cfg(feature = "consumer")]
        "consumer_solana_scalar_wide" => Some((64, 32, consumer_solana_scalar_wide)),
        #[cfg(feature = "consumer")]
        "consumer_solana_scalar_product" => Some((64, 32, consumer_solana_scalar_product)),
        #[cfg(feature = "consumer")]
        "consumer_solana_point_double" => Some((32, 33, consumer_solana_point_double)),
        #[cfg(feature = "consumer")]
        "consumer_solana_base_mul" => Some((32, 32, consumer_solana_base_mul)),
        #[cfg(feature = "consumer")]
        "consumer_solana_sha512" => Some((32, 64, consumer_solana_sha512)),
        #[cfg(feature = "consumer")]
        "consumer_solana_public_key" => Some((32, 32, consumer_solana_public_key)),
        #[cfg(feature = "consumer")]
        "consumer_solana_base58" => Some((32, 65, consumer_solana_base58)),
        #[cfg(feature = "consumer")]
        "consumer_solana_address" => Some((32, 97, consumer_solana_address)),
        _ => None,
    }
}
