#![no_std]
//! Runtime-input RSA arithmetic using the consumer's crypto-bigint release.
#[cfg(feature = "consumer")]
use crypto_bigint::{
    Encoding, NonZero, U256, U512, U1024, U2048,
    modular::runtime_mod::{DynResidue, DynResidueParams},
};
pub type Probe = unsafe extern "C" fn(*const u8, *mut u8);

/// # Safety
/// Disjoint readable input[64] and writable output[64], byte aligned.
/// Big integers are big endian; shift counts and bit lengths are little endian.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_mul256(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 64]>().read_unaligned() };
    let result: [u8; 64] = {
        let a = U256::from_be_slice(&bytes[..32]);
        let b = U256::from_be_slice(&bytes[32..]);
        let product: U512 = a.mul(&b);
        product.to_be_bytes()
    };
    unsafe { output.cast::<[u8; 64]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[256] and writable output[256], byte aligned.
/// Big integers are big endian; shift counts and bit lengths are little endian.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_mul1024(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 256]>().read_unaligned() };
    let result: [u8; 256] = {
        let a = U1024::from_be_slice(&bytes[..128]);
        let b = U1024::from_be_slice(&bytes[128..]);
        let product: U2048 = a.mul(&b);
        product.to_be_bytes()
    };
    unsafe { output.cast::<[u8; 256]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[256] and writable output[257], byte aligned.
/// Big integers are big endian; shift counts and bit lengths are little endian.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_add_sub1024(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 256]>().read_unaligned() };
    let result: [u8; 257] = {
        let a = U1024::from_be_slice(&bytes[..128]);
        let b = U1024::from_be_slice(&bytes[128..]);
        let mut result = [0; 257];
        result[0] = u8::from(a < b);
        result[1..129].copy_from_slice(&a.wrapping_add(&b).to_be_bytes());
        result[129..].copy_from_slice(&a.wrapping_sub(&b).to_be_bytes());
        result
    };
    unsafe { output.cast::<[u8; 257]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[132] and writable output[264], byte aligned.
/// Big integers are big endian; shift counts and bit lengths are little endian.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_shifts1024(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 132]>().read_unaligned() };
    let result: [u8; 264] = {
        let a = U1024::from_be_slice(&bytes[..128]);
        let shift = u32::from_le_bytes(bytes[128..].try_into().unwrap()) as usize;
        let mut result = [0; 264];
        result[..128].copy_from_slice(&a.shl_vartime(shift).to_be_bytes());
        result[128..256].copy_from_slice(&a.shr_vartime(shift).to_be_bytes());
        result[256..260].copy_from_slice(&(a.bits_vartime() as u32).to_le_bytes());
        result[260..].copy_from_slice(&(a.trailing_zeros() as u32).to_le_bytes());
        result
    };
    unsafe { output.cast::<[u8; 264]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[256] and writable output[257], byte aligned.
/// Big integers are big endian; shift counts and bit lengths are little endian.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_divrem1024(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 256]>().read_unaligned() };
    let result: [u8; 257] = {
        let a = U1024::from_be_slice(&bytes[..128]);
        let b = U1024::from_be_slice(&bytes[128..]);
        let mut result = [0; 257];
        if let Some(divisor) = Option::<NonZero<U1024>>::from(NonZero::new(b)) {
            let (q, r) = a.div_rem(&divisor);
            result[0] = 1;
            result[1..129].copy_from_slice(&q.to_be_bytes());
            result[129..].copy_from_slice(&r.to_be_bytes());
        }
        result
    };
    unsafe { output.cast::<[u8; 257]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[384] and writable output[385], byte aligned.
/// Big integers are big endian; shift counts and bit lengths are little endian.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_modular1024(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 384]>().read_unaligned() };
    let result: [u8; 385] = {
        let a = U1024::from_be_slice(&bytes[..128]);
        let b = U1024::from_be_slice(&bytes[128..256]);
        let n = U1024::from_be_slice(&bytes[256..]);
        let mut result = [0; 385];
        if n > U1024::ONE && bytes[383] & 1 == 1 {
            let params = DynResidueParams::new(&n);
            let a = DynResidue::new(&a, params);
            let b = DynResidue::new(&b, params);
            result[0] = 1;
            result[1..129].copy_from_slice(&(a * b).retrieve().to_be_bytes());
            result[129..257].copy_from_slice(&a.square().retrieve().to_be_bytes());
            result[257..].copy_from_slice(&a.retrieve().to_be_bytes());
        }
        result
    };
    unsafe { output.cast::<[u8; 385]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[384] and writable output[129], byte aligned.
/// Big integers are big endian; shift counts and bit lengths are little endian.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_pow32(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 384]>().read_unaligned() };
    let result: [u8; 129] = {
        let a = U1024::from_be_slice(&bytes[..128]);
        let e = U1024::from_be_slice(&bytes[128..256]);
        let n = U1024::from_be_slice(&bytes[256..]);
        let mut result = [0; 129];
        if n > U1024::ONE && bytes[383] & 1 == 1 {
            result[0] = 1;
            let value = DynResidue::new(&a, DynResidueParams::new(&n))
                .pow_bounded_exp(&e, 32)
                .retrieve();
            result[1..].copy_from_slice(&value.to_be_bytes());
        }
        result
    };
    unsafe { output.cast::<[u8; 129]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[384] and writable output[129], byte aligned.
/// Big integers are big endian; shift counts and bit lengths are little endian.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_pow1024(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 384]>().read_unaligned() };
    let result: [u8; 129] = {
        let a = U1024::from_be_slice(&bytes[..128]);
        let e = U1024::from_be_slice(&bytes[128..256]);
        let n = U1024::from_be_slice(&bytes[256..]);
        let mut result = [0; 129];
        if n > U1024::ONE && bytes[383] & 1 == 1 {
            result[0] = 1;
            let value = DynResidue::new(&a, DynResidueParams::new(&n))
                .pow(&e)
                .retrieve();
            result[1..].copy_from_slice(&value.to_be_bytes());
        }
        result
    };
    unsafe { output.cast::<[u8; 129]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable input[128] and writable output[1], byte aligned.
/// Big integers are big endian; shift counts and bit lengths are little endian.
#[cfg(feature = "consumer")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_prime1024(input: *const u8, output: *mut u8) {
    let bytes = unsafe { input.cast::<[u8; 128]>().read_unaligned() };
    let result: [u8; 1] = {
        [u8::from(vanity_logic::modes::rsa_modulus::probable_p(
            &bytes,
        ))]
    };
    unsafe { output.cast::<[u8; 1]>().write_unaligned(result) };
}

pub fn probe(entry: &str) -> Option<(usize, usize, Probe)> {
    match entry {
        #[cfg(feature = "consumer")]
        "consumer_rsa_mul256" => Some((64, 64, consumer_rsa_mul256)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_mul1024" => Some((256, 256, consumer_rsa_mul1024)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_add_sub1024" => Some((256, 257, consumer_rsa_add_sub1024)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_shifts1024" => Some((132, 264, consumer_rsa_shifts1024)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_divrem1024" => Some((256, 257, consumer_rsa_divrem1024)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_modular1024" => Some((384, 385, consumer_rsa_modular1024)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_pow32" => Some((384, 129, consumer_rsa_pow32)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_pow1024" => Some((384, 129, consumer_rsa_pow1024)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_prime1024" => Some((128, 1, consumer_rsa_prime1024)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_generate" => Some((1080, 129, search::consumer_rsa_generate)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_progression" => Some((1200, 257, search::consumer_rsa_progression)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_sample_q" => Some((1208, 129, search::consumer_rsa_sample_q)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_eligible_pair" => Some((772, 1, search::consumer_rsa_eligible_pair)),
        #[cfg(feature = "consumer")]
        "consumer_rsa_candidate" => Some((1596, 260, search::consumer_rsa_candidate)),
        _ => None,
    }
}

#[cfg(all(feature = "consumer", not(target_arch = "nvptx64")))]
pub mod corpus;
#[cfg(feature = "consumer")]
pub mod factors;

#[cfg(feature = "consumer")]
pub mod search;
#[cfg(all(feature = "consumer", not(target_arch = "nvptx64")))]
pub mod search_corpus;
