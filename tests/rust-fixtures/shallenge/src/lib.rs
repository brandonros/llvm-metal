#![no_std]

use vanity_logic::{modes::shallenge, search::xoroshiro};

#[cfg(feature = "compiler-probes")]
mod probes;
#[cfg(feature = "compiler-probes")]
pub use probes::*;

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

/// # Safety
/// Disjoint readable 16-byte input (lane, seed: LE u64) and writable 21-byte output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shallenge_nonce(input: *const u8, output: *mut u8) {
    let input = unsafe { &*input.cast::<[u8; 16]>() };
    let lane = u64::from_le_bytes(input[..8].try_into().unwrap()) as usize;
    let seed = u64::from_le_bytes(input[8..].try_into().unwrap());
    let mut nonce = [0; 21];
    xoroshiro::generate_base64_nonce(lane, seed, &mut nonce);
    unsafe {
        core::ptr::copy_nonoverlapping(nonce.as_ptr(), output, 21);
    }
}

/// # Safety
/// Disjoint readable 64-byte input (two hashes) and writable 4-byte LE i32 output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shallenge_compare(input: *const u8, output: *mut u8) {
    let a = unsafe { &*input.cast::<[u8; 32]>() };
    let b = unsafe { &*input.add(32).cast::<[u8; 32]>() };
    let result = shallenge::compare_hashes(a, b).to_le_bytes();
    unsafe {
        core::ptr::copy_nonoverlapping(result.as_ptr(), output, 4);
    }
}

/// Single candidate: lane LE u64, seed LE u64, username length u8, 30 username
/// bytes, 32 target bytes. Output: hash[32], nonce[30], nonce length, is_better,
/// valid. Invalid username lengths produce 65 zero bytes.
///
/// # Safety
/// Disjoint readable 79-byte input and writable 65-byte output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shallenge_candidate(input: *const u8, output: *mut u8) {
    let input = unsafe { &*input.cast::<[u8; 79]>() };
    let mut bytes = [0; 65];
    let length = input[16] as usize;
    if (1..=30).contains(&length) {
        let result = shallenge::generate_and_check_shallenge(&shallenge::ShallengeRequest {
            thread_idx: u64::from_le_bytes(input[..8].try_into().unwrap()) as usize,
            rng_seed: u64::from_le_bytes(input[8..16].try_into().unwrap()),
            username: &input[17..47],
            username_len: length,
            target_hash: input[47..79].try_into().unwrap(),
        });
        bytes[..32].copy_from_slice(&result.hash);
        bytes[32..62].copy_from_slice(&result.nonce[..30]);
        bytes[62] = result.nonce_len as u8;
        bytes[63] = result.is_better as u8;
        bytes[64] = 1;
    }
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), output, 65);
    }
}

#[cfg(target_arch = "nvptx64")]
unsafe extern "C" {
    #[link_name = "llvm_metal.linear_thread_index"]
    fn thread_index() -> u32;
    #[link_name = "llvm_metal.atomic_add_device_u32"]
    fn atomic_add(pointer: *mut u32, value: u32) -> u32;
}

/// Indexed candidate batch. Input is count LE u32 followed by count requests.
/// Each lane writes its 65-byte result. Two zero-initialized atomic counters
/// count valid candidates and matches; matching lanes publish their indices in
/// unique winner slots. CPU readers must wait for command completion.
///
/// # Safety
/// Disjoint buffers: input >= 4+79*count, output >= 65*count, counters >= 8,
/// winners >= 4*count. Counters/winners are u32 aligned. Dispatch is 1D.
#[cfg(target_arch = "nvptx64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shallenge_batch(
    input: *const u8,
    output: *mut u8,
    counters: *mut u32,
    winners: *mut u32,
) {
    unsafe {
        let index = thread_index();
        let count = input.cast::<u32>().read_unaligned();
        if index >= count {
            return;
        }
        let result = output.add(index as usize * 65);
        shallenge_candidate(input.add(4 + index as usize * 79), result);
        if *result.add(64) != 0 {
            atomic_add(counters, 1);
            if *result.add(63) != 0 {
                let slot = atomic_add(counters.add(1), 1);
                winners.add(slot as usize).write(index);
            }
        }
    }
}
