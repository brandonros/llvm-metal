//! Byte-buffer adapters for feature-gated production SHA diagnostics.
use vanity_logic::crypto::sha256::compiler_probes as sha;

unsafe fn words<const N: usize>(input: *const u8) -> [u32; N] {
    core::array::from_fn(|i| unsafe { input.add(i * 4).cast::<u32>().read_unaligned() })
}
unsafe fn write_words(output: *mut u8, words: &[u32]) {
    for (i, word) in words.iter().enumerate() {
        unsafe {
            output.add(i * 4).cast::<u32>().write_unaligned(*word);
        }
    }
}

/// # Safety
/// Disjoint initialized 12-byte input and writable 32-byte output, LE words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sha_ops(input: *const u8, output: *mut u8) {
    let [x, y, z] = unsafe { words(input) };
    unsafe {
        write_words(output, &sha::operations(x, y, z));
    }
}

/// # Safety
/// Disjoint initialized 16-byte input and writable 4-byte output, LE words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sha_schedule(input: *const u8, output: *mut u8) {
    unsafe {
        write_words(output, &[sha::schedule_word(&words(input))]);
    }
}

/// # Safety
/// Disjoint initialized 40-byte input and writable 32-byte output, LE words.
/// Input is state[8], schedule word, round constant.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sha_round(input: *const u8, output: *mut u8) {
    let state = unsafe { words(input) };
    let [word, constant] = unsafe { words(input.add(32)) };
    unsafe {
        write_words(output, &sha::round(&state, word, constant));
    }
}

/// # Safety
/// Disjoint initialized 96-byte input and writable 32-byte output, LE words.
/// Input is block[16], initial state[8].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sha_compress(input: *const u8, output: *mut u8) {
    let block = unsafe { words(input) };
    let mut state = unsafe { words(input.add(64)) };
    sha::compression(&block, &mut state);
    unsafe {
        write_words(output, &state);
    }
}
