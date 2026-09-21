//! Shallenge: find a nonce whose SHA-256 of `prefix ‖ nonce` is small. Written
//! for the GPU: the message is always one block, which the host lays out once;
//! a thread tries many nonces and keeps its own best, so one record comes back
//! for many hashes; the nonce is a counter, turned into text by arithmetic.
#![cfg_attr(target_arch = "nvptx64", no_std)]

mod compress;

use llvm_metal_kernel::{In, Out, Plain, Thread, kernel};

/// Characters in a nonce, six bits of the counter and the salt each.
pub const NONCE: usize = 16;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Request {
    /// The first counter of thread 0. Thread `t` tries
    /// `base + t * attempts ..` for `attempts` counters.
    pub base: u64,
    /// The nonce's bits above the counter's 64.
    pub salt: u64,
    /// The padded block as big-endian words, its nonce bytes zero.
    pub block: [u32; 16],
    /// Where the nonce starts in the block, in bytes.
    pub offset: u32,
    pub attempts: u32,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Record {
    /// The smallest hash this thread found, as big-endian words.
    pub hash: [u32; 8],
    /// The counter that gave it.
    pub counter: u64,
}

// SAFETY: `repr(C)`; 88 and 40 bytes of integers, each a multiple of the
// alignment of 8, with no gaps between fields.
unsafe impl Plain for Request {}
unsafe impl Plain for Record {}

kernel! {
    pub fn shallenge(thread: Thread, request: In<Request, 0>, records: Out<[Record], 1>) {
        let Some(request) = request.read() else { return };
        let index = thread.index();
        records.write(index as usize, search(&request, index));
    }
}

/// Character `i` of the nonce: six bits of `salt ‖ counter`, as base64 text.
/// The alphabet is four runs, so the character is the value plus a step at each
/// run's start: no table, and no branch.
pub fn character(counter: u64, salt: u64, i: usize) -> u32 {
    let bit = 6 * i as u32;
    let low = counter.checked_shr(bit).unwrap_or(0);
    let high = if bit >= 64 {
        salt >> (bit - 64)
    } else {
        (salt << 1) << (63 - bit)
    };
    let v = (low | high) as u32 & 63;
    let step = |from: u32, by: u32| ((v >= from) as u32).wrapping_mul(by);
    (v + b'A' as u32)
        .wrapping_add(step(26, 6)) // 'a' - 'Z' - 1
        .wrapping_sub(step(52, 75)) // '0' - 'z' - 1
        .wrapping_sub(step(62, 15)) // '+' - '9' - 1
        .wrapping_add(step(63, 3)) // '/' - '+' - 1
}

/// The best of this thread's attempts. Every thread runs the same loop the same
/// number of times, and "is this hash smaller" selects by mask.
fn search(request: &Request, thread: u32) -> Record {
    let first = (request.base).wrapping_add((thread as u64).wrapping_mul(request.attempts as u64));
    let mut best = Record {
        hash: [u32::MAX; 8],
        counter: first,
    };
    for attempt in 0..request.attempts {
        let counter = first.wrapping_add(attempt as u64);
        let mut block = request.block;
        for i in 0..NONCE {
            let byte = request.offset as usize + i;
            block[(byte / 4) & 15] |= character(counter, request.salt, i) << (24 - 8 * (byte % 4));
        }
        let hash = compress::compress(&block);

        // smaller = hash < best.hash, most significant word first.
        let (mut smaller, mut equal) = (0u32, u32::MAX);
        for w in 0..8 {
            smaller |= equal & ((hash[w] < best.hash[w]) as u32).wrapping_neg();
            equal &= ((hash[w] == best.hash[w]) as u32).wrapping_neg();
        }
        for w in 0..8 {
            best.hash[w] = (hash[w] & smaller) | (best.hash[w] & !smaller);
        }
        let wide = smaller as i32 as i64 as u64;
        best.counter = (counter & wide) | (best.counter & !wide);
    }
    best
}
