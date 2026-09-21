//! Shallenge: find a nonce whose SHA-256 of `prefix ‖ nonce` is small. Each
//! thread tries one nonce and records its hash; the host keeps the best.
#![cfg_attr(target_arch = "nvptx64", no_std)]

mod sha256;
mod xoroshiro;

use llvm_metal_kernel::{In, Out, Plain, Thread, kernel};

pub const PREFIX: usize = 32;
pub const NONCE: usize = 16;
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Request {
    pub seed: u64,
    pub prefix_length: u64,
    pub prefix: [u8; PREFIX],
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Record {
    pub hash: [u8; 32],
    pub nonce: [u8; NONCE],
}

// SAFETY: `repr(C)`, integers and byte arrays, no padding.
unsafe impl Plain for Request {}
unsafe impl Plain for Record {}

kernel! {
    pub fn shallenge(thread: Thread, request: In<Request, 0>, records: Out<[Record], 1>) {
        let Some(request) = request.read() else { return };
        let index = thread.index();
        records.write(index as usize, attempt(&request, index));
    }
}

fn attempt(request: &Request, thread: u32) -> Record {
    let mut random = xoroshiro::Xoroshiro::seeded(request.seed.wrapping_add(thread.into()));
    let mut nonce = [0; NONCE];
    for byte in &mut nonce {
        *byte = ALPHABET[(random.next() % 64) as usize];
    }
    let length = (request.prefix_length as usize).min(PREFIX);
    let mut message = [0; PREFIX + NONCE];
    message[..length].copy_from_slice(&request.prefix[..length]);
    message[length..length + NONCE].copy_from_slice(&nonce);
    Record { hash: sha256::digest(&message[..length + NONCE]), nonce }
}
