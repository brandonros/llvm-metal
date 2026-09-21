//! Bulk ECDSA P-256 signing, written for the GPU's execution model: one thread
//! signs one message, and every thread runs the same instruction stream. Point
//! addition has one code path, k*G is 64 table additions with no doublings,
//! and both inversions are fixed ladders.
//!
//! TOY CODE. The host supplies the hashed message and the nonce `k`; real ECDSA
//! derives `k` (RFC 6979), and a repeated or biased `k` gives the key away. The
//! table read is indexed by secret data. Not audited for side channels.
#![cfg_attr(target_arch = "nvptx64", no_std)]

pub mod field;
mod modular;
pub mod point;

use field::{Field, L, Number};
use llvm_metal_kernel::{In, Out, Plain, Thread, kernel};
pub use modular::{modadd, montmul, reduce};
use point::Point;

/// Bits of `k` per table window, and the multiples of G each window holds.
pub const WINDOW: usize = 4;
pub const MULTIPLES: usize = 1 << WINDOW;
pub const WINDOWS: usize = 256 / WINDOW;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Key {
    /// The coordinate field.
    pub p: Field,
    /// The scalar field: integers modulo the order of G.
    pub n: Field,
    /// 3b, in Montgomery form modulo p.
    pub b3: Number,
    /// The private key, in Montgomery form modulo n.
    pub d: Number,
}

// SAFETY: `repr(C)` and `u32` throughout, so no padding.
unsafe impl Plain for Field {}
unsafe impl Plain for Key {}
unsafe impl Plain for Point {}
unsafe impl Plain for Request {}
unsafe impl Plain for Signature {}

/// What one thread signs.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Request {
    /// The hashed message.
    pub z: Number,
    /// The nonce.
    pub k: Number,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Signature {
    pub r: Number,
    pub s: Number,
}

kernel! {
    pub fn sign(
        thread: Thread,
        key: In<Key, 0>,
        table: In<[Point], 1>,
        requests: In<[Request], 2>,
        signatures: Out<[Signature], 3>,
    ) {
        let Some(key) = key.read() else { return };
        let index = thread.index() as usize;
        let Some(Request { z, k }) = requests.get(index) else { return };

        // k*G = sum of table[i][k_i], where k_i is the i-th 4-bit digit of k and
        // table[i][w] = w * 16^i * G. The digit picks an address, never a path.
        let mut sum = point::identity(&key.p);
        for window in 0..WINDOWS {
            let digit = (k[window / 8] >> (WINDOW * (window % 8))) as usize & (MULTIPLES - 1);
            let Some(multiple) = table.get(window * MULTIPLES + digit) else { return };
            sum = point::add(&sum, &multiple, &key.b3, &key.p);
        }
        signatures.write(index, finish(&sum, &z, &k, &key));
    }
}

/// `r = x(k*G) mod n` and `s = (z + r d) / k mod n`.
#[inline(always)]
fn finish(sum: &Point, z: &Number, k: &Number, key: &Key) -> Signature {
    let (p, n) = (&key.p, &key.n);
    let x = p.leave(&p.mul(&sum.x, &p.invert(&sum.z)));
    // x < p < 2n, so one subtraction reduces it.
    let r = reduce(&x, 0, &n.modulus);
    let rd = n.mul(&n.enter(&r), &key.d);
    let s = n.mul(&n.invert(&n.enter(k)), &n.add(&n.enter(z), &rd));
    Signature { r, s: n.leave(&s) }
}

const _: () = assert!(L * 32 == 256 && WINDOWS * WINDOW == 256);
