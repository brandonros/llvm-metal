//! Bulk RSA-2048 signing, written for the GPU's execution model: one thread
//! signs one message, and every thread runs the same instruction stream. Loop
//! bounds are constants, the exponent ladder always multiplies, and a mask
//! selects where CPU code would branch.
//!
//! TOY CODE. A message is an already-padded 2048-bit integer. Not audited for
//! side channels. Do not point this at a key you care about.
#![cfg_attr(target_arch = "nvptx64", no_std)]

mod modular;

use llvm_metal_kernel::{In, Out, Plain, Thread, kernel};
pub use modular::{modadd, modsub, montmul};

/// 32-bit limbs in a 1024-bit CRT half. Limbs are 32 bits because the GPU has
/// no 128-bit product: the carry lives in the top half of a `u64`.
pub const HL: usize = 32;
/// 32-bit limbs in a 2048-bit value.
pub const FL: usize = 64;
/// Exponent bits consumed per ladder step.
const WINDOW: usize = 4;
const TBL: usize = 1 << WINDOW;

/// Everything one CRT half needs.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HalfKey {
    /// p or q.
    pub modulus: [u32; HL],
    /// dp or dq.
    pub exponent: [u32; HL],
    /// R mod m: the number 1 in Montgomery form.
    pub one: [u32; HL],
    /// R^2 mod m.
    pub r2: [u32; HL],
    /// R^3 mod m.
    pub r3: [u32; HL],
    /// -m^-1 mod 2^32.
    pub n0inv: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SignKey {
    pub p: HalfKey,
    pub q: HalfKey,
    /// q^-1 mod p, in Montgomery form.
    pub qinv_r: [u32; HL],
}

// SAFETY: `repr(C)` and `u32` throughout, so no padding.
unsafe impl Plain for HalfKey {}
unsafe impl Plain for SignKey {}

kernel! {
    pub fn sign(
        thread: Thread,
        key: In<SignKey, 0>,
        messages: In<[[u32; FL]], 1>,
        signatures: Out<[[u32; FL]], 2>,
    ) {
        let Some(key) = key.read() else { return };
        let index = thread.index() as usize;
        let Some(message) = messages.get(index) else { return };
        signatures.write(index, sign_one(&message, &key));
    }
}

/// `message ^ key.exponent mod key.modulus`, from the full 2048-bit message.
fn modexp_half(message: &[u32; FL], key: &HalfKey) -> [u32; HL] {
    let (m, n0inv) = (&key.modulus, key.n0inv);
    // Reduce 2048 -> 1024 bits and enter Montgomery form, with no division:
    //   message = lo + hi * R   =>   message * R = lo*R + hi*R^2   (mod m)
    let (mut lo, mut hi) = ([0; HL], [0; HL]);
    for j in 0..HL {
        lo[j] = message[j];
        hi[j] = message[HL + j];
    }
    let x = modadd(
        &montmul(&lo, &key.r2, m, n0inv),
        &montmul(&hi, &key.r3, m, n0inv),
        m,
    );

    // Window table: x^0 .. x^15, 2 KB of thread memory.
    let mut table = [key.one; TBL];
    table[1] = x;
    for i in 2..TBL {
        table[i] = montmul(&table[i - 1], &x, m, n0inv);
    }

    // Every step is 4 squarings and 1 multiply: a zero window multiplies by
    // `table[0]` (= 1) instead of being skipped. The window value picks an
    // address, never a code path.
    let mut acc = key.one;
    for limb in (0..HL).rev() {
        for step in 0..32 / WINDOW {
            for _ in 0..WINDOW {
                acc = montmul(&acc, &acc, m, n0inv);
            }
            let shift = 32 - WINDOW * (step + 1);
            let window = (key.exponent[limb] >> shift) as usize & (TBL - 1);
            acc = montmul(&acc, &table[window], m, n0inv);
        }
    }

    // Leave Montgomery form: multiply by plain 1.
    let mut plain_one = [0; HL];
    plain_one[0] = 1;
    montmul(&acc, &plain_one, m, n0inv)
}

/// One signature: two half-size exponentiations and Garner's recombination.
pub fn sign_one(message: &[u32; FL], key: &SignKey) -> [u32; FL] {
    let m1 = modexp_half(message, &key.p);
    let m2 = modexp_half(message, &key.q);

    // h = (m1 - m2) * qinv mod p. The key has p > q, so m2 < p already.
    let h = modsub(&m1, &m2, &key.p.modulus);
    let h = montmul(&h, &key.qinv_r, &key.p.modulus, key.p.n0inv);

    // signature = m2 + h * q, which is below n: no reduction.
    let mut out = [0u32; FL];
    for j in 0..HL {
        out[j] = m2[j];
    }
    for i in 0..HL {
        let mut carry = 0u64;
        let hi = h[i] as u64;
        for j in 0..HL {
            let s = out[i + j] as u64 + hi * key.q.modulus[j] as u64 + carry;
            out[i + j] = s as u32;
            carry = s >> 32;
        }
        // A fixed-length carry ripple.
        for j in i + HL..FL {
            let s = out[j] as u64 + carry;
            out[j] = s as u32;
            carry = s >> 32;
        }
    }
    out
}
