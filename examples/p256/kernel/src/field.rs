//! Arithmetic modulo one 256-bit prime, in Montgomery form.
use crate::modular::{modadd, modsub, montmul};

/// 32-bit limbs in a 256-bit value.
pub const L: usize = 8;
pub type Number = [u32; L];

const WINDOW: usize = 4;
const TBL: usize = 1 << WINDOW;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Field {
    pub modulus: Number,
    /// R mod m: the number 1 in Montgomery form.
    pub one: Number,
    /// R^2 mod m.
    pub r2: Number,
    /// m - 2: by Fermat, `x^(m-2)` is `1/x`.
    pub inverse_exponent: Number,
    /// -m^-1 mod 2^32.
    pub n0inv: u32,
}

impl Field {
    pub fn mul(&self, a: &Number, b: &Number) -> Number {
        montmul(a, b, &self.modulus, self.n0inv)
    }

    pub fn add(&self, a: &Number, b: &Number) -> Number {
        modadd(a, b, &self.modulus)
    }

    pub fn sub(&self, a: &Number, b: &Number) -> Number {
        modsub(a, b, &self.modulus)
    }

    pub fn triple(&self, a: &Number) -> Number {
        self.add(&self.add(a, a), a)
    }

    /// Into Montgomery form. `a` may be any 256-bit value.
    pub fn enter(&self, a: &Number) -> Number {
        self.mul(a, &self.r2)
    }

    /// Out of Montgomery form: multiply by plain 1.
    pub fn leave(&self, a: &Number) -> Number {
        let mut plain_one = [0; L];
        plain_one[0] = 1;
        self.mul(a, &plain_one)
    }

    /// `1/a`. The exponent is the same on every thread and so is the path: 4
    /// squarings and 1 multiply per window, and a zero window multiplies by 1.
    pub fn invert(&self, a: &Number) -> Number {
        let mut table = [self.one; TBL];
        table[1] = *a;
        for i in 2..TBL {
            table[i] = self.mul(&table[i - 1], a);
        }
        let mut acc = self.one;
        for limb in (0..L).rev() {
            for step in 0..32 / WINDOW {
                for _ in 0..WINDOW {
                    acc = self.mul(&acc, &acc);
                }
                let shift = 32 - WINDOW * (step + 1);
                let window = (self.inverse_exponent[limb] >> shift) as usize & (TBL - 1);
                acc = self.mul(&acc, &table[window]);
            }
        }
        acc
    }
}
