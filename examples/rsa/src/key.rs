//! A toy key, generated once with `openssl genrsa 2048`, and the Montgomery
//! constants the kernel needs, derived here with the kernel's own arithmetic.
use rsa_kernel::{FL, HL, HalfKey, SignKey, modadd, montmul};

const P: &str = "faceea173a5de3929c127a39d48be075a752651742444b541a3dab00bd34ca779ccb093e2761124febf5aa9a166a9aee2ff917adc19907a6cd0ac4a49b063753e586cac3f41c9726aa38cee7d6683e51182ac74914193f7cf7d28a2f9a301034eaffd1a252e899c4f03b17928fc58ed513ffe9ae333c17f6969795d16619ca09";
const Q: &str = "deed782fbfce71c956311871cb474260c45668de8704e93a5da22f288e7e8e294b4bd409579d4645541fdd7142e9d2dab464307f432a347ab61c4a50bc03937eb1694d9b7ec17da60b5fe76159267db9badebd40d9d403f7e87c96a4a38c93d780f45bdba7753ea7d868c43fed5e2202bf29460c92f4ae4ead477203d83ef19b";
const DP: &str = "11df38b31b07a1b5cac54e4c5ca6f301af40a1cf7c7b5d5acadbe6199161f7a37a5ac577d65a86718780e3fd42e7a9ce9b4086bd6cf438a55c2b0e44247fd6e5758f9b574747da45790fbf3ea9fa97a633b0a8aebe6de626438a8f2a413477932dc3b8ee7635f8ef1da73850cb49ea99a8692dffa9caf8722bdcf5620c827df1";
const DQ: &str = "aa63ffb51f79ffe6d0067e949bb73fb90ed8ad1749442baffcd9760a1dc00590f28866ed2d167d1b888d4288cb88452dfd2b8715fe9447c073697433f941127f87c2e11ab4ebd7ca0e6fa33ef9113e8fb391843e0940d037b06f6a05352cc1e3ba210c04fc1dd5621d3b16a5761cb90a386aa7abfcb72073ea65ed739ad9a179";
const N: &str = "da68164ea1c0f486d574b1b5e49e7fa57bc95de7889e4b8aa0e783c84e348ff9971133b4f59b8124e07f5dfe19899b472c1f7fa4b8a4eb2d2841103bc59dfaa0eac31a49e30daebbbe44035c3cb46f0303b314f1b6dc198d0c1a18020c3602512285e7d484474178526088ec91203801ab296bd40d7e05a2c06737608ca754db0b479e2c3996c363b3382fd5e40e680d6cacc89828b6770cf621d024c352ef395581027ab153134e958e3f3afe37cb32a58ac99052c1e0363b15226b7708a4be4fc066d7a2f7efc2b6281ade0b7ad3ff485557a39a90b642df7a1c41132a1713627ce0ee86786bd632c23b4c893082c94a9bd0231e0c7189a472b3b89efdcc73";

/// Little-endian limbs of a big-endian hex number of exactly `8 * L` digits.
fn limbs<const L: usize>(hex: &str) -> [u32; L] {
    assert_eq!(hex.len(), 8 * L);
    std::array::from_fn(|j| {
        let end = hex.len() - 8 * j;
        u32::from_str_radix(&hex[end - 8..end], 16).expect("hex")
    })
}

/// `-m^-1 mod 2^32` for odd `m`: Newton's iteration doubles the correct bits.
fn n0inv(m: u32) -> u32 {
    let mut x = 1u32;
    for _ in 0..5 {
        x = x.wrapping_mul(2u32.wrapping_sub(m.wrapping_mul(x)));
    }
    x.wrapping_neg()
}

/// `2^bits mod m`, by doubling.
fn power_of_two<const L: usize>(bits: usize, m: &[u32; L]) -> [u32; L] {
    let mut x = [0; L];
    x[0] = 1;
    for _ in 0..bits {
        x = modadd(&x, &x, m);
    }
    x
}

fn half(modulus: &str, exponent: &str) -> HalfKey {
    let modulus = limbs(modulus);
    HalfKey {
        modulus,
        exponent: limbs(exponent),
        one: power_of_two(32 * HL, &modulus),
        r2: power_of_two(64 * HL, &modulus),
        r3: power_of_two(96 * HL, &modulus),
        n0inv: n0inv(modulus[0]),
    }
}

/// The signing key. `p > q`, as Garner's step in the kernel requires.
pub fn sign_key() -> SignKey {
    let (p, q) = (half(P, DP), half(Q, DQ));
    assert!(p.modulus.iter().rev().gt(q.modulus.iter().rev()));
    // q^-1 = q^(p-2) mod p, square-and-multiply in Montgomery form; the result
    // stays in that form. p ends in ...09, so p - 2 does not borrow.
    let mut exponent = p.modulus;
    exponent[0] -= 2;
    let base = montmul(&q.modulus, &p.r2, &p.modulus, p.n0inv);
    let mut qinv_r = p.one;
    for bit in (0..32 * HL).rev() {
        qinv_r = montmul(&qinv_r, &qinv_r, &p.modulus, p.n0inv);
        if exponent[bit / 32] >> (bit % 32) & 1 == 1 {
            qinv_r = montmul(&qinv_r, &base, &p.modulus, p.n0inv);
        }
    }
    SignKey { p, q, qinv_r }
}

/// The public key, `e = 65537`, ready for `montmul` at 64 limbs.
pub struct PublicKey {
    n: [u32; FL],
    r2: [u32; FL],
    n0inv: u32,
}

pub fn public_key() -> PublicKey {
    let n = limbs(N);
    PublicKey {
        n,
        r2: power_of_two(64 * FL, &n),
        n0inv: n0inv(n[0]),
    }
}

impl PublicKey {
    /// `signature ^ 65537 mod n == message`
    pub fn verifies(&self, signature: &[u32; FL], message: &[u32; FL]) -> bool {
        let mul = |a: &[u32; FL], b: &[u32; FL]| montmul(a, b, &self.n, self.n0inv);
        let s = mul(signature, &self.r2);
        let mut acc = s;
        for _ in 0..16 {
            acc = mul(&acc, &acc);
        }
        let mut one = [0; FL];
        one[0] = 1;
        mul(&mul(&acc, &s), &one) == *message
    }
}
