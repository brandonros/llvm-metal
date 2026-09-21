//! P-256's constants and the host's side of the curve: the key, the table the
//! kernel adds from, and ECDSA verification. All of it runs on the kernel's own
//! field and point code, along paths the signer does not take.
use p256_kernel::field::{Field, L, Number};
use p256_kernel::point::{self, Point};
use p256_kernel::{Key, MULTIPLES, Signature, WINDOWS, modadd, reduce};

const P: &str = "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff";
const N: &str = "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551";
const B: &str = "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b";
const GX: &str = "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296";
const GY: &str = "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5";

/// Little-endian limbs of a 64-digit big-endian hex number.
pub fn number(hex: &str) -> Number {
    assert_eq!(hex.len(), 8 * L);
    std::array::from_fn(|j| {
        let end = hex.len() - 8 * j;
        u32::from_str_radix(&hex[end - 8..end], 16).expect("hex")
    })
}

fn field(modulus: &str) -> Field {
    let modulus = number(modulus);
    // 2^bits mod m, by doubling.
    let power_of_two = |bits| {
        let mut x = [0; L];
        x[0] = 1;
        for _ in 0..bits {
            x = modadd(&x, &x, &modulus);
        }
        x
    };
    // -m^-1 mod 2^32: Newton's iteration doubles the correct bits.
    let mut inverse = 1u32;
    for _ in 0..5 {
        inverse = inverse.wrapping_mul(2u32.wrapping_sub(modulus[0].wrapping_mul(inverse)));
    }
    // Both moduli end in a limb above 1, so m - 2 does not borrow.
    let mut inverse_exponent = modulus;
    inverse_exponent[0] -= 2;
    Field {
        modulus,
        one: power_of_two(32 * L),
        r2: power_of_two(64 * L),
        inverse_exponent,
        n0inv: inverse.wrapping_neg(),
    }
}

/// The curve with the private key `d`.
pub fn key(d: &Number) -> Key {
    let (p, n) = (field(P), field(N));
    Key {
        b3: p.triple(&p.enter(&number(B))),
        d: n.enter(d),
        p,
        n,
    }
}

pub fn generator(key: &Key) -> Point {
    Point {
        x: key.p.enter(&number(GX)),
        y: key.p.enter(&number(GY)),
        z: key.p.one,
    }
}

/// `scalar * point`, by double-and-add.
pub fn multiply(scalar: &Number, point: &Point, key: &Key) -> Point {
    let mut sum = point::identity(&key.p);
    for bit in (0..32 * L).rev() {
        sum = point::add(&sum, &sum, &key.b3, &key.p);
        if scalar[bit / 32] >> (bit % 32) & 1 == 1 {
            sum = point::add(&sum, point, &key.b3, &key.p);
        }
    }
    sum
}

/// The plain coordinates of a point, unless it is the identity.
pub fn affine(point: &Point, key: &Key) -> Option<[Number; 2]> {
    let p = &key.p;
    let inverse = p.invert(&point.z);
    (point.z != [0; L]).then(|| [&point.x, &point.y].map(|c| p.leave(&p.mul(c, &inverse))))
}

/// `table[i * MULTIPLES + w] = w * 16^i * G`.
pub fn table(key: &Key) -> Vec<Point> {
    let add = |a: &Point, b: &Point| point::add(a, b, &key.b3, &key.p);
    let mut table = Vec::with_capacity(WINDOWS * MULTIPLES);
    let mut base = generator(key);
    for _ in 0..WINDOWS {
        let mut multiple = point::identity(&key.p);
        for _ in 0..MULTIPLES {
            table.push(multiple);
            multiple = add(&multiple, &base);
        }
        base = multiple;
    }
    table
}

/// ECDSA verification: `x(z/s * G + r/s * Q) mod n == r`.
pub fn verifies(Signature { r, s }: &Signature, z: &Number, public: &Point, key: &Key) -> bool {
    let n = &key.n;
    let inverse = n.invert(&n.enter(s));
    let u1 = n.leave(&n.mul(&n.enter(z), &inverse));
    let u2 = n.leave(&n.mul(&n.enter(r), &inverse));
    let sum = point::add(
        &multiply(&u1, &generator(key), key),
        &multiply(&u2, public, key),
        &key.b3,
        &key.p,
    );
    affine(&sum, key).is_some_and(|[x, _]| reduce(&x, 0, &n.modulus) == *r)
}
