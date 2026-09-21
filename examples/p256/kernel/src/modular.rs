//! Branchless modular arithmetic on little-endian 32-bit limbs.

/// `t - n` if `carry` is set or `t >= n`, else `t`: compute both, then select.
/// On a CPU this is a branch; here a branch would split the SIMD group.
pub fn reduce<const N: usize>(t: &[u32; N], carry: u32, n: &[u32; N]) -> [u32; N] {
    let mut d = [0u32; N];
    let mut borrow = 0u64;
    for j in 0..N {
        let s = (t[j] as u64).wrapping_sub(n[j] as u64).wrapping_sub(borrow);
        d[j] = s as u32;
        borrow = (s >> 32) & 1;
    }
    let mask = (carry | (1 - borrow as u32)).wrapping_neg();
    let mut r = [0u32; N];
    for j in 0..N {
        r[j] = (d[j] & mask) | (t[j] & !mask);
    }
    r
}

/// Montgomery multiplication, CIOS form: `a * b / R mod n`, where `R = 2^(32 N)`.
/// No division, and both loops have a constant trip count. Needs `a * b < n * R`.
/// `n0inv` is `-n^-1 mod 2^32`.
pub fn montmul<const N: usize>(a: &[u32; N], b: &[u32; N], n: &[u32; N], n0inv: u32) -> [u32; N] {
    // The accumulator is N + 2 limbs: `t`, then `top`, then one carry bit.
    let mut t = [0u32; N];
    let mut top = 0u32;
    for i in 0..N {
        // t += a * b[i]
        let bi = b[i] as u64;
        let mut c = 0u64;
        for j in 0..N {
            let s = t[j] as u64 + a[j] as u64 * bi + c; // cannot overflow
            t[j] = s as u32;
            c = s >> 32;
        }
        let s = top as u64 + c;
        top = s as u32;
        let over = (s >> 32) as u32;

        // t = (t + m * n) >> 32, where m makes the low limb cancel.
        let m = t[0].wrapping_mul(n0inv) as u64;
        let mut c = (t[0] as u64 + m * n[0] as u64) >> 32;
        for j in 1..N {
            let s = t[j] as u64 + m * n[j] as u64 + c;
            t[j - 1] = s as u32;
            c = s >> 32;
        }
        let s = top as u64 + c;
        t[N - 1] = s as u32;
        top = over + (s >> 32) as u32;
    }
    reduce(&t, top, n)
}

/// `a + b mod n`, for `a, b < n`.
pub fn modadd<const N: usize>(a: &[u32; N], b: &[u32; N], n: &[u32; N]) -> [u32; N] {
    let mut s = [0u32; N];
    let mut c = 0u64;
    for j in 0..N {
        let x = a[j] as u64 + b[j] as u64 + c;
        s[j] = x as u32;
        c = x >> 32;
    }
    reduce(&s, c as u32, n)
}

/// `a - b mod n`, for `a, b < n`.
pub fn modsub<const N: usize>(a: &[u32; N], b: &[u32; N], n: &[u32; N]) -> [u32; N] {
    let mut d = [0u32; N];
    let mut borrow = 0u64;
    for j in 0..N {
        let x = (a[j] as u64).wrapping_sub(b[j] as u64).wrapping_sub(borrow);
        d[j] = x as u32;
        borrow = (x >> 32) & 1;
    }
    // Went negative: add n back.
    let mask = (borrow as u32).wrapping_neg();
    let mut r = [0u32; N];
    let mut c = 0u64;
    for j in 0..N {
        let x = d[j] as u64 + (n[j] & mask) as u64 + c;
        r[j] = x as u32;
        c = x >> 32;
    }
    r
}
