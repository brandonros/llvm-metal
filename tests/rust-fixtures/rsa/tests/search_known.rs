#![cfg(feature = "consumer")]
use llvm_metal_fixture_rsa::{probe, search_corpus::cases};
use num_bigint::BigUint;
use sha2::{Digest, Sha256};
use vanity_logic::{modes::rsa_modulus::SearchConfig, search::device_record::DeviceRecord};
fn be(b: &[u8]) -> BigUint {
    BigUint::from_bytes_be(b)
}
fn fixed<const N: usize>(n: &BigUint) -> [u8; N] {
    let raw = n.to_bytes_be();
    let mut out = [0; N];
    out[N - raw.len()..].copy_from_slice(&raw);
    out
}
fn decode<T: DeviceRecord>(b: &[u8]) -> T {
    assert!(b.len() >= size_of::<T>());
    // SAFETY: size checked, DeviceRecord accepts every integer bit pattern.
    unsafe { b.as_ptr().cast::<T>().read_unaligned() }
}
fn run(entry: &str, input: &[u8]) -> Vec<u8> {
    let (n, m, f) = probe(entry).unwrap();
    assert_eq!(n, input.len());
    let mut output = vec![0; m];
    unsafe {
        f(input.as_ptr(), output.as_mut_ptr());
    }
    output
}
// Independent HMAC over sha2, not the consumer's incremental SHA implementation.
fn sample(c: &SearchConfig, id: u64, label: &[u8], bound: &BigUint) -> Option<BigUint> {
    let zero = BigUint::from(0u8);
    let one = BigUint::from(1u8);
    if *bound == zero {
        return None;
    }
    if *bound == one {
        return Some(zero);
    }
    let mask = (&one << (bound - &one).bits() as usize) - &one;
    for attempt in 0u32..128 {
        let mut result = Vec::new();
        for block in 0u128..4 {
            let mut ipad = [0x36; 64];
            let mut opad = [0x5c; 64];
            for i in 0..32 {
                ipad[i] ^= c.seed[i];
                opad[i] ^= c.seed[i];
            }
            let mut inner = Sha256::new();
            inner.update(ipad);
            inner.update(b"vanity-miner/crypto-search/v1\0");
            inner.update(label);
            inner.update([0]);
            inner.update([0; 64]);
            inner.update(c.worker.to_be_bytes());
            inner.update(((u128::from(id) << 2) | block).to_be_bytes());
            inner.update(attempt.to_be_bytes());
            let mut outer = Sha256::new();
            outer.update(opad);
            outer.update(inner.finalize());
            result.extend(outer.finalize());
        }
        let n = be(&result) & &mask;
        if n < *bound {
            return Some(n);
        }
    }
    None
}
fn progression(c: &SearchConfig, p: &BigUint) -> Option<(BigUint, BigUint)> {
    let one = BigUint::from(1u8);
    let zero = BigUint::from(0u8);
    let bits = c.suffix_bits as usize;
    if bits == 0 || bits > 2048 || p.bits() != 1024 || p % 2u8 == zero {
        return None;
    }
    let lower = be(&c.lower);
    let upper = be(&c.upper);
    if upper < lower {
        return None;
    }
    let low = ((&lower + p - &one) / p).max(&one << 1023);
    let high = (&upper / p).min((&one << 1024) - &one);
    if low > high {
        return None;
    }
    let stride = &one << bits;
    // Euler's theorem: odd p is a unit modulo 2^bits (including bits=1).
    let inverse = p.modpow(&((&one << (bits - 1)) - &one), &stride);
    let residue = (be(&c.suffix) * inverse) % &stride;
    let first = &low + (residue + &stride - (&low % &stride)) % &stride;
    if first > high {
        None
    } else {
        let count = (high - &first) / stride + one;
        Some((first, count))
    }
}
#[test]
fn candidate_prf_and_range_match_independent_arithmetic() {
    for entry in ["consumer_rsa_generate", "consumer_rsa_progression"] {
        for input in cases(entry).unwrap() {
            let c: SearchConfig = decode(&input);
            let actual = run(entry, &input);
            let expected = if entry.ends_with("generate") {
                let id = u64::from_le_bytes(input[1072..].try_into().unwrap());
                let p = sample(&c, id, b"rsa-factor", &be(&c.p_count))
                    .map(|offset| be(&c.p_min) + offset * 2u8)
                    .filter(|p| p.bits() == 1024 && p % 2u8 == BigUint::from(1u8));
                if let Some(p) = p {
                    [vec![1], fixed::<128>(&p).to_vec()].concat()
                } else {
                    vec![0; 129]
                }
            } else {
                if let Some((first, count)) = progression(&c, &be(&input[1072..])) {
                    [
                        vec![1],
                        fixed::<128>(&first).to_vec(),
                        fixed::<128>(&count).to_vec(),
                    ]
                    .concat()
                } else {
                    vec![0; 257]
                }
            };
            assert_eq!(actual, expected, "{entry}");
        }
    }
}
#[test]
fn sampled_q_matches_independent_progression_and_hmac() {
    let one = BigUint::from(1u8);
    let zero = BigUint::from(0u8);
    for input in cases("consumer_rsa_sample_q").unwrap() {
        let c: SearchConfig = decode(&input);
        let p = be(&input[1072..1200]);
        let id = u64::from_le_bytes(input[1200..].try_into().unwrap());
        let mut expected = vec![0; 129];
        if let Some((first, total)) = progression(&c, &p) {
            let distance = &one << 924usize;
            let stride = &one << c.suffix_bits as usize;
            let last = &first + (&total - &one) * &stride;
            let low = (&p - &distance).max(first.clone());
            let high = (&p + &distance).min(last);
            let (skip_start, skip_count) = if low <= high {
                let begin = (&low - &first + &stride - &one) / &stride;
                let end = (&high - &first) / &stride;
                if begin <= end {
                    (begin.clone(), &end - &begin + &one)
                } else {
                    (zero.clone(), zero.clone())
                }
            } else {
                (zero.clone(), zero.clone())
            };
            let eligible = &total - &skip_count;
            if eligible != zero {
                let mut index = sample(&c, id, b"rsa-range-start", &eligible).unwrap();
                if index >= skip_start {
                    index += skip_count;
                }
                let q = &first + index * stride;
                assert_eq!(q.bits(), 1024);
                expected = [vec![1], fixed::<128>(&q).to_vec()].concat();
            }
        }
        assert_eq!(run("consumer_rsa_sample_q", &input), expected);
    }
}
