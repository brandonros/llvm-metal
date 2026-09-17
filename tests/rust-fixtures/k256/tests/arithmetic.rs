use llvm_metal_fixture_k256::probe;
use num_bigint::BigUint;

fn run(entry: &str, input: &[u8]) -> Vec<u8> {
    let (n, m, function) = probe(entry).unwrap();
    assert_eq!(input.len(), n);
    let mut output = vec![0xa5; m];
    unsafe { function(input.as_ptr(), output.as_mut_ptr()) };
    output
}

fn bytes(value: &BigUint) -> [u8; 32] {
    let value = value.to_bytes_be();
    let mut result = [0; 32];
    result[32 - value.len()..].copy_from_slice(&value);
    result
}

#[test]
fn scalar_boundaries_distinguish_order_from_field_modulus() {
    let n = BigUint::parse_bytes(
        b"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
        16,
    )
    .unwrap();
    for x in [
        BigUint::from(0u32),
        BigUint::from(1u32),
        &n - 1u32,
        n.clone(),
        &n + 1u32,
        (BigUint::from(1u32) << 256) - 1u32,
    ] {
        let input = bytes(&x);
        let mut expected = vec![0; 33];
        if x < n {
            expected[0] = 1;
            expected[1..].copy_from_slice(&input);
        }
        assert_eq!(run("k256_scalar_roundtrip", &input), expected);
    }
}

#[test]
fn wide_products_include_all_carries() {
    for a in [0u64, 1, u32::MAX as u64, 1 << 32, 1 << 63, u64::MAX] {
        for b in [0u64, 1, u32::MAX as u64, 1 << 32, 1 << 63, u64::MAX] {
            let input = [a.to_le_bytes(), b.to_le_bytes()].concat();
            let mut expected = (BigUint::from(a) * BigUint::from(b)).to_bytes_le();
            expected.resize(16, 0);
            assert_eq!(run("wide_mul", &input), expected);
        }
    }
}

#[test]
fn actual_field_multiplication_matches_independent_big_integers() {
    let p: BigUint = (BigUint::from(1u32) << 256usize) - (BigUint::from(1u32) << 32usize) - 977u32;
    let mut cases = vec![
        BigUint::from(0u32),
        BigUint::from(1u32),
        &p - 1u32,
        p.clone(),
        &p + 1u32,
    ];
    let mut state = 0x123456789abcdef0u64;
    for _ in 0..32 {
        let mut input = [0u8; 32];
        for part in input.as_chunks_mut::<8>().0 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            part.copy_from_slice(&state.to_be_bytes());
        }
        cases.push(BigUint::from_bytes_be(&input));
    }
    for a in &cases {
        for (entry, inverse) in [("k256_field_square", false), ("k256_field_invert", true)] {
            let mut expected = vec![0; 33];
            if a < &p && (!inverse || a != &BigUint::from(0u32)) {
                expected[0] = 1;
                let value = if inverse {
                    a.modpow(&(&p - 2u32), &p)
                } else {
                    (a * a) % &p
                };
                expected[1..].copy_from_slice(&bytes(&value));
            }
            assert_eq!(run(entry, &bytes(a)), expected, "{entry} a={a:x}");
        }
        for b in &cases {
            let input = [bytes(a), bytes(b)].concat();
            let mut expected = vec![0; 33];
            if a < &p && b < &p {
                expected[0] = 1;
                expected[1..].copy_from_slice(&bytes(&((a * b) % &p)));
            }
            assert_eq!(run("k256_field_mul", &input), expected, "a={a:x}, b={b:x}");
        }
    }
}
