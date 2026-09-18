#![cfg(feature = "consumer")]
use llvm_metal_fixture_rsa::{corpus::cases, probe};
use num_bigint::BigUint;
fn be(bytes: &[u8]) -> BigUint {
    BigUint::from_bytes_be(bytes)
}
fn bytes(n: BigUint, size: usize) -> Vec<u8> {
    let b = n.to_bytes_be();
    let mut v = vec![0; size];
    v[size - b.len()..].copy_from_slice(&b);
    v
}
#[test]
fn arithmetic_matches_independent_biguint() {
    let one = BigUint::from(1u8);
    let zero = BigUint::from(0u8);
    let space = &one << 1024;
    for suffix in [
        "mul256",
        "mul1024",
        "add_sub1024",
        "shifts1024",
        "divrem1024",
        "modular1024",
        "pow32",
        "pow1024",
    ] {
        let entry = format!("consumer_rsa_{suffix}");
        let (n, m, f) = probe(&entry).unwrap();
        for input in cases(&entry) {
            assert_eq!(input.len(), n);
            let mut actual = vec![0; m];
            // SAFETY: exact-sized disjoint buffers.
            unsafe {
                f(input.as_ptr(), actual.as_mut_ptr());
            }
            let expected = match suffix {
                "mul256" => bytes(be(&input[..32]) * be(&input[32..]), 64),
                "mul1024" => bytes(be(&input[..128]) * be(&input[128..]), 256),
                "add_sub1024" => {
                    let a = be(&input[..128]);
                    let b = be(&input[128..]);
                    [
                        vec![u8::from(a < b)],
                        bytes((&a + &b) % &space, 128),
                        bytes((&a + &space - &b) % &space, 128),
                    ]
                    .concat()
                }
                "shifts1024" => {
                    let a = be(&input[..128]);
                    let n = u32::from_le_bytes(input[128..].try_into().unwrap()) as usize;
                    let (left, right) = if n >= 1024 {
                        (zero.clone(), zero.clone())
                    } else {
                        ((&a << n) % &space, &a >> n)
                    };
                    [
                        bytes(left, 128),
                        bytes(right, 128),
                        (a.bits() as u32).to_le_bytes().to_vec(),
                        (a.trailing_zeros().unwrap_or(1024) as u32)
                            .to_le_bytes()
                            .to_vec(),
                    ]
                    .concat()
                }
                "divrem1024" => {
                    let a = be(&input[..128]);
                    let b = be(&input[128..]);
                    if b == zero {
                        vec![0; m]
                    } else {
                        [vec![1], bytes(&a / &b, 128), bytes(&a % &b, 128)].concat()
                    }
                }
                _ => {
                    let a = be(&input[..128]);
                    let b = be(&input[128..256]);
                    let modulus = be(&input[256..]);
                    if modulus <= one || input[383] & 1 == 0 {
                        vec![0; m]
                    } else if suffix == "modular1024" {
                        [
                            vec![1],
                            bytes((&a * &b) % &modulus, 128),
                            bytes((&a * &a) % &modulus, 128),
                            bytes(&a % &modulus, 128),
                        ]
                        .concat()
                    } else {
                        let exponent = if suffix == "pow32" {
                            b % (&one << 32)
                        } else {
                            b
                        };
                        [vec![1], bytes(a.modpow(&exponent, &modulus), 128)].concat()
                    }
                }
            };
            assert_eq!(actual, expected, "{entry}, input={input:02x?}");
        }
    }
}
#[test]
fn prime_filter_matches_known_primes_and_composites() {
    let entry = "consumer_rsa_prime1024";
    let (_, _, f) = probe(entry).unwrap();
    for (input, expected) in cases(entry).iter().zip([
        false, false, true, false, true, true, false, false, false, true, true, false,
    ]) {
        let mut result = [0];
        // SAFETY: exact-sized disjoint buffers.
        unsafe {
            f(input.as_ptr(), result.as_mut_ptr());
        }
        assert_eq!(result, [u8::from(expected)]);
    }
}
