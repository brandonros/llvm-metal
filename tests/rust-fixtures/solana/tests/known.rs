#![cfg(feature = "consumer")]
use llvm_metal_fixture_solana::probe;
use num_bigint::BigUint;
use sha2::{Digest, Sha512};

fn run(name: &str, input: &[u8]) -> Vec<u8> {
    let (n, m, f) = probe(&format!("consumer_solana_{name}")).unwrap();
    assert_eq!(input.len(), n);
    let mut out = vec![0; m];
    // SAFETY: exact-sized disjoint buffers.
    unsafe { f(input.as_ptr(), out.as_mut_ptr()) };
    out
}
fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
fn padded(n: BigUint) -> Vec<u8> {
    let mut bytes = n.to_bytes_le();
    bytes.resize(32, 0);
    bytes
}
#[test]
fn independent_hash_encoding_and_scalar_arithmetic() {
    let order = BigUint::from_bytes_le(&hex(
        "edd3f55c1a631258d69cf7a2def9de1400000000000000000000000000000010",
    ));
    let mut values = vec![vec![0; 32], vec![255; 32], (0..32).collect()];
    for bit in 0..256 {
        let mut v = vec![0; 32];
        v[bit / 8] = 1 << (bit % 8);
        values.push(v);
    }
    for delta in [0u8, 1] {
        values.push(padded(&order + BigUint::from(delta)));
    }
    values.push(padded(&order - BigUint::from(1u8)));
    let alphabet = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    for input in &values {
        assert_eq!(run("sha512", input), Sha512::digest(input).as_slice());
        let n = BigUint::from_bytes_le(input);
        assert_eq!(run("scalar_reduce", input), padded(&n % &order));
        let mut wide = input.clone();
        wide.extend(input.iter().rev());
        assert_eq!(
            run("scalar_wide", &wide),
            padded(BigUint::from_bytes_le(&wide) % &order)
        );
        assert_eq!(
            run("scalar_product", &wide),
            padded((&n * BigUint::from_bytes_le(&wide[32..])) % &order)
        );
        let mut clamped = input.clone();
        clamped[0] &= 248;
        clamped[31] = (clamped[31] & 127) | 64;
        assert_eq!(run("clamp", input), clamped);
        // Independent big-integer Base58 reference (consumer uses limb division).
        let mut n = BigUint::from_bytes_be(input);
        let mut expected = Vec::new();
        while n != BigUint::from(0u8) {
            let rem = (&n % BigUint::from(58u8)).to_bytes_le();
            expected.push(alphabet[rem.first().copied().unwrap_or(0) as usize]);
            n /= 58u8;
        }
        expected.extend(std::iter::repeat_n(
            b'1',
            input.iter().take_while(|&&b| b == 0).count(),
        ));
        expected.reverse();
        let encoded = run("base58", input);
        assert_eq!(encoded[0] as usize, expected.len());
        assert_eq!(&encoded[1..1 + expected.len()], expected);
        assert!(encoded[1 + expected.len()..].iter().all(|&x| x == 0));
    }
}
#[test]
fn rfc8032_and_consumer_known_answers() {
    // RFC 8032 section 7.1, first two Ed25519 signing keys.
    for (secret, public) in [
        (
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        ),
        (
            "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        ),
    ] {
        assert_eq!(run("public_key", &hex(secret)), hex(public));
    }
    let secret = hex("fa9ce9b02dc28a48f7e9d15506d3d2c443d596565fa05214b0ff7c5ab5e7956b");
    let result = run("address", &secret);
    assert_eq!(
        &result[..32],
        hex("089a23ffc422f53d114587012bb2c028492fabdabe1266bc9ad6698ac43016bb")
    );
    let address = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";
    assert_eq!(result[32] as usize, address.len());
    assert_eq!(&result[33..33 + address.len()], address);
    let mut one = [0; 32];
    one[0] = 1;
    let base = hex("5866666666666666666666666666666666666666666666666666666666666666");
    assert_eq!(run("base_mul", &one), base);
    one[0] = 2;
    let doubled = run("point_double", &base);
    assert_eq!(doubled[0], 1);
    assert_eq!(&doubled[1..], run("base_mul", &one));
    let mut identity = [0; 32];
    identity[0] = 1;
    assert_eq!(run("base_mul", &[0; 32]), identity);
}
