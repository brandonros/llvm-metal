#![cfg(feature = "consumer")]
use bech32::ToBase32;
use llvm_metal_fixture_k256::probe;
use sha2::{Digest, Sha256};

fn run(entry: &str, input: &[u8]) -> Vec<u8> {
    let (n, m, f) = probe(entry).unwrap();
    assert_eq!(input.len(), n);
    let mut output = vec![0xa5; m];
    // SAFETY: exact-sized disjoint buffers for the reviewed fixture.
    unsafe { f(input.as_ptr(), output.as_mut_ptr()) };
    output
}
fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
fn corpus(n: usize) -> Vec<Vec<u8>> {
    let mut cases = vec![vec![0; n], vec![255; n], (0..n as u8).collect()];
    for bit in 0..n * 8 {
        let mut bytes = vec![0; n];
        bytes[bit / 8] = 1 << (bit % 8);
        cases.push(bytes);
    }
    cases
}

#[test]
fn hashing_matches_independent_rustcrypto_implementations() {
    for input in corpus(32) {
        assert_eq!(
            run("consumer_bitcoin_ripemd160", &input),
            &ripemd::Ripemd160::digest(&input)[..]
        );
    }
    for input in corpus(33) {
        let sha = Sha256::digest(&input);
        let hash = ripemd::Ripemd160::digest(sha);
        assert_eq!(
            run("consumer_bitcoin_hash160", &input),
            [&sha[..], &hash[..]].concat()
        );
    }
}

#[test]
fn bech32_matches_independent_encoder_and_preserves_zero_padding() {
    for input in corpus(20) {
        let mut data = vec![bech32::u5::try_from_u8(0).unwrap()];
        data.extend(input.to_base32());
        let expected = bech32::encode("bc", data, bech32::Variant::Bech32).unwrap();
        let result = run("consumer_bitcoin_bech32", &input);
        assert_eq!(result[0] as usize, expected.len());
        assert_eq!(&result[1..1 + expected.len()], expected.as_bytes());
        assert!(result[1 + expected.len()..].iter().all(|&b| b == 0));
    }
}

#[test]
fn bitcoin_address_matches_existing_known_answer_and_rejects_invalid_keys() {
    let key = hex("23a33f35737ab1abc16cc1d17555c8dc751833ac76cf4bc9e32faf3d7352e930");
    let result = run("consumer_bitcoin_address", &key);
    assert_eq!(result[0], 1);
    assert_eq!(
        &result[1..34],
        hex("03bd954ff18736033d7eb34a760a16e7096a0f3a00e74f541d17e55619e6510b16")
    );
    assert_eq!(
        &result[34..54],
        hex("46047c8a3d8edb134c3f1a3e7d65b0fd7421f127")
    );
    assert_eq!(result[54], 42);
    assert_eq!(
        &result[55..97],
        b"bc1qgcz8ez3a3md3xnplrgl86edsl46zruf8mwx56m"
    );
    assert!(result[97..].iter().all(|&b| b == 0));
    for key in [
        vec![0; 32],
        vec![255; 32],
        hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"),
    ] {
        assert_eq!(run("consumer_bitcoin_address", &key), vec![0; 119]);
    }
}
