#![cfg(feature = "consumer")]
use llvm_metal_fixture_k256::probe;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn run(entry: &str, input: &[u8]) -> Vec<u8> {
    let (n, m, f) = probe(entry).unwrap();
    assert_eq!(input.len(), n);
    let mut output = vec![0xa5; m];
    // SAFETY: exact-sized disjoint buffers for the selected fixture contract.
    unsafe { f(input.as_ptr(), output.as_mut_ptr()) };
    output
}

#[test]
fn keccak_matches_existing_consumer_known_answer() {
    let half = hex("61a314b0183724ea0e5f237584cb76092e253b99783d846a5b10db155128eafd");
    assert_eq!(
        run("consumer_keccak256", &[half.clone(), half].concat()),
        hex("0f439a9830558b9cd6842328dd11585401c34321a53b29422aacde310643d373")
    );
}

#[test]
fn ethereum_addresses_have_fixed_answers_and_exclude_sec1_tag() {
    for (key, address) in [
        (
            "0000000000000000000000000000000000000000000000000000000000000001",
            "7e5f4552091a69125d5dfcb7b8c2659029395bdf",
        ),
        (
            "0000000000000000000000000000000000000000000000000000000000000002",
            "2b5ad5c4795c026514f8317c7a215e218dccd6cf",
        ),
        (
            "23a33f35737ab1abc16cc1d17555c8dc751833ac76cf4bc9e32faf3d7352e930",
            "55e56b7b70dc37a7a1419e1e84ea4e6e237ef602",
        ),
    ] {
        let key = hex(key);
        let output = run("consumer_ethereum_address", &key);
        assert_eq!(output[0], 1);
        assert_eq!(&output[65..], hex(address));
        let public = run("consumer_public_keys", &key);
        assert_eq!(&output[1..65], &public[35..99]);
        let digest = run("consumer_keccak256", &output[1..65]);
        assert_eq!(&output[65..], &digest[12..]);
    }
    for invalid in [
        vec![0; 32],
        vec![0xff; 32],
        hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"),
    ] {
        assert_eq!(run("consumer_ethereum_address", &invalid), vec![0; 85]);
    }
}
