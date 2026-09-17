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
    let mut out = vec![0xa5; m];
    unsafe {
        f(input.as_ptr(), out.as_mut_ptr());
    }
    out
}
const G: &str = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";
const TWO_G: &str = "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee51ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a";

#[test]
fn generator_doubles_to_known_two_g_and_bad_coordinates_are_rejected() {
    let expected = [vec![1, 4], hex(TWO_G)].concat();
    assert_eq!(run("k256_point_double", &hex(G)), expected);
    assert_eq!(run("k256_point_double", &[0; 64]), vec![0; 66]);
    let mut bad = hex(G);
    bad[63] ^= 1;
    assert_eq!(run("k256_point_double", &bad), vec![0; 66]);
}

#[test]
fn runtime_scalar_multiplication_has_fixed_public_key_answers() {
    for (private, point, prefix) in [
        (
            "0000000000000000000000000000000000000000000000000000000000000001",
            G,
            2,
        ),
        (
            "0000000000000000000000000000000000000000000000000000000000000002",
            TWO_G,
            2,
        ),
        (
            "152d53723da4203478574b153143a7eaa921a8d82c629517d6b18949f0111abb",
            "9163ab449d4b90de13ce60b504bfc27a4aed378c1f8338686156b91445637c8d33272b79994dae54da4011cc3e3491ccdf3bd3fd92978a00873727f99beb4375",
            3,
        ),
    ] {
        let xy = hex(point);
        let expected = [vec![1, prefix], xy[..32].to_vec(), vec![4], xy].concat();
        assert_eq!(run("k256_scalar_mul", &hex(private)), expected);
        #[cfg(feature = "consumer")]
        assert_eq!(run("consumer_public_keys", &hex(private)), expected);
    }
    assert_eq!(run("k256_scalar_mul", &[0; 32]), vec![0; 99]);
    assert_eq!(run("k256_scalar_mul", &[0xff; 32]), vec![0; 99]);
}
