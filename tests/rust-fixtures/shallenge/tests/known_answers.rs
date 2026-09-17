use llvm_metal_fixture_shallenge::shallenge_sha256_32;

#[test]
fn fixed_length_sha256_matches_independent_known_answers_and_preserves_guards() {
    // SHA-256 answers independently generated with Python hashlib/OpenSSL.
    let cases = [
        (
            [0; 32],
            "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925",
        ),
        (
            [0xff; 32],
            "af9613760f72635fbdb44a5a0a63c39f12af30f950a6ee5c971be188e89c4051",
        ),
        (
            core::array::from_fn(|i| i as u8),
            "630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd",
        ),
    ];
    for (input, expected) in cases {
        let mut guarded_input = [0x5a; 34];
        guarded_input[1..33].copy_from_slice(&input);
        let original = guarded_input;
        let mut output = [0xa5; 34];
        // SAFETY: exactly 32 readable/writable bytes, byte alignment, disjoint.
        unsafe {
            shallenge_sha256_32(guarded_input.as_ptr().add(1), output.as_mut_ptr().add(1));
        }
        let actual: String = output[1..33].iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(actual, expected);
        assert_eq!(guarded_input, original);
        assert_eq!(output[0], 0xa5);
        assert_eq!(output[33], 0xa5);
    }
}
