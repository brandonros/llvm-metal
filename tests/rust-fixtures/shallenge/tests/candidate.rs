use llvm_metal_fixture_shallenge::{shallenge_candidate, shallenge_compare};

#[test]
fn candidate_matches_existing_known_answer_and_rejects_invalid_lengths() {
    let mut input = [0; 79];
    input[8..16].copy_from_slice(&12345_u64.to_le_bytes());
    input[16] = 10;
    input[17..27].copy_from_slice(b"brandonros");
    input[47..].fill(255);
    let mut output = [0; 65];
    // SAFETY: fixed lengths and disjoint byte buffers.
    unsafe {
        shallenge_candidate(input.as_ptr(), output.as_mut_ptr());
    }
    let hash: String = output[..32].iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hash,
        "c3750f8711bf809f46de1f01eceb6f4e6fde670ad8a3e2a600a0e0b7357654c9"
    );
    assert_eq!(&output[62..], &[21, 1, 1]);
    for length in [0, 31, 255] {
        input[16] = length;
        output.fill(0xa5);
        unsafe {
            shallenge_candidate(input.as_ptr(), output.as_mut_ptr());
        }
        assert_eq!(output, [0; 65]);
    }
}

#[test]
fn comparison_checks_the_last_byte_and_equality() {
    let mut input = [0; 64];
    let mut output = [0; 4];
    for (a, b, expected) in [(0, 1, -1_i32), (1, 0, 1), (1, 1, 0)] {
        input[31] = a;
        input[63] = b;
        unsafe {
            shallenge_compare(input.as_ptr(), output.as_mut_ptr());
        }
        assert_eq!(i32::from_le_bytes(output), expected);
    }
}
