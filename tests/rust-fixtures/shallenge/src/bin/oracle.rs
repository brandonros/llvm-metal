//! Host-only oracle using the same pinned production logic; binary stdin/stdout.
use llvm_metal_fixture_shallenge::*;
use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let entry = std::env::args().nth(1);
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input)?;
    if entry.as_deref() == Some("shallenge_batch") {
        if input.len() < 4 {
            return Err("missing batch count".into());
        }
        let count = u32::from_le_bytes(input[..4].try_into()?) as usize;
        if input.len() != 4 + count * 79 {
            return Err("invalid batch input length".into());
        }
        let mut output = vec![0; count * 65];
        for (request, result) in input[4..]
            .as_chunks::<79>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<65>().0.iter_mut())
        {
            unsafe {
                shallenge_candidate(request.as_ptr(), result.as_mut_ptr());
            }
        }
        std::io::stdout().write_all(&output)?;
        return Ok(());
    }
    let (input_len, output_len, function): (
        usize,
        usize,
        unsafe extern "C" fn(*const u8, *mut u8),
    ) = match entry.as_deref() {
        #[cfg(feature = "compiler-probes")]
        Some("sha_ops") => (12, 32, sha_ops),
        #[cfg(feature = "compiler-probes")]
        Some("sha_schedule") => (16, 4, sha_schedule),
        #[cfg(feature = "compiler-probes")]
        Some("sha_round") => (40, 32, sha_round),
        #[cfg(feature = "compiler-probes")]
        Some("sha_compress") => (96, 32, sha_compress),
        Some("shallenge_nonce") => (16, 21, shallenge_nonce),
        Some("shallenge_compare") => (64, 4, shallenge_compare),
        Some("shallenge_candidate") => (79, 65, shallenge_candidate),
        _ => return Err("unknown oracle entry".into()),
    };
    if input.len() != input_len {
        return Err("incorrect input length".into());
    }
    let mut output = vec![0; output_len];
    // SAFETY: length-checked disjoint buffers; these wrappers require byte alignment.
    unsafe {
        function(input.as_ptr(), output.as_mut_ptr());
    }
    std::io::stdout().write_all(&output)?;
    Ok(())
}
