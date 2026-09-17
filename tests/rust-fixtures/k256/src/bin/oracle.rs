//! Host-only oracle; accepts a sequence of fixed-size records on stdin.
use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let entry = std::env::args().nth(1).ok_or("missing entry")?;
    let (input_len, output_len, function) =
        llvm_metal_fixture_k256::probe(&entry).ok_or("unknown entry")?;
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input)?;
    if input.len() % input_len != 0 {
        return Err("incomplete input record".into());
    }
    for record in input.chunks_exact(input_len) {
        let mut output = vec![0; output_len];
        // SAFETY: exact-sized disjoint byte-aligned buffers.
        unsafe { function(record.as_ptr(), output.as_mut_ptr()) };
        std::io::stdout().write_all(&output)?;
    }
    Ok(())
}
