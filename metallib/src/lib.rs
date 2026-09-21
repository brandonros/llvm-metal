//! Minimal single-compute-function Metal library container.
//! Format adapted from Metal.jl's compiler/library.jl (MIT); see THIRD_PARTY_NOTICES.
//! Emits macOS metallib 1.2.4, AIR 2.4, Metal 3.0, without reflection records.

use sha2::{Digest, Sha256};

fn tag(out: &mut Vec<u8>, name: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(name);
    out.extend_from_slice(&(data.len() as u16).to_le_bytes());
    out.extend_from_slice(data);
}

/// Package already legalized and downgraded AIR. This does not validate its IR.
pub fn package(name: &str, air: &[u8]) -> Result<Vec<u8>, String> {
    if name.is_empty() || name.len() >= u16::MAX as usize || name.contains('\0') {
        return Err("entry name must be nonempty, NUL-free and fit a u16 tag".into());
    }
    if !air.starts_with(b"BC\xc0\xde") && !air.starts_with(b"\xde\xc0\x17\x0b") {
        return Err("AIR must be raw or wrapped LLVM bitcode".into());
    }
    let mut tags = Vec::new();
    tag(&mut tags, b"NAME", &[name.as_bytes(), &[0]].concat());
    tag(&mut tags, b"TYPE", &[2]); // compute kernel
    tag(&mut tags, b"HASH", &Sha256::digest(air));
    tag(&mut tags, b"OFFT", &[0; 24]); // section-relative offsets, one function
    tag(&mut tags, b"VERS", &[2, 0, 4, 0, 3, 0, 0, 0]);
    tag(&mut tags, b"MDSZ", &(air.len() as u64).to_le_bytes());
    tags.extend_from_slice(b"ENDT");
    let group_size = u32::try_from(tags.len() + 4).map_err(|_| "function tag group too large")?;
    let mut functions = Vec::new();
    functions.extend_from_slice(&1_u32.to_le_bytes());
    functions.extend_from_slice(&group_size.to_le_bytes());
    functions.extend_from_slice(&tags);
    let public_offset = 88 + functions.len() + 4; // header extension ENDT
    let private_offset = public_offset + 8;
    let module_offset = private_offset + 8;
    let total = module_offset
        .checked_add(air.len())
        .ok_or("library too large")?;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"MTLB");
    for value in [0x8001_u16, 2, 4] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&[0; 6]); // executable + reserved platform fields
    for value in [
        total,
        88,
        functions.len() - 4,
        public_offset,
        8,
        private_offset,
        8,
        module_offset,
        air.len(),
    ] {
        out.extend_from_slice(&(value as u64).to_le_bytes());
    }
    out.extend_from_slice(&functions);
    out.extend_from_slice(b"ENDT");
    for _ in 0..2 {
        out.extend_from_slice(&8_u32.to_le_bytes());
        out.extend_from_slice(b"ENDT");
    }
    out.extend_from_slice(air);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_bad_names_and_non_bitcode() {
        for name in ["".to_owned(), "a\0b".to_owned(), "x".repeat(65535)] {
            assert!(package(&name, b"BC\xc0\xde").is_err());
        }
        assert!(package("entry", b"text LLVM IR").is_err());
    }
}
