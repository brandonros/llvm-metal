// Source operation for byte-iterator.ll, reduced from crypto-bigint encoding.
// No toolchain-dependent regeneration is used by the regression tests.
#![no_std]
pub fn encode(limbs: &[u64; 4]) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (limb, chunk) in limbs.iter().rev().zip(bytes.chunks_exact_mut(8)) {
        chunk.copy_from_slice(&limb.to_be_bytes());
    }
    bytes
}
