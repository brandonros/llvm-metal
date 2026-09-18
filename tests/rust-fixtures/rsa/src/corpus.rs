//! Host-only runtime corpora shared by the oracle and independent native tests.
extern crate std;
use crate::factors::{P, Q};
use std::{vec, vec::Vec};

fn small(n: u64) -> Vec<u8> {
    let mut b = vec![0; 128];
    b[120..].copy_from_slice(&n.to_be_bytes());
    b
}
pub fn cases(entry: &str) -> Vec<Vec<u8>> {
    #[cfg(feature = "consumer")]
    if let Some(cases) = crate::search_corpus::cases(entry) {
        return cases;
    }

    if entry.ends_with("prime1024") {
        let mut v: Vec<_> = [
            0,
            1,
            2,
            4,
            137,
            65537,
            561,
            341550071728321,
            3825123056546413051,
        ]
        .into_iter()
        .map(small)
        .collect();
        v.extend([P.to_vec(), Q.to_vec(), vec![255; 128]]);
        return v;
    }
    let mut values = vec![
        vec![0; 128],
        small(1),
        vec![255; 128],
        (0..128).collect(),
        P.to_vec(),
        Q.to_vec(),
    ];
    for limb in 0..16 {
        for bit in [0, 63] {
            let mut v = vec![0; 128];
            let pos = limb * 64 + bit;
            v[127 - pos / 8] = 1 << (pos % 8);
            values.push(v);
        }
    }
    let mut state = 0x123456789abcdef0u64;
    for _ in 0..8 {
        let mut v = vec![0; 128];
        for p in v.chunks_exact_mut(8) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            p.copy_from_slice(&state.to_be_bytes());
        }
        values.push(v);
    }
    if entry.ends_with("shifts1024") {
        let mut out = Vec::new();
        for v in &values[..6] {
            for n in [0u32, 1, 63, 64, 65, 511, 1023, 1024, 2048, u32::MAX] {
                let mut b = v.clone();
                b.extend(n.to_le_bytes());
                out.push(b);
            }
        }
        return out;
    }
    if entry.ends_with("pow32") || entry.ends_with("pow1024") {
        let exponents = [small(0), small(1), small(65537), vec![255; 128], {
            let mut v = small(37);
            v[0] = 128;
            v
        }];
        let mut out = Vec::new();
        for (i, e) in exponents.into_iter().enumerate() {
            out.push([values[(i + 2) % values.len()].clone(), e, P.to_vec()].concat());
        }
        for n in [small(0), small(1), small(4)] {
            out.push([P.to_vec(), small(65537), n].concat());
        }
        return out;
    }
    if entry.ends_with("modular1024") {
        let mut out = Vec::new();
        for (i, a) in values.iter().enumerate() {
            out.push(
                [
                    a.clone(),
                    values[(i + 2) % values.len()].clone(),
                    P.to_vec(),
                ]
                .concat(),
            );
        }
        for n in [small(0), small(1), small(4), small(3), Q.to_vec()] {
            out.push([vec![255; 128], P.to_vec(), n].concat());
        }
        return out;
    }
    let mut out = Vec::new();
    for (i, a) in values.iter().enumerate() {
        let b = values[(i + 2) % values.len()].clone();
        if entry.ends_with("mul256") {
            out.push([a[96..].to_vec(), b[96..].to_vec()].concat());
        } else {
            out.push([a.clone(), b].concat());
        }
    }
    if !entry.ends_with("mul256") {
        for divisor in [small(0), small(1), small(2), small(65537), vec![255; 128]] {
            out.push([vec![255; 128], divisor].concat());
        }
    }
    out
}
