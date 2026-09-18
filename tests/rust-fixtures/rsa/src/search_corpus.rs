extern crate std;
use crate::factors::{P, Q};
use crypto_bigint::{Encoding, U1024, U2048};
use std::{vec, vec::Vec};
use vanity_logic::{
    modes::rsa_modulus::SearchConfig,
    search::{device_record::DeviceRecord, hex_pattern::HexPattern},
};
pub fn bytes<T: DeviceRecord>(value: &T) -> Vec<u8> {
    // SAFETY: DeviceRecord is padding-free initialized integer storage.
    unsafe {
        std::slice::from_raw_parts(core::ptr::from_ref(value).cast(), size_of::<T>()).to_vec()
    }
}
pub fn config() -> SearchConfig {
    let n: U2048 = U1024::from_be_slice(&P).mul(&U1024::from_be_slice(&Q));
    let mut suffix = [0; 256];
    suffix[255] = 1;
    SearchConfig {
        lower: n.to_be_bytes(),
        upper: n.to_be_bytes(),
        p_min: P,
        p_count: U1024::ONE.to_be_bytes(),
        suffix,
        seed: [42; 32],
        worker: 7,
        suffix_bits: 1,
        reserved: 0,
    }
}
pub fn cases(entry: &str) -> Option<Vec<Vec<u8>>> {
    let mut cases = Vec::new();
    match entry {
        "consumer_rsa_generate" => {
            for bound in [0, 1, 2, 3, 17, 257, 65537] {
                for id in [0u64, 1, 63, u64::MAX] {
                    let mut c = config();
                    c.p_count = U1024::from_u32(bound).to_be_bytes();
                    cases.push([bytes(&c), id.to_le_bytes().to_vec()].concat());
                }
            }
            let mut c = config();
            c.p_min = [0xff; 128];
            c.p_count = U1024::from_u32(65537).to_be_bytes();
            cases.push([bytes(&c), 1u64.to_le_bytes().to_vec()].concat());
            for worker in [0, 1, u64::MAX] {
                for seed in [[0; 32], [0xff; 32]] {
                    let mut c = config();
                    c.worker = worker;
                    c.seed = seed;
                    c.p_count = U1024::from_u32(65537).to_be_bytes();
                    cases.push([bytes(&c), 7u64.to_le_bytes().to_vec()].concat());
                }
            }
        }
        "consumer_rsa_progression" => {
            for bits in [0, 1, 8, 63, 64, 65, 1023, 1024, 2048, 2049] {
                let mut c = config();
                c.suffix_bits = bits;
                c.suffix = c.lower;
                cases.push([bytes(&c), P.to_vec()].concat());
                c.upper = U2048::from_be_slice(&c.upper)
                    .wrapping_add(&U1024::MAX.resize())
                    .to_be_bytes();
                cases.push([bytes(&c), P.to_vec()].concat());
            }
            let c = config();
            cases.push([bytes(&c), vec![0; 128]].concat());
            let mut even = P;
            even[127] &= 0xfe;
            cases.push([bytes(&c), even.to_vec()].concat());
            let mut inverted = c;
            inverted.upper = [0; 256];
            cases.push([bytes(&inverted), P.to_vec()].concat());
        }
        "consumer_rsa_sample_q" => {
            for bits in [1, 8, 1024, 2048] {
                for id in [0u64, 17, u64::MAX] {
                    let mut c = config();
                    c.suffix = c.lower;
                    c.suffix_bits = bits;
                    cases.push([bytes(&c), P.to_vec(), id.to_le_bytes().to_vec()].concat());
                    c.upper = [0xff; 256];
                    cases.push([bytes(&c), P.to_vec(), id.to_le_bytes().to_vec()].concat());
                }
            }
            let mut c = config();
            let square: U2048 = U1024::from_be_slice(&P).mul(&U1024::from_be_slice(&P));
            c.lower = square.to_be_bytes();
            c.upper = c.lower;
            cases.push([bytes(&c), P.to_vec(), 0u64.to_le_bytes().to_vec()].concat());
            c.upper = [0; 256];
            cases.push([bytes(&c), P.to_vec(), 1u64.to_le_bytes().to_vec()].concat());
        }
        "consumer_rsa_eligible_pair" => {
            let any = HexPattern::new("", "", 256).unwrap();
            let miss = HexPattern::new("00", "", 256).unwrap();
            for (p, q, pattern) in [
                (P, Q, any),
                (P, Q, miss),
                (P, P, any),
                ([0; 128], Q, any),
                (P, [0; 128], any),
            ] {
                cases.push([p.to_vec(), q.to_vec(), bytes(&pattern)].concat());
            }
            let mut composite = Q;
            composite[127] &= 0xfe;
            cases.push([P.to_vec(), composite.to_vec(), bytes(&any)].concat());
        }
        "consumer_rsa_candidate" => {
            let c = config();
            let any = HexPattern::new("", "", 256).unwrap();
            let encode = |c: &SearchConfig, p: &HexPattern, id: u64| {
                [bytes(c), bytes(p), id.to_le_bytes().to_vec()].concat()
            };
            // Repeated IDs separated by other work must return identical records.
            for id in [0, 7, 0, u64::MAX] {
                cases.push(encode(&c, &any, id));
            }
            cases.push(encode(&c, &HexPattern::new("00", "", 256).unwrap(), 0));
            for change in 0..4 {
                let mut invalid = c;
                match change {
                    0 => invalid.reserved = 1,
                    1 => invalid.p_count = [0; 128],
                    2 => invalid.upper = [0; 256],
                    _ => invalid.suffix_bits = 0,
                }
                cases.push(encode(&invalid, &any, 0));
            }
        }
        _ => return None,
    }
    Some(cases)
}
