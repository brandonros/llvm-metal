extern crate std;
use crate::factors::{P, Q};
use crypto_bigint::{Encoding, U1024, U2048};
use std::{vec, vec::Vec};
use vanity_logic::{
    modes::rsa_modulus::{self as rsa, SearchConfig, Task},
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
        "consumer_rsa_prepare" => {
            for bits in [1, 8, 1024, 2048] {
                for id in [0u64, 17] {
                    let mut c = config();
                    c.suffix = c.lower;
                    c.suffix_bits = bits;
                    let mut task = Task::EMPTY;
                    task.p = P;
                    task.id = id;
                    cases.push([bytes(&c), bytes(&task)].concat());
                    c.upper = [0xff; 256];
                    cases.push([bytes(&c), bytes(&task)].concat());
                }
            }
            let mut c = config();
            let p = U1024::from_be_slice(&P);
            let square: U2048 = p.mul(&p);
            c.lower = square.to_be_bytes();
            c.upper = c.lower;
            let mut task = Task::EMPTY;
            task.p = P;
            cases.push([bytes(&c), bytes(&task)].concat());
        }
        "consumer_rsa_advance" => {
            let c = config();
            let mut task = Task::EMPTY;
            task.p = P;
            task.first = Q;
            task.state = 2;
            task.count = U1024::from_u8(5).to_be_bytes();
            task.remaining = task.count;
            task.cursor = U1024::from_u8(4).to_be_bytes();
            for offset in [0u32, 1, 4, 5, u32::MAX] {
                for assigned in [0u32, 1, 4, 5, u32::MAX] {
                    cases.push(
                        [
                            bytes(&c),
                            bytes(&task),
                            offset.to_le_bytes().to_vec(),
                            assigned.to_le_bytes().to_vec(),
                        ]
                        .concat(),
                    );
                }
            }
            for state in [0u32, 1, 3] {
                task.state = state;
                cases.push(
                    [
                        bytes(&c),
                        bytes(&task),
                        0u32.to_le_bytes().to_vec(),
                        1u32.to_le_bytes().to_vec(),
                    ]
                    .concat(),
                );
            }
        }
        "consumer_rsa_mine" => {
            let c = config();
            let pattern = HexPattern::new("", "", 256).unwrap();
            let encode = |c: &SearchConfig,
                          p: &HexPattern,
                          t: &Task,
                          start: u64,
                          stride: u32,
                          steps: u32| {
                [
                    bytes(c),
                    bytes(p),
                    bytes(t),
                    start.to_le_bytes().to_vec(),
                    stride.to_le_bytes().to_vec(),
                    steps.to_le_bytes().to_vec(),
                ]
                .concat()
            };
            cases.push(encode(&c, &pattern, &Task::EMPTY, 0, 1, 1));
            cases.push(encode(&c, &pattern, &Task::EMPTY, 7, 1, 2));
            let mut active = Task::EMPTY;
            active.p = P;
            active.id = 17;
            assert_eq!(rsa::prepare_range(&c, &mut active), Ok(true));
            cases.push(encode(&c, &pattern, &active, 20, 1, 1));
            let miss = HexPattern::new("00", "", 256).unwrap();
            cases.push(encode(&c, &miss, &active, 20, 1, 1));
            for (start, stride, steps) in [(0, 0, 1), (0, 1, 0), (u64::MAX, 2, 2)] {
                cases.push(encode(&c, &pattern, &Task::EMPTY, start, stride, steps));
            }
            active.state = 3;
            cases.push(encode(&c, &pattern, &active, 0, 1, 1));
        }
        _ => return None,
    }
    Some(cases)
}
