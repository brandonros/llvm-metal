//! The GPU finds what the host finds, for every thread, however rustc was asked
//! to optimize the kernel; and a separate, general SHA-256 on the host agrees
//! with every hash reported, and that each is its thread's smallest.
use shallenge_kernel::{Record, Request};

const PREFIX: &[u8] = b"xxxxx";
const ATTEMPTS: u32 = 8;

/// The hash of `prefix ‖ nonce` for a counter, by the general implementation.
fn hash(request: &Request, counter: u64) -> [u8; 32] {
    let record = Record {
        hash: [0; 8],
        counter,
    };
    let mut text = PREFIX.to_vec();
    text.extend(shallenge::nonce(request, &record));
    shallenge::sha256::digest(&text)
}

#[test]
fn the_nonce_is_base64_text_in_counter_order() {
    let request = shallenge::request(PREFIX, 0, 0, 1);
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (counter, &letter) in alphabet.iter().enumerate() {
        let record = Record {
            hash: [0; 8],
            counter: counter as u64,
        };
        assert_eq!(shallenge::nonce(&request, &record)[0], letter);
    }
    // Bits 60 to 65 straddle the counter and the salt; the salt fills the rest.
    let record = Record {
        hash: [0; 8],
        counter: u64::MAX,
    };
    let request = shallenge::request(PREFIX, 0, u64::MAX, 1);
    assert_eq!(&shallenge::nonce(&request, &record), b"////////////////");
}

#[test]
fn the_host_finds_each_threads_smallest_hash() {
    let request = shallenge::request(PREFIX, 1000, 7, ATTEMPTS);
    for (thread, record) in shallenge::on_host(&request, 50).iter().enumerate() {
        let first = 1000 + thread as u64 * ATTEMPTS as u64;
        let (smallest, counter) = (first..first + ATTEMPTS as u64)
            .map(|counter| (hash(&request, counter), counter))
            .min()
            .unwrap();
        assert_eq!(record.counter, counter);
        assert_eq!(record.hash.map(u32::to_be_bytes).concat(), smallest);
    }
}

#[test]
#[ignore = "requires an Apple GPU"]
fn gpu_matches_host_at_every_optimization_level() {
    let request = shallenge::request(PREFIX, 1000, 7, ATTEMPTS);
    let threads = 1000;
    let expected = shallenge::on_host(&request, threads);
    for level in ["0", "s", "3"] {
        // SAFETY: the other tests in this binary do not read the environment.
        unsafe { std::env::set_var("CARGO_PROFILE_RELEASE_OPT_LEVEL", level) };
        let directory = shallenge::root()
            .join("../../target/examples/shallenge")
            .join(level);
        let matches = shallenge::Gpu::compile(&directory, threads)
            .and_then(|mut gpu| {
                gpu.request.write(|slot| slot[0] = request);
                gpu.search(threads)?;
                Ok(gpu.records.read(|records| records == expected))
            })
            .unwrap_or_else(|error| panic!("opt-level {level}: {error}"));
        assert!(matches, "opt-level {level}: GPU and host differ");
    }
}
