//! Grind: launch after launch over fresh nonces, printing each new best.
//!
//!   cargo run --release -p shallenge -- [prefix] [threads] [attempts]
use std::hash::{BuildHasher, Hasher, RandomState};
use std::time::Instant;

/// Shapes from 16,384 x 1,024 to 262,144 x 1,024 all sustain the same rate on an
/// M5, within 5%. This one launches about four times a second.
const THREADS: usize = 1 << 16;
const ATTEMPTS: usize = 1 << 10;

fn main() -> Result<(), String> {
    let usage = "usage: shallenge [prefix] [threads] [attempts]";
    let mut arguments = std::env::args().skip(1);
    let prefix = arguments.next().unwrap_or("llvm-metal/".into());
    let mut number = |default| match arguments.next() {
        Some(argument) => argument.parse::<usize>().map_err(|_| usage),
        None => Ok(default),
    };
    let (threads, attempts) = (number(THREADS)?, number(ATTEMPTS)?);

    // A random corner of the 96-bit nonce space for this run, searched in
    // order: runs do not repeat each other, and no thread needs an RNG.
    let random = || RandomState::new().build_hasher().finish();
    let mut request = shallenge::request(prefix.as_bytes(), random(), random(), attempts as u32);
    let directory = shallenge::root().join("../../target/examples/shallenge/bench");
    let mut gpu = shallenge::Gpu::compile(&directory, threads)?;

    let (start, mut best) = (Instant::now(), [u32::MAX; 8]);
    for launch in 1u64.. {
        gpu.request.write(|slot| slot[0] = request);
        gpu.search(threads)?;
        let found = (gpu.records)
            .read(|records| records.iter().min_by_key(|record| record.hash).copied())
            .ok_or("no threads")?;
        let rate = (launch * (threads * attempts) as u64) as f64 / start.elapsed().as_secs_f64();
        if found.hash < best {
            best = found.hash;
            let mut text = prefix.clone().into_bytes();
            text.extend(shallenge::nonce(&request, &found));
            if shallenge::sha256::digest(&text) != best.map(u32::to_be_bytes).concat()[..] {
                return Err("the host's SHA-256 disagrees".into());
            }
            let hash: String = best.iter().map(|word| format!("{word:08x}")).collect();
            println!(
                "{}  {hash}  ({:.0}M hashes/s)",
                String::from_utf8_lossy(&text),
                rate / 1e6
            );
        }
        eprint!("\r{:.0}M hashes/s, {launch} launches ", rate / 1e6);
        request.base = request.base.wrapping_add((threads * attempts) as u64);
    }
    Ok(())
}
