//! Hashes per second, and the best nonce of one launch.
use std::hash::{BuildHasher, Hasher, RandomState};

fn main() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1).map(|a| a.parse::<usize>());
    let threads = arguments
        .next()
        .unwrap_or(Ok(1 << 16))
        .map_err(|_| "usage: shallenge [threads] [attempts]")?;
    let attempts = arguments
        .next()
        .unwrap_or(Ok(1 << 10))
        .map_err(|_| "usage: shallenge [threads] [attempts]")?;
    let prefix = b"llvm-metal/";
    // A random corner of the 96-bit nonce space for each launch, searched in
    // order: launches do not repeat each other, and no thread needs an RNG.
    let random = || RandomState::new().build_hasher().finish();
    let request = shallenge::request(prefix, random(), random(), attempts as u32);
    let directory = shallenge::root().join("../../target/examples/shallenge/bench");
    let mut gpu = shallenge::Gpu::compile(&directory, threads)?;

    gpu.request.write(|slot| slot[0] = request);
    gpu.search(1)?; // warm up
    let kernel = gpu.search(threads)?;
    let hashes = (threads * attempts) as f64;
    println!(
        "GPU  threads={threads} attempts={attempts}  kernel {kernel:.2?}, {:.1}M hashes/s",
        hashes / kernel.as_secs_f64() / 1e6
    );

    let best = (gpu.records)
        .read(|records| records.iter().min_by_key(|record| record.hash).copied())
        .ok_or("no threads")?;
    let mut text = prefix.to_vec();
    text.extend(shallenge::nonce(&request, &best));
    let hash: String = best.hash.iter().map(|word| format!("{word:08x}")).collect();
    println!("{}  {hash}", String::from_utf8_lossy(&text));
    match shallenge::sha256::digest(&text) == best.hash.map(u32::to_be_bytes).concat()[..] {
        true => Ok(()),
        false => Err("the host's SHA-256 disagrees".into()),
    }
}
