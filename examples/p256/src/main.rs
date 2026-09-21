//! Signatures per second: the GPU against one host core running the same code.
use p256::curve::{self, number};
use std::time::Instant;

fn main() -> Result<(), String> {
    let batch: usize = match std::env::args().nth(1) {
        Some(argument) => argument.parse().map_err(|_| "usage: p256 [batch]")?,
        None => 1 << 20,
    };
    // The private key of RFC 6979's test vectors: public, so a toy by construction.
    let d = number("c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721");
    let key = curve::key(&d);
    let table = curve::table(&key);
    let public = curve::multiply(&d, &curve::generator(&key), &key);
    let requests = p256::requests(batch);
    let gpu = p256::Gpu::compile(&p256::root().join("../../target/examples/p256/bench"))?;

    gpu.sign(&key, &table, &requests[..1])?; // warm up
    let start = Instant::now();
    let (signatures, kernel) = gpu.sign(&key, &table, &requests)?;
    let total = start.elapsed();
    println!(
        "GPU  batch={batch}  kernel {kernel:.2?}, {:.0} signatures/s; end to end {total:.2?}, {:.0}/s",
        batch as f64 / kernel.as_secs_f64(),
        batch as f64 / total.as_secs_f64(),
    );

    let sample = &requests[..batch.min(2048)];
    let start = Instant::now();
    let expected = p256::on_host(&key, &table, sample);
    let rate = sample.len() as f64 / start.elapsed().as_secs_f64();
    println!(
        "host batch={}  {rate:.0} signatures/s on one core",
        sample.len()
    );

    let bad = (signatures.iter().zip(sample))
        .filter(|(signature, request)| !curve::verifies(signature, &request.z, &public, &key))
        .count();
    println!(
        "public key rejected {bad} of the first {}; GPU matches host: {}",
        sample.len(),
        signatures[..sample.len()] == expected
    );
    if bad == 0 {
        Ok(())
    } else {
        Err("bad signatures".into())
    }
}
