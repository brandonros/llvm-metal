//! Signatures per second: the GPU against one host core running the same code.
use std::time::Instant;

fn main() -> Result<(), String> {
    let batch: usize = match std::env::args().nth(1) {
        Some(argument) => argument.parse().map_err(|_| "usage: rsa [batch]")?,
        None => 65536,
    };
    let key = rsa::key::sign_key();
    let public = rsa::key::public_key();
    let messages = rsa::messages(batch);
    let gpu = rsa::Gpu::compile(&rsa::root().join("../../target/examples/rsa/bench"))?;

    gpu.sign(&key, &messages[..1])?; // warm up
    let start = Instant::now();
    let (signatures, kernel) = gpu.sign(&key, &messages)?;
    let total = start.elapsed();
    println!(
        "GPU  batch={batch}  kernel {kernel:.2?}, {:.0} signatures/s; end to end {total:.2?}, {:.0}/s",
        batch as f64 / kernel.as_secs_f64(),
        batch as f64 / total.as_secs_f64(),
    );

    let sample = &messages[..batch.min(256)];
    let start = Instant::now();
    let expected = rsa::on_host(&key, sample);
    let rate = sample.len() as f64 / start.elapsed().as_secs_f64();
    println!(
        "host batch={}  {rate:.0} signatures/s on one core",
        sample.len()
    );

    let bad = (signatures.iter().zip(&messages))
        .filter(|(signature, message)| !public.verifies(signature, message))
        .count();
    println!(
        "public key rejected {bad} of {batch}; GPU matches host: {}",
        signatures[..sample.len()] == expected
    );
    if bad == 0 {
        Ok(())
    } else {
        Err("bad signatures".into())
    }
}
