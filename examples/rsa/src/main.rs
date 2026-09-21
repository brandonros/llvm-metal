//! Signatures per second: the GPU against one host core running the same code.
//! `runtime::run` loads the library and builds the pipeline on every call, so
//! the GPU rate is the slope between two batch sizes, which cancels that cost.
use std::time::{Duration, Instant};

fn timed<T>(work: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    (work(), start.elapsed())
}

fn main() -> Result<(), String> {
    let batch: usize = match std::env::args().nth(1) {
        Some(argument) => argument.parse().map_err(|_| "usage: rsa [batch]")?,
        None => 65536,
    };
    let key = rsa::key::sign_key();
    let public = rsa::key::public_key();
    let messages = rsa::messages(batch);
    let compiled = rsa::compile(&rsa::root().join("../../target/examples/rsa/bench"))?;

    rsa::on_gpu(&compiled, &key, &messages[..1])?; // the pipeline cache is warm after this
    let (small, small_time) = timed(|| rsa::on_gpu(&compiled, &key, &messages[..batch / 4]));
    let (signatures, time) = timed(|| rsa::on_gpu(&compiled, &key, &messages));
    let (small, signatures) = (small?, signatures?);
    let slope = (batch - small.len()) as f64 / (time - small_time).as_secs_f64();
    println!(
        "GPU  batch={batch}  {time:.2?} ({small_time:.2?} for a quarter)  {slope:.0} signatures/s"
    );

    let sample = &messages[..batch.min(256)];
    let (expected, host_time) = timed(|| rsa::on_host(&key, sample));
    let rate = sample.len() as f64 / host_time.as_secs_f64();
    println!(
        "host batch={}  {host_time:.2?}  {rate:.0} signatures/s on one core",
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
