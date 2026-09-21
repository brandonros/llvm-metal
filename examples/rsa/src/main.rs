//! Signatures per second: the GPU against one host core running the same code.
use std::time::Instant;

fn main() -> Result<(), String> {
    let batch: usize = match std::env::args().nth(1) {
        Some(argument) => argument.parse().map_err(|_| "usage: rsa [batch]")?,
        None => 65536,
    };
    let key = rsa::key::sign_key();
    let public = rsa::key::public_key();
    let directory = rsa::root().join("../../target/examples/rsa/bench");
    let mut gpu = rsa::Gpu::compile(&directory, &key, batch)?;

    gpu.messages.write(rsa::fill);
    gpu.sign(1)?; // warm up
    let start = Instant::now();
    let kernel = gpu.sign(batch)?;
    let total = start.elapsed();
    println!(
        "GPU  batch={batch}  kernel {kernel:.2?}, {:.0} signatures/s; end to end {total:.2?}, {:.0}/s",
        batch as f64 / kernel.as_secs_f64(),
        batch as f64 / total.as_secs_f64(),
    );

    let sample = batch.min(256);
    let (messages, signatures) = (&gpu.messages, &gpu.signatures);
    messages.read(|messages| {
        signatures.read(|signatures| {
            let start = Instant::now();
            let expected = rsa::on_host(&key, &messages[..sample]);
            let rate = sample as f64 / start.elapsed().as_secs_f64();
            println!("host batch={sample}  {rate:.0} signatures/s on one core");

            let bad = (signatures.iter().zip(messages))
                .filter(|(signature, message)| !public.verifies(signature, message))
                .count();
            println!(
                "public key rejected {bad} of {batch}; GPU matches host: {}",
                signatures[..sample] == expected
            );
            if bad == 0 {
                Ok(())
            } else {
                Err("bad signatures".into())
            }
        })
    })
}
