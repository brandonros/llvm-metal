use shallenge_kernel::{PREFIX, Request};

fn main() -> Result<(), String> {
    let name = b"llvm-metal/";
    let mut prefix = [0; PREFIX];
    prefix[..name.len()].copy_from_slice(name);
    let request = Request {
        seed: 1,
        prefix_length: name.len() as u64,
        prefix,
    };
    let directory = shallenge::root().join("../../target/examples/shallenge");
    let records = shallenge::on_gpu(&request, 1 << 16, &directory)?;
    let best = records
        .iter()
        .min_by_key(|record| record.hash)
        .ok_or("no threads")?;
    let hash: String = best.hash.iter().map(|byte| format!("{byte:02x}")).collect();
    println!(
        "{}{}  {hash}",
        String::from_utf8_lossy(name),
        String::from_utf8_lossy(&best.nonce)
    );
    Ok(())
}
