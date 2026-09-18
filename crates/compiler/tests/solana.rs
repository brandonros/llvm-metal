#![cfg(target_os = "macos")]
use llvm_metal_runtime::{Buffer, Kernel};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

fn be(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

fn cases(entry: &str) -> Vec<Vec<u8>> {
    let n = if entry.ends_with("wide") || entry.ends_with("product") {
        64
    } else {
        32
    };
    let mut cases = vec![vec![0; n], vec![255; n], (0..n as u8).collect()];
    // Walking bits cover every limb, carry and byte position.
    for bit in 0..n * 8 {
        let mut value = vec![0; n];
        if entry.ends_with("product") {
            // Exercise both operands against a nonzero partner; multiplying
            // walking bits by zero would conceal the wide arithmetic.
            let other = if bit < 256 { 32..64 } else { 0..32 };
            value[other].fill(255);
        }
        value[bit / 8] = 1 << (bit % 8);
        cases.push(value);
    }
    let order = be("edd3f55c1a631258d69cf7a2def9de1400000000000000000000000000000010");
    for delta in [-1i16, 0, 1] {
        let mut value = vec![0; n];
        value[..32].copy_from_slice(&order);
        if entry.ends_with("product") {
            value[32..].fill(255);
        }
        value[0] = (i16::from(value[0]) + delta) as u8;
        cases.push(value);
    }
    let mut state = 0x123456789abcdef0u64;
    for _ in 0..16 {
        let mut value = vec![0; n];
        for part in value.chunks_exact_mut(8) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            part.copy_from_slice(&state.to_le_bytes());
        }
        cases.push(value);
    }
    if n == 32 {
        cases.extend([
            be("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"),
            be("5866666666666666666666666666666666666666666666666666666666666666"),
            be("089a23ffc422f53d114587012bb2c028492fabdabe1266bc9ad6698ac43016bb"),
        ]);
    }
    cases
}

fn build_kernel(entry: &str) -> (PathBuf, Kernel, llvm_metal_abi::KernelInterface) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut producer = Command::new("python3");
    producer.current_dir(&root).args([
        "tests/rust-fixtures/build.py",
        "--fixture",
        "solana",
        "--entry",
        entry,
    ]);
    if entry.starts_with("consumer_") {
        producer.arg("--consumer-path").arg(
            std::env::var_os("LLVM_METAL_CONSUMER_PATH")
                .expect("set LLVM_METAL_CONSUMER_PATH to the isolated consumer worktree"),
        );
    }
    let build = producer.output().unwrap();
    assert!(
        build.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    let dir = root.join("target/rust-fixtures/solana").join(entry);
    let context = inkwell::context::Context::create();
    let module = llvm_metal_compiler::parse_bitcode(
        &context,
        &fs::read(dir.join("kernel.bc")).unwrap(),
        entry,
    )
    .unwrap();
    let interface =
        serde_json::from_slice(&fs::read(dir.join("kernel.interface.json")).unwrap()).unwrap();
    let artifact = llvm_metal_compiler::compile::compile(&module, &interface).unwrap();
    fs::write(dir.join("kernel.air.ll"), &artifact.air_ir).unwrap();
    fs::write(dir.join("kernel.air.bc"), &artifact.air_bitcode).unwrap();
    let library = dir.join("kernel.metallib");
    fs::write(&library, &artifact.metallib).unwrap();
    eprintln!("{entry}: LLVM/AIR compiled; loading Metal pipeline");
    let kernel = Kernel::load(&library, &artifact.bindings).unwrap();
    (root, kernel, interface)
}

fn oracle(root: &std::path::Path, entry: &str, input: Vec<u8>) -> Vec<u8> {
    let mut oracle =
        Command::new(root.join("target/rust-fixtures/solana/cargo-host/release/oracle"))
            .arg(entry)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
    let mut stdin = oracle.stdin.take().unwrap();
    // Write concurrently: the complete corpus can exceed pipe capacity.
    let writer = std::thread::spawn(move || stdin.write_all(&input).unwrap());
    let expected = oracle.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(expected.status.success());
    expected.stdout
}

fn check(entry: &str) {
    let (root, kernel, interface) = build_kernel(entry);
    let cases = cases(entry);
    let expected = oracle(&root, entry, cases.concat());
    let output_size = interface.arguments[1].bytes;
    assert_eq!(expected.len(), cases.len() * output_size);
    for (index, (input, expected)) in cases
        .iter()
        .zip(expected.chunks_exact(output_size))
        .enumerate()
    {
        let mut source = vec![0x5a; 256];
        source.extend(input);
        source.extend([0x5a; 256]);
        let original = source.clone();
        let mut buffers = [
            Buffer {
                bytes: source,
                offset: 256,
            },
            Buffer {
                bytes: vec![0xa5; 512 + output_size],
                offset: 256,
            },
        ];
        // SAFETY: reviewed single-invocation fixtures with exact-sized disjoint buffers.
        unsafe {
            kernel.run(&mut buffers, 1, 1).unwrap();
        }
        assert_eq!(
            buffers[0].bytes, original,
            "{entry} input/guards case {index}"
        );
        let mut output = vec![0xa5; 256];
        output.extend(expected);
        output.extend([0xa5; 256]);
        assert_eq!(
            buffers[1].bytes, output,
            "{entry} case {index}, input={input:02x?}"
        );
    }
    eprintln!(
        "{entry}: {} CPU/GPU comparisons on {}",
        cases.len(),
        kernel.device_name()
    );
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_02_clamp() {
    check("consumer_solana_clamp");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_03_scalar_reduce() {
    check("consumer_solana_scalar_reduce");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_04_scalar_wide() {
    check("consumer_solana_scalar_wide");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_05_scalar_product() {
    check("consumer_solana_scalar_product");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_06_point_double() {
    check("consumer_solana_point_double");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_07_base_mul() {
    check("consumer_solana_base_mul");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_01_sha512() {
    check("consumer_solana_sha512");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_08_public_key() {
    check("consumer_solana_public_key");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_09_base58() {
    check("consumer_solana_base58");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_10_address() {
    check("consumer_solana_address");
}
