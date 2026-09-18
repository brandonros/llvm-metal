#![cfg(target_os = "macos")]
use llvm_metal_runtime::{Buffer, Kernel};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

fn build_kernel(entry: &str) -> (PathBuf, Kernel, llvm_metal_abi::KernelInterface) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut producer = Command::new("python3");
    producer.current_dir(&root).args([
        "tests/rust-fixtures/build.py",
        "--fixture",
        "rsa",
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
    let dir = root.join("target/rust-fixtures/rsa").join(entry);
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
    let mut oracle = Command::new(root.join("target/rust-fixtures/rsa/cargo-host/release/oracle"))
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
    let corpus = Command::new(root.join("target/rust-fixtures/rsa/cargo-host/release/oracle"))
        .args([entry, "--cases"])
        .output()
        .unwrap();
    assert!(corpus.status.success());
    assert_eq!(corpus.stdout.len() % interface.arguments[0].bytes, 0);
    let cases: Vec<_> = corpus
        .stdout
        .chunks_exact(interface.arguments[0].bytes)
        .map(|v| v.to_vec())
        .collect();
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
fn stage_01_mul256() {
    check("consumer_rsa_mul256");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_02_mul1024() {
    check("consumer_rsa_mul1024");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_03_add_sub1024() {
    check("consumer_rsa_add_sub1024");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_04_shifts1024() {
    check("consumer_rsa_shifts1024");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_05_divrem1024() {
    check("consumer_rsa_divrem1024");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_06_modular1024() {
    check("consumer_rsa_modular1024");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_07_pow32() {
    check("consumer_rsa_pow32");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_08_pow1024() {
    check("consumer_rsa_pow1024");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_09_prime1024() {
    check("consumer_rsa_prime1024");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_10_generate() {
    check("consumer_rsa_generate");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_11_progression() {
    check("consumer_rsa_progression");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_12_sample_q() {
    check("consumer_rsa_sample_q");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_13_eligible_pair() {
    check("consumer_rsa_eligible_pair");
}

#[test]
#[ignore = "requires Apple GPU, local consumer and pinned .#rust-fixtures shell"]
fn stage_14_candidate() {
    check("consumer_rsa_candidate");
}
