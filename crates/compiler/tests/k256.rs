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
    if entry == "k256_point_double" {
        let g = be(
            "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",
        );
        let two = be(
            "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee51ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a",
        );
        let mut bad = g.clone();
        bad[63] ^= 1;
        return vec![g, two, vec![0; 64], vec![0xff; 64], bad];
    }
    if entry == "wide_mul" {
        let values = [
            0u64,
            1,
            u32::MAX as u64,
            1 << 32,
            1 << 63,
            u64::MAX,
            0xaaaaaaaa55555555,
        ];
        return values
            .iter()
            .flat_map(|a| {
                values
                    .iter()
                    .map(move |b| [a.to_le_bytes(), b.to_le_bytes()].concat())
            })
            .collect();
    }
    let mut values = vec![
        vec![0; 32],
        {
            let mut x = vec![0; 32];
            x[31] = 1;
            x
        },
        vec![0xff; 32],
        {
            let mut x = vec![0; 32];
            x[31] = 2;
            x
        },
        be("152d53723da4203478574b153143a7eaa921a8d82c629517d6b18949f0111abb"),
    ];
    for modulus in [
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
    ] {
        let x = be(modulus);
        let mut below = x.clone();
        below[31] -= 1;
        let mut above = x.clone();
        above[31] += 1;
        values.extend([below, x, above]);
    }
    let mut state = 0x123456789abcdef0u64;
    for _ in 0..16 {
        let mut x = vec![0; 32];
        for part in x.chunks_exact_mut(8) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            part.copy_from_slice(&state.to_be_bytes());
        }
        values.push(x);
    }
    if entry == "k256_field_mul" {
        values
            .iter()
            .flat_map(|a| {
                values
                    .iter()
                    .map(move |b| [a.as_slice(), b.as_slice()].concat())
            })
            .collect()
    } else {
        values
    }
}

fn build_kernel(entry: &str) -> (PathBuf, Kernel, llvm_metal_abi::KernelInterface) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut producer = Command::new("python3");
    producer.current_dir(&root).args([
        "tests/rust-fixtures/build.py",
        "--fixture",
        "k256",
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
    let dir = root.join("target/rust-fixtures/k256").join(entry);
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
    let mut oracle = Command::new(root.join("target/rust-fixtures/k256/cargo-host/release/oracle"))
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
#[ignore = "requires Apple GPU and pinned .#rust-fixtures shell"]
fn scalar_roundtrip() {
    check("k256_scalar_roundtrip");
}

#[test]
#[ignore = "requires Apple GPU and pinned .#rust-fixtures shell"]
fn wide_multiplication() {
    check("wide_mul");
}

#[test]
#[ignore = "requires Apple GPU and pinned .#rust-fixtures shell"]
fn field_multiplication() {
    check("k256_field_mul");
}

#[test]
#[ignore = "requires Apple GPU and pinned .#rust-fixtures shell"]
fn field_square() {
    check("k256_field_square");
}
#[test]
#[ignore = "requires Apple GPU and pinned .#rust-fixtures shell"]
fn field_invert() {
    check("k256_field_invert");
}
#[test]
#[ignore = "requires Apple GPU and pinned .#rust-fixtures shell"]
fn point_double() {
    check("k256_point_double");
}

#[test]
#[ignore = "requires Apple GPU and pinned .#rust-fixtures shell"]
fn scalar_multiply() {
    check("k256_scalar_mul");
}

#[test]
#[ignore = "requires Apple GPU, .#rust-fixtures, and LLVM_METAL_CONSUMER_PATH"]
fn consumer_public_keys() {
    check("consumer_public_keys");
}

#[test]
#[ignore = "requires Apple GPU, .#rust-fixtures, and LLVM_METAL_CONSUMER_PATH"]
fn consumer_public_key_batches() {
    let (root, kernel, _) = build_kernel("consumer_public_keys_batch");
    let inputs = cases("consumer_public_keys");
    for count in [0usize, 1, 31, 32, 33, 65, 129, 257] {
        let keys: Vec<_> = (0..count)
            .flat_map(|i| inputs[i % inputs.len()].iter().copied())
            .collect();
        let expected = oracle(&root, "consumer_public_keys", keys.clone());
        assert_eq!(expected.len(), count * 99);
        for group in [32usize, 64] {
            let mut source = vec![0x5a; 256];
            source.extend((count as u32).to_le_bytes());
            source.extend(&keys);
            source.extend([0x5a; 256]);
            let original = source.clone();
            let capacity = count.max(1) * 99;
            let mut buffers = [
                Buffer {
                    bytes: source,
                    offset: 256,
                },
                Buffer {
                    bytes: vec![0xa5; 512 + capacity],
                    offset: 256,
                },
            ];
            // SAFETY: count matches allocated inputs/outputs. Excess lanes are bounded by the kernel.
            unsafe {
                kernel.run(&mut buffers, count.max(1), group).unwrap();
            }
            assert_eq!(buffers[0].bytes, original);
            let mut output = vec![0xa5; 512 + capacity];
            output[256..256 + expected.len()].copy_from_slice(&expected);
            assert_eq!(buffers[1].bytes, output, "count={count} group={group}");
        }
    }
}
