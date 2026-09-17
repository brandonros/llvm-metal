#![cfg(target_os = "macos")]

use llvm_metal_runtime::{Buffer, Kernel};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn run(program: &str, args: &[&Path]) {
    let output = Command::new(program).args(args).output().unwrap();
    assert!(
        output.status.success(),
        "{program}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn package_air(source: &Path, entry: &str, directory: &Path) -> PathBuf {
    fs::create_dir_all(directory).unwrap();
    let modern = directory.join("modern.bc");
    let air = directory.join("air.bc");
    run("llvm-as", &[source, Path::new("-o"), &modern]);
    run(
        "llvm-downgrade",
        &[
            &modern,
            Path::new("--bitcode-version=14.0"),
            Path::new("-o"),
            &air,
        ],
    );
    let library = directory.join("kernel.metallib");
    fs::write(
        &library,
        llvm_metal_metallib::package(entry, &fs::read(air).unwrap()).unwrap(),
    )
    .unwrap();
    library
}

fn check_add42(library: &Path, bindings: &llvm_metal_abi::MetalBindings) {
    let kernel = Kernel::load(library, bindings).unwrap();
    eprintln!("Metal device: {}", kernel.device_name());
    for input in [0_u32, 1, u32::MAX, 0x80000000, 0x12345678] {
        let mut input_bytes = vec![0x5a; 516];
        input_bytes[256..260].copy_from_slice(&input.to_le_bytes());
        let original = input_bytes.clone();
        let mut buffers = [
            Buffer {
                bytes: input_bytes,
                offset: 256,
            },
            Buffer {
                bytes: vec![0xa5; 516],
                offset: 256,
            },
        ];
        // SAFETY: this reviewed kernel reads/writes one u32 in separate buffers.
        unsafe {
            kernel.run(&mut buffers, 1, 1).unwrap();
        }
        assert_eq!(buffers[0].bytes, original);
        let mut expected = vec![0xa5; 516];
        expected[256..260].copy_from_slice(&input.wrapping_add(42).to_le_bytes());
        assert_eq!(buffers[1].bytes, expected, "input={input:#x}");
    }
}

#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade from the Nix shell"]
fn reference_air_executes_through_our_packager_and_runtime() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = package_air(
        &root.join("tests/fixtures/air/add42.ll"),
        "add42",
        &root.join("target/metal-tests/reference"),
    );
    let interface = buffer_interface("add42", 4, 4);
    check_add42(&library, &interface.validate().unwrap());
}

#[test]
#[ignore = "requires Apple GPU and the stock Rust producer in .#rust-fixtures"]
fn stock_rust_buffer_kernel_executes_on_metal() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let directory = root.join("target/metal-tests/rust-buffer");
    fs::create_dir_all(&directory).unwrap();
    let bc = directory.join("kernel.bc");
    run(
        "rustc",
        &[
            &root.join("tests/rust-fixtures/buffer.rs"),
            Path::new("--edition=2024"),
            Path::new("--crate-type=rlib"),
            Path::new("--target=nvptx64-nvidia-cuda"),
            Path::new("-Copt-level=3"),
            Path::new("--emit=llvm-bc"),
            Path::new("-o"),
            &bc,
        ],
    );
    let context = inkwell::context::Context::create();
    let module =
        llvm_metal_compiler::parse_bitcode(&context, &fs::read(bc).unwrap(), "rust-buffer")
            .unwrap();
    let interface = llvm_metal_abi::KernelInterface {
        schema: 1,
        entry: "buffer_add42".into(),
        calling_convention: "C".into(),
        invocations: Some(1),
        dispatch: llvm_metal_abi::Dispatch::Single,
        aliasing: "disjoint buffers".into(),
        arguments: vec![
            llvm_metal_abi::BufferArgument {
                name: "input".into(),
                kind: "buffer".into(),
                access: llvm_metal_abi::Access::Read,
                bytes: 4,
                alignment: 4,
            },
            llvm_metal_abi::BufferArgument {
                name: "output".into(),
                kind: "buffer".into(),
                access: llvm_metal_abi::Access::Write,
                bytes: 4,
                alignment: 4,
            },
        ],
    };
    let (air, bindings) = llvm_metal_compiler::air::legalize(&module, &interface).unwrap();
    assert_eq!(bindings.buffers[0].index, 0);
    assert_eq!(bindings.buffers[1].index, 1);
    let source = directory.join("air.ll");
    air.print_to_file(&source).unwrap();
    let library = package_air(&source, &bindings.entry, &directory);
    check_add42(&library, &bindings);
}

fn buffer_interface(
    entry: &str,
    bytes: usize,
    alignment: usize,
) -> llvm_metal_abi::KernelInterface {
    use llvm_metal_abi::*;
    KernelInterface {
        schema: 1,
        entry: entry.into(),
        calling_convention: "C".into(),
        invocations: Some(1),
        dispatch: llvm_metal_abi::Dispatch::Single,
        aliasing: "disjoint".into(),
        arguments: vec![
            BufferArgument {
                name: "input".into(),
                kind: "buffer".into(),
                access: Access::Read,
                bytes,
                alignment,
            },
            BufferArgument {
                name: "output".into(),
                kind: "buffer".into(),
                access: Access::Write,
                bytes,
                alignment,
            },
        ],
    }
}

#[test]
#[ignore = "requires Apple GPU and the pinned stock Rust producer in .#rust-fixtures"]
fn pinned_shallenge_sha256_executes_on_metal() {
    use sha2::{Digest, Sha256};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    run("python3", &[&root.join("tests/rust-fixtures/build.py")]);
    let source = root.join("target/rust-fixtures/shallenge");
    let context = inkwell::context::Context::create();
    let module = llvm_metal_compiler::parse_bitcode(
        &context,
        &fs::read(source.join("kernel.bc")).unwrap(),
        "shallenge",
    )
    .unwrap();
    let interface =
        serde_json::from_slice(&fs::read(source.join("kernel.interface.json")).unwrap()).unwrap();
    let compiled = llvm_metal_compiler::compile::compile(&module, &interface).unwrap();
    let output = root.join("target/metal-tests/shallenge");
    fs::create_dir_all(&output).unwrap();
    fs::write(output.join("kernel.air.ll"), &compiled.air_ir).unwrap();
    fs::write(output.join("kernel.air.bc"), &compiled.air_bitcode).unwrap();
    fs::write(
        output.join("kernel.bindings.json"),
        serde_json::to_vec_pretty(&compiled.bindings).unwrap(),
    )
    .unwrap();
    let library = output.join("kernel.metallib");
    fs::write(&library, compiled.metallib).unwrap();
    let kernel = Kernel::load(&library, &compiled.bindings).unwrap();
    let mut cases = vec![[0; 32], [0xff; 32], core::array::from_fn(|i| i as u8)];
    let mut random = 0x123456789abcdef0_u64;
    for _ in 0..64 {
        cases.push(core::array::from_fn(|_| {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random as u8
        }));
    }
    for input in cases {
        let mut bytes = vec![0x5a; 544];
        bytes[256..288].copy_from_slice(&input);
        let original = bytes.clone();
        let mut buffers = [
            Buffer { bytes, offset: 256 },
            Buffer {
                bytes: vec![0xa5; 544],
                offset: 256,
            },
        ];
        // SAFETY: reviewed fixed-length SHA wrapper; separate 32-byte ranges.
        unsafe {
            kernel.run(&mut buffers, 1, 1).unwrap();
        }
        let mut expected = vec![0xa5; 544];
        expected[256..288].copy_from_slice(&Sha256::digest(input));
        assert_eq!(buffers[0].bytes, original);
        assert_eq!(buffers[1].bytes, expected, "input={input:02x?}");
    }
    eprintln!(
        "67 fixed-length SHA cases and buffer guards passed on {}",
        kernel.device_name()
    );
}

fn check_public_fixture(entry: &str, cases: Vec<Vec<u8>>) {
    check_fixture(entry, cases, None);
}

fn check_fixture(entry: &str, cases: Vec<Vec<u8>>, independent: Option<Vec<Vec<u8>>>) {
    use std::io::Write;
    use std::process::Stdio;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let script = root.join("tests/rust-fixtures/build.py");
    let mut args = vec![script.as_path(), Path::new("--entry"), Path::new(entry)];
    let consumer = std::env::var_os("LLVM_METAL_CONSUMER_PATH").map(PathBuf::from);
    if entry.starts_with("sha_") {
        args.extend([
            Path::new("--consumer-path"),
            consumer
                .as_deref()
                .expect("set LLVM_METAL_CONSUMER_PATH for unpublished SHA probes"),
        ]);
    }
    run("python3", &args);
    let source = root.join("target/rust-fixtures/shallenge").join(entry);
    let context = inkwell::context::Context::create();
    let module = llvm_metal_compiler::parse_bitcode(
        &context,
        &fs::read(source.join("kernel.bc")).unwrap(),
        entry,
    )
    .unwrap();
    let interface =
        serde_json::from_slice(&fs::read(source.join("kernel.interface.json")).unwrap()).unwrap();
    let compiled = llvm_metal_compiler::compile::compile(&module, &interface).unwrap();
    let output = root.join("target/metal-tests").join(entry);
    fs::create_dir_all(&output).unwrap();
    fs::write(output.join("kernel.air.ll"), &compiled.air_ir).unwrap();
    fs::write(output.join("kernel.air.bc"), &compiled.air_bitcode).unwrap();
    fs::write(
        output.join("kernel.bindings.json"),
        serde_json::to_vec_pretty(&compiled.bindings).unwrap(),
    )
    .unwrap();
    let library = output.join("kernel.metallib");
    fs::write(&library, compiled.metallib).unwrap();
    let kernel = Kernel::load(&library, &compiled.bindings).unwrap();
    for (case, input) in cases.iter().enumerate() {
        let mut oracle =
            Command::new(root.join("target/rust-fixtures/shallenge/cargo-host/release/oracle"))
                .arg(entry)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
        oracle.stdin.take().unwrap().write_all(input).unwrap();
        let result = oracle.wait_with_output().unwrap();
        assert!(result.status.success());
        if let Some(expected) = &independent {
            assert_eq!(
                result.stdout, expected[case],
                "independent CPU answer for {entry}/{case}"
            );
        }
        let mut bytes = vec![0x5a; input.len() + 512];
        bytes[256..256 + input.len()].copy_from_slice(input);
        let original = bytes.clone();
        let mut expected = vec![0xa5; result.stdout.len() + 512];
        expected[256..256 + result.stdout.len()].copy_from_slice(&result.stdout);
        let mut buffers = [
            Buffer { bytes, offset: 256 },
            Buffer {
                bytes: vec![0xa5; expected.len()],
                offset: 256,
            },
        ];
        // SAFETY: fixed reviewed wrapper contracts, exact buffer sizes, one invocation.
        unsafe {
            kernel.run(&mut buffers, 1, 1).unwrap();
        }
        assert_eq!(buffers[0].bytes, original);
        assert_eq!(buffers[1].bytes, expected, "{entry}, input={input:02x?}");
    }
    eprintln!(
        "{entry}: {} CPU/GPU comparisons and guards passed on {}",
        cases.len(),
        kernel.device_name()
    );
}

#[test]
#[ignore = "requires Apple GPU and .#rust-fixtures"]
fn production_nonce_and_comparison_execute_on_metal() {
    let mut nonce_inputs = Vec::new();
    for lane in [0_u64, 1, 7, u32::MAX as u64, u64::MAX] {
        for seed in [0_u64, 12345, u64::MAX] {
            nonce_inputs.push([lane.to_le_bytes(), seed.to_le_bytes()].concat());
        }
    }
    check_public_fixture("shallenge_nonce", nonce_inputs);
    let mut comparisons = vec![vec![0; 64], vec![0xff; 64]];
    for index in [0, 15, 31] {
        for direction in [0, 32] {
            let mut input = vec![0; 64];
            input[index + direction] = 1;
            comparisons.push(input);
        }
    }
    check_public_fixture("shallenge_compare", comparisons);
}

#[test]
#[ignore = "requires Apple GPU, .#rust-fixtures and LLVM_METAL_CONSUMER_PATH with compiler-probes"]
fn local_sha_intermediates_execute_on_metal() {
    use sha2::{Digest, Sha256};
    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
    fn small0(x: u32) -> u32 {
        x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
    }
    fn small1(x: u32) -> u32 {
        x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
    }
    fn big0(x: u32) -> u32 {
        x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
    }
    fn big1(x: u32) -> u32 {
        x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
    }
    // Independent expressions from the SHA-256 definition; the device calls
    // vanity-miner's helpers. Compression additionally uses RustCrypto's oracle.
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for x in [0_u32, 1, u32::MAX, 0x80000000, 0x12345678] {
        for y in [0_u32, 1, 31, 32, 63, u32::MAX] {
            let z = 0x87654321;
            inputs.push(bytes(&[x, y, z]));
            expected.push(bytes(&[
                small0(x),
                small1(x),
                big0(x),
                big1(x),
                (x & y) | (!x & z),
                (x & y) | (x & z) | (y & z),
                x.swap_bytes(),
                x.rotate_right(y),
            ]));
        }
    }
    check_fixture("sha_ops", inputs, Some(expected));
    let words = [
        [0; 4],
        [u32::MAX; 4],
        [1, 2, 3, 4],
        [0x80000000, 0x12345678, 0x87654321, u32::MAX],
    ];
    check_fixture(
        "sha_schedule",
        words.iter().map(|w| bytes(w)).collect(),
        Some(
            words
                .iter()
                .map(|w| {
                    bytes(&[small1(w[0])
                        .wrapping_add(w[1])
                        .wrapping_add(small0(w[2]))
                        .wrapping_add(w[3])])
                })
                .collect(),
        ),
    );
    let initial = [
        0x6a09e667_u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for state in [[0; 8], [u32::MAX; 8], initial] {
        for word in [0, u32::MAX, 0x61626380] {
            let constant = 0x428a2f98;
            inputs.push(bytes(&[state.as_slice(), &[word, constant]].concat()));
            let [a, b, c, d, e, f, g, h] = state;
            let t1 = [h, big1(e), (e & f) | (!e & g), constant, word]
                .into_iter()
                .fold(0_u32, u32::wrapping_add);
            let t2 = big0(a).wrapping_add((a & b) | (a & c) | (b & c));
            expected.push(bytes(&[
                t1.wrapping_add(t2),
                a,
                b,
                c,
                d.wrapping_add(t1),
                e,
                f,
                g,
            ]));
        }
    }
    check_fixture("sha_round", inputs, Some(expected));
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for message in [
        vec![],
        b"abc".to_vec(),
        vec![0; 32],
        vec![255; 32],
        (0..55).collect(),
    ] {
        let mut padded = [0_u8; 64];
        padded[..message.len()].copy_from_slice(&message);
        padded[message.len()] = 0x80;
        padded[56..].copy_from_slice(&((message.len() as u64) * 8).to_be_bytes());
        let mut words: Vec<_> = padded
            .chunks_exact(4)
            .map(|b| u32::from_be_bytes(b.try_into().unwrap()))
            .collect();
        words.extend(initial);
        inputs.push(bytes(&words));
        expected.push(bytes(
            &Sha256::digest(&message)
                .chunks_exact(4)
                .map(|b| u32::from_be_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(),
        ));
    }
    check_fixture("sha_compress", inputs, Some(expected));
}

#[test]
#[ignore = "requires Apple GPU and .#rust-fixtures"]
fn complete_shallenge_candidate_executes_on_metal() {
    let mut cases = Vec::new();
    for length in [0_u8, 1, 10, 30, 31, 255] {
        for target in [0_u8, 255] {
            for lane in [0_u64, 7] {
                let mut input = vec![0; 79];
                input[..8].copy_from_slice(&lane.to_le_bytes());
                input[8..16].copy_from_slice(&12345_u64.to_le_bytes());
                input[16] = length;
                input[17..47].fill(b'a');
                input[47..].fill(target);
                cases.push(input);
            }
        }
    }
    check_public_fixture("shallenge_candidate", cases);
}

#[test]
#[ignore = "requires Apple GPU and .#rust-fixtures"]
fn reduced_nonce_operations_execute_on_metal() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let directory = root.join("target/metal-tests/memory");
    fs::create_dir_all(&directory).unwrap();
    let bc = directory.join("kernel.bc");
    run(
        "rustc",
        &[
            &root.join("tests/rust-fixtures/memory.rs"),
            Path::new("--edition=2024"),
            Path::new("--crate-type=rlib"),
            Path::new("--target=nvptx64-nvidia-cuda"),
            Path::new("-Copt-level=3"),
            Path::new("--emit=llvm-bc"),
            Path::new("-o"),
            &bc,
        ],
    );
    let context = inkwell::context::Context::create();
    let module =
        llvm_metal_compiler::parse_bitcode(&context, &fs::read(bc).unwrap(), "memory").unwrap();
    let cases = [
        (
            "rotate64",
            0x123456789abcdef0_u64.to_le_bytes().to_vec(),
            0x123456789abcdef0_u64
                .wrapping_mul(5)
                .rotate_left(24)
                .to_le_bytes()
                .to_vec(),
        ),
        ("lookup", vec![63], vec![b'/']),
        (
            "copy21",
            (0..21).collect::<Vec<u8>>(),
            (0..21).collect::<Vec<u8>>(),
        ),
        ("fill21", vec![0x37], vec![0x37; 21]),
    ];
    for (entry, input, expected) in cases {
        let mut interface = buffer_interface(entry, input.len(), 1);
        interface.arguments[1].bytes = expected.len();
        let compiled = llvm_metal_compiler::compile::compile(&module, &interface).unwrap();
        fs::write(directory.join(format!("{entry}.air.ll")), &compiled.air_ir).unwrap();
        let library = directory.join(format!("{entry}.metallib"));
        fs::write(&library, compiled.metallib).unwrap();
        let kernel =
            Kernel::load(&library, &compiled.bindings).unwrap_or_else(|e| panic!("{entry}: {e}"));
        let mut buffers = [
            Buffer {
                bytes: input.clone(),
                offset: 0,
            },
            Buffer {
                bytes: vec![0xa5; expected.len()],
                offset: 0,
            },
        ];
        unsafe {
            kernel.run(&mut buffers, 1, 1).unwrap();
        }
        assert_eq!(buffers[0].bytes, input);
        assert_eq!(buffers[1].bytes, expected, "{entry}");
        eprintln!("{entry}: passed on {}", kernel.device_name());
    }
}

#[test]
#[ignore = "requires Apple GPU and .#rust-fixtures"]
fn device_atomic_tickets_cover_partial_and_multiple_threadgroups() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let directory = root.join("target/metal-tests/atomic");
    fs::create_dir_all(&directory).unwrap();
    let bc = directory.join("kernel.bc");
    run(
        "rustc",
        &[
            &root.join("tests/rust-fixtures/dispatch.rs"),
            Path::new("--edition=2024"),
            Path::new("--crate-type=rlib"),
            Path::new("--target=nvptx64-nvidia-cuda"),
            Path::new("-Copt-level=3"),
            Path::new("--emit=llvm-bc"),
            Path::new("-o"),
            &bc,
        ],
    );
    let context = inkwell::context::Context::create();
    let module =
        llvm_metal_compiler::parse_bitcode(&context, &fs::read(bc).unwrap(), "atomic").unwrap();
    let mut interface = buffer_interface("atomic_tickets", 4, 4);
    interface.dispatch = llvm_metal_abi::Dispatch::Grid1d;
    interface.invocations = None;
    interface.arguments.push(llvm_metal_abi::BufferArgument {
        name: "counter".into(),
        kind: "buffer".into(),
        access: llvm_metal_abi::Access::ReadWrite,
        bytes: 4,
        alignment: 4,
    });
    let compiled = llvm_metal_compiler::compile::compile(&module, &interface).unwrap();
    fs::write(directory.join("kernel.air.ll"), compiled.air_ir).unwrap();
    let library = directory.join("kernel.metallib");
    fs::write(&library, compiled.metallib).unwrap();
    let kernel = Kernel::load(&library, &compiled.bindings).unwrap();
    for count in [0_u32, 1, 31, 32, 33, 64, 257] {
        for base in [0_u32, u32::MAX - 128] {
            let size = count.max(1) as usize * 4;
            let mut input = vec![0x5a; 516];
            input[256..260].copy_from_slice(&count.to_le_bytes());
            let original = input.clone();
            let mut counter = vec![0x5a; 516];
            counter[256..260].copy_from_slice(&base.to_le_bytes());
            let mut expected_counter = counter.clone();
            expected_counter[256..260].copy_from_slice(&base.wrapping_add(count).to_le_bytes());
            let mut buffers = [
                Buffer {
                    bytes: input,
                    offset: 256,
                },
                Buffer {
                    bytes: vec![0xa5; size + 512],
                    offset: 256,
                },
                Buffer {
                    bytes: counter,
                    offset: 256,
                },
            ];
            // SAFETY: bounds-checked reviewed kernel, enough output slots, only atomic counter access.
            unsafe {
                kernel.run(&mut buffers, count as usize + 7, 32).unwrap();
            }
            assert_eq!(buffers[0].bytes, original);
            assert_eq!(buffers[2].bytes, expected_counter);
            assert!(
                buffers[1].bytes[..256]
                    .iter()
                    .chain(buffers[1].bytes[256 + count as usize * 4..].iter())
                    .all(|&x| x == 0xa5)
            );
            let mut tickets: Vec<u32> = buffers[1].bytes[256..256 + count as usize * 4]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()).wrapping_sub(base))
                .collect();
            tickets.sort();
            assert_eq!(tickets, (0..count).collect::<Vec<_>>());
        }
    }
}

#[test]
#[ignore = "requires Apple GPU and .#rust-fixtures"]
fn shallenge_batches_publish_complete_cpu_checked_results() {
    use std::{io::Write, process::Stdio};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    run(
        "python3",
        &[
            &root.join("tests/rust-fixtures/build.py"),
            Path::new("--entry"),
            Path::new("shallenge_batch"),
        ],
    );
    let source = root.join("target/rust-fixtures/shallenge/shallenge_batch");
    let context = inkwell::context::Context::create();
    let module = llvm_metal_compiler::parse_bitcode(
        &context,
        &fs::read(source.join("kernel.bc")).unwrap(),
        "batch",
    )
    .unwrap();
    let interface =
        serde_json::from_slice(&fs::read(source.join("kernel.interface.json")).unwrap()).unwrap();
    let compiled = llvm_metal_compiler::compile::compile(&module, &interface).unwrap();
    let output = root.join("target/metal-tests/shallenge_batch");
    fs::create_dir_all(&output).unwrap();
    fs::write(output.join("kernel.air.ll"), &compiled.air_ir).unwrap();
    fs::write(output.join("kernel.air.bc"), &compiled.air_bitcode).unwrap();
    fs::write(
        output.join("kernel.bindings.json"),
        serde_json::to_vec_pretty(&compiled.bindings).unwrap(),
    )
    .unwrap();
    let library = output.join("kernel.metallib");
    fs::write(&library, compiled.metallib).unwrap();
    let kernel = Kernel::load(&library, &compiled.bindings).unwrap();
    for count in [0_u32, 1, 31, 32, 33, 64, 65, 129, 257] {
        let mut input = count.to_le_bytes().to_vec();
        for lane in 0..count {
            let mut request = [0; 79];
            request[..8].copy_from_slice(&(lane as u64).to_le_bytes());
            request[8..16].copy_from_slice(&12345_u64.to_le_bytes());
            request[16] = if lane % 11 == 0 {
                0
            } else {
                [1, 10, 30][lane as usize % 3]
            };
            request[17..47].fill(b'a');
            request[47..].fill(if lane % 3 == 0 { 0 } else { 255 });
            input.extend_from_slice(&request);
        }
        let mut oracle =
            Command::new(root.join("target/rust-fixtures/shallenge/cargo-host/release/oracle"))
                .arg("shallenge_batch")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
        // Drain stdout concurrently with stdin for batches larger than a pipe.
        let mut stdin = oracle.stdin.take().unwrap();
        let sent = input.clone();
        let writer = std::thread::spawn(move || stdin.write_all(&sent).unwrap());
        let expected = oracle.wait_with_output().unwrap();
        writer.join().unwrap();
        assert!(expected.status.success());
        let records = expected.stdout;
        let valid = records.chunks_exact(65).filter(|r| r[64] != 0).count() as u32;
        let expected_winners: Vec<u32> = records
            .chunks_exact(65)
            .enumerate()
            .filter(|(_, r)| r[63] != 0 && r[64] != 0)
            .map(|(i, _)| i as u32)
            .collect();
        let guarded = |data: &[u8], size: usize, fill: u8| {
            let mut bytes = vec![fill; size + 512];
            bytes[256..256 + data.len()].copy_from_slice(data);
            Buffer { bytes, offset: 256 }
        };
        let mut buffers = [
            guarded(&input, input.len(), 0x5a),
            guarded(&[], records.len().max(1), 0xa5),
            guarded(&[0; 8], 8, 0x5a),
            guarded(&[], count.max(1) as usize * 4, 0xa5),
        ];
        let original = buffers[0].bytes.clone();
        // SAFETY: each candidate gets disjoint ranges, counts start at zero,
        // winner capacity >= count, and CPU reads follow command completion.
        unsafe {
            kernel.run(&mut buffers, count as usize + 7, 32).unwrap();
        }
        assert_eq!(buffers[0].bytes, original);
        assert_eq!(
            buffers[1].bytes,
            guarded(&records, records.len().max(1), 0xa5).bytes,
            "count={count}"
        );
        let counters = [
            valid.to_le_bytes(),
            (expected_winners.len() as u32).to_le_bytes(),
        ]
        .concat();
        assert_eq!(
            buffers[2].bytes,
            guarded(&counters, 8, 0x5a).bytes,
            "count={count}"
        );
        let used = expected_winners.len() * 4;
        let mut winners: Vec<u32> = buffers[3].bytes[256..256 + used]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        winners.sort();
        assert_eq!(winners, expected_winners, "count={count}");
        assert!(
            buffers[3].bytes[..256]
                .iter()
                .chain(buffers[3].bytes[256 + used..].iter())
                .all(|&b| b == 0xa5)
        );
    }
    eprintln!(
        "9 Shallenge batch sizes, exact results, counters, winner sets and guards passed on {}",
        kernel.device_name()
    );
}

#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn prepared_buffers_reuse_storage_and_reject_changed_shapes() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = package_air(
        &root.join("tests/fixtures/air/add42.ll"),
        "add42",
        &root.join("target/metal-tests/prepared"),
    );
    let bindings = buffer_interface("add42", 4, 4).validate().unwrap();
    let kernel = Kernel::load(&library, &bindings).unwrap();
    let mut buffers = [
        Buffer {
            bytes: vec![0; 4],
            offset: 0,
        },
        Buffer {
            bytes: vec![0; 4],
            offset: 0,
        },
    ];
    let mut prepared = kernel.prepare(&buffers).unwrap();
    for value in [0_u32, 7, u32::MAX, 42] {
        buffers[0].bytes.copy_from_slice(&value.to_le_bytes());
        // SAFETY: the reference entry accesses one disjoint u32 per buffer.
        let timings = unsafe { prepared.run(&mut buffers, 1, 1).unwrap() };
        assert_eq!(buffers[1].bytes, value.wrapping_add(42).to_le_bytes());
        assert!(timings.wall >= timings.upload + timings.download);
    }
    buffers[1].bytes.push(0);
    // SAFETY: the shape check rejects this before any dispatch.
    assert!(
        unsafe { prepared.run(&mut buffers, 1, 1) }
            .unwrap_err()
            .contains("stay fixed")
    );
    buffers[1].bytes.pop();
    // SAFETY: these dispatch contracts are rejected before executing.
    assert!(unsafe { prepared.run(&mut buffers, 2, 1) }.is_err());
    assert!(unsafe { prepared.run(&mut buffers, 1, 0) }.is_err());
}
