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
#[ignore = "requires Apple GPU and the stock Rust producer in the Nix shell"]
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
#[ignore = "requires Apple GPU and the Nix shell"]
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
#[ignore = "requires Apple GPU and the Nix shell"]
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
    // A changed shape can be explicitly reconfigured without recompiling the pipeline.
    assert!(prepared.reconfigure(&buffers).unwrap());
    assert!(!prepared.reconfigure(&buffers).unwrap());
    prepared.clear();
    // Clearing the retained allocations must not change host mirrors or prevent reuse.
    let before = buffers[0].bytes.clone();
    unsafe {
        prepared.run(&mut buffers, 1, 1).unwrap();
    }
    assert_eq!(buffers[0].bytes, before);
    assert_eq!(&buffers[1].bytes[..4], &84_u32.to_le_bytes());
    let offset = buffers[0].offset;
    buffers[0].offset = buffers[0].bytes.len();
    assert!(prepared.reconfigure(&buffers).is_err());
    buffers[0].offset = offset;
    assert!(!prepared.reconfigure(&buffers).unwrap());
    assert!(unsafe { prepared.run(&mut buffers, 2, 1) }.is_err());
    assert!(unsafe { prepared.run(&mut buffers, 1, 0) }.is_err());
}

#[test]
#[ignore = "requires Apple GPU and the Nix shell"]
fn local_bytes_read_back_as_a_word_execute_on_metal() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let directory = root.join("target/metal-tests/punned-local");
    fs::create_dir_all(&directory).unwrap();
    let context = inkwell::context::Context::create();
    let source = fs::read(root.join("tests/fixtures/gpu/punned-local.ll")).unwrap();
    let module = llvm_metal_compiler::parse_ir(&context, &source, "punned").unwrap();
    let compiled =
        llvm_metal_compiler::compile::compile(&module, &buffer_interface("punned", 8, 1)).unwrap();
    fs::write(directory.join("kernel.air.ll"), &compiled.air_ir).unwrap();
    let library = directory.join("kernel.metallib");
    fs::write(&library, compiled.metallib).unwrap();
    let kernel = Kernel::load(&library, &compiled.bindings).unwrap();
    let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut buffers = [
        Buffer {
            bytes: input.clone(),
            offset: 0,
        },
        Buffer {
            bytes: vec![0xa5; 8],
            offset: 0,
        },
    ];
    unsafe {
        kernel.run(&mut buffers, 1, 1).unwrap();
    }
    assert_eq!(buffers[1].bytes, input);
}
