use inkwell::context::Context;
use llvm_metal_compiler::{descriptor::extract, parse_ir};
use llvm_metal_kernel as k;

fn encoded() -> k::Encoded {
    k::encode(
        "kernel",
        k::Dispatch::Single,
        &[k::argument::<u32>(
            "value",
            k::Access::ReadWrite,
            k::Shape::Fixed,
        )],
    )
}
fn source(bytes: &[u8], entry: &str, parameters: &str) -> String {
    let data: String = bytes.iter().map(|b| format!("\\{b:02X}")).collect();
    format!(
        "@__llvm_metal_descriptor_{entry} = constant [{} x i8] c\"{data}\"\n@keep = global i32 3\n@llvm.compiler.used = appending global [2 x ptr] [ptr @__llvm_metal_descriptor_{entry}, ptr @keep], section \"llvm.metadata\"\ndefine void @{entry}({parameters}) {{ ret void }}",
        bytes.len()
    )
}
#[test]
fn extract_preserves_unrelated_retention_and_rejects_wrong_entries() {
    let c = Context::create();
    let e = encoded();
    let m = parse_ir(
        &c,
        source(e.as_bytes(), "kernel", "ptr %p").as_bytes(),
        "valid",
    )
    .unwrap();
    let (stripped, ds) = extract(&m).unwrap();
    assert_eq!(ds.len(), 1);
    assert!(
        stripped
            .get_global("__llvm_metal_descriptor_kernel")
            .is_none()
    );
    assert!(
        stripped
            .print_to_string()
            .to_string()
            .contains("[ptr @keep]")
    );
    assert!(m.get_global("__llvm_metal_descriptor_kernel").is_some());
    for (entry, args) in [("other", "ptr %p"), ("kernel", ""), ("kernel", "i32 %p")] {
        let m = parse_ir(&c, source(e.as_bytes(), entry, args).as_bytes(), "bad").unwrap();
        assert!(extract(&m).is_err());
    }
    let mut bad = e.as_bytes().to_vec();
    bad[4] = 2;
    let m = parse_ir(
        &c,
        source(&bad, "kernel", "ptr %p").as_bytes(),
        "bad version",
    )
    .unwrap();
    assert!(extract(&m).is_err());
    let missing = parse_ir(&c, b"define void @kernel(ptr %p) { ret void }", "missing").unwrap();
    assert!(extract(&missing).is_err());
    let used_source = source(e.as_bytes(), "kernel", "ptr %p").replace(
        "{ ret void }",
        "{ %v = load i8, ptr @__llvm_metal_descriptor_kernel\n store i8 %v, ptr %p\n ret void }",
    );
    let used = parse_ir(&c, used_source.as_bytes(), "executable metadata use").unwrap();
    assert!(extract(&used).is_err());
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires stock Rust NVPTX producer, LLVM tools and Apple GPU in .#rust-fixtures"]
fn stock_rust_descriptor_survives_and_executes() {
    use llvm_metal_runtime::{Buffer, Kernel};
    use std::{fs, path::PathBuf, process::Command};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("target/metal-tests/descriptor");
    fs::create_dir_all(&dir).unwrap();
    let run = |args: Vec<String>| {
        let out = Command::new(&args[0]).args(&args[1..]).output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    let kernel = dir.join("libllvm_metal_kernel.rlib");
    run(vec![
        "rustc".into(),
        root.join("crates/kernel/src/lib.rs").display().to_string(),
        "--crate-name=llvm_metal_kernel".into(),
        "--crate-type=rlib".into(),
        "--edition=2024".into(),
        "--target=nvptx64-nvidia-cuda".into(),
        "-Copt-level=3".into(),
        "-o".into(),
        kernel.display().to_string(),
    ]);
    let bc = dir.join("fixture.bc");
    run(vec![
        "rustc".into(),
        root.join("tests/rust-fixtures/descriptor.rs")
            .display()
            .to_string(),
        "--crate-type=rlib".into(),
        "--edition=2024".into(),
        "--target=nvptx64-nvidia-cuda".into(),
        "-Copt-level=3".into(),
        "-Cno-vectorize-loops".into(),
        "-Cno-vectorize-slp".into(),
        "--extern".into(),
        format!("llvm_metal_kernel={}", kernel.display()),
        "--emit=llvm-bc".into(),
        "-o".into(),
        bc.display().to_string(),
    ]);
    let c = Context::create();
    let m = llvm_metal_compiler::parse_bitcode(&c, &fs::read(&bc).unwrap(), "stock descriptor")
        .unwrap();
    let (_stripped, ds) = extract(&m).unwrap();
    let d = &ds["descriptor_add"];
    // Independent native compilation of the exact declaration, not a numeric table.
    let host = dir.join("host.rs");
    fs::write(&host, format!("#[path={:?}] mod fixture; fn main() {{ use std::io::Write; std::io::stdout().write_all(&fixture::contract::DESCRIPTOR).unwrap(); }}", root.join("tests/rust-fixtures/descriptor.rs"))).unwrap();
    let host_kernel = dir.join("libhost_kernel.rlib");
    run(vec![
        "rustc".into(),
        root.join("crates/kernel/src/lib.rs").display().to_string(),
        "--crate-name=llvm_metal_kernel".into(),
        "--crate-type=rlib".into(),
        "--edition=2024".into(),
        "-o".into(),
        host_kernel.display().to_string(),
    ]);
    let host_bin = dir.join("host");
    run(vec![
        "rustc".into(),
        host.display().to_string(),
        "--edition=2024".into(),
        "--extern".into(),
        format!("llvm_metal_kernel={}", host_kernel.display()),
        "-o".into(),
        host_bin.display().to_string(),
    ]);
    let expected = Command::new(host_bin).output().unwrap();
    assert!(expected.status.success());
    let bindings = d.bindings().unwrap();
    bindings.validate_host_descriptor(&expected.stdout).unwrap();
    let mut wrong = bindings.clone();
    wrong.buffers.swap(1, 2);
    assert!(wrong.validate_host_descriptor(&expected.stdout).is_err());
    let mut wrong = bindings.clone();
    wrong.descriptor.as_mut().unwrap().arguments.swap(1, 2);
    assert!(wrong.validate_host_descriptor(&expected.stdout).is_err());
    // Exercise the public no-interface CLI, including serialized descriptor bindings.
    run(vec![
        env!("CARGO_BIN_EXE_llvm-metalc").into(),
        "compile".into(),
        bc.display().to_string(),
        "--entry".into(),
        "descriptor_add".into(),
        "--output".into(),
        dir.display().to_string(),
    ]);
    assert!(
        !fs::read_to_string(dir.join("kernel.air.ll"))
            .unwrap()
            .contains(k::PREFIX)
    );
    let emitted: llvm_metal_abi::MetalBindings =
        serde_json::from_slice(&fs::read(dir.join("kernel.bindings.json")).unwrap()).unwrap();
    assert_eq!(emitted, bindings);
    emitted.validate_host_descriptor(&expected.stdout).unwrap();
    let library = dir.join("kernel.metallib");
    let kernel = Kernel::load(&library, &bindings).unwrap();
    for count in [0usize, 1, 7, 33] {
        let add = 0xfffffffeu32;
        let guard = |n| Buffer {
            bytes: vec![0xa5; 512 + n],
            offset: 256,
        };
        let mut buffers = [guard(8), guard(count.max(1) * 4), guard(count.max(1) * 4)];
        buffers[0].bytes[256..260].copy_from_slice(&(count as u32).to_le_bytes());
        buffers[0].bytes[260..264].copy_from_slice(&add.to_le_bytes());
        for i in 0..count {
            buffers[1].bytes[256 + i * 4..260 + i * 4].copy_from_slice(&(i as u32).to_le_bytes());
        }
        let originals: Vec<_> = buffers.iter().map(|b| b.bytes.clone()).collect();
        d.validate_lengths(&[8, count.max(1) * 4, count.max(1) * 4], &[1, count, count])
            .unwrap();
        // SAFETY: validated disjoint buffers for the reviewed fixture's count.
        unsafe {
            kernel.run(&mut buffers, 1, 1).unwrap();
        }
        assert_eq!(buffers[0].bytes, originals[0]);
        assert_eq!(buffers[1].bytes, originals[1]);
        let mut expected = originals[2].clone();
        for i in 0..count {
            expected[256 + i * 4..260 + i * 4]
                .copy_from_slice(&(i as u32).wrapping_add(add).to_le_bytes());
        }
        assert_eq!(buffers[2].bytes, expected);
    }
}
