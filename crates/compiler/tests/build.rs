use inkwell::{context::Context, module::Linkage};
use llvm_metal_compiler::{build::llvm, parse_ir};
use std::{fs, path::PathBuf, process::Command};

const MODULE: &str = r#"
target triple = "nvptx64-nvidia-cuda"
@kept = global i32 0
@table = global i32 1
@hidden = hidden global i32 2
@external = external global i32
@llvm.used = appending global [1 x ptr] [ptr @kept], section "llvm.metadata"
@alias = alias i32, ptr @table
declare void @llvm_metal.linear_thread_index()
declare void @missing_runtime()
declare void @unused_declaration()
declare i32 @llvm.ctpop.i32(i32)
define void @entry() {
  call void @llvm_metal.linear_thread_index()
  call void @missing_runtime()
  %x = load i32, ptr @external
  %y = call i32 @llvm.ctpop.i32(i32 %x)
  ret void
}
define void @other_entry() { ret void }
define internal void @already_local() { ret void }
"#;

#[test]
fn internalize_keeps_only_the_entry_and_what_llvm_would() {
    let context = Context::create();
    let module = parse_ir(&context, MODULE.as_bytes(), "module").unwrap();
    llvm::internalize(&module, "entry").unwrap();
    let function = |name| module.get_function(name).unwrap().get_linkage();
    let global = |name| module.get_global(name).unwrap().get_linkage();
    assert_eq!(function("entry"), Linkage::External);
    assert_eq!(function("other_entry"), Linkage::Internal);
    assert_eq!(function("already_local"), Linkage::Internal);
    // Declarations, llvm.used and its members stay visible; the rest do not.
    assert_eq!(function("missing_runtime"), Linkage::External);
    assert_eq!(global("external"), Linkage::External);
    assert_eq!(global("llvm.used"), Linkage::Appending);
    assert_eq!(global("kept"), Linkage::External);
    assert_eq!(global("table"), Linkage::Internal);
    assert_eq!(global("hidden"), Linkage::Internal);
    let text = module.print_to_string().to_string();
    assert!(text.contains("@hidden = internal global"), "{text}");
    assert!(text.contains("@alias = internal alias"), "{text}");
    module.verify().unwrap();
    // The same selection as LLVM's pass with the entry as its public API.
    let reference = parse_ir(&context, MODULE.as_bytes(), "module").unwrap();
    llvm::set_options(&["-internalize-public-api-list=entry"]);
    llvm::passes(&reference, "internalize").unwrap();
    assert_eq!(text, reference.print_to_string().to_string());
}

#[test]
fn undefined_lists_symbols_and_not_llvm_intrinsics() {
    let context = Context::create();
    let module = parse_ir(&context, MODULE.as_bytes(), "module").unwrap();
    let mut undefined = llvm::undefined(&module);
    undefined.sort();
    assert_eq!(
        undefined,
        [
            "external",
            "llvm_metal.linear_thread_index",
            "missing_runtime",
            "unused_declaration"
        ]
    );
}

fn metalc(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_llvm-metalc"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn build_rejects_ambiguous_or_missing_inputs() {
    for args in [
        &["build", "--output", "unused"][..],
        &[
            "build", "--crate", ".", "--rlib", "x.rlib", "--output", "unused",
        ],
        &["build", "--rlib", "x.rlib"],
        &[
            "build",
            "--rlib",
            "x.rlib",
            "--output",
            "unused",
            "--unknown",
            "flag",
        ],
        &["build-unit"],
    ] {
        let output = metalc(args);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).starts_with("build failed: "),
            "{args:?}"
        );
    }
}

#[test]
#[ignore = "requires the stock Rust NVPTX producer and llvm-downgrade in the Nix shell"]
fn builds_a_stock_rust_kernel_reproducibly() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let directory = root.join("target/metal-tests/build");
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    let rustc = |source: PathBuf, name: &str, externs: &[String]| {
        let archive = directory.join(format!("lib{name}.rlib"));
        let output = Command::new("rustc")
            .arg(source)
            .args([
                "--crate-type=rlib",
                "--edition=2024",
                "--target=nvptx64-nvidia-cuda",
            ])
            .args([
                "-Copt-level=3",
                "-Cno-vectorize-loops",
                "-Cno-vectorize-slp",
            ])
            .arg(format!("--crate-name={name}"))
            .args(externs)
            .arg("-o")
            .arg(&archive)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        archive.display().to_string()
    };
    let kernel = rustc(
        root.join("crates/kernel/src/lib.rs"),
        "llvm_metal_kernel",
        &[],
    );
    let fixture = rustc(
        root.join("tests/rust-fixtures/descriptor.rs"),
        "fixture",
        &["--extern".into(), format!("llvm_metal_kernel={kernel}")],
    );
    let build = |name: &str, policy: &str| {
        let output = directory.join(name);
        let result = metalc(&[
            "build",
            "--rlib",
            &kernel,
            "--rlib",
            &fixture,
            "--inlining",
            policy,
            "--output",
            output.to_str().unwrap(),
        ]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        output
    };
    let (first, second) = (build("first", "selective"), build("second", "selective"));
    for name in [
        "kernel.bc",
        "kernel.air.ll",
        "kernel.bindings.json",
        "kernel.descriptor.json",
    ] {
        assert_eq!(
            fs::read(first.join(name)).unwrap(),
            fs::read(second.join(name)).unwrap(),
            "{name}"
        );
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(first.join("kernel.build.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], 1);
    assert_eq!(manifest["inlining"], "selective");
    assert!(manifest["artifacts"]["kernel.metallib"].is_string());
    assert!(
        fs::read(first.join("kernel.metallib"))
            .unwrap()
            .starts_with(b"MTLB")
    );
    // Nothing of the staging directory is left behind.
    let left: Vec<_> = fs::read_dir(&first)
        .unwrap()
        .flatten()
        .map(|e| e.file_name())
        .collect();
    assert_eq!(left.len(), 8, "{left:?}");
    let forced = build("forced", "all");
    assert!(
        fs::read(forced.join("kernel.metallib"))
            .unwrap()
            .starts_with(b"MTLB")
    );
    // A second entry name that the module does not declare is refused.
    let refused = metalc(&[
        "build",
        "--rlib",
        &kernel,
        "--rlib",
        &fixture,
        "--entry",
        "absent",
        "--output",
        directory.join("refused").to_str().unwrap(),
    ]);
    assert!(!refused.status.success());
}
