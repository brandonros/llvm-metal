use std::{
    path::PathBuf,
    process::{Command, Output},
};

fn fixture(category: &str, name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(category)
        .join(name)
        .to_string_lossy()
        .into_owned()
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_llvm-metalc"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn inspect_reports_definitions_and_declarations_without_claiming_metal_support() {
    let output = run(&[
        "inspect",
        &fixture("positive", "02-rotate.ll"),
        "--entry",
        "fixture",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("definition: fixture"));
    assert!(stdout.contains("declaration: llvm.fshl.i64"));
    assert!(stdout.contains("Metal legality and GPU execution have not been checked"));
}

#[test]
fn inspect_accepts_a_generated_bitcode_file() {
    let context = inkwell::context::Context::create();
    let source = std::fs::read(fixture("positive", "01-wrapping-add.ll")).unwrap();
    let module = llvm_metal_compiler::parse_ir(&context, &source, "addition.ll").unwrap();
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join(format!("cli-input-{}.bc", std::process::id()));
    std::fs::write(&path, module.write_bitcode_to_memory().as_slice()).unwrap();
    let output = run(&["inspect", path.to_str().unwrap(), "--entry", "fixture"]);
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("definition: fixture"));
}

#[test]
fn invalid_ir_fails_with_the_verification_stage() {
    let output = run(&[
        "inspect",
        &fixture("negative", "dominance.ll"),
        "--entry",
        "fixture",
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("LLVM verification error"));
}

#[test]
fn missing_input_and_entry_fail_without_success_output() {
    for args in [
        vec![
            "inspect".to_owned(),
            fixture("positive", "absent.ll"),
            "--entry".to_owned(),
            "fixture".to_owned(),
        ],
        vec![
            "inspect".to_owned(),
            fixture("positive", "01-wrapping-add.ll"),
            "--entry".to_owned(),
            "absent".to_owned(),
        ],
    ] {
        let arguments: Vec<_> = args.iter().map(String::as_str).collect();
        let output = run(&arguments);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn usage_errors_are_distinct_from_help() {
    assert!(run(&["--help"]).status.success());
    for args in [vec![], vec!["compile"], vec!["inspect", "file.ll"]] {
        assert_eq!(run(&args).status.code(), Some(2));
    }
}
