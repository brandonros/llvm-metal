use inkwell::context::Context;
use llvm_metal_compiler::{InputError, parse_bitcode, parse_ir, require_entry};
use std::{fs, path::PathBuf};

mod support;

#[test]
fn every_positive_fixture_verifies_and_round_trips_through_bitcode() {
    let context = Context::create();
    for name in support::POSITIVE_FIXTURES {
        let source = support::fixture(name);
        let module = parse_ir(&context, &source, name).unwrap();
        require_entry(&module, "fixture").unwrap();
        let original = module.print_to_string().to_string();
        let bitcode = module.write_bitcode_to_memory();
        let decoded = parse_bitcode(&context, bitcode.as_slice(), name).unwrap();
        require_entry(&decoded, "fixture").unwrap();
        assert_eq!(original, decoded.print_to_string().to_string(), "{name}");
    }
}

#[test]
fn all_committed_positive_ir_files_are_registered() {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/positive");
    let mut actual: Vec<_> = fs::read_dir(directory)
        .unwrap()
        .map(|file| file.unwrap().file_name().into_string().unwrap())
        .collect();
    let mut expected: Vec<_> = support::POSITIVE_FIXTURES
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    actual.sort();
    expected.sort();
    assert_eq!(
        actual, expected,
        "new fixtures must be wired into the tests"
    );
}

#[test]
fn syntax_errors_are_parse_errors() {
    let context = Context::create();
    let source = support::negative_fixture("syntax.ll");
    assert!(matches!(
        parse_ir(&context, &source, "syntax.ll"),
        Err(InputError::Parse(_))
    ));
}

#[test]
fn invalid_ssa_is_a_verifier_error() {
    let context = Context::create();
    let source = support::negative_fixture("dominance.ll");
    assert!(matches!(
        parse_ir(&context, &source, "dominance.ll"),
        Err(InputError::Verify(_))
    ));
}

#[test]
fn missing_and_declaration_only_entries_are_rejected() {
    let context = Context::create();
    let module = parse_ir(&context, b"declare void @external()\n", "declaration.ll").unwrap();
    for entry in ["missing", "external"] {
        assert!(matches!(
            require_entry(&module, entry),
            Err(InputError::Entry(_))
        ));
    }
}

#[test]
fn malformed_bitcode_is_rejected() {
    let context = Context::create();
    assert!(matches!(
        parse_bitcode(&context, b"not bitcode", "broken.bc"),
        Err(InputError::Parse(_))
    ));
}
