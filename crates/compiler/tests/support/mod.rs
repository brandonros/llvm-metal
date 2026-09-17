// Shared helpers are compiled into multiple integration-test binaries.
#![allow(dead_code)]

use std::{fs, path::PathBuf};

pub const POSITIVE_FIXTURES: &[&str] = &[
    "01-wrapping-add.ll",
    "02-rotate.ll",
    "03-clz.ll",
    "04-add-carry.ll",
    "05-branch-phi.ll",
    "06-loop-phi.ll",
    "07-helper-call.ll",
    "08-pointer-helper.ll",
    "09-byte-copy.ll",
    "10-sha256-sigma0.ll",
    "11-device-index.ll",
    "12-device-atomic.ll",
];

pub fn fixture(name: &str) -> Vec<u8> {
    read("positive", name)
}

pub fn negative_fixture(name: &str) -> Vec<u8> {
    read("negative", name)
}

fn read(category: &str, name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(category)
            .join(name),
    )
    .unwrap()
}
