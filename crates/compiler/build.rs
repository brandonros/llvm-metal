use std::{env, path::PathBuf, process::Command};
fn main() {
    let sources = ["native/llvm_ext.cpp"];
    for source in sources {
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-env-changed=LLVM_SYS_211_PREFIX");
    let config = env::var_os("LLVM_SYS_211_PREFIX")
        .map(|p| PathBuf::from(p).join("bin/llvm-config"))
        .unwrap_or_else(|| "llvm-config".into());
    let output = Command::new(config)
        .arg("--includedir")
        .output()
        .expect("LLVM 21 llvm-config");
    assert!(output.status.success());
    let include = String::from_utf8(output.stdout).unwrap();
    cc::Build::new()
        .cpp(true)
        .files(sources)
        .include(include.trim())
        .flag("-std=c++17")
        .flag("-fno-rtti")
        .compile("llvm_metal_passes");
}
