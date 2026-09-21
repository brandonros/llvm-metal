//! Build a kernel crate for the GPU target and collect its bitcode. On that
//! target rustc's object files are LLVM bitcode, so the crates' rlibs hold
//! everything; libcore is deliberately left out (see `Rule::KnownFunction`).
use std::{fs, io::Read, path::Path, process::Command};

const TARGET: &str = "nvptx64-nvidia-cuda";

/// The bitcode of every crate in the kernel's dependency graph.
pub fn bitcode(manifest: &Path, target_dir: &Path) -> Result<Vec<Vec<u8>>, String> {
    let output = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--lib",
            "--target",
            TARGET,
            "--message-format=json",
        ])
        .arg("--manifest-path")
        .arg(manifest)
        .arg("--target-dir")
        .arg(target_dir)
        // Line tables put a Rust source location on every refusal.
        .env("CARGO_PROFILE_RELEASE_DEBUG", "line-tables-only")
        .output()
        .map_err(|error| format!("cargo: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let mut modules = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let message: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let files = message["filenames"].as_array().into_iter().flatten();
        for archive in files
            .filter_map(|file| file.as_str())
            .filter(|file| file.ends_with(".rlib"))
        {
            let file = fs::File::open(archive).map_err(|error| format!("{archive}: {error}"))?;
            let mut archive = ar::Archive::new(file);
            while let Some(member) = archive.next_entry() {
                let mut bytes = Vec::new();
                member
                    .map_err(|e| e.to_string())?
                    .read_to_end(&mut bytes)
                    .map_err(|e| e.to_string())?;
                if bytes.starts_with(b"BC\xC0\xDE") {
                    modules.push(bytes);
                }
            }
        }
    }
    Ok(modules)
}
