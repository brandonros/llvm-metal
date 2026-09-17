#!/usr/bin/env python3
"""Build the pinned Shallenge fixture; no PTX, AIR or GPU execution."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/rust-fixtures/shallenge"
OUTPUT = ROOT / "target/rust-fixtures/shallenge"
TARGET = "nvptx64-nvidia-cuda"
COMMANDS = []


def run(*args, capture=False):
    command = [str(arg) for arg in args]
    COMMANDS.append(command)
    return subprocess.run(
        command, cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # External compiler overrides would make this a different fixture producer.
    for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC", "RUSTC_WRAPPER",
                 "RUSTC_WORKSPACE_WRAPPER", "RUSTC_BOOTSTRAP"):
        if os.environ.get(name):
            raise RuntimeError(f"unset {name}; use nix develop .#rust-fixtures")
    rust_version = run("rustc", "-vV", capture=True)
    if "release: 1.93.0\n" not in rust_version or "LLVM version: 21.1.8\n" not in rust_version:
        raise RuntimeError("expected stable Rust 1.93.0 / LLVM 21.1.8; use the rust-fixtures Nix shell")
    llvm_version = run("llvm-link", "--version", capture=True)
    if "LLVM version 21.1.8" not in llvm_version:
        raise RuntimeError("expected LLVM tools 21.1.8")
    interface = json.loads((FIXTURE / "kernel.interface.json").read_text())
    entry = interface["entry"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    common = ["--locked", "--manifest-path", FIXTURE / "Cargo.toml"]
    run("cargo", "test", *common, "--release", "--target-dir", OUTPUT / "cargo-host")
    messages = run(
        "cargo", "rustc", *common, "--release", "--lib", "--target", TARGET,
        "--target-dir", OUTPUT / "cargo-device", "--message-format=json",
        "--", "--emit=llvm-bc", "-Cembed-bitcode=yes", capture=True,
    )
    # Use Cargo's current artifact list, never a glob that can pick up old builds.
    archives = []
    root_archive = None
    for line in messages.splitlines():
        message = json.loads(line)
        if message.get("reason") != "compiler-artifact":
            continue
        for filename in message["filenames"]:
            path = Path(filename)
            if path.suffix == ".rlib":
                archives.append(path)
                if message["target"]["name"] == "llvm_metal_fixture_shallenge":
                    root_archive = path
    if root_archive is None:
        raise RuntimeError("Cargo did not report the fixture archive")
    with tempfile.TemporaryDirectory(prefix="build-", dir=OUTPUT) as temporary:
        stage = Path(temporary)
        modules = []
        ordered_archives = [root_archive] + [p for p in archives if p != root_archive]
        for index, archive in enumerate(ordered_archives):
            members = run("llvm-ar", "t", archive, capture=True).splitlines()
            for member_index, member in enumerate(members):
                if not member.endswith(".o"):
                    continue  # rlib also contains Rust metadata, not LLVM bitcode.
                command = ["llvm-ar", "p", str(archive), member]
                COMMANDS.append(command)
                data = subprocess.check_output(command, cwd=ROOT)
                if not data.startswith(b"BC\xc0\xde"):
                    raise RuntimeError(f"expected NVPTX bitcode in {archive.name}:{member}")
                path = stage / f"dependency-{index}-{member_index}.bc"
                path.write_bytes(data)
                modules.append(path)
        if not modules:
            raise RuntimeError("Cargo's archives contained no LLVM modules")
        run("llvm-link", *modules, "-o", stage / "linked.bc")
        run(
            "opt", "-passes=internalize,globaldce,default<O2>,globaldce,strip-dead-prototypes,verify",
            f"-internalize-public-api-list={entry}", stage / "linked.bc",
            "-o", stage / "kernel.bc",
        )
        unresolved = run("llvm-nm", "--undefined-only", stage / "kernel.bc", capture=True).strip()
        if unresolved:
            raise RuntimeError(f"fixture has unresolved symbols; no runtime stubs are supplied:\n{unresolved}")
        run("llvm-dis", stage / "kernel.bc", "-o", stage / "kernel.ll")
        (stage / "kernel.interface.json").write_bytes((FIXTURE / "kernel.interface.json").read_bytes())
        inputs = [
            Path(__file__).resolve(), FIXTURE / "Cargo.toml", FIXTURE / "Cargo.lock",
            FIXTURE / "src/lib.rs", FIXTURE / "tests/known_answers.rs",
            FIXTURE / "kernel.interface.json", ROOT / "flake.nix", ROOT / "flake.lock",
        ]
        artifacts = ["kernel.bc", "kernel.ll", "kernel.interface.json"]
        provenance = {
            "schema": 1,
            "target": TARGET,
            "rustc": rust_version,
            "llvm": llvm_version,
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
            "archive_sha256": {str(p): digest(p) for p in archives},
            "artifacts": {name: {"sha256": digest(stage / name), "bytes": (stage / name).stat().st_size}
                          for name in artifacts},
            "commands": COMMANDS,
            "checks": {"cpu_known_answers": "passed", "llvm_verify": "passed",
                       "undefined_symbols": [], "metal_execution": "not implemented"},
        }
        (stage / "kernel.build.json").write_text(json.dumps(provenance, indent=2) + "\n")
        # Publish only after all build checks pass; provenance is written last.
        for name in artifacts + ["kernel.build.json"]:
            (stage / name).replace(OUTPUT / name)
    print(f"Built {OUTPUT / 'kernel.bc'} ({(OUTPUT / 'kernel.bc').stat().st_size} bytes)")


if __name__ == "__main__":
    main()
