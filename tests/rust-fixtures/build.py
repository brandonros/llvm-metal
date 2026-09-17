#!/usr/bin/env python3
"""Build the pinned Shallenge fixture; no PTX, AIR or GPU execution."""

import argparse
import hashlib
import json
import os
import shutil
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
    interfaces = {p.stem: p for p in (FIXTURE / "interfaces").glob("*.json")}
    interfaces["shallenge_sha256_32"] = FIXTURE / "kernel.interface.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", choices=sorted(interfaces), default="shallenge_sha256_32")
    parser.add_argument("--consumer-path", type=Path, help="explicit local diagnostic build; records copied consumer source hashes")
    options = parser.parse_args()
    selected = options.entry
    interface_path = interfaces[selected]
    interface = json.loads(interface_path.read_text())
    output = OUTPUT if selected == "shallenge_sha256_32" else OUTPUT / selected
    entry = interface["entry"]
    output.mkdir(parents=True, exist_ok=True)
    source_fixture = FIXTURE
    local_consumer_hashes = None
    if options.consumer_path:
        consumer = options.consumer_path.resolve() / "crates/logic"
        source_fixture = output / "local-source/fixture"
        snapshot = output / "local-source/logic"
        if (output / "local-source").exists():
            shutil.rmtree(output / "local-source")
        shutil.copytree(FIXTURE, source_fixture, dirs_exist_ok=True)
        shutil.copytree(consumer / "src", snapshot / "src", dirs_exist_ok=True)
        shutil.copyfile(consumer / "Cargo.toml", snapshot / "Cargo.toml")
        manifest = source_fixture / "Cargo.toml"
        lines = manifest.read_text().splitlines()
        lines = [line for line in lines if not line.startswith("rev = ")]
        lines = [f"path = {json.dumps(str(snapshot))}" if line.startswith("git = ") else line for line in lines]
        lines = ['compiler-probes = ["vanity-logic/compiler-probes"]' if line == "compiler-probes = []" else line for line in lines]
        manifest.write_text("\n".join(lines) + "\n")
        run("cargo", "generate-lockfile", "--offline", "--manifest-path", manifest)
        local_consumer_hashes = {str(p.relative_to(snapshot)): digest(p) for p in [snapshot / "Cargo.toml", *sorted((snapshot / "src").rglob("*.rs"))]}
    common = ["--locked", "--manifest-path", source_fixture / "Cargo.toml"]
    if selected.startswith("sha_"):
        if not options.consumer_path:
            raise RuntimeError("private SHA exports are not published in the pinned revision; use --consumer-path for an explicitly local diagnostic build")
        common += ["--features", "compiler-probes"]
    run("cargo", "test", *common, "--release", "--target-dir", OUTPUT / "cargo-host")
    run("cargo", "build", *common, "--release", "--bin", "oracle", "--target-dir", OUTPUT / "cargo-host")
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
    with tempfile.TemporaryDirectory(prefix="build-", dir=output) as temporary:
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
            "opt", "-passes=internalize,globaldce,default<O3>,globaldce,strip-dead-prototypes,verify",
            "-inline-threshold=10000",
            f"-internalize-public-api-list={entry}", stage / "linked.bc",
            "-o", stage / "kernel.bc",
        )
        unresolved = run("llvm-nm", "--undefined-only", stage / "kernel.bc", capture=True).strip()
        device_operations = {"llvm_metal.linear_thread_index", "llvm_metal.atomic_add_device_u32"}
        undefined = [line.split()[-1] for line in unresolved.splitlines()]
        if any(name not in device_operations for name in undefined):
            raise RuntimeError(f"fixture has unresolved symbols; no runtime stubs are supplied:\n{unresolved}")
        run("llvm-dis", stage / "kernel.bc", "-o", stage / "kernel.ll")
        (stage / "kernel.interface.json").write_bytes(interface_path.read_bytes())
        inputs = [
            Path(__file__).resolve(), FIXTURE / "Cargo.toml", FIXTURE / "Cargo.lock",
            *sorted((FIXTURE / "src").rglob("*.rs")), *sorted((FIXTURE / "tests").rglob("*.rs")),
            interface_path, ROOT / "flake.nix", ROOT / "flake.lock",
        ]
        artifacts = ["kernel.bc", "kernel.ll", "kernel.interface.json"]
        provenance = {
            "schema": 1,
            "target": TARGET,
            "rustc": rust_version,
            "llvm": llvm_version,
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
            "archive_sha256": {str(p): digest(p) for p in archives},
            "local_consumer_sources": local_consumer_hashes,
            "fixture_lock_sha256": digest(source_fixture / "Cargo.lock"),
            "artifacts": {name: {"sha256": digest(stage / name), "bytes": (stage / name).stat().st_size}
                          for name in artifacts},
            "commands": COMMANDS,
            "checks": {"cpu_known_answers": "passed", "llvm_verify": "passed",
                       "device_operations": undefined, "undefined_runtime_symbols": [],
                       "metal_execution": "not performed by this producer"},
        }
        (stage / "kernel.build.json").write_text(json.dumps(provenance, indent=2) + "\n")
        # Publish only after all build checks pass; provenance is written last.
        for name in artifacts + ["kernel.build.json"]:
            (stage / name).replace(output / name)
    print(f"Built {output / 'kernel.bc'} ({(output / 'kernel.bc').stat().st_size} bytes)")


if __name__ == "__main__":
    main()
