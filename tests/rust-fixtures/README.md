# Pinned Rust integration fixtures

This lane exercises a real Rust dependency through the LLVM input boundary.
`shallenge/` is an independent Cargo workspace depending on vanity-miner's `logic`
crate at `f26605ebddc829dd3d6b5f3f4c3cd6276c14f2de`, with only the `shallenge`
feature enabled. The compiler crate has no dependency on vanity-miner.

Run the integration test from the repository root:

```sh
nix develop .#rust-fixtures --command cargo test --locked \
  -p llvm-metal-compiler --test rust_fixture -- --ignored --nocapture
```

Use `path:.#rust-fixtures` while newly added files are untracked. The test is
explicitly ignored by ordinary `cargo test`: this lane downloads/builds a pinned
Git dependency and needs the device producer toolchain. An explicit run fails
if the build, CPU answers, dependency closure or LLVM checks fail.

To generate artifacts without running the compiler reader test:

```sh
nix develop .#rust-fixtures --command python3 tests/rust-fixtures/build.py
```

## What is tested

The `no_std` wrapper exposes one ordinary C function taking two disjoint byte
buffers. It calls the actual production `sha256_32_from_bytes` with 32 runtime
input bytes and writes the 32-byte digest. CPU tests check independent SHA-256
answers for zero bytes, all-one bytes and ascending bytes, input preservation,
and guards around the output. No crypto implementation is copied here.

The producer is official **stable Rust 1.93.0**, pinned through `flake.nix` and
`flake.lock`. Its LLVM 21.1.8 matches our native LLVM reader; its prebuilt
`nvptx64-nvidia-cuda` libraries avoid building `core` with nightly. This small
fixture uses neither experimental kernel ABI nor device intrinsics. Passing it
does not establish that every vanity-miner dependency works on stable Rust.

`build.py` asks Cargo for the current build's archive paths. NVPTX object members
are LLVM bitcode: it extracts those members, links them with `llvm-link`, and
internalizes everything except the declared entry before optimizing and pruning.
Rust archive metadata is excluded. The build fails on surviving undefined
symbols; it supplies no allocator, panic handler or other runtime stubs. The
initial SHA fixture has no reachable allocation/panic calls and needs no patched
`zeroize`; its own lockfile intentionally resolves registry dependencies.
Extending to a routine that needs those facilities requires explicit support.

The compiler test reads and verifies both generated formats with Inkwell,
checks the target and entry signature, rejects unresolved non-intrinsic functions
and globals, and verifies a bitcode round trip. The logical buffer contract in
`kernel.interface.json` is a fixture description, not a finalized Metal ABI.

## Artifacts and limits

Generated files live under `target/rust-fixtures/shallenge/`:

- `kernel.bc` and `kernel.ll`: linked, optimized device module.
- `kernel.interface.json`: single-invocation buffer contract.
- `kernel.build.json`: tool versions, commands, source/lock/archive hashes and
  output hashes, with the checks actually performed.

Cargo may reuse matching cached builds. Each invocation relinks and verifies
the reported artifacts; it does not choose bitcode from a directory glob.
Source, lockfiles and wrappers belong in Git; these generated binaries do not.
This is a reproducible build recipe, not a claim of byte-identical output across
different hosts and checkout paths.

There is no PTX generation, AIR lowering, metallib packaging or GPU execution.
Native CPU answers and structurally valid device bitcode are separate checks;
device numerical correctness remains a future Metal execution test. This
complete hash fixture does not replace the smaller SHA operation/round fixtures
in the [phase-2 progression](../../docs/phase-2.md).
