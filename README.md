# llvm-metal
Compile LLVM bitcode into Metal libraries for Apple GPUs.

The initial integer/buffer pipeline runs real Rust Shallenge routines on an
Apple M5: SHA-256, nonce generation, candidate checking, and indexed batches
with atomic match counts and winner indices. This is an experimental compiler
profile, not general CUDA compatibility or a performance-tested miner backend.

```text
stock Rust → linked LLVM 21 bitcode + interface.json
           → our AIR legalization → llvm-downgrade → our metallib packager
           → objc2-metal runtime → checked Apple GPU output
```

No PTX, Rust-CUDA, CUDA toolkit or Metal source compiler is used. Inkwell/native
LLVM supplies parsing, verification and optimization. The pinned external
llvm-downgrade supplies older bitcode serialization. The four workspace crates
are `llvm-metal-abi`, `llvm-metal-compiler`, `llvm-metal-metallib` and
`llvm-metal-runtime`; attribution is in [third-party](third-party/README.md).

## Build and run

```sh
nix develop --command cargo test --locked --workspace
nix develop .#rust-fixtures --command python3 tests/rust-fixtures/build.py
nix develop --command cargo run --locked -p llvm-metal-compiler --bin llvm-metalc -- \
  compile target/rust-fixtures/shallenge/kernel.bc \
  --interface target/rust-fixtures/shallenge/kernel.interface.json \
  --output target/compiled/shallenge
```

Compilation writes `kernel.air.ll`, `kernel.air.bc`, `kernel.metallib` and
`kernel.bindings.json`. The CLI does not execute input. `inspect <file.ll|file.bc>
--entry <name>` only parses/verifies and reports definitions and declarations.

The explicit GPU suite requires macOS and an Apple GPU:

```sh
nix develop .#rust-fixtures --command cargo test --locked \
  -p llvm-metal-compiler --test metal -- --ignored --nocapture \
  --test-threads=1 --skip local_sha_intermediates
```

Use `path:.` instead of `.` in Nix commands while new files are untracked. The
flake pins native LLVM 21.1.8 and the downgrader. Its separate fixture shell pins
stable Rust 1.93.0 with prebuilt NVPTX libraries; the default shell uses the newer
Rust compiler from locked nixpkgs. Non-Nix builds need equivalent LLVM development
libraries, libffi, a C++ compiler, and `llvm-downgrade` on PATH; set
`LLVM_SYS_211_PREFIX` to the LLVM prefix containing `bin/llvm-config`.

## Scope and tests

The supported entry is a C function returning void with explicit buffer pointers.
The compiler retargets layouts and address spaces, inlines helpers, moves constant
tables, lowers declared thread-index/device-atomic operations, and writes AIR
metadata. It rejects unsupported operations rather than providing runtime stubs.
See [the profile and limitations](docs/air-profile.md).

GPU tests are opt-in and fail if their prerequisites or comparisons fail. Results
have been checked on Apple M5/macOS 26.6.2 only. Buffer guards, CPU comparisons,
independent SHA answers, partial groups and concurrent atomic results are tested;
throughput and other Apple GPU/OS combinations have not been measured.

The [fixture guide](tests/rust-fixtures/README.md) explains pinned dependencies,
source hashes and local SHA diagnostics. Keep small source fixtures in Git and
generate ordinary bitcode under `target/`. The [phase-2 plan](docs/phase-2.md)
tracks the remaining crypto progression; wide integer arithmetic, elliptic
curves and RSA are still future work.
