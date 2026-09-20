# llvm-metal

Compile LLVM 22 IR or bitcode into Metal libraries for Apple GPUs.
Experimental support for integer kernels with explicit buffer arguments;
see the [supported profile and limitations](docs/air-profile.md).

## Build and compile

Use Nix with flakes enabled. The development shells supply LLVM, Rust, and
`llvm-downgrade`; the fixture shell pins the Rust bitcode producer separately.

```sh
nix develop --command cargo build --locked --workspace
nix develop .#rust-fixtures --command cargo run --locked -p llvm-metal-compiler --bin llvm-metalc -- \
  build --crate path/to/kernel-crate --output target/compiled/kernel
```

`build` compiles the crate for `nvptx64-nvidia-cuda` with the Rust on `PATH` and
produces one bundle per entry. Consumers need no LLVM of their own: the flake
exports the compiler with its LLVM and `llvm-downgrade` as
`packages.<system>.llvm-metalc`. To compile bitcode that already exists:

```sh
nix develop .#rust-fixtures --command python3 tests/rust-fixtures/build.py
nix develop --command cargo run --locked -p llvm-metal-compiler --bin llvm-metalc -- \
  compile target/rust-fixtures/shallenge/kernel.bc \
  --interface target/rust-fixtures/shallenge/kernel.interface.json \
  --output target/compiled/shallenge
```

Compilation writes AIR, `kernel.metallib`, and `kernel.bindings.json` to the
output directory. Run `llvm-metalc --help` for CLI usage. Execute libraries
through the `llvm-metal-runtime` crate on macOS with an Apple GPU.

Apple's GPU compiler accepts only LLVM 14-encoded bitcode; newer encodings
crash the OS compiler service. `llvm-downgrade` re-serializes the LLVM 22 AIR
module into that encoding. This constraint is undocumented and was determined
empirically (see Metal.jl/GPUCompiler.jl).

## Tests

```sh
nix develop --command cargo test --locked --workspace
```

GPU tests are opt-in and require macOS with an Apple GPU:

```sh
nix develop .#rust-fixtures --command cargo test --locked \
  -p llvm-metal-compiler --test metal -- --ignored --nocapture \
  --test-threads=1 --skip local_sha_intermediates
```

See the [fixture guide](tests/rust-fixtures/README.md) for other workloads and
local SHA diagnostics. Use `path:.` or `path:.#rust-fixtures` in Nix commands
when testing untracked source files.
