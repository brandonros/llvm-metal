# llvm-metal
Compile LLVM bitcode into Metal libraries for Apple GPUs

The first implemented stage reads and verifies LLVM IR/bitcode using Inkwell
and LLVM 21. AIR lowering, metallib generation and Metal execution are not yet
implemented. Native execution tests validate the fixture oracles, not GPU support.

Use the pinned stable Rust / LLVM development environment:

```sh
nix develop
cargo test --locked --workspace
cargo run --locked -p llvm-metal-compiler --bin llvm-metalc -- \
  inspect tests/fixtures/positive/11-device-index.ll --entry fixture
```

For a checkout with untracked flake files, use `nix develop path:.`.
Without Nix, install stable Rust (1.85 or newer), LLVM 21 development libraries and libffi,
then set `LLVM_SYS_211_PREFIX` to the prefix containing `bin/llvm-config`.
`Cargo.lock` and `flake.lock` record the tested dependency/toolchain selections.

`inspect` accepts `.ll` or `.bc`, requires a defined entry, and lists module
definitions/declarations. It does not establish a complete dependency closure
or validate a kernel ABI. The CLI never executes input.

See [the fixture policy and progression](tests/fixtures/README.md) for the steps
between scalar operations and complete crypto kernels. Only the compiler crate
exists today; ABI, packaging and runtime crates will be added with their first
tested implementation.

The [pinned Rust integration fixture](tests/rust-fixtures/README.md) builds the
real vanity-miner Shallenge SHA-256 routine into complete NVPTX LLVM bitcode.
