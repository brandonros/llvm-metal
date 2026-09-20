# Rust kernel fixtures

Single-file kernels that test llvm-metal's own crates from stock Rust: buffer
binding (`buffer.rs`), descriptors (`descriptor.rs`), device atomics and dispatch
(`dispatch.rs`) and private memory (`memory.rs`). They depend on nothing outside
this repository. The tests in `crates/compiler/tests` compile them with the
`rustc` in the Nix shell for `nvptx64-nvidia-cuda`; `build.rs` builds one through
`llvm-metalc build`.

Application workloads are not tested here. vanity-miner-rs is the acceptance
bench: its 16 kernel crates are built with `llvm-metalc build` and every
self-test case runs on an Apple GPU (see `AGENTS.md`). When one of its kernels
exposes a compiler bug, reduce it to an IR fixture under `tests/fixtures/` with a
focused test; do not add a workload kernel here.
