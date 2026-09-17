# Rust integration fixtures

`shallenge/` is an independent Cargo workspace depending on vanity-miner's
`logic` crate at `f26605ebddc829dd3d6b5f3f4c3cd6276c14f2de`, with only the
`shallenge` feature enabled by default. The compiler has no vanity-miner dependency.
Wrappers call the real production code, accepting runtime byte buffers and
returning complete values. Host tests include independent SHA answers and the
existing Shallenge candidate known answer.

## Pinned builds and Metal execution

```sh
nix develop .#rust-fixtures --command python3 tests/rust-fixtures/build.py
nix develop .#rust-fixtures --command python3 tests/rust-fixtures/build.py --entry shallenge_batch
nix develop .#rust-fixtures --command cargo test --locked \
  -p llvm-metal-compiler --test rust_fixture -- --ignored --nocapture
nix develop .#rust-fixtures --command cargo test --locked \
  -p llvm-metal-compiler --test metal -- --ignored --nocapture \
  --test-threads=1 --skip local_sha_intermediates
```

Use `path:.#rust-fixtures` while files are untracked. GPU tests require macOS and
an Apple GPU. Run them serially: fixture builds share Cargo output directories
and the selected CPU oracle executable. An explicit test fails on missing
prerequisites, compilation errors or wrong results; ordinary workspace tests
report these external-tool/GPU tests as ignored.

| Entry | Input/output contract | GPU checks |
|---|---|---|
| `shallenge_sha256_32` (default) | 32-byte input/digest | 67 edge/deterministic random cases vs RustCrypto SHA-256 |
| `shallenge_nonce` | LE u64 lane + seed → 21 nonce bytes | 15 lane/seed combinations |
| `shallenge_compare` | Two 32-byte hashes → LE i32 comparison | Equality and first/middle/last-byte differences |
| `shallenge_candidate` | 79-byte request → 65-byte result | 24 length/target/lane cases, including invalid lengths |
| `shallenge_batch` | Count + requests → results, two counters, winner indices | Nine sizes, partial/multiple groups, mixed matches/invalid inputs |

A request is lane[8], seed[8], username length[1], username storage[30], target[32].
A result is hash[32], nonce storage[30], nonce length[1], match[1], valid[1].
Invalid username lengths produce a zero result. Batch buffers hold a LE u32 count
followed by requests, contiguous results, two LE u32 counters (valid/matching),
and LE u32 winning lane indices. Counters must start at zero and winner capacity
must cover the request count. CPU comparison waits for GPU completion; there is
no claim of concurrent GPU consumer publication ordering. Buffer guards and
input preservation are checked for the production routines.

`buffer.rs`, `memory.rs` and `dispatch.rs` are smaller standalone stock Rust
fixtures for the first buffer operation, rotations/constant tables/copies/fills,
and indexed atomic fetch-add. They precede the complete workload in diagnosis.

## Local SHA diagnostics

Private production SHA helpers are exposed by an opt-in `compiler-probes` change
in the local vanity-miner checkout. That change has not been published or added
to the pinned Git dependency. The default lane above remains pinned and usable
without it. To test the prepared local change explicitly:

```sh
nix develop .#rust-fixtures --command python3 tests/rust-fixtures/build.py \
  --entry sha_ops --consumer-path ../vanity-miner-rs
nix develop .#rust-fixtures --command env LLVM_METAL_CONSUMER_PATH=/absolute/path/to/vanity-miner-rs \
  cargo test --locked -p llvm-metal-compiler --test metal local_sha_intermediates \
  -- --ignored --nocapture --test-threads=1
```

This creates an isolated source snapshot under `target/`, wires its local Cargo
feature, and records hashes of the consumer manifest and Rust source. It does
not rewrite the committed Git pin or lockfile. Do not enable `compiler-probes`
directly against the old Git dependency: that revision lacks the exports.

The four stages are `sha_ops` (30 cases), `sha_schedule` (four), `sha_round`
(nine), and `sha_compress` (five). Buffers contain LE u32 words; compression takes
16 message words followed by eight state words. Tests compare all outputs with
both the native consumer and independent expressions/RustCrypto answers. This
keeps the steps between scalar operations and a complete crypto kernel visible.

## Producer and artifacts

The producer is official stable Rust 1.93.0 with LLVM 21.1.8 and prebuilt
`nvptx64-nvidia-cuda` libraries. It uses no nightly build-std, Rust-CUDA,
CUDA toolkit or PTX. The producer script itself stops at LLVM bitcode; compiler
and runtime tests subsequently establish AIR/GPU correctness.

`build.py` takes archive paths from the current Cargo build, extracts LLVM object
members (excluding Rust metadata), links them, internalizes other exports and
runs LLVM O3 with a higher inlining threshold before pruning. It rejects unresolved
runtime symbols. Only the explicitly declared linear index and device fetch-add
operations are permitted for the batch wrapper. No allocator/panic/runtime stubs
or patched zeroize are supplied. Passing these fixtures does not establish that
all vanity-miner dependencies work on stable Rust.

Files live under `target/rust-fixtures/shallenge/`, with named subdirectories for
nondefault entries: `kernel.bc`, `kernel.ll`, `kernel.interface.json`, and
`kernel.build.json`. Provenance includes versions, commands, source/lock/archive
hashes, selected device operations and output hashes. Each invocation relinks and
verifies current artifacts even when Cargo reuses cached compilation. This is a
reproduction recipe, not a byte-identical-build claim across checkout paths.

Generated AIR, bindings and libraries from GPU tests live in `target/metal-tests/`.
Source wrappers, interface descriptions and lockfiles belong in Git; ordinary
binary artifacts do not. Promote a small captured IR/bitcode file only when its
exact shape/encoding is needed for a regression, following the
[fixture policy](../fixtures/README.md).
