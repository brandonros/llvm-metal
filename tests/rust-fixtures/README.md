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

### k256 arithmetic probes

`k256/` pins k256 0.13.4, the version used by vanity-miner, with `arithmetic`
and `expose-field`. Field operations use k256's actual 64-bit implementation.
Host tests compare field multiplication, squaring and inversion against independent
`num-bigint` arithmetic modulo p. Point tests include generator/2G and vanity-miner's
existing public-key known answers.

The `consumer_*` entries call vanity-miner's checked production APIs from an
explicit local source snapshot. They retain the consumer's k256 features and its
existing zeroize fork at the commit in the consumer Cargo.lock. That fork disables
its optional atomic-fence feature; the compiler preserves its volatile writes.
This does not establish compatibility with crates.io zeroize's inline assembly.
The checked APIs reject invalid keys; legacy APIs retain their panic-on-invalid
behavior. Compressed serialization uses k256's public coordinate accessors.

```sh
nix develop path:.#rust-fixtures --command python3 tests/rust-fixtures/build.py \
  --fixture k256 --entry k256_field_mul
nix develop path:.#rust-fixtures --command env \
  LLVM_METAL_CONSUMER_PATH=/absolute/path/to/vanity-miner-rs-metal cargo test --locked \
  -p llvm-metal-compiler --test k256 -- --ignored --nocapture --test-threads=1
```

| Entry | Input/output | GPU corpus |
|---|---|---|
| `k256_scalar_roundtrip` | BE scalar[32] → validity[1] + scalar[32] | 27 cases: zero, one/two, n/p boundaries, known key, deterministic random |
| `wide_mul` | Two LE u64s → exact LE u128 product | 49 products, including maximum/carry cases |
| `k256_field_mul` | Two canonical BE field elements → validity[1] + normalized product[32] | 729 pairs versus native k256 |
| `k256_field_square`, `k256_field_invert` | BE field element → validity + normalized result | 27 each, including zero/invalid encodings |
| `k256_point_double` | Affine x/y[64] → validity + uncompressed point[65] | G, 2G, and three invalid points |
| `k256_scalar_mul` | Private scalar[32] → validity + compressed[33] + uncompressed[65] | 27 runtime keys |
| `consumer_public_keys` | Same contract, actual checked production functions | 27 runtime keys |
| `consumer_public_keys_batch` | LE count + keys[32] → records[99] | Counts 0/1/31/32/33/65/129/257, group sizes 32/64 |
| `consumer_keccak256` | Input[64] → Keccak-256 digest[32] | 516 cases: known answer, zero/ones/ramp, every single input bit |
| `consumer_ethereum_address` | Private key[32] → validity + public XY[64] + address[20] | 27 runtime keys, including invalid scalars |
| `consumer_ethereum_address_batch` | LE count + keys[32] → records[85] | Counts 0/1/31/32/33/65/129/257, group sizes 32/64 |
| `consumer_bitcoin_ripemd160` | Input[32] → RIPEMD-160 digest[20] | 259 cases: zero/ones/ramp and every single input bit |
| `consumer_bitcoin_hash160` | Compressed key bytes[33] → SHA-256[32] + HASH160[20] | 267 cases; hashing does not validate SEC1 encoding |
| `consumer_bitcoin_bech32` | HASH160[20] → length[1] + padded address[64] | 163 cases; mainnet P2WPKH (`bc1q…`) |
| `consumer_bitcoin_address` | Private key[32] → validity + public key[33] + HASH160[20] + length + address[64] | 27 runtime keys, including invalid scalars |

Invalid scalar/field encodings produce all-zero output. Scalar zero is valid in
the round-trip probe, but rejected by public-key probes and field inversion.
Inputs arrive in runtime buffers,
each invocation checks all result bytes, input preservation, and output guards.
The batch kernels assign one independent output record per lane and check bounds.
Tests round dispatches up to full threadgroups to exercise excess-lane guards.
Ethereum fixtures call the same checked address function used by the consumer's
existing Ethereum mode. They hash the 64 coordinate bytes without the SEC1 tag,
then select the last 20 digest bytes. CPU known answers cover keys one/two and
the consumer's existing Ethereum vector; invalid keys zero the complete record.
Bitcoin leaf fixtures compare the consumer's hashes against RustCrypto `sha2`
and `ripemd`, and its address encoder against the independent `bech32` crate.
The complete address fixture calls the checked production helper, preserves
zero padding, and returns all zeros for invalid private keys. Host known answers
include the consumer's existing Bitcoin vector. The fixture harness does not
exercise candidate matching or winner publication; those belong to the miner's
separate Metal application tests.
The production public-key path uses ordinary generator multiplication; k256's
lazy precomputed generator table is not reached, despite the enabled feature.
Artifacts, including AIR and metallib from the GPU tests, are under
`target/rust-fixtures/k256/<entry>/`. Source/lock/interface hashes and producer
commands follow the same provenance contract as Shallenge.

### Shared producer

The producer is official stable Rust 1.93.0 with LLVM 21.1.8 and prebuilt
`nvptx64-nvidia-cuda` libraries. It uses no nightly build-std, Rust-CUDA,
CUDA toolkit or PTX. The producer script itself stops at LLVM bitcode; compiler
and runtime tests subsequently establish AIR/GPU correctness.

`build.py` takes archive paths from the current Cargo build, extracts LLVM object
members (excluding Rust metadata), links them, internalizes other exports and
runs LLVM O3 with a higher inlining threshold before pruning. A second O3 pass
with normal inlining and an unroll threshold of 1000 exposes fixed-size SEC1
encoding invariants. Device-side Rust/LLVM vectorization is disabled in this
producer configuration; Metal still performs its own code generation. These are
correctness fixtures, not tuned performance builds. It rejects unresolved
runtime symbols. Only the explicitly declared linear index and device fetch-add
operations are permitted for the batch wrapper. No allocator/panic/runtime stubs
are supplied. Only explicit local k256 consumer snapshots apply the consumer's
locked zeroize patch. Passing these fixtures does not establish that
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
