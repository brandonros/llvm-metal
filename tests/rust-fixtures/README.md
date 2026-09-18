# Rust integration fixtures

`descriptor.rs` proves the typed declaration path without a handwritten interface.
The test independently compiles its declaration for the native host and NVPTX,
extracts the device constant, rejects incompatible bindings, and compares runtime
inputs/outputs and guards on Metal, including empty and varying-length slices:

```sh
nix develop path:.#rust-fixtures --command cargo test --locked \
  -p llvm-metal-compiler --test descriptor -- --include-ignored --nocapture
```

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
runs LLVM O3 with a higher inlining threshold before pruning. Post-inline cleanup
first canonicalizes loop counters (`loop-simplify,lcssa,loop(indvars)`), unrolls,
then runs SROA/InstCombine/SimplifyCFG before a second O3 pass. The unroll threshold
is 1000 to expose fixed-size SEC1 encoding invariants. This ordering avoids the
minutes-long ScalarEvolution predicate analysis triggered by Bech32's partly
unrolled 8-to-5-bit counter loops. Device-side Rust/LLVM vectorization is disabled
in this producer configuration; Metal still performs its own code generation. These are
correctness fixtures, not tuned performance builds. It rejects unresolved
runtime symbols. Three bounded SROA/GVN cleanup rounds with a 10,000-instruction
memory scan budget expose SHA buffer lengths across large inlined blocks; this
eliminates proven dead panic paths without stubbing them. Only the explicitly declared linear index and device fetch-add
operations are permitted for the batch wrapper. No allocator/panic/runtime stubs
are supplied. Explicit local k256, Solana and RSA consumer snapshots apply the consumer's
locked zeroize patch. Passing these fixtures does not establish that
all vanity-miner dependencies work on stable Rust.

Files live under `target/rust-fixtures/shallenge/`, with named subdirectories for
nondefault entries: `kernel.bc`, `kernel.ll`, `kernel.interface.json`, and
`kernel.build.json`. Provenance includes versions, commands, source/lock/archive
hashes, selected device operations, output hashes and separate Rust compilation,
archive extraction/linking, initial O3 and post-inline timings. Each invocation relinks and
verifies current artifacts even when Cargo reuses cached compilation. This is a
reproduction recipe, not a byte-identical-build claim across checkout paths.

Generated AIR, bindings and libraries from GPU tests live in `target/metal-tests/`.
Source wrappers, interface descriptions and lockfiles belong in Git; ordinary
binary artifacts do not. Promote a small captured IR/bitcode file only when its
exact shape/encoding is needed for a regression, following the
[fixture policy](../fixtures/README.md).

The reduced Bech32 counter regression calls each producer's real `post_inline`
helper with a 10-second deadline, then checks native writes and guard bytes before
and after optimization. It does not substitute for complete encoder GPU tests:

```sh
nix develop path:.#rust-fixtures --command env \
  LLVM_METAL_CONSUMER_PATH=/absolute/path/to/vanity-miner-rs-metal cargo test --locked \
  -p llvm-metal-compiler --test optimizer -- --ignored --nocapture
```

## Solana

`solana` is a separate fixture workspace using curve25519-dalek 4.1.3 and an
explicit local consumer snapshot with `logic/test-vectors`. It applies the
consumer's locked zeroize fork and imports the consumer's fixed Solana vector.
The ten `consumer_solana_*` entries cover SHA-512, clamp, scalar reduction from
32/64 bytes, scalar product, point decompression/doubling, base multiplication,
public-key derivation, Base58 and full addresses. Each consumes runtime bytes
and returns intermediate data rather than a pass/fail constant. Interfaces
specify exact input/output sizes; all outputs include padding checks on the GPU.

The native tests use independent SHA-512, BigUint modular arithmetic and Base58,
RFC 8032 public keys and existing consumer vectors. GPU tests cover walking bits,
scalar-order boundaries, zero/all-ones/ramp and deterministic random inputs,
with 3,316 guarded CPU/GPU comparisons. Only source, interfaces and locks are
committed; generated bitcode, AIR and metallib remain under `target/`.

```sh
nix develop path:.#rust-fixtures --command env \
  LLVM_METAL_CONSUMER_PATH=/absolute/path/to/vanity-miner-rs-metal \
  cargo test --locked -p llvm-metal-compiler --test solana \
  -- --ignored --nocapture --test-threads=1
```

The separate consumer application tests cover batched candidates, patterns,
errors, winner reporting and CLI execution. `metal_solana_bench` in that worktree
measures warm throughput with audit disabled, separating GPU, transfers,
submission, pipeline creation and a single-thread CPU baseline.

## RSA modulus

The `rsa` workspace requires an explicit local consumer snapshot exposing the
independent-candidate API and the opt-in `logic/test-vectors` feature. Public P/Q
factors are imported from that module; the fixture does not maintain a copy.
The fixture contract is provided by vanity-miner-rs commit
[`48ea297`](https://github.com/brandonros/vanity-miner-rs/commit/48ea297)
on `codex/gpu-test-consolidation`; compatible descendants can be selected explicitly.
The consumer's reviewed `crypto-bigint` source and locked zeroize fork are
snapshotted and hashed with the wrapper sources.

Fourteen runtime-input fixtures cover multiplication, carry/borrow/comparison,
shifts/counts, division, Montgomery arithmetic, exponentiation, prime filtering,
HMAC factor derivation, interval/residue construction, eligible-q sampling,
pair eligibility, and complete independent candidates. Native oracles retain
independent BigUint and SHA-256/HMAC calculations. Fixed public factors check
match/miss/error payloads and repeatability across intervening candidate IDs.
GPU tests compare every output byte and preserve inputs and buffer guards.

The former `prepare`, `advance`, and `mine` entries targeted a deleted resumable
API. They are replaced by `sample_q`, `eligible_pair`, and `candidate`; cursor
retirement is no longer an application behavior. The explicit mapping is in
[`coverage.json`](coverage.json). Current producer errors identify missing
consumer fixture features before compilation instead of implying compatibility
with the old pinned dependency.

```sh
nix develop path:.#rust-fixtures --command env \
  LLVM_METAL_CONSUMER_PATH=/absolute/path/to/compatible-vanity-miner-rs \
  cargo test --locked -p llvm-metal-compiler --test rsa \
  -- --ignored --nocapture --test-threads=1
```

## Coverage ownership

The miner's `scripts/test-gpu.sh` owns production grid dispatch, candidate matching,
winner publication, cancellation, CLI exports, and complete self-test execution.
Its logic self-tests own the fixed crypto vectors. RSA and Solana local fixtures
reuse `logic/test-vectors`; pinned Shallenge fixtures remain explicitly historical
at the revision named above, and standalone k256 probes retain their independent
public curve constants without introducing a miner dependency.

Compiler GPU suites retain runtime-input intermediate corpora, guards, and reduced
LLVM regressions needed to locate a lowering defect. A pass on a historical pin
is not a pass of the current application. No intermediate or LLVM regression is
removed merely because a larger miner kernel also reaches that operation.
Use [`coverage.json`](coverage.json) for the fixture ownership/change mapping and
the miner's `crates/cli/tests/gpu-coverage.json` for application test selection.
