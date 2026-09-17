# Phase 2: grow llvm-metal from the vanity-miner dependency graph

Phase 1 supplies stock-rustc device bitcode. Phase 2 builds LLVM → AIR → Metal
execution in small, independently testable steps drawn from the real workloads.
The first working GPU kernel is the infrastructure milestone, not the end of
this phase. PTX and NVIDIA execution are not dependencies.

This plan is based on reading vanity-miner-rs at
`f26605ebddc829dd3d6b5f3f4c3cd6276c14f2de`, particularly its self-test registry,
Shallenge SHA-256 probes, Solana arithmetic/Scalar52 probes, Bitcoin layout/k256
probes, and RSA-PSS/RSA-modulus implementations. Paths below are relative to
`vanity-miner-rs/crates/logic/src/`. Existing probes are source material, not
evidence that they work through llvm-metal.

## Current checkpoint

Implemented and checked on Apple M5/macOS 26.6.2:

- Reference AIR → LLVM downgrade → metallib → runtime, then a stock Rust buffer kernel.
- Reduced rotate/lookup/copy/fill kernels and production fixed-length SHA-256.
- SHA operations, schedule word, one round and compression, using local
  feature-gated consumer exports with independent answers. These exports are
  not yet in the pinned Git dependency; publication and repinning remain pending.
- Production nonce generation, hash comparison and a complete Shallenge candidate.
- A fixture batch kernel with bounds checks, device fetch-add counters and winner
  indices, checked after command completion against the CPU oracle.

The tested batch wrapper is not yet integrated into vanity-miner's application
backend. Performance, other GPU/OS combinations, general memory ordering,
wide-integer support and the other crypto workloads remain open. See the
[implemented AIR profile](air-profile.md) and [commands](../tests/rust-fixtures/README.md).

## 2.0 Establish the execution and isolation machinery

Complete the output compatibility experiment: known-valid AIR → selected LLVM
writer/downgrader → metallib packager → Metal pipeline → checked output. Then
connect a phase-1 Rust-generated store/index kernel through our own lowering.
Exercise explicit buffer/scalar bindings and guarded outputs. This establishes
the execution path shared by every later test.

In parallel, make one selected Rust operation produce one kernel artifact:

- The current `self_test/registration.rs` gives CPU execution named selection,
  but GPU `run_device` invokes the entire owning group. A CLI filter does not
  create a smaller GPU compilation unit.
- Add narrow fixture-facing wrappers inside the owning Rust crate, where the
  existing private modules can be accessed. Select a case at build time and
  call it directly; do not dispatch through the full registry at runtime.
- Link the selected entry's closure, internalize other exports and remove
  unreachable definitions. Inspect retained functions/globals to verify the
  isolation. If upstream compilation still generates the entire group, narrow
  source inclusion/features as well; post-link pruning alone does not solve
  frontend build cost.
- For diagnostic stages, accept runtime input buffers and return intermediate
  values, not only a boolean. Keep the original named checks as integration
  tests. Constant inputs and a returned `1` can conceal eliminated computation.
- Existing probes often use `black_box`. Inspect its actual LLVM representation;
  do not silently discard assembly/barriers or remove it and permit constant
  folding. Runtime input wrappers provide a separate way to keep work live.
- Preserve case identity by stable name, not registry slot number. Record
  source revision, selected entry, dependency features, rustc/LLVM versions,
  optimization settings, ABI and fixture hashes.

The registry uses `1` for success, `0` for failure and `2` for a GPU skip. A
skipped case remains incomplete. RSA-PSS `end_to_end` currently has an explicit
GPU compilation-cost skip; its status must not be silently inherited as success.

## 2.1 Shared integer, layout and call foundations

Port these existing small probes before whole crypto routines. Keep both a
small handwritten LLVM regression and a captured real Rust version when their
IR shapes differ. Determine required LLVM transformations from emitted modules,
not old PTX comments in the source.

| Existing probes | Next smaller/additional cases | Compiler behavior to establish |
|---|---|---|
| `solana.arith_overflowing_add`, `arith_overflowing_sub`, `arith_carry_chain_3limb` | One limb → two → three; both carry/borrow paths; full output limbs | Overflow aggregates, boolean extension and carry propagation |
| `solana.arith_u64_mul_hi`, `arith_u128_mul`, `arith_widening_mul_pair`, `arith_widening_mul_chain_3term` | Low/high product halves; u128 shifts/truncations; accumulation | Wide integer legalization and support routines |
| `solana.arith_u32_div_var`, `arith_u64_div_var`, remainder variants, `arith_divrem_by_58_pow_5` | Nonzero divisors and quotient/remainder boundaries | Division lowering without importing NVIDIA runtime helpers |
| `bitcoin.static_u64_array_lookup`, `static_struct_wrapped_u64_lookup` | Runtime index into raw and wrapped constants | Constant memory and aggregate layout |
| `bitcoin.generic_array_basic_index`, `generic_array_copy_from_slice`, `generic_array_copy_from_ga_source` | Length 0/1/31/32/33; source/destination guard regions | Rust wrappers, slices, copies and pointer provenance |
| `solana.dynamic_index_write`, `named_field_struct_return`, `slice_reverse_partial` | Return an aggregate; mutate through a helper; pointer PHI | Helper ABI, aggregate returns, write footprint |
| `bitcoin.subtle_choice_u8_into_bool`, `subtle_conditional_select_u64` | Both choices and nontrivial high bits | Actual library-generated masks/selects |

The existing llvm-metal scalar fixtures are starting points for these checks;
their native CPU success is not Metal coverage. Each new feature must pass the
Apple execution path before a workload depends on it.

## 2.2 Shared hashing and Shallenge: first complete workload

Source: `crypto/sha256.rs`, `search/xoroshiro.rs`, `modes/shallenge.rs`, and
`self_test/shallenge/{mod,sha256_probes,comparison_probes,candidate_probes}.rs`.

Build this chain with separate artifacts and expected intermediate outputs:

```text
byte/word endian conversion
    → rotate + choose/majority + big/small sigma
    → one message-schedule expansion word
    → one SHA-256 round with runtime state/message word
    → a few rounds → complete compression block
    → fixed 32-byte hashing path
    → variable-length padding and streaming path
```

The source has both `sha256_32_from_bytes` and `Sha256::update/finalize`.
Shallenge's fixed-size preimage uses the first; passing the generic streaming
path does not establish the specialized path, or vice versa. Keep the original
unrolled implementation as the acceptance target. A looped diagnostic version
does not replace it. Generic padding/streaming coverage is a separate branch;
it is not a prerequisite for Shallenge's first complete fixed-length candidate.

Reuse exact tests `shallenge.sha256_padding_0`, `_55`, `_56`, `_63`, `_64`, `_65`,
`sha256_streaming_boundary`, `sha256_multiblock`, and `sha256_streaming_chunks`.
Their fixtures include independently generated hashlib answers. Check padding
bytes and intermediate state separately when a whole digest fails.

Develop nonce generation and comparison as independent branches:

```text
splitmix/xoroshiro state step → output bytes → base64 alphabet mapping → nonce
byte comparison → equal/prefix-different/last-byte-different hash comparison
```

Then compose nonce → `username || '/' || nonce` → hash → comparison → structured
candidate result. Reuse `xoroshiro_base64_nonce`, `compare_hashes_lt/gt/eq`,
`compare_hashes_last_byte`, and candidate match/miss/invalid tests. Only after a
single candidate is correct add indexed batches and atomic result publication.

**Milestone:** a complete Shallenge candidate runs correctly on Metal, with each
intermediate stage still available as a small regression.

## 2.3 PSS encoding without RSA arithmetic

Source: `crypto/rsa_pss.rs`, `search/{salt_counter,message_window}.rs`, and
`self_test/rsa_pss/{mod,salt_probes}.rs`.

This branch depends on SHA-256 and memory support, not big-integer exponentiation:

```text
big-endian MGF counter → SHA256(seed || counter)
    → one mask block → multiple blocks → final partial block
    → PSS H calculation → masked DB → unused-bit mask/trailer
    → complete EMSA-PSS encoded message
```

Reuse `rsa_pss.mgf1_partial_block` (50 output bytes), `salt32_encoding`,
`empty_salt_encoding`, `maximum_salt_encoding` and `oversized_salt_rejected`.
Add small mask lengths 0/1/31/32/33 and verify unchanged guard bytes. Use the
existing invalid-width tests in `crypto/rsa_pss.rs`: rejected inputs must leave
output untouched. Check counter endianness, salt carry and exhaustion separately.

Keep the encoded message as an observable output. Do not attach CRT/private-key
operations until this entire branch is independently correct.

## 2.4 Solana: split hashes, scalar arithmetic, points and encoding

Source: `self_test/solana/{arithmetic,ed25519_probes,base58_probes,layout_probes}.rs`,
`self_test/solana/bisect_scalar52.rs`, `crypto/{sha512,ed25519}.rs`.

Run three branches before composing the Solana candidate:

1. **SHA-512:** 64-bit rotate/sigma → one round → schedule/compression → existing
   fixed 32-byte-input hash. Reuse `solana.primitive_sha512`.
2. **Scalar and point operations:** clamp bytes → byte/52-bit-limb unpack/pack →
   subtraction with and without borrow → widening multiplication → Montgomery
   reduction without final subtraction → with final subtraction → real dalek
   scalar APIs → point operations → base multiplication → compressed point.
3. **Base58:** one divide/remainder → one limb division → inner mutation →
   digit construction/reversal → full encoding, including zeros/leading zeros.

The arithmetic branch already has targeted checks:
`dalek_clamp_integer`, `dalek_scalar52_from_bytes`, `dalek_scalar52_as_bytes_one`,
`dalek_scalar52_sub_no_underflow`, `dalek_scalar52_sub_with_underflow`,
`dalek_scalar52_montgomery_reduce_r`,
`dalek_scalar52_mul_internal_then_reduce_one_r`, and
`dalek_scalar52_montgomery_reduce_with_sub`.

`bisect_scalar52.rs` is a local reproduction of dalek internals with diagnostic
variants. Passing it does not replace tests of the actual dependency. Follow
with `dalek_scalar_round_trip_zero/one`, `dalek_order_boundaries`,
`dalek_wide_nonzero` and `dalek_mul_base_scalar_one`; add nontrivial runtime
scalars rather than relying only on zero/one identities.

Base58 already has `base58_div_by_58`, `base58_limb_divrem`,
`base58_inner_mutate_phase`, `base58_min_nonzero`, and `base58_all_zeros`.
Return intermediate limbs and digit lengths when diagnosing failures.

**Composition:** seed/counter → private bytes → SHA-512 → Ed25519 public bytes →
Base58 → match. Compare each boundary before calling the entire candidate.

## 2.5 Bitcoin: k256, HASH160 and address encodings

Source: `self_test/bitcoin/{secp256k1_probes,layout_probes,base58_probes,mod}.rs`,
`crypto/{secp256k1,ripemd160}.rs`, `encoding/{bech32,base58}.rs`.

Separate these branches:

- **k256:** byte/scalar validation → order boundaries → scalar serialization →
  affine coordinates to encoded point → affine generator encoding →
  projective-to-affine conversion → doubling → scalar multiplication with
  runtime scalars. Reuse `k256_secret_from_bytes_one`,
  `k256_scalar_order_boundaries`, `k256_scalar_one_round_trip`,
  `k256_encoded_point_from_affine_coords`, `k256_affine_generator_encode`,
  `k256_encode_generator`, `k256_double_generator`, and `k256_derive_scalar_one/two`.
- **Hashing:** reuse SHA-256, then RIPEMD160 boolean/rotate steps → one round →
  compression → existing 32-byte-input primitive → HASH160 of a supplied public
  key. This does not require curve multiplication to be in the same kernel.
- **Encoding:** Bech32 bit conversion → polymod → checksum → complete P2WPKH
  address. Independently cover WIF's version/compression bytes → double SHA-256
  checksum → Base58Check using the existing four WIF checks.

The current Bitcoin candidate returns a **Bech32 P2WPKH address**. Base58 is
still needed for WIF and its probes, but must not be mistaken for the production
address encoder. Test `bech32_p2wpkh`, matching and candidate rejection cases.

**Composition:** seed/counter → private bytes → compressed public key → HASH160 →
Bech32 → match. Preserve existing intermediate fields/checks as compatibility
requirements; do not remove work while migrating the compiler.

## 2.6 RSA arithmetic, modulus search and CRT

Source: `crypto/{rsa_prime,rsa_crt}.rs`, `modes/rsa_modulus.rs`,
`self_test/rsa_modulus/{mod,range_probes}.rs`, `self_test/rsa_pss/mod.rs`.

Use two independent size axes: limb width and operation complexity. Small
numeric inputs inside a U1024 operation do not necessarily produce small IR.
Diagnostic smaller-width instantiations supplement, not replace, the real widths.

```text
limb add/subtract/carry → widening product → short multi-limb multiplication
    → full U1024 × U1024 → U2048
    → comparison / shifts / CLZ-CTZ / division-remainder
    → modular conversion → multiply/square → retrieve
    → bounded exponentiation → actual-width exponentiation
```

Reuse `rsa_modulus.multiplication_carry` and `progression_carry` at full width.
Add reduced modular-operation fixtures using the same crypto-bigint APIs that
the real code calls, with independent integer reference answers.

Split modulus search into:

```text
HMAC candidate block → factor derivation
range bounds → q-at-offset → cursor advance → tile retirement
small-prime sieve → n-1 decomposition → one Miller-Rabin base → all bases
    → eligible factor pair → resumable search
```

Existing anchors include `device_derivation`, `device_range`, `device_cursor`,
`range_empty`, `range_multiple`, `range_exhausted`, `range_partial_tile`,
`prime_filter` and `pseudoprime_rejected`. They should not all be linked into the
same early diagnostic kernel.

Split CRT into input reduction → p exponentiation → q exponentiation → modular
difference/inverse multiplication → recombination → public-operation check.
Reuse `rsa_pss.crt_known_answer`, `crt_fault_rejected` and `crt_modulus_rejected`.
The internal textbook-key tests are diagnostic only; do not weaken the real
constructor's full-width validation to accommodate them.

Only then join the already passing PSS encoder to CRT and candidate enumeration.
The existing skipped RSA-PSS end-to-end case remains an explicit final target.

## 2.7 Remaining consumers and full workload acceptance

Ethereum reuses k256 but needs uncompressed point encoding plus Keccak. Build
Keccak permutation steps/one round before its existing 64-byte-input hash, then
address extraction and candidate tests.

P-256 public-key work reuses hashing/layout support but must test its own field
and scalar arithmetic. Its current `point_double` check calls public-key
derivation with scalar 2; add a truly isolated doubling/field-operation probe
when that distinction is needed. Follow HMAC derivation, scalar boundaries,
generator/point output and x/y encoding with whole candidates.

P-256 signatures add message-window mutation/hash, nonce derivation, ephemeral-r,
signature arithmetic and low/high-S representation. Use existing RFC6979 known
answers only after their smaller dependencies pass. Do not infer P-256 support
from k256 success.

Finally run single candidates, then indexed batches, atomic contention and
result publication, then all eight production and eight self-test groups.
When several lanes can win, compare allowed results and counts instead of a
fixed scheduling-dependent winning lane.

## Work order and test artifacts

Start with 2.0 and a small selection from 2.1. Prioritize the SHA-256/Shallenge
branch for the first complete workload. After SHA-256, PSS encoding can advance
without waiting for RSA arithmetic. Scalar/layout work feeds the independent
Solana, k256 and RSA branches. One integrator owns shared pass ordering and ABI;
agents can own separate fixture families and specific lowering changes.

The first real-Rust fixture queue, after the Metal execution path works, is:

1. Overflow add/subtract and the three-limb carry chain.
2. Wrapped constant lookup and GenericArray copy with guard bytes.
3. A wide product and high-half extraction.
4. SHA-256 sigma, one schedule word and one round returning the full state.
5. A full compression and both fixed-32-byte/variable-padding hash paths.
6. Shallenge nonce generation, last-byte comparison and one candidate.
7. MGF1's 50-byte partial-block case and PSS encoding boundaries.
8. Scalar52 unpack/pack, borrow and the two reduction variants.

This is a dependency order, not an estimate of which source function yields the
smallest IR. Measure functions, instructions, bitcode size, compile time, peak
memory and Metal pipeline time for each artifact. If a supposed leaf brings in
a large graph, stop and isolate its actual dependency before proceeding.

Each case should retain minimal Rust source/wrapper, source/dependency provenance,
reviewable captured `.ll`, ABI/entry description, runtime inputs, expected raw
outputs and declared prerequisite cases. Generate ordinary `.bc` in tests;
preserve exact binary fixtures only for encoding compatibility regressions.
Expose a stage's diagnostic output through a test wrapper, without changing the
production algorithm solely to make the compiler's job easier.

Use a separate, explicitly invoked Cargo workspace under llvm-metal's
`tests/rust-fixtures/` for wrappers around public vanity-miner functions. Pin the
Git revision, dependency lockfile and stock Rust producer; the initial Shallenge
fixture follows this approach. Narrow exports for currently private operations
still belong in vanity-miner-rs. Keep reduced, self-contained compiler regression
fixtures in llvm-metal, and keep the reusable compiler independent of vanity-miner.
Do not duplicate entire crypto implementations. Preserve source licensing and
attribution for copied reductions.

For each case: establish independent expected outputs → show the failing target
stage → implement that stage → check raw GPU outputs/guards → rerun immediate
parent cases. Use published or independently generated answers as well as the
CPU implementation. The compiler must lower the operations generically, never
recognize a fixture name or substitute a canned answer.

Record parsing, legalization, AIR serialization, library loading, pipeline
creation and execution separately. Unsupported cases and current GPU skips are
visible backlog entries, not ignored passing tests. A milestone is accepted
only when its output has been checked on Apple GPUs. Native-only tests are
useful fixture validation but do not satisfy GPU milestones; use the explicit
Metal suite for the implemented checkpoint above.
