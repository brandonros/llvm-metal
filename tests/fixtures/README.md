# Fixture policy and progression

Commit small, readable `.ll` files with an explicit semantic contract. Ordinary
`.bc` is generated in memory during tests and passed through the real bitcode
reader. This keeps reviewable source canonical and avoids duplicating every
fixture as a version-dependent binary.

Commit `.bc` only when the exact producer encoding is part of a regression or
compatibility test. Such fixtures need a producing toolchain/command, source or
reduction provenance, expected result, and SHA-256 digest. Keep the original
bytes: reassembling them with the current LLVM would destroy that test's purpose.
When a Rust-generated pattern matters, retain the minimal Rust source and
reproduction command alongside its captured IR. Do not make ordinary tests
depend on recompiling with an arbitrary installed rustc.

The original `positive/` fixtures are hand-written for this project; they are not extracted from
vanity-miner's compiler output. Their expected values live in
`crates/compiler/tests/semantics.rs` as Rust reference operations and fixed
answers. A round-trip text comparison checks preservation, not correctness by
itself. No large application artifacts or private mining inputs belong here.

`optimizer/bech32-counters.ll` is a 71-line reduction of the consumer's real
Bech32 conversion loops, with source/toolchain provenance in its header. It was
reduced with LLVM 21.1.8 `llvm-reduce` using a slow second-O3 invocation as the
interestingness condition, then repaired to use a valid output pointer and
terminate. Its contract is only the remaining counter loops and three byte
writes, not Bech32 encoding. `crates/compiler/tests/optimizer.rs` checks guarded
native execution before/after the actual producer cleanup and imposes a
10-second deadline. The old cleanup exceeds that deadline; the ordered cleanup
takes a fraction of a second. Full consumer encoder fixtures remain the GPU
acceptance test.

## What runs today

| Fixtures | LLVM parse / verify / bitcode round trip | Native CPU oracle | AIR / Apple GPU |
|---|---|---|---|
| 01–04: wrapping add, rotate, zero-defined CLZ, carry | Tested | Boundary values + deterministic cases | Separate GPU fixtures |
| 05–07: branch PHI, loop PHI, helper calls | Tested | Both paths, zero/many iterations, overflow | Separate GPU fixtures |
| 08–09: pointer helper and byte copy | Tested | Guard elements/bytes and exact write footprint | Separate GPU fixtures |
| 10: SHA-256 small sigma0 | Tested | Fixed answers + Rust reference | Separate GPU fixtures |
| 11–12: invocation index and device atomic | Tested | Not executable on CPU through this harness | Separate GPU fixtures |
| Negative syntax / SSA dominance | Rejected at the expected stage | Not applicable | Not applicable |

The first ten fixtures intentionally have no device triple: the test harness
can execute their trusted, portable IR natively with checked signatures. Scalar
oracles run from textual and bitcode inputs at two JIT code-generation optimization
levels. This does not test an LLVM IR optimization pipeline, address-space
conversion, Metal dispatch, or concurrency. Fixtures 11–12 preserve NVPTX target
information, a proposed device-operation declaration, metadata and device-space
atomics through input parsing. These original parser fixtures are not the Metal ABI;
the explicit device operations and GPU tests are described in
[the AIR profile](../../docs/air-profile.md).

Positive means structurally valid LLVM, not supported Metal. Unsupported but
valid LLVM is covered by separate legalization-rejection tests.
Never JIT an arbitrary submitted module: the native harness executes only the
reviewed fixtures with exact function signatures and bounded inputs.

Actual Metal execution is covered in `crates/compiler/tests/metal.rs`: reference
AIR, a stock Rust buffer kernel, reduced memory operations, SHA intermediates,
pinned production SHA/nonce/comparison/candidate routines, and guarded batch
execution with device atomic counters. Those tests require an explicit run on
an Apple GPU; the tables above describe the original input fixtures only.

## Build coverage in these steps

The [phase-2 implementation subplan](https://github.com/brandonros/llvm-metal/issues/2) maps this progression
to specific vanity-miner probes and smaller operations within each workload.
It also specifies one-case kernel isolation: the current GPU self-test runner
executes entire groups even when the CPU selector names one check.

Each feature gets a reduced fixture, independent expected results, a failing
test, the implementation, and then an affected larger case. Add the missing
test at the stage being implemented; do not mark skipped GPU work as passed.

1. **Input boundary:** the current fixtures, malformed input, entry validation
   and bitcode compatibility. Next add captured stock-rustc patterns as phase 1
   produces them; hand-written IR cannot establish Rust frontend compatibility.
2. **First Metal execution:** independently prove AIR downgrade/packaging with
   a known-valid reference, then lower a Rust-generated buffer store/vecadd.
   Test empty, partial and multi-threadgroup dispatches.
3. **Scalar operations on Metal:** wrapping 32/64-bit arithmetic, carry/borrow,
   shifts/rotates including zero and full-width counts, CLZ/CTZ and byte swaps.
   Preserve defined versus poison-producing input semantics explicitly.
4. **Control flow and calls on Metal:** select, branches, PHIs, bounded loops,
   early returns, nested helpers and aggregate returns. Introduce one feature
   per fixture before combinations.
5. **Memory on Metal:** buffer offsets, alignment, arrays/structs, stack arrays,
   constants, pointer PHIs, helpers returning pointers and writes through helper
   arguments. Use guard regions and CPU-checked write footprints.
6. **Atomics and publication:** returned old values, ordering/scope, contention
   and result publication across invocations. Host scalar execution is not an
   atomic-concurrency test.
7. **Small crypto building blocks:** SHA-256 sigma/choose/majority → one round →
   one compression block with published known answers. Separately exercise
   multi-limb add/subtract → multiply → modular reduction. Keep both paths small
   enough to inspect and diagnose.
8. **Reduced real Rust crypto routines:** capture emitted IR for one hash block,
   field operation or a bounded elliptic-curve step, with source and toolchain
   provenance. Exercise dependency linking and layouts, not just handwritten IR.
9. **Real application tests:** selected Solana/RSA/P-256 self-tests, then all
   production and self-test groups. Reduce every failure into the smallest
   meaningful regression before changing the compiler.

The existing CPU oracle stays fixed when adding AIR tests. Compare GPU results
against it and known-answer vectors; do not derive expected results from the new
backend. Keep parsing, legalization, packaging, pipeline creation and execution
results separate so a frontend success never stands in for GPU correctness.

Application workloads are accepted in vanity-miner-rs. Promote a reduced capture
of one of its kernels here when it becomes a specific compiler regression.
