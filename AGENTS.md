# llvm-metal

**This is a GPU dialect of Rust.** Kernels are written for the GPU. Compiling
arbitrary crates is not a goal: a compiler that accepts arbitrary Rust has to
repair the IR until Apple takes it, and then whether a kernel builds depends on
what the optimizer happened to do. The compiler makes GPU code correct and says
precisely, at build time, what is not allowed. When `verify` refuses what a
crate does, the kernel is rewritten; "make crate X build" is out of scope.

The rules below are tripwires: when one trips, stop and report; do not reason
past it.

## Invariants

1. **Acceptance never depends on optimization.** `verify` decides, before any
   pass runs. An opt level, attribute or pass order that changes whether a
   kernel builds or what it computes is a design bug, not something to tune.
2. **Device memory is touched only through the buffer intrinsics.** Every other
   pointer is thread memory by construction. No address-space inference.
3. **Every refusal is a `Rule`:** one enum variant whose doc comment is the spec,
   one test, and the Rust source location in the message.
4. **`lower` is a fixed list of rewrites,** each run once, deterministic, with a
   stated pre- and postcondition. No retries, thresholds or policies.
5. **The host build of the same kernel source is the reference.** Every GPU test
   is differential against it, and every example also has a check that shares
   no code with what it checks (a published vector, a second implementation).

## Stop and report; do not code around it

- `verify` refuses a kernel. The kernel or the spec changes, and the user decides
  which. Never add a lowering in the same session.
- Host and GPU disagree. Reduce it before changing anything.
- The fix is keyed on an opcode, a symbol name, a bit width or a count.
- You are about to add a CLI flag, a mode, a policy or a fallback, or to make
  something pass by making it optional.
- A source file passes 400 lines, or `verify` + `lower` together pass 1,200.
- The task has become a different task. One session, one issue, one scope.

## Before writing code

- State what is wrong with the approach and whether you would build it this way
  from scratch. First, unasked.
- A change to an invariant, a `Rule` or the kernel interface is agreed with the
  user first, and an interface proposal comes with a number from a kernel that
  would benefit. Intuition is not evidence.
- Done means: invariants hold, the diff is as small as it can be, and the report
  lists what was not run. A claim that turns out wrong is corrected where it
  was made.
- Defensive code states what it guards and the evidence; with none, delete it.

## Written for the GPU

One job per thread, all parallelism across jobs. Constant loop bounds. A
mask-select where CPU code would branch on data. The host lays data out once;
large shared tables go in an `In<[T]>` buffer, never a Rust `const` (a `const`
is copied into every thread). A thread keeps its own best; do not send
everything back. Buffers are filled and checked in place. Toy cryptography says
so at the top of the file. Each example carries its own code: no shared crates.

## Known facts about the target

A fact is a guess until a test in `compiler/tests/target_facts.rs` shows it. A
test that can hang the GPU stays out of the suite: keep its reproducer file.

- Apple's compiler accepts only LLVM 14-encoded bitcode (`llvm-downgrade`).
- Pointers carry one of three address spaces: device, constant, thread.
- No i128, no recursion, no indirect calls, no trap.
- `ptrtoint` on a thread pointer is accepted; `memcpy` from constant to thread
  memory is accepted.
- rustc's `PIC Level` module flag can crash Apple's compiler service
  (`XPC_ERROR_CONNECTION_INTERRUPTED`; `tests/facts/pic_relocation.ll`); `emit`
  clears it. A crash leaves `~/Library/Logs/DiagnosticReports/MTLCompilerService-*.ips`:
  read it first.
- A `memcpy` of run-time length zero never returns, and no watchdog ends it; a
  zero-length `memset` writes anyway (`tests/facts/zero_copy.ll`). **Nothing
  guards this yet:** an empty `copy_from_slice` in a kernel can hang the GPU.
- Pipeline compile time grows with code size, on one core per pipeline.
- rustc's release LLVM must be the major this project links, and no newer.

Measured on an M5; re-measure before relying on one:

- A threadgroup is one SIMD group wide; the widest was 2.6 times slower.
- A kernel built at opt-level 0 runs 7 to 20 times slower; `s` is close to `3`.
- A call to a function with a multi-KB frame cost 1.7 times (unexplained).
- Throughput drifts about 30% with the machine's state. Compare variants by
  interleaved runs of the GPU's own kernel time, never one before and one after.

## Tooling

- Use the development shell: `nix develop --command` (`path:.` when inputs
  include untracked files).
- One Cargo workspace, all Rust: no second language, no scripts, no `build.rs`.
  LLVM only through its C API (`llvm-sys`, Inkwell).
- Tests are `cargo test`; a GPU test is `#[ignore]` with the reason. Nothing is
  kept that the code does not read: no inventories, no hand-kept lists.
- GPU runs are serial. Report which stages ran, on what hardware, and what was
  skipped.
- Leave nothing uncommitted. Park unfinished work on a pushed branch whose
  message states its condition.

## Documentation

`README.md` is a short entry point, under 100 lines. The spec is the `Rule` doc
comments. No planning, validation or handoff files: durable context goes on the
issue.
