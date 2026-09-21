# The rewrite

**v2 is a GPU dialect of Rust.** Kernels are written for the GPU. Compiling
arbitrary crates (crypto-bigint, for example) is not a goal: that is a CPU-to-GPU
port, and it is the goal that sank `master`. The compiler's job is to make GPU
code correct and to tell the author precisely, at build time, what is not
allowed. When `verify` refuses what an existing crate does, the kernel is
rewritten; "make crate X build" is out of scope.

This branch started from an empty tree. The compiler on `master` is a failed
design: it accepts arbitrary Rust and repairs the IR until Apple takes it, so
whether a kernel builds depends on what the optimizer happened to do. `master`
is evidence of what Apple's compiler requires, not a source of code. Anything
brought over is copied in deliberately, in its own commit, with the reason it
earns its place. Extract; never paste a file whole.

The design, its open questions and its scope are in issue #30. The rules below
are tripwires, not values: when one trips, stop and report; do not reason past it.

## Invariants

1. **Acceptance never depends on optimization.** `verify` decides, before any
   pass runs. If an opt level, attribute or pass order changes whether a kernel
   builds or what it computes, that is a bug in the design, not something to tune.
2. **Device memory is touched only through the buffer intrinsics.** Every other
   pointer is thread memory by construction. No address-space inference, no
   pointer provenance, no specialization by analysis.
3. **Every refusal is a `Rule`:** one enum variant, with a doc comment that is the
   spec, one test, and the Rust source location in the message.
4. **`lower` is a fixed list of rewrites.** Each runs once, is deterministic and
   states its pre- and postcondition. No retries, no fixpoints over refusals, no
   thresholds, no policies.
5. **The host build of the same kernel source is the reference.** Every GPU test
   is differential against it.

## Stop and report; do not code around it

- `verify` refuses a kernel. The kernel changes or the spec changes, and the user
  decides which. Never add a lowering in the same session.
- Host and GPU disagree. Reduce it. Do not change flags, attributes or pass order
  until the cause is known.
- The fix you are about to write is keyed on an opcode, a symbol name, a bit
  width or an instruction count.
- You are about to add a CLI flag, a mode, a policy or a fallback.
- You are about to make something pass by making it optional.
- A source file passes 400 lines, or `verify` + `lower` together pass 1,200.
  Size is the alarm for accumulated special cases.
- The task has become a different task. One session, one issue, one scope.
- A spike question comes back "no". The design is wrong; say so.

## Before writing code

- State what is wrong with the approach and whether you would build it this way
  from scratch. Do this first, unasked.
- A design change is a change to the invariants or to a `Rule`'s doc comment,
  agreed with the user, before the code that needs it.
- Done means: invariants hold, the diff is as small as it can be, and the report
  lists what was not run. Passing tests are necessary, not the goal.
- Never describe defensive code as rigor. Say what it guards and what evidence
  says the guard is needed; if there is none, delete it.

## Known facts about the target

Each cost days to find on `master`. Verify before relying on one; add to the
list only with the evidence.

- Apple's compiler accepts only LLVM 14-encoded bitcode (`llvm-downgrade`).
- Metal pointers carry one of three address spaces: device, constant, thread.
- No i128, no recursion, no indirect calls, no trap.
- `ptrtoint` on a thread pointer is accepted and correct, folded or not
  (`compiler/tests/target_facts.rs`, M5, macOS 27). `master` refused it in its own
  validator and never asked Apple: an inherited "fact" is a guess until a test
  in that file shows it.
- Kernels carrying `optsize`/`minsize` returned wrong answers on Apple M5.
- `llvm.memcpy` from constant to thread memory is accepted (`target_facts.rs`).
- rustc's `PIC Level` module flag can kill Apple's compiler service: a null
  dereference in a machine function pass, reported as
  `XPC_ERROR_CONNECTION_INTERRUPTED`. Reproducer: `tests/facts/pic_relocation.ll`.
  `emit` sets the level to zero. A compiler-service crash leaves a report in
  `~/Library/Logs/DiagnosticReports/MTLCompilerService-*.ips`: read it first.
- A `llvm.memcpy` whose length is zero at run time never returns, and no watchdog
  ends the command; a zero-length `llvm.memset` writes anyway
  (`tests/facts/zero_copy.ll`, `target_facts.rs`, M5, macOS 27). Nothing guards
  against either yet: an empty `copy_from_slice` in a kernel can hang the GPU.
- Pipeline compile time grows with code size and runs on one core per pipeline.
- rustc's release LLVM must be the major this project links, and no newer.

## Tooling

- Use the development shell: `nix develop --command`. Use `path:.` when inputs
  include untracked files.
- Everything is Rust in one Cargo workspace. No Python, no shell scripts, no
  second language, no C/C++/Objective-C, no `build.rs`. LLVM is reached only
  through its C API (`llvm-sys`, Inkwell).
- Tests are `cargo test`. No runner around the runner, no inventories, coverage
  or provenance JSON. A test that needs a GPU is `#[ignore]` with the reason.
- `examples/` holds kernels written against the public interface, built by the
  workspace and run by its tests. Never vendor or pin a consumer.
- No metadata the code does not read. No hand-kept lists, no checker that
  compares two of them.
- GPU runs are serial. Report which stages ran (verify, library and pipeline
  creation, checked execution), on what hardware, and what was skipped.
- Leave nothing uncommitted. Park unfinished work on a pushed branch whose
  message states its condition.

## Documentation

- `README.md` is a short entry point, under 100 lines. No narratives, histories,
  timings or handoffs. The spec is the `Rule` doc comments, not a Markdown file.
- No per-task, planning, validation or handoff Markdown files. Durable context
  goes on the issue.
