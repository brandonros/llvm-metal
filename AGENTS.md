# Development

- Use the development shell selected by `flake.lock`: `nix develop --command`.
  Use `path:.` when inputs include untracked files.
- Start with focused regression tests, then run affected integration tests and
  `nix develop --command cargo test --locked --workspace` after code changes.
- Run GPU tests explicitly on macOS with an Apple GPU, serially: they share
  output directories.
- Distinguish LLVM verification, Metal library/pipeline creation, and checked GPU
  execution. Report which stages ran and any failures or missing prerequisites.
- Scope validation claims to the tested source, toolchain, inputs, GPU, and OS.
  Keep historical results labeled when dependencies or source revisions change.
- Keep small source fixtures in Git and generated artifacts under `target/`.
  Follow `tests/fixtures/README.md` when a regression needs captured IR or bitcode.

## Working rules

Each of these comes from a real mistake. #24 has the baselines to re-measure
before and after a change.

1. **One passing kernel proves nothing.** A change to what reaches Apple's
   compiler is accepted only by vanity-miner's full run (`just test` in
   `nix develop --override-input llvm-metal path:<this checkout>`: 203 self-test
   cases). Passing PIC/PIE flags through unchanged passed 212 tests and
   one GPU kernel, then crashed Apple's pipeline compiler on two others.
2. **Decide which layer is wrong before fixing it.** `urem i128` in a kernel was
   a kernel bug, not a reason to add 128-bit division here. `llvm.usub.sat` from
   an ordinary `.take(n)` was a compiler gap, not a reason to contort the kernel.
3. **Identical text is not identical output.** `kernel.air.ll` can match while
   `kernel.air.bc` and the metallib differ. Compare the bytes Metal loads.
4. **Keep the old implementation selectable until the new one is proven.** Run
   both on clones of one module and require the same printed IR, refusals and
   predicate results. Compare clone with clone: cloning reorders use lists.
5. **Measure before explaining.** Sample the process, check CPU per process,
   read the recorded stage timings. Slow GPU tests were blamed on compiling; the
   time was in Apple's `MTLCompilerService`.
6. **Do not grow a pass one refused operation at a time.** If a workload needs a
   new lowering, first ask whether the workload should avoid the construct, and
   whether the pass is the right design at all (#18, #19).
7. **Never weaken a safety or verification check to gain speed** without saying
   so in the PR title and getting explicit agreement.
8. **One GPU run at a time.** GPU and fixture tests share output directories;
   concurrent runs corrupt each other. One issue, one branch, one PR.
9. **Leave nothing uncommitted in a worktree, and say what you did not run.**
   Park unfinished work on a pushed branch whose message states its condition.
   Every PR lists which checks ran, on what hardware, and which were skipped.

## Tooling rules

Each of these removes something that was built here and had to be deleted.

- **No Python, no shell scripts, no second language.** Everything is Rust in this
  workspace: a `#[test]`, a library function, or an `llvm-metalc` subcommand.
  What used to be `opt`, `llvm-link`, `llvm-ar` and `llvm-nm` calls from a script
  is `crates/compiler/src/build`.
- **Tests are `cargo test`.** No runner around the runner: no test inventories,
  coverage or ownership JSON, discovery cross-checks, or provenance files for
  fixtures. A test that must not run by default is `#[ignore]` with its
  requirement in the reason.
- **No workload kernels here.** vanity-miner-rs is the acceptance bench. This
  repository keeps small IR tests and the single-file kernels that test its own
  crates. A compiler bug found by a workload becomes a reduced IR fixture under
  `tests/fixtures/`, never a fixture workspace, a consumer pin or a
  `*_CONSUMER_PATH` variable.
- **Consumers supply bitcode and nothing else.** Anything needed to turn a kernel
  crate into a metallib goes into `llvm-metalc build` and the flake package. A
  consumer's flake must never need LLVM, llvm-downgrade or this source tree.
- **No metadata the code does not read.** No hand-kept catalogs or manifests, and
  no checker that compares two hand-written lists.
- **Fix forward.** When an LLVM or Rust bump breaks something, find the cause and
  fix it on the new toolchain. Do not retreat to the old one to keep tests green,
  and decide the layer first (rule 2): the LLVM 22 break was one function in the
  workload, not a missing lowering here.

## No native code

- This repository has no C, C++ or Objective-C, no `build.rs`, and no `cc`,
  `bindgen`, `cxx` or `autocxx` dependency. Keep it that way. LLVM is reached
  only through its C API (`llvm-sys`, and Inkwell above it).
- Rust cannot call LLVM's C++ API: the symbols are mangled, templated or inline.
  When the C API lacks something, do not add a C++ shim, do not fork
  `llvm-sys` or Inkwell to hold one, do not call mangled symbols, and do not
  rewrite textual IR. First look for a Rust route. Every gap met so far had one:

  | Missing from the C API | What to do in Rust |
  |---|---|
  | `dropAllReferences` | Replace all uses with poison, then erase. |
  | Turn a constant expression into an instruction (the builder folds all-constant operations) | Build around a frozen-poison stand-in, then `LLVMSetOperand` the real operand. See `src/phi_constants.rs`. |
  | `removeDeadConstantUsers` | Point dead constant users at null before deleting the global. See `src/descriptor.rs`. |
  | Retype an `alloca` or GEP | Rebuild it in place with the same name, alignment, flags and debug location. See `src/wide.rs`. |
  | `CloneFunctionInto` | `src/clone.rs`. PHIs are rebuilt, because a copied PHI's incoming blocks cannot be changed. |
  | `DominatorTree`, `LoopInfo` | `src/loops.rs`. |
  | `getUnderlyingObject(s)` | `src/pointer_provenance.rs`, bounded and fail-closed. |
  | A pass parameter the textual pipeline does not accept | Look for an LLVM option and set it with `LLVMParseCommandLineOptions`. See `run_inference` in `src/address_spaces.rs`. |
  | Remove a module flag or named-metadata operand | Replace the operand instead, with `LLVMReplaceMDNodeOperandWith`. See `neutralize_codegen_flags` in `src/air.rs`. |
  | Read a constant wider than 64 bits | Split it through LLVM's constant folding. See `halves` in `src/wide.rs`. |
  | Read `inalloca`/`swifterror`, create a `distinct` metadata node, clone a `DISubprogram` | No route. Refuse the input or drop the fact, and say so in a comment. |

- If no Rust route exists, stop and raise it on an issue with the evidence: the
  exact LLVM C++ API, the C API functions checked, and what was tried. Native
  code is added only after that discussion, as one generic `LLVMExt*` function
  that wraps one LLVM utility, holds no Metal policy, and states in a comment
  which gap it covers and what would let it be deleted.
## Documentation

- Keep `docs/air-profile.md` as the supported-profile reference and
  `tests/rust-fixtures/README.md` as the fixture guide. Update existing sections
  instead of appending task histories or repeated evidence.
- Do not create per-task, planning, validation, or handoff Markdown files unless
  the user explicitly requests one.
- Keep raw logs, commands, hashes, and structured results in ignored artifact
  directories. Put durable technical context on existing issues/PRs when the
  user has authorized GitHub updates.

## README maintenance

- Keep `README.md` a short entry point: purpose, supported scope, and essential
  setup, build, compile, and test commands. Keep it under 100 lines unless the
  user explicitly requests a longer guide; do not cram paragraphs onto long lines.
- Edit it only when essential user-facing instructions change or become wrong.
  Completing an internal change or investigation is not a reason to add a section.
- Correct or replace existing instructions instead of appending updates. Use
  plain, direct wording; omit promotional language and repeated explanations.
- Do not add implementation narratives, debugging history, validation reports,
  timings, hashes, artifact inventories, agent handoffs, or migration chronicles.
- Use `--help` for CLI options and link to existing documentation for details.
  Do not duplicate those details or create extra Markdown files to hold material
  removed from the README.
- Before finishing a README edit, review the whole file for stale instructions,
  duplication, and unnecessary detail. Preserve the commands needed to get started.
