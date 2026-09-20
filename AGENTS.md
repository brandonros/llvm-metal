# Development

- Use the development shells selected by `flake.lock`: `nix develop --command`
  for compiler work and `nix develop .#rust-fixtures --command` for Rust fixtures.
  Use `path:.` or `path:.#rust-fixtures` when inputs include untracked files.
- Start with focused regression tests, then run affected integration tests and
  `nix develop --command cargo test --locked --workspace` after code changes.
- Run GPU tests explicitly on macOS with an Apple GPU. Run fixture tests serially
  because they share output directories. Follow `tests/rust-fixtures/README.md`
  for prerequisites and commands.
- Distinguish LLVM verification, Metal library/pipeline creation, and checked GPU
  execution. Report which stages ran and any failures or missing prerequisites.
- Scope validation claims to the tested source, toolchain, inputs, GPU, and OS.
  Keep historical results labeled when dependencies or source revisions change.
- Keep small source fixtures in Git and generated artifacts under `target/`.
  Follow `tests/fixtures/README.md` when a regression needs captured IR or bitcode.

## No native code

- This repository has no C, C++ or Objective-C, no `build.rs`, and no `cc`,
  `bindgen`, `cxx` or `autocxx` dependency. Keep it that way. LLVM is reached
  only through its C API (`llvm-sys`, and Inkwell above it).
- Rust cannot call LLVM's C++ API: the symbols are mangled, templated or inline.
  When the C API lacks something, do not add a C++ shim, do not move one into
  the `llvm-sys` or Inkwell forks, do not call mangled symbols, and do not
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
- A single passing kernel does not prove a behaviour is safe to drop. Dropping
  PIC/PIE flag handling passed one GPU test and then failed Apple's pipeline
  compiler on two other kernels.

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
