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
