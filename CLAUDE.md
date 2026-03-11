# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read CLAUDE.md carefully before starting any task. If you're about to write documentation, benchmarks, or make git commits, check the relevant section of CLAUDE.md for my conventions first.

## Project Context

This is a Rust project with Python bindings (PyO3). Primary language is Rust. When writing documentation, take a Python-first perspective targeting data scientists, similar to Polars documentation style. Do not make docs too Rust-centric.

## Project Overview

hotcoco is a pure Rust port of [pycocotools](https://github.com/ppwwyyxx/cocoapi) with PyO3 Python bindings. It provides 11-26x speedups over pycocotools for bbox, segmentation, and keypoint evaluation.

- **Primary language:** Rust. All core logic lives in `hotcoco`.
- **Python bindings:** PyO3/maturin in `hotcoco-pyo3`, exposed as the `hotcoco` Python package.
- **CLI:** `hotcoco-cli` binary wrapping the Rust library.

## Workspace Structure

```
crates/hotcoco/      # Pure Rust library — types, mask ops, COCO API, evaluation
crates/hotcoco-cli/  # CLI binary
crates/hotcoco-pyo3/ # PyO3 Python bindings (cdylib, built with maturin)
```

### Key Architecture

- `hotcoco-pyo3` uses `hotcoco-core` as the Cargo dependency alias for `hotcoco` to avoid name collision with the `hotcoco` Python module name
- Python bindings return plain dicts (not wrapped Rust structs) matching pycocotools conventions
- Mask operations handle numpy row-major <-> Rust column-major transposition in the PyO3 layer
- `cargo build --workspace` will fail at link time for hotcoco-pyo3 (expected — cdylib needs Python). Use `cargo check` instead, or build via maturin.

## Metric Parity

All 12 COCO evaluation metrics (AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl) must match pycocotools. Keypoints has 10 metrics (no small area range).

- **Always ensure exact parity when modifying evaluation logic.** Run `cargo test` after Rust changes.
- Verified on val2017: keypoints exact, bbox within 0.0001, segm within 0.0002.
- When in doubt, run differential tests against pycocotools on real COCO data before declaring a task complete.

### Verification Sequence (after any eval.rs change)

```bash
cargo test                                                    # 1. All tests pass
cd crates/hotcoco-pyo3 && uv run python data/parity.py       # 2. Parity vs pycocotools
uv run python data/bench.py                                   # 3. (optional) speed check
```

Tolerances: bbox ≤1e-4, segm ≤2e-4, kpts exact.

## Benchmarking

- **Use wall clock time**, not CPU time.
- **Only scale detections** when creating synthetic benchmarks (never scale ground truth).
- Format benchmark tables consistently: columns are `[Eval Type | pycocotools | faster-coco-eval | hotcoco]`, times in seconds with 2 decimal places, speedups in parentheses vs pycocotools.
- Always verify all 12 metrics still match before reporting timing results.

## Testing

- Run `cargo test` after any Rust code changes and verify all tests pass before committing.
- For Python binding changes, run from `crates/hotcoco-pyo3/`: `uv run maturin develop --release && uv run python -c 'import hotcoco'` as a smoke test.

## Skills

Custom skills for this project (invoke with `/skill-name`):
- `/parity` — run parity check vs pycocotools
- `/bench` — run benchmarks and update README tables
- `/docs` — documentation workflow for new API surface
- `/ship` — feature-complete: sync docs/changelog, then commit
- `/adversarial-parity` — attacker/fixer loop to find parity bugs
- `/voice` — audit tone and style in a single doc file

## Workflow Preferences

- Before making large-scale changes (docs revamps, major refactors), present a concrete preview or small example for approval first. Do not rewrite everything at once.

## Build Commands

```bash
cargo build                    # Build all crates
cargo test                     # Run all tests
cargo test -p hotcoco          # Run library tests only
cargo check -p hotcoco-pyo3    # Check pyo3 crate (can't link without Python)
cargo clippy                   # Lint
cargo fmt --all                # Format (use --all, not --workspace)
cargo fmt --all -- --check     # Check formatting

# Python bindings (run from crates/hotcoco-pyo3/)
# One-time setup: cd crates/hotcoco-pyo3 && uv venv && uv pip install maturin ".[dev]"
uv run maturin develop --release  # Build + install into .venv
```

## Refactoring

Use a task agent to find every file and line that references the old naming convention, then summarize what needs to change before making any edits.

## Documentation

- This project targets Python users first, Rust users second. Documentation, README, and examples should lead with Python usage in a Python-first tone similar to Polars. Do not be Rust-centric.
- Before writing the full documentation, show me an outline with 2-3 example sections so I can confirm the tone, structure, and audience focus. Do not generate all pages until I approve.

Docs are built with Zensical (config: `zensical.toml`). Preview locally with `zensical serve`.

When updating documentation (`docs/`) or `README.md`, always ensure both reflect the same information. Any change to one must be checked against the other — benchmark numbers, API examples, CLI flags, installation instructions, and feature descriptions must stay consistent across both.

## Pre-Commit Checks

A git pre-commit hook in `.github/hooks/pre-commit` runs these three checks automatically — they mirror CI exactly. All three must pass or the commit is rejected.

```bash
cargo fmt --all -- --check                              # 1. Formatting
cargo clippy --workspace --all-targets -- -D warnings   # 2. Lint (warnings are errors)
cargo test                                              # 3. Tests
```

To install the hook (one-time setup):

```bash
ln -sf ../../.github/hooks/pre-commit .git/hooks/pre-commit
```

If formatting fails, run `cargo fmt --all` to fix, then re-commit. If clippy fails, fix the warning before committing. Never suppress clippy warnings with allows. Never skip the hook with `--no-verify`.

## Git Workflow

- **Never commit or push unless explicitly asked.** Wait for the user to say "commit", "push", or "ship it" before running any git commit/push commands.
- When committing and pushing, always verify the current git status first to avoid trying to commit already-committed changes. Check `git status` and `git log --oneline -3` before any commit/push operation.
- Keep commits clean: never include build artifacts, compiled files, or `__pycache__` directories. Review staged files carefully before committing. If unsure, ask before committing.
- Commit message body: use bullet points, not prose paragraphs.
- Main branch: `main`.
