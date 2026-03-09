# Contributing to hotcoco

Thanks for your interest in contributing! This guide covers everything you need to build, test, and submit changes.

## Getting started

### Prerequisites

- [Rust](https://rustup.rs/) (stable, 1.70+)
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- Python 3.9+

### Build the Rust library

```bash
git clone https://github.com/derekallman/hotcoco.git
cd hotcoco
cargo build
cargo test
```

### Build the Python bindings

```bash
cd crates/hotcoco-pyo3
uv venv
uv pip install maturin ".[dev]"
uv run maturin develop --release
```

Verify the install:

```bash
uv run python -c "import hotcoco; print(hotcoco.__version__)"
```

## Making changes

### Pre-commit hook

A hook in `.github/hooks/pre-commit` runs formatting, lint, and tests automatically before every commit. Install it once:

```bash
ln -sf ../../.github/hooks/pre-commit .git/hooks/pre-commit
```

The hook runs:

1. `cargo fmt --all -- --check` — formatting
2. `cargo clippy --workspace --all-targets -- -D warnings` — lint (warnings are errors)
3. `cargo test` — all tests

If formatting fails, run `cargo fmt --all` and re-commit. Fix all clippy warnings before committing — never suppress them with `#[allow(...)]`.

### After changing evaluation logic

Verify metric parity against pycocotools on COCO val2017:

```bash
cd crates/hotcoco-pyo3
uv run python data/parity.py
```

Tolerances: bbox ≤ 1e-4, segm ≤ 2e-4, keypoints exact.

### After changing Python bindings

Smoke test:

```bash
cd crates/hotcoco-pyo3
uv run maturin develop --release
uv run python -c "import hotcoco"
```

## Code style

- **Rust:** `cargo fmt --all`. No clippy warnings.
- **Python:** No formatter enforced, but keep style consistent with existing code.
- Don't add comments where the logic is self-evident. Comments should explain *why*, not *what*.

## Tests

```bash
cargo test                    # All Rust tests
cargo test -p hotcoco         # Library tests only
cd crates/hotcoco-pyo3 && uv run pytest data/test_parity.py -v
```

Test fixtures live in `crates/hotcoco/tests/fixtures/`. When adding a new feature that touches evaluation, add a corresponding Rust integration test.

## Submitting a pull request

1. Fork the repo and create a branch from `main`.
2. Make your changes and ensure all pre-commit checks pass.
3. If you changed evaluation logic, include parity output in the PR description.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Open a PR — the template will guide you through the checklist.

## Reporting bugs

Use the [bug report template](https://github.com/derekallman/hotcoco/issues/new?template=bug_report.md). Include your OS, Python version, hotcoco version, and a minimal reproducer.

## Questions

Open a [GitHub Discussion](https://github.com/derekallman/hotcoco/discussions) for usage questions, feature ideas, or anything that isn't a clear bug.
