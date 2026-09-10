# Contributing to hotcoco

Thanks for your interest in contributing! This guide covers everything you need to build, test, and submit changes.

## Getting started

### Prerequisites

- [Rust](https://rustup.rs/) (stable, 1.70+)
- [uv](https://docs.astral.sh/uv/) — Python dependency management (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [just](https://just.systems/) — task runner (`cargo install just`)
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
uv sync --all-extras  # creates .venv at repo root with all dev deps
just build            # builds and installs the extension into .venv
```

Verify the install:

```bash
uv run python -c "import hotcoco; print(hotcoco.__version__)"
```

## Architecture

All core logic lives in the Rust library. The Python package and Rust CLI are thin wrappers.

The core is split by **what a function produces**, not by which family calls it, because
detection is the first metric family rather than the only one. Panoptic and tracking are
siblings of `detection/`, composing the same `primitives/` and `metrics/` layers — so a
kernel or a formula that a second family would need belongs in a shared layer the day it
is written, not the day the second family arrives.

```
                       ┌──→ PyO3 ──→ hotcoco (Python library + CLI)
Rust Core (all logic) ─┤
                       └──→ hotcoco-cli (Rust CLI, eval only)
```

- **Rust core** (`crates/hotcoco/`) — types, masks, eval, dataset ops, format conversion. Layered by what a function produces: `primitives/` (matching kernels), `metrics/` (numbers from matches), `detection/` (the COCO/LVIS/Open Images driver), `quality/` (dataset introspection). `tests/architecture.rs` enforces the layering and fails the build on duplicated kernels.
- **Python CLI** (primary) — all subcommands: `coco eval`, `coco stats`, `coco merge`, and the rest. Rich formatting and plots.
- **Rust CLI** (`crates/hotcoco-cli/`) — evaluation only, no Python. New features added on request.

| Registry | Package | Contents |
|----------|---------|----------|
| PyPI | `hotcoco` | Python library + Python CLI + compiled Rust core |
| crates.io | `hotcoco` | Rust library |
| crates.io | `hotcoco-cli` | Rust CLI binary (eval only, no Python) |

## Making changes

### Pre-commit hook

A hook in `.github/hooks/pre-commit` runs formatting, lint, and tests automatically before every commit. Install it once:

```bash
git config core.hooksPath .github/hooks
```

(The relative path resolves in both the main repo and git worktrees; a symlink under `.git/hooks` does not.)

The hook runs:

1. `cargo fmt --all -- --check` — formatting
2. `cargo clippy --workspace --all-targets -- -D warnings` — lint (warnings are errors)
3. `cargo test` — all tests
4. `pre-commit run --all-files` — when Python files are staged; runs the configured checks across the whole tree

Step 4 needs `pre-commit` on `PATH`; the hook falls back to `uvx pre-commit` and fails if neither is available. Install it with `uv tool install pre-commit`. Don't run `pre-commit install` — `core.hooksPath` points at `.github/hooks`, and pre-commit refuses to install over it. The hook runs `pre-commit run --all-files` when Python files are staged.

Some of those hooks rewrite the file they fix (trailing whitespace, missing final newline). They exit nonzero when they do, so the commit is rejected rather than silently amended — re-stage the file and commit again.

If formatting fails, run `cargo fmt --all` (Rust) or `pre-commit run --all-files`, re-stage modified files, and re-commit. Fix all clippy warnings before committing — never suppress them with `#[allow(...)]`.

### After changing evaluation logic

Verify metric parity against pycocotools on COCO val2017:

```bash
just parity
```

Tolerance: 1e-12 for every iou_type — sized to floating-point noise and nothing else; the measured worst case is recorded in [Metric parity](docs/benchmarks.md#metric-parity).

Specialized protocols have their own gates: `just parity-lvis`, `just parity-tide`, `just parity-mask`, and `just parity-oid` (Open Images vs the TF Object Detection API; `just gen-oid-fixtures` regenerates its fixtures). `just parity-all` runs everything.

### After changing Python bindings

Smoke test:

```bash
just build
uv run python -c "import hotcoco"
```

Then run `uv run pytest scripts/test_stubs.py`. The four hand-written `.pyi` files
don't update themselves, and the test compares them against the extension in **both**
directions — a member the stub omits is invisible to autocomplete, and one the stub
invents is worse, because an IDE offers it and the call fails at runtime.

**Add a camelCase alias only when pycocotools has a method of that exact name.** That
is the whole reason the convention exists: hotcoco is a drop-in replacement, so
`getAnnIds` and `loadRes` must resolve. LVIS is not a reason — its API is snake_case.
Anything hotcoco invented gets one snake_case spelling and no alias; converters,
diagnostics, and Open Images parameters all have no pycocotools counterpart to match.
Four converter aliases were shipped against this rule and later removed.

Every `#[pyfunction]` alias needs an explicit `#[pyo3(name = "...")]`. Without one PyO3
exports the Rust identifier, which is how `fr_py_objects_snake` became public API while
the documented `fr_py_objects` did not exist.

## Code style

- **Rust:** `cargo fmt --all`. No clippy warnings.
- **Python:** `pre-commit run ruff-format --all-files` and `pre-commit run ruff-check --all-files` — enforced by the pre-commit hook and CI. `just py-fmt` and `just py-fmt-check` run the formatting hook, which fixes files and fails when changes are needed. Use `just py-lint` for Python linting.
- **Ruff version:** pinned only in `.pre-commit-config.yaml`. Update that revision to change the version for local commands and CI; keep rule settings in `pyproject.toml`.
- Don't add comments where the logic is self-evident. Comments should explain *why*, not *what* — and "why" means a constraint or invariant the code can't show, not the history of how the code got here. Never narrate a change ("used to", "previously", "the old version...") or justify it against alternatives the reader can't see; that rationale belongs in the commit message.

## Documentation style

Published prose follows the [Google developer documentation style guide](https://developers.google.com/style).
The rules that come up most — sentence-case headings, no `e.g.`/`i.e.`/`etc.`, `might` for
possibility, no "see below" — plus hotcoco's deliberate deviations from Google are recorded
in [STYLE.md](STYLE.md). Read it before writing docs, doc comments, or error messages.

Run `just docs-links` after any page split, rename, or heading change.

## Tests

```bash
cargo test            # All Rust tests
cargo test -p hotcoco # Library tests only
just test             # Build + cargo test + pytest hypothesis suite
```

Test fixtures live in `crates/hotcoco/tests/fixtures/`. The Rust integration
suites are split by area: `integration_test.rs` (end-to-end evaluation),
`convert_fixes.rs` / `core_fixes.rs` / `detection_fixes.rs` (regression tests
pinning fixed defects in the converters, the core data layer, and detection),
and `architecture.rs` (layering conformance — fails the build on duplicated
kernels or layer violations). When adding a feature that touches evaluation, add
a corresponding Rust integration test; when fixing a bug, pin it in the matching
`*_fixes.rs` suite. Python-side regression tests live in `scripts/test_*.py`
(`test_parity.py`, `test_stubs.py`, `test_browse.py`, `test_adversarial.py`).

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
