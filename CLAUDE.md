# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read CLAUDE.md carefully before starting any task. If you're about to write documentation, benchmarks, or make git commits, check the relevant section of CLAUDE.md for my conventions first.

## Project Overview

hotcoco is a pure Rust port of [pycocotools](https://github.com/ppwwyyxx/cocoapi) with PyO3 Python bindings. It provides 13-23x speedups over pycocotools for bbox, segmentation, and keypoint evaluation.

- **Primary language:** Rust. All core logic lives in `hotcoco`.
- **Python bindings:** PyO3/maturin in `hotcoco-pyo3`, exposed as the `hotcoco` Python package.
- **CLI:** `hotcoco-cli` binary wrapping the Rust library.

## Workspace Structure

```
crates/hotcoco/      # Pure Rust library — types, mask ops, COCO API, evaluation
crates/hotcoco-cli/  # CLI binary
crates/hotcoco-pyo3/ # PyO3 Python bindings (cdylib, built with maturin)
python/              # Python package source (hotcoco/__init__.py, cli.py, etc.)
scripts/             # Dev scripts: parity.py, bench.py, test_parity.py, fuzz_parity.py, etc.
scripts/fixtures/    # Hypothesis database and generated output (adversarial/ gitignored)
data/                # Large COCO data files (gitignored — see installation docs)
Justfile             # Task runner: just build / test / parity / bench / lint / fmt
```

### Key Architecture

- Single root `pyproject.toml` acts as both maturin build config and Python package definition; `manifest-path = "crates/hotcoco-pyo3/Cargo.toml"` points maturin at the cdylib. `[tool.uv] package = false` means uv won't auto-build — always use `just build` explicitly.
- `hotcoco-pyo3` uses `hotcoco-core` as the Cargo dependency alias for `hotcoco` to avoid name collision with the `hotcoco` Python module name
- Python bindings return plain dicts (not wrapped Rust structs) matching pycocotools conventions
- Mask operations handle numpy row-major <-> Rust column-major transposition in the PyO3 layer
- `cargo build --workspace` will fail at link time for hotcoco-pyo3 (expected — cdylib needs Python). Use `cargo check` instead, or build via maturin.
- **Type stubs:** `python/hotcoco/__init__.pyi` is hand-written and must be updated whenever the Python API changes (new methods, renamed parameters, changed return types). Run `uv run pytest scripts/test_stubs.py` to check for drift. The test catches missing names but not signature mismatches — review the stub manually when changing signatures.

## Metric Parity

All COCO evaluation metrics must match pycocotools: 12 for bbox/segm, 10 for keypoints (no small area range), 13 for LVIS (adds APr/APc/APf/AR@300).

- **Always ensure exact parity when modifying evaluation logic.** Run `cargo test` after Rust changes.
- Verified on val2017: keypoints exact, bbox within 0.0001, segm within 0.0002.
- When in doubt, run differential tests against pycocotools on real COCO data before declaring a task complete.

### Verification Sequence (after any eval.rs change)

```bash
just parity          # build + parity vs pycocotools (bbox ≤1e-4, segm ≤2e-4, kpts exact)
just test            # build + cargo test + fast pytest regression suite
just bench           # (optional) speed comparison
```

## Benchmarking

- **Use wall clock time**, not CPU time.
- **Only scale detections** when creating synthetic benchmarks (never scale ground truth).
- Format benchmark tables consistently: columns are `[Eval Type | pycocotools | faster-coco-eval | hotcoco]`, times in seconds with 2 decimal places, speedups in parentheses vs pycocotools.
- Always verify all 12 metrics still match before reporting timing results.

## Testing

- Run `cargo test` after any Rust code changes and verify all tests pass before committing.
- For Python binding changes: `just build` as a smoke test, then `just parity` to verify metrics.
- `just test` runs `cargo test` + fast Python regression tests (`scripts/test_parity.py`) — safe for CI, completes in under 30s.
- `just fuzz` runs the hypothesis-based fuzzer (`scripts/fuzz_parity.py`) — use to hunt for parity bugs, not in CI. Takes several minutes.
- Model: use the fuzzer to *find* bugs, then prove fixes with Rust integration tests in `crates/hotcoco/tests/`.

## Skills

Custom skills for this project (invoke with `/skill-name`):
- `/parity` — run parity check vs pycocotools
- `/bench` — run benchmarks and update README tables
- `/docs` — documentation workflow for new API surface
- `/ship` — feature-complete: sync docs/changelog, then commit
- `/adversarial-parity` — attacker/fixer loop to find parity bugs
- `/voice` — audit tone and style in a single doc file
- `/plot` — guide design and implementation of matplotlib plots
- `/infographic` — create publication-quality charts and figures
- `/review` — pre-commit quality sweep (fmt, clippy, typos, tests)

Generic skills (from plugins): `/commit`, `/simplify`, `/review-pr`, and others — see plugin list.

## Agents

Nine specialized agents for quality gates (see `.claude/agents/`):
- **docs-consistency-checker** — validates README.md and docs/ stay in sync
- **parity-validator** — runs `just parity` and reports pass/fail with tolerances
- **changelog-drafter** — drafts CHANGELOG.md entries from git diff
- **pyo3-api-reviewer** — reviews PyO3 binding changes for convention violations
- **benchmark-formatter** — formats raw benchmark output as README tables
- **pre-commit-validator** — simulates the pre-commit hook before committing
- **stub-coverage-checker** — validates `__init__.pyi` covers all PyO3 bindings
- **type-cascade-detector** — maps affected modules when core types change
- **release-preflight** — comprehensive pre-release validation gate

## Hooks

Four Claude Code hooks in `.claude/hooks/` enforce conventions automatically:
- **Pre-edit:** blocks lock file edits, warns on commits without CHANGELOG staged, blocks bare `python`
- **Post-edit:** single dispatcher formats code (cargo fmt / ruff), runs cargo check on the right crate, warns when eval.rs changes

These run on every tool use — no manual invocation needed.

## Tool Preferences

- **Always use context7 for library/crate documentation lookups.** Use `mcp__context7__resolve-library-id` then `mcp__context7__query-docs`. Never use WebFetch to browse documentation sites (docs.rs, python.org, PyPI, GitHub READMEs, etc.). WebFetch is for user-provided URLs only.
- **Always use `uv run python` — never bare `python` or `python3`.** This project uses uv-managed Python; the OS Python is not the project environment.

## Build Commands

```bash
# Common workflows (use just from the repo root)
just build       # Build Python extension (maturin develop --release)
just test        # Build + cargo test + fast pytest regression suite
just fuzz        # Build + hypothesis parity fuzzer (slow — bug hunting only)
just parity      # Build + parity check vs pycocotools
just bench       # Build + benchmark
just lint        # cargo clippy (warnings as errors)
just fmt         # cargo fmt --all
just fmt-check   # cargo fmt --all -- --check

# Raw Rust commands (when just isn't needed)
cargo build                    # Build all crates
cargo test                     # Run all tests
cargo test -p hotcoco          # Run library tests only
cargo check -p hotcoco-pyo3    # Check pyo3 crate (can't link without Python)

# One-time Python setup (run this before anything else)
just setup                     # = uv sync --all-extras (installs maturin + all dev deps)
just build                     # Build the Rust extension into .venv
```

**Important:** `uv sync` alone (without `--all-extras`) only installs base deps and will not install `maturin`, causing `just build` to fail. Always use `just setup` for first-time setup.

`uv run python` works from anywhere in the repo (no need to cd first).

The `coco` CLI is installed into `.venv/bin/coco` by `just build`. Run it as `uv run coco <subcommand>` (or activate the venv with `source .venv/bin/activate` for bare `coco`).

## Documentation

- This project targets Python users first, Rust users second. Documentation, README, and examples should lead with Python usage in a Python-first tone similar to Polars. Do not be Rust-centric.
- Before making large-scale changes (docs revamps, major refactors), present a concrete preview or small example for approval first. Do not rewrite everything at once. For small additions (a single new page, a new section), just write it directly.

Docs are built with Zensical (config: `zensical.toml`). Preview locally with `zensical serve`.

When updating documentation (`docs/`) or `README.md`, always ensure both reflect the same information. Any change to one must be checked against the other — benchmark numbers, API examples, CLI flags, installation instructions, and feature descriptions must stay consistent across both.

## Design System ("Cold Brew")

All visual surfaces share the **Cold Brew** theme. The canonical spec lives at `.claude/skills/theme-factory/themes/cold-brew.md` — consult it before changing any colors, fonts, or chart palettes.

**Surfaces and their token files:**

| Surface | Token file | Notes |
|---------|-----------|-------|
| Browse UI | `python/hotcoco/static/style.css` | CSS custom properties in `:root` |
| Docs site | `docs/stylesheets/extra.css` + `zensical.toml` | Zensical (MkDocs Material) variables |
| Matplotlib | `python/hotcoco/plot/theme.py` | `"cold-brew"` is the default theme |
| Dashboard | `python/hotcoco/dashboard.py` | Plotly colorway + surface constants |
| Bundled fonts | `python/hotcoco/_fonts/` | DM Sans TTF for matplotlib |

**Key rules:**
- Accent is Dusty Steel (`#8694A8` dark / `#566272` light), not warm caramel.
- Chart series use the 10-color infographic palette (defined in the spec), not the UI accent colors.
- Fonts are DM Serif Display (display), DM Sans (body), JetBrains Mono (code) on all surfaces.
- When adding a new visual surface, pull colors from the spec — do not eyeball new values.

## Pre-Commit Checks

A git pre-commit hook in `.github/hooks/pre-commit` runs three checks automatically. All must pass or the commit is rejected.

```bash
cargo fmt --all -- --check                                              # 1. Formatting
cargo clippy --workspace --all-targets -- -D warnings                   # 2. Lint (all crates)
cargo test -p hotcoco -p hotcoco-cli                                    # 3. Tests
```

The hook excludes `hotcoco-pyo3` from tests (it's a cdylib with no Rust tests). Clippy runs on the full workspace to match CI.

To install the hook (one-time setup — works in both main repo and worktrees):

```bash
git config core.hooksPath .github/hooks
```

If formatting fails, run `cargo fmt --all` to fix, then re-commit. If clippy fails, fix the warning before committing. Never suppress clippy warnings with allows. Never skip the hook with `--no-verify`.

## Git Workflow

- **Never commit or push unless explicitly asked.** Wait for the user to say "commit", "push", or "ship it" before running any git commit/push commands.
- When committing and pushing, always verify the current git status first to avoid trying to commit already-committed changes. Check `git status` and `git log --oneline -3` before any commit/push operation.
- Keep commits clean: never include build artifacts, compiled files, or `__pycache__` directories. Review staged files carefully before committing. If unsure, ask before committing.
- Commit message body: use bullet points, not prose paragraphs.
- Main branch: `main`.
- **Standard pre-commit sequence:** `/simplify` → `/review` → `/ship` → `/commit`. Run all four in order when a feature is done.

### Worktrees

Sessions may run in a git worktree (e.g. `.claude/worktrees/<name>/`). Worktrees share the same `.git` history but have their own branch and working directory.

- **Commits land on the worktree branch**, not `main`. When the user says "push", they likely mean push to `main`. Use `git -C <main-repo> cherry-pick <hash>` then push from the main repo, or ask to confirm.
- **`data/` is gitignored** and won't exist in a fresh worktree. Scripts that need COCO data (`parity.py`, `bench.py`) require a symlink: `ln -s /Users/derek/coco-rust/data data`
- **`.venv`** is created per-worktree by `just setup`. Run it once in a new worktree.
- **Pre-commit hook** uses `git config core.hooksPath` (not a symlink), so it works in both the main repo and worktrees.
- **All paths in skills and scripts must be relative** — never hardcode the main repo path.

### Before every commit — communication surfaces

Run `/ship` before committing. It enforces these gates in order:

1. **CHANGELOG.md** — every meaningful change has a bullet under `[Unreleased]`, grouped Added / Changed / Fixed. Be specific.
2. **ROADMAP.md** — newly completed items marked `**Shipped.**` with sub-bullets struck through.
3. **Docs sync** — for any new or changed public API:
   - `cargo doc --no-deps 2>&1 | grep warning` is clean
   - PyO3 `#[doc]` strings present; `help(hotcoco.Thing)` looks correct
   - `docs/` site has a page or section for the feature
   - `README.md` and `docs/` are consistent (numbers, flags, examples, install steps)
4. **Parity** — if eval logic changed, `just parity` must pass first.

Do not commit and then update docs/CHANGELOG after. Update everything first, then commit once.
