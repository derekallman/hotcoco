# AGENTS.md

## Project Overview

hotcoco is a perception evaluation toolkit — a pure-Rust engine with PyO3 Python
bindings. Two families ship: detection (COCO, LVIS, and Open Images protocols over
bbox, segm, keypoints, and OBB), which doubles as a drop-in
[pycocotools](https://github.com/ppwwyyxx/cocoapi) replacement, tens of times faster
on COCO val2017 (`docs/benchmarks.md` owns the current numbers), and panoptic (PQ,
SQ, RQ against panopticapi). Tracking is the next planned family on the same engine —
see `plans/PLAN.md` for the ladder, `plans/POSITIONING.md` for how the docs say so.

### Build and binding mechanics

- Single root `pyproject.toml` acts as both maturin build config and Python package definition; `manifest-path = "crates/hotcoco-pyo3/Cargo.toml"` points maturin at the cdylib. `[tool.uv] package = false` means uv won't auto-build — always use `just build` explicitly.
- `hotcoco-pyo3` uses `hotcoco-core` as the Cargo dependency alias for `hotcoco` to avoid name collision with the `hotcoco` Python module name
- Python bindings return plain dicts (not wrapped Rust structs) matching pycocotools conventions
- Mask operations handle numpy row-major <-> Rust column-major transposition in the PyO3 layer
- `cargo build --workspace` will fail at link time for hotcoco-pyo3 (expected — cdylib needs Python). Use `cargo check` instead, or build via maturin.
- **Type stubs:** five hand-written files must track the Python API — `python/hotcoco/{__init__,detection,metrics,primitives,panoptic}.pyi`. Run `uv run pytest tests/test_stubs.py` for drift, and `uv run pyright` to type-check the shipped package against them. The tests check **names only, never signatures** — review those by hand when changing a signature. For `metrics` they check both directions (stub ⊇ `__all__`, and `__all__` ⊇ the extension module), because a new `#[pyfunction]` that the facade forgets to re-export is otherwise unreachable from Python with the suite green.

### Core-crate architecture

Core-crate architecture rules (layers, single-owner conventions, conformance tests) live
in `crates/hotcoco/AGENTS.md`, which loads when you work under `crates/hotcoco/`. Read it
before adding metric or evaluation code anywhere. **Put metric math in `metrics`, never
in a `COCOeval` method.**

## Metric Parity

All COCO evaluation metrics must match pycocotools: 12 for bbox/segm, 10 for keypoints (no small area range), 13 for LVIS (adds APr/APc/APf/AR@300).

- **Always ensure exact parity when modifying evaluation logic.** Run `cargo test` after Rust changes.
- Verified on val2017 for bbox, segm, and keypoints; `docs/benchmarks.md` (Metric parity) owns the measured figures and CONTRIBUTING.md owns the tolerance.
- When in doubt, run differential tests against pycocotools on real COCO data before declaring a task complete.
- After any change to evaluation logic, run `just test`, then `just parity` if `data/` exists. Do not commit or update docs until both are clean.
- **Not everything has a *checked* reference.** `scripts/parity.py` covers pycocotools (bbox/segm/keypoints) on val2017, `scripts/parity_tide.py` covers tidecv, and `scripts/parity_panoptic.py` covers panopticapi on panoptic val2017; all three need `data/` (`just download-panoptic` fetches the panoptic part). In CI, `tests/test_parity_lvis.py` covers LVIS, `tests/test_mask_parity.py` covers `pycocotools.mask`, `tests/test_parity_panoptic.py` covers panoptic against panopticapi run live on synthetic label maps, and `tests/test_parity_oid.py` covers Open Images against frozen output of the TF Object Detection API. Oriented boxes have **no parity script**, which is why `report()` marks them `Provenance::Extension`; Open Images is `Extension` too, but for a different reason — the Challenge's non-exhaustive image-level-label rule is unimplemented, not uncompared.

  Say *no parity script*, not *no reference implementation* — the distinction matters. Open Images had two reference implementations all along (the TF Object Detection API, which the official protocol page points to, and FiftyOne), and a group-of defect went unnoticed for the life of the feature partly because the "no reference exists" framing made a comparison look impossible rather than merely unwritten. Oriented boxes genuinely have no reference protocol, but their IoU kernel is checked against Shapely.
- **Panoptic follows panopticapi, quirks included.** Ground-truth area comes from the JSON, not the PNG; a prediction more than half on void plus the *last* crowd segment of its category is ignored; a PNG id missing from the GT JSON is neither a segment nor void. Where the reference raises, `run()` errors; where it would divide by zero (an empty things or stuff split), hotcoco reports `-1.0`. The formula home is `primitives/panoptic.rs` — a second entry in the architecture test's `FORMULA_HOMES`, because the void-discounted union is a different formula, not a copy.
- **Open Images follows the Challenge protocol**, not V2: a group-of box counts as one ground truth, its best-scoring enclosed detection is a TP, surplus detections are ignored, and an undetected group-of box is a miss. "Inside" is IoA (intersection ÷ *detection* area), the same measure as COCO `iscrowd`. Equivalent to TF `group_of_weight = 1.0`. Both protocols are real — see `docs/guide/lvis-open-images.md` — so name which one before changing anything here.

## Testing

- For Python binding changes: `just build` as a smoke test, then `just parity` to verify metrics.
- **Layout rule:** if pytest collects it, it lives in `tests/`; if you run it with `python`, it lives in `scripts/`. CONTRIBUTING.md's Tests section owns the rest: which file checks which reference, and where the fixtures and shared helpers live.
- `just test` runs `cargo test` + the whole Python suite (bare `pytest`) — the same set CI runs, no `data/` needed, under 30s.
- `just fuzz` runs the hypothesis-based fuzzer (`tests/fuzz_parity.py`) — use to hunt for parity bugs, not in CI. Takes several minutes.
- `just fuzz-dropin` runs `tests/fuzz_dropin.py` — one dataset under many in-memory spellings (bytes `counts`, numpy scalars, tuples, missing optional keys) versus pycocotools. `fuzz_parity.py` goes through JSON files and cannot see any of those; this is the fuzzer that finds the issue #5 class. It fails only on a spelling that changes the numbers and prints loud gaps as a summary.
- `just fuzz-torchmetrics` runs `scripts/fuzz_torchmetrics.py` — torchmetrics' `MeanAveragePrecision` with hotcoco swapped in, the way RF-DETR evaluates, versus the pycocotools backend. It is the only fuzzer that sees what torchmetrics does around the evaluator: `float32` threshold grids, `mask.encode` per predicted mask, `mask.area` per annotation, `COCO()` + `dataset =` + `createIndex()`. Its dependencies are the `torchmetrics` dependency group, which `just setup` does not install.
- Model: use the fuzzer to *find* bugs, then prove fixes with Rust integration tests in `crates/hotcoco/tests/`.

### What CI does and does not check

- `just semver` gates the public Rust API (needs `cargo install cargo-semver-checks --locked`).
  **Do not pin its baseline.** A pinned baseline plus a major version bump runs 0 checks
  and still prints "no semver update required" — a gate that passes while checking
  nothing. The `semver` recipe carries a comment explaining this; leave it there.
- **`typos` runs through `.pre-commit-config.yaml`** — on staged files in the git
  hook, over the whole tree in CI and `just hooks`. The hook is **report-only**: the
  upstream default `--write-changes` is dropped on purpose, because auto-rewriting
  would corrupt the CHANGELOG entries that quote British spellings. Fix the hits your
  change introduces by hand; don't rewrite shipped CHANGELOG entries.
- **Real-data parity is local-only.** `data/` is gitignored, so `just parity` cannot run
  in CI. The Python CI job asserts only that 12 metrics come out and one is positive —
  it would not catch a wrong number. Run parity locally before claiming metrics hold.

### Verifying a check can fail

When you add or change a guard — a conformance test, a parity script, a CI gate — prove
it fails on a real violation before trusting it. This repo has produced the same defect
repeatedly: `parity_tide.py` exited 0 in both failure modes, a pinned semver baseline
ran 0 checks, the layering tests passed against four spellings of the import they
banned. In each case the check looked green and verified nothing. Injecting a violation
and watching it fail takes a minute and is not optional.

## Tool Preferences

- **Always use `uv run python` — never bare `python` or `python3`.** This project uses uv-managed Python; the OS Python is not the project environment.
- For library/crate documentation, prefer a docs MCP server (e.g. context7) when one is connected. None is configured by default, so WebFetch against docs.rs, python.org, and PyPI is the working fallback — those domains are already in the project allowlist.

## Build Gotchas

`just` is the task runner — run `just --list` for the current recipes. The non-obvious parts:

- **`uv sync` alone is not enough.** Without `--all-extras` it skips `maturin`, and `just build` then fails. Always use `just setup` for first-time setup.
- The `coco` CLI is installed into `.venv/bin/coco` by `just build`. Run it as `uv run coco <subcommand>` (or activate the venv with `source .venv/bin/activate` for bare `coco`).

### Keep `target/` bounded

**Cargo has no eviction policy.** Nothing in the toolchain removes stale build
artifacts — not by age, not by size, not by LRU. A long-lived working copy grows
monotonically until the disk fills. Assume nothing cleans up for you.

- `just disk` reports what `target/` costs; `just clean` reclaims it. A clean is
  always safe: it costs one rebuild (~25s cold for all test binaries) and nothing
  else. The compiled Python extension lives at `python/hotcoco/hotcoco.abi3.so`,
  **outside** `target/`, so `import hotcoco` survives a clean.
- A full cold build of every target is ~1.1 GB. That is the number to compare
  against — if `just disk` reports several GB, something is wrong, not merely used.
- **Do not remove `split-debuginfo`/`debug` from `[profile.dev]` and
  `[profile.test]`.** Cargo's macOS default (`unpacked`, full debuginfo) writes
  ~200 loose `.o` files per crate build into `target/debug/deps` and never collects
  them. With the pre-commit hook running `cargo test` on every commit, this repo
  reached **87,134 object files / 18 GB — 96% of `target/`**. The profile comments
  in `Cargo.toml` record why each value is set; the pre-commit hook warns (never
  blocks) if stray `.o` files reappear, so a regression is caught on the next
  commit rather than after weeks of growth.
- Each git worktree under `.claude/worktrees/` carries its **own** `target/`. They
  inherit the workspace profiles, but `just clean` only cleans the one you are in.

## Documentation

`STYLE.md` at the repo root is the style authority: hotcoco follows the
[Google developer documentation style guide](https://developers.google.com/style), and
`STYLE.md` records the rules that come up most plus the deliberate deviations (spaced em
dashes stay). It governs `docs/`, `README.md`, `CONTRIBUTING.md`, `///` and `//!` doc
comments, PyO3 `#[doc]` strings, Python docstrings, `.pyi` stubs, CLI `help=` text, and
user-facing warning and error messages — but not `//` implementation comments. No linter
enforces it; grep your own diff.

- This project targets Python users first, Rust users second. Documentation, README, and examples should lead with Python usage in a Python-first tone similar to Polars. Do not be Rust-centric.
- Before making large-scale changes (docs revamps, major refactors), present a concrete preview or small example for approval first. Do not rewrite everything at once. For small additions (a single new page, a new section), just write it directly.

When updating documentation (`docs/`) or `README.md`, always ensure both reflect the same information. Any change to one must be checked against the other — benchmark numbers, API examples, CLI flags, installation instructions, and feature descriptions must stay consistent across both.

Two structural rules, both learned the hard way (2026-08 docs audit):

- **Every fact has one owning page**; other surfaces link, never restate. The owner map is in `STYLE.md` (Canonical homes). The audit found the same content in up to 9 places, already drifting.
- **ROADMAP.md is forward-looking only.** When something ships, delete its roadmap entry — the CHANGELOG is the record. No `**Shipped.**` markers, no strikethrough.

Internal planning docs live in `plans/` at the repo root (gitignored). Never put them under `docs/` — zensical builds every `.md` there, nav or not, so they would be published.

`just docs-links` checks every internal link, heading anchor, and nav entry. Run it after
any page split, rename, or heading change — a moved heading breaks inbound anchors
silently, and the site build does not catch it.

## Design System ("Cyanotype")

All visual surfaces (browse UI, docs site, matplotlib, Plotly dashboard) share the **Cyanotype** theme. The canonical spec — colors, fonts, the 10-color chart palette, and which file owns each token — is the `cyanotype` skill. Consult it before changing any colors, fonts, or chart palettes, and when adding a new visual surface. Never eyeball new values.

## Pre-Commit Checks

A git pre-commit hook in `.github/hooks/pre-commit` runs `pre-commit run` on the staged
files, then formatting, clippy, and tests. All must pass or the commit is rejected.
CI runs every hook in the same pre-commit configuration over the whole tree.

To install the hook (one-time setup — works in both main repo and worktrees):

```bash
git config core.hooksPath .github/hooks
```

`pre-commit` itself comes from the dev extra, so `just setup` installs it.

**Never run `pre-commit install`.** It refuses to install while `core.hooksPath` is set,
and the bash hook already invokes it — `pre-commit` on `PATH`, else `uv run`.
`.pre-commit-config.yaml` holds the trivial hygiene hooks (whitespace, line endings,
YAML/TOML/JSON syntax) plus `ruff` and `typos`. It is the sole source of Ruff versions
for local commands and CI; `pyproject.toml` holds Ruff settings only. Vendored and
generated trees (`external/`, `tests/fixtures/`, `python/hotcoco/_fonts/`, minified
bundles under `python/hotcoco/static/`) are excluded so the whitespace fixers cannot
rewrite them. Check the whole tree with `just hooks`.

If formatting fails, run `cargo fmt --all` to fix, then re-commit. If clippy fails, fix the warning before committing. **Never suppress clippy warnings with allows. Never skip the hook with `--no-verify`.**

## Git Workflow

- **Never commit or push unless explicitly asked.** Wait for the user to say "commit", "push", or "ship it" before running any git commit/push commands.
- When committing and pushing, always verify the current git status first to avoid trying to commit already-committed changes. Check `git status` and `git log --oneline -3` before any commit/push operation.
- Keep commits clean: never include build artifacts, compiled files, or `__pycache__` directories. Review staged files carefully before committing. If unsure, ask before committing.
- Commit message body: use bullet points, not prose paragraphs.
- **Standard pre-commit sequence:** `/simplify` → `/code-review` → `/ship` → `/commit`. Run all four in order when a feature is done.
- `/ship` is the gate for communication surfaces — CHANGELOG, ROADMAP, docs sync, and parity. Do not commit and then update docs/CHANGELOG after; update everything first, then commit once.

### Worktrees

Sessions may run in a git worktree (e.g. `.claude/worktrees/<name>/`).

- **Commits land on the worktree branch**, not `main`. When the user says "push", they likely mean push to `main`. Use `git -C <main-repo> cherry-pick <hash>` then push from the main repo, or ask to confirm.
- **`data/` is gitignored** and won't exist in a fresh worktree. Scripts that need COCO data (`parity.py`, `bench.py`) require a symlink to the main repo's `data/`.
- **`.venv`** is created per-worktree by `just setup`. Run it once in a new worktree.
- **Pre-commit hook** uses `git config core.hooksPath` set to the *relative* path `.github/hooks`, so it resolves in both the main repo and worktrees. Verify with `git config core.hooksPath` — it must print `.github/hooks`. An absolute path, or a symlink under `.git/hooks`, works in the main repo but is not worktree-portable; re-run the setup command in Pre-Commit Checks to correct it.
- **All paths in skills and scripts must be relative** — never hardcode the main repo path.
