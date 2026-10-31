---
name: ship
description: Feature-complete workflow for hotcoco — sync CHANGELOG, ROADMAP, and docs, verify parity, then commit. Use whenever a feature is done and before every commit. Trigger on "ship it", "ship this", "/ship", or when the user says a change is complete and ready to commit.
---

# Ship

Feature-complete workflow: update all communication surfaces, then commit.
Run this before every commit. Do not commit first and update docs/changelog after.

## Steps

Work through these in order. Each step is a gate — do not proceed until it's done.

### 1. CHANGELOG.md

Add an entry under `[Unreleased]` for every meaningful change. Use bullet points,
grouped under `Added`, `Changed`, `Fixed`, or `Removed` as appropriate. **One
bullet is at most five lines.** Name what changed and where to read more; the
guide page owns the explanation, the docstring owns the signature, and a reader
skimming the release should finish a bullet in one breath. The 1.3 and 1.4 entries
ran to 190 lines of paragraphs that restated the docs, which nobody reads twice. Be specific: name
the function, file, flag, or behavior that changed. Vague entries like "improved
performance" are not acceptable.

Even infrastructure changes (Justfile recipes, layout changes, script moves) belong
here if they affect contributors or users.

**Structure rules — enforce every time you touch CHANGELOG.md:**
- Each `## [version]` section must have at most one `### Added`, one `### Changed`,
  one `### Fixed`, and one `### Removed`. Never create a second `### Added` block;
  merge all new items into the existing one.
- Only these four group names are valid. Never write `### Added (previously)`,
  `### Added (earlier)`, or any variant.
- When adding items across multiple sessions, always find the existing group and
  append to it — do not create a new group with the same name.

### 2. ROADMAP.md

The roadmap is forward-looking only — the CHANGELOG is the record of what shipped.

- For any item that is now complete, **delete it from ROADMAP.md entirely**. Never
  mark items `**Shipped.**` or strike them through with `~~...~~` — that habit is
  how the roadmap once grew into a 333-line changelog and had to be rewritten.
- If only part of an item shipped, delete the shipped part and keep the rest.
- Keep entries brief: a bold name plus 1–5 lines. No architecture diagrams, no
  reference tables — that material lives in CONTRIBUTING.md.

If nothing is newly complete, no change is needed — but you must check.

### 3. Docs sync

For **every** new or changed public API surface (Python or Rust), all of the
following must be true before committing:

- **Links resolve** — run `just docs-links`. It fails on a link to a missing file, a
  heading anchor that no longer exists (the usual casualty of splitting or renaming a
  page), a nav entry with no file, and a page missing from nav. Must be clean.
- **Rust `///` docstrings** — every new/modified public item in `crates/hotcoco/src/`
  has a doc comment. Run `cargo doc --no-deps 2>&1 | grep warning` — must be clean.
- **PyO3 `#[doc]` strings** — every new/modified item in `crates/hotcoco-pyo3/src/lib.rs`
  has Python-audience docstrings. Verify with
  `just build && uv run python -c "import hotcoco; help(hotcoco.ChangedThing)"`.
- **`docs/` site** — a guide page in `docs/guide/` or an API page in `docs/api/` covers
  the feature. If not, write it now. A new page needs a nav entry in `zensical.toml`.
  Check `STYLE.md` (Canonical homes) for the fact's owner before writing a paragraph.
  For bulk rewrites, get outline approval before writing; for small additions, write directly.
- **`README.md`** — user-facing features are mentioned. Cross-check that benchmark
  numbers, API examples, CLI flags, and installation instructions match `docs/` exactly.
- **Style** — prose follows `STYLE.md` (voice and mechanics). Grep the diff for the
  words-to-avoid list:
  ```bash
  git diff HEAD -U0 | grep -nE '^\+.*(\be\.g\.|\bi\.e\.|\betc\.|\bsimply\b|\bplease\b|see below|shown below|allows you to|in order to|[Nn]ote that|and/or)'
  ```

If any of these are incomplete, **stop** and finish them before continuing.

For pure infra/tooling changes with no API surface, skip the docstring and PyO3 checks
but still verify `cargo doc --no-deps` is clean and README is consistent with docs/.

### 4. Parity check

If any evaluation logic changed — anything under `crates/hotcoco/src/detection/`,
`primitives/`, or `metrics/`, or `mask.rs`, `params.rs`, `types.rs`, or the `coco.rs`
load paths:
- Run `just test`, then `just parity` if `data/` exists. Both must pass before committing.
- If `data/` is missing, say so plainly — real-data parity was not checked.
- Do not proceed until parity is confirmed.

### 5. Diff hygiene

The git pre-commit hook already runs formatting, clippy, typos, and tests. These two
greps cover what it does not; a hit is a flag to resolve, not an automatic failure.

```bash
# New TODO/FIXME markers
git diff HEAD -U0 | grep -E '^\+.*\b(TODO|FIXME|HACK|XXX)\b' || echo "No new TODOs"

# Comments that narrate the change instead of stating a constraint — that
# rationale belongs in the commit message, not the file
git diff HEAD -U0 | grep -E '^\+\s*(//|#).*\b(used to|previously|no longer|the old |replaces the)\b' || echo "No history-narrating comments"
```

`git diff` does not list untracked files, so read any brand-new file for the same two
patterns by hand.

### 6. Toolchain freshness (soft nudge)

The Rust toolchain is pinned in `rust-toolchain.toml` and bumped manually. Check
how long since the last bump:

```bash
git log -1 --format='%cd' --date=relative -- rust-toolchain.toml
```

If it has been more than ~3 months, remind the user (do **not** block the commit):
suggest `rustup update stable`, then bumping `channel` in `rust-toolchain.toml` to
the new version and running `cargo clippy --workspace --all-targets -- -D warnings`
to surface and fix any newly-stabilized lints. If the pin is recent or the user
declines, just proceed.

Whenever the channel is bumped, also tell the user to re-run `just setup`. Switching
channels drops every component not listed in `rust-toolchain.toml`, and rust-analyzer
is deliberately not listed there (it would make CI download it too). Its absence is
silent — `~/.cargo/bin/rust-analyzer` is a rustup proxy, so `which` still resolves
while every spawn fails — which is exactly how it went unnoticed before.

Note: this only fires when `/ship` runs, so a long-dormant repo won't be nudged
until the next ship — an accepted limitation of keeping this tool-free (the daily
`cargo-deny` job covers the security side of dormancy regardless).

### 7. Commit

Use the `commit-commands` plugin (`/commit`). Before confirming the staged files:
- Never include `target/`, `*.so`, `.venv/`, `__pycache__`, or data files.
- Confirm the commit message accurately names every changed surface.
