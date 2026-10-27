# hotcoco documentation style

hotcoco documentation follows the
[Google developer documentation style guide](https://developers.google.com/style),
with the deviations recorded at the end of this file. Google is the default; when
this file is silent, follow Google. When Google is silent, follow the
[Google word list](https://developers.google.com/style/word-list) and then the
Chicago Manual of Style.

The choice was cross-shopped in 2026-08 against the alternatives — Microsoft,
Red Hat, GitLab, and the Diátaxis framework — and Google was reaffirmed. Don't
re-litigate it. Diátaxis is a structure framework, not a style guide; the
guide/API page split already follows its discipline, and no further adoption is
planned.

**Scope.** These rules govern published prose: `docs/`, `README.md`, `CONTRIBUTING.md`,
Rust `///` and `//!` doc comments, PyO3 `#[doc = "..."]` strings, Python docstrings,
`.pyi` stubs, CLI `help=` text, and user-facing warning and error messages. They do not
govern `//` and `#` implementation comments, which answer to the *why, not what* rule in
`CONTRIBUTING.md` (Code style) instead.

## Voice

Google supplies the mechanical rules; this section is the voice on top of them.

- **Python-first, data-scientist audience.** Doc comments on public API read like the
  Polars or numpy docs — plain English, example-driven, no Rust jargon in the prose.
  Type signatures handle that.
- **Plain, not promotional.** hotcoco is an open-source project, not a pitch. No
  taglines, no paired slogans, no sales rhetoric. The test for every sentence: you could
  say it to a colleague without wincing. State concrete facts in plain verbs — the
  register of the ruff and uv READMEs. The perception-toolkit framing stays; the sales
  voice does not.
- **Singular, confident tone.** Direct imperative: "Run `evaluate()` before calling
  this."
- **Lean prose.** Every sentence earns its place. No "It should be noted", no "This
  function".
- **Why, not what.** The signature shows what. The doc comment explains intent,
  tradeoffs, and non-obvious invariants.

For a voice-only pass on a file: fix prose in place, do not touch logic or structure,
and leave doc comments that are already good alone.

## Canonical homes

Every fact has exactly one page that owns it; every other surface links there instead of
restating it. Before writing a paragraph, check whether its owner already exists.

| Fact | Owner | Everyone else |
|---|---|---|
| Benchmark tables, timing numbers, parity figures | `docs/benchmarks.md` | README and index keep one headline number and a link |
| Provenance and `"extension"` semantics | `docs/guide/results.md` | one line and a link |
| Warning categories (`HotcocoWarning` and its subclasses) | `docs/guide/results.md` (Filtering the warnings) | name the category and link |
| camelCase alias table | `docs/getting-started/migration.md` | one note at the top of an API page, never per-method admonitions |
| `init_as_pycocotools()` usage | `docs/guide/frameworks.md` | a code snippet is fine in README and quickstart; explanation links |
| RLE `counts` bytes-to-str gotcha | `docs/guide/masks.md` | link |
| LVIS metric key set | `docs/guide/lvis-open-images.md` | link |
| Repo-checkout workflows (`just download-coco`, build from source, parity scripts) | `CONTRIBUTING.md` and the benchmarks "Reproducing" section | user-facing pages never include them |

**Guide pages** hold worked examples and interpretation; **API pages** hold signatures,
parameters, and return shapes. Never copy a return-shape table into a guide. The guide
splits three ways: the core pipeline in `docs/guide/evaluation.md`, LVIS and Open Images
protocols in `docs/guide/lvis-open-images.md`, and confusion, TIDE, calibration, and
comparison in `docs/guide/diagnostics.md`. Metrics composed from `primitives` and
`metrics` live in `docs/guide/custom-metrics.md`, which links to diagnostics for reading
what it builds.

Admonitions flag genuine traps only. If the same admonition would appear twice on one
page, replace all of them with one note at the top.

## Rules

### Headings and titles

Sentence case. Capitalize the first word and proper nouns only.

```
# Mask operations          # Working with results          ## Format conversion
```

Acronyms and product names keep their capitalization: `# LVIS and Open Images`,
`## Ultralytics YOLO`, `### From the CLI`, `## What is RLE?`.

Nav labels in `zensical.toml` are titles too, and must match the page's H1.

Spell out `and` in headings and in bold bullet labels — never `&`. Product names are the
exception: `### Weights & Biases`, `W&B`.

Changing `&` to `and` **moves the heading's anchor**, because `&` slugifies to nothing.
Run `just docs-links` and repoint inbound links. Sentence-casing alone is anchor-safe;
the slugifier lowercases either way.

### Words to avoid

| Don't write | Write instead |
|---|---|
| `e.g.` | `for example`, `such as`, or recast the sentence |
| `i.e.` | `that is`, or recast |
| `etc.` | `and so on`, name the rest, or recast |
| `simply`, `easily`, `just`, `obviously`, `straightforward` | delete it |
| `please` | delete it |
| `allows you to`, `enables you to` | say what it does |
| `in order to` | `to` |
| `Note that` | delete it |
| `and/or` | name the combinations |
| `sanity check` | `confidence check`, `quick check` |
| `dummy` | `placeholder` |
| `abort` | `stop`, `cancel` |
| `blindly` | `without checking` |
| `we`, `our` | second person, or name the actor |

### may, might, can

Google splits three senses that English blurs:

- **`may`** — permission. `metrics` **may not** import a family driver. Correct, keep it.
- **`might`** — possibility. Some metrics **might** show `-1.000`.
- **`can`** — ability. These numbers **can** be compared against a leaderboard.

`you may need to` is almost always `you might need to`.

### Tense

Present tense. Describe what the software does, not what it will do.

- **No:** a 4-core laptop will see less
- **Yes:** a 4-core laptop sees less

### Spatial references

Never `above` or `below` to point at other content on a page — content reflows, and a
screen reader user has no "above." Use `preceding` / `following`, or link to the section
by name.

- **No:** see below · as shown below · the run above · see `image_id` issue above
- **Yes:** see [Verification](#verification) · the preceding run · the next section

`above` and `below` are fine for real comparisons: *scores below the threshold*, *anything
above ~1e-12*. They are also fine when describing actual on-screen layout: *the banner
above the KPI tiles*.

### Link text

Link text names the destination. Never `here`, `this`, `this link`, or `more`. Match the
target page's title so a renamed page is a visible, greppable break.

### Lists, tables, code

- Sentence case in table headers; no terminal punctuation in cells unless they hold full
  sentences.
- Serial (Oxford) comma.
- Every code fence carries a language tag: `python`, `bash`, `json`, `toml`.
- Numbered lists for ordered procedures; each step starts with an imperative verb.

### Error and warning messages

Lowercase, no trailing period, specific about what went wrong and what to do:
`iou_type must be one of: bbox, segm, keypoints`. The word rules above apply — a
message reading `lines may show -1.000` becomes `lines might show -1.000`.

A message string is often quoted verbatim in `docs/`. Change both in the same commit, and
check `cargo test` — some tests assert on message text.

Downstream code matches on warning text too. Before the warning categories, every
hotcoco warning was a plain `UserWarning`, so text was the only thing a filter could
match, and those filters still depend on it: RF-DETR silences the `max_dets` warning with
`^hotcoco: max_dets differ from expected`, and rewording that message brings it back
on every validation epoch of theirs.

## Deviations from Google

**Em dashes are spaced: ` — `, not `—`.** Google specifies no surrounding spaces. hotcoco
uses spaced em dashes throughout, deliberately: the docs are read as often in a terminal
and in raw Markdown as on the rendered site, and an unspaced em dash closes up the line in
monospace. This is the house style; do not "fix" it.

**`vs` is allowed inside code identifiers and short table cells** where `versus` would not
fit — `bytes vs string`. Use `versus` in prose.

**British spellings survive in shipped CHANGELOG entries.** `_typos.toml` sets
`locale = "en-us"`, and `typos` runs through pre-commit. Fix the spellings your change
introduces; leave released CHANGELOG text alone.

## Checking

No linter enforces this file — Vale was evaluated in 2026-03 and removed as too noisy for a
single-contributor repo. The checks that do run:

```bash
just docs-links                      # internal links, heading anchors, nav coverage
uv run pre-commit run typos --all-files  # spelling
cargo doc --no-deps 2>&1 | grep warning
```

Before finishing a docs change, grep your own diff for the words-to-avoid table. It takes a
minute and catches most of what this file is for.
