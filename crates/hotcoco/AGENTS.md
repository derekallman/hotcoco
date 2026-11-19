# hotcoco core crate

## The layered architecture

`crates/hotcoco/src/` splits by **what a function produces**, not by which family calls it:

| Layer | Produces | Contents |
|---|---|---|
| `primitives` | matches and similarities | `sim`, `greedy`, `assign`, `panoptic` |
| `metrics` | numbers from matches | `counts`, `calibration`, `confusion`, `bootstrap`, `panoptic`, `semantic` |
| `report` | the cross-family output contract | `EvalReport`, `Provenance` |
| `detection` | the detection family driver | `COCOeval` + AP/AR, LVIS, Open Images, TIDE |
| `panoptic` | the panoptic family driver | `PanopticEval` + PQ/SQ/RQ, PNG and mask inputs |
| `semantic` | the semantic family driver | `SemanticEval` + mIoU/fwIoU/mAcc/aAcc, array and PNG label-map inputs |
| `quality` | dataset introspection | health checks, statistics |

`primitives` and `metrics` are **free functions over flat arrays** — callable with no
evaluator, the way `sklearn.metrics` and `torchmetrics.functional` are. `COCOeval`'s
analysis methods (`calibration`, `confusion_matrix`, `compare`, `f_scores`) are
*adapters*: they decide which detections count, marshal them into arrays, and call the
shared function. **Put metric math in `metrics`, never in a `COCOeval` method** — the
alternative is what 1.0 spent its whole cycle undoing.

Dependencies run one way: `detection` → `metrics` → `primitives`, and `panoptic` →
`metrics` → `primitives` and `semantic` → `metrics` → `primitives` beside it —
`tests/architecture.rs` holds `panoptic/` and `semantic/` to an allowlist that
excludes `detection`. `metrics` may reach `primitives::sim` for the one count-based
IoU kernel (`iou_from_counts`); a second IoU formula anywhere else still fails the
build. Tracking and concepts will be further siblings,
composing the same two layers, each with an allowlist entry of its own.

Detection keeps what is genuinely detection-shaped: TIDE's Cls/Loc/Both/Dupe/Bkg
taxonomy is about box localization vs classification, and `image_diagnostics` reports
per-image fields. Those would need redesigning, not moving, to serve another family.

## The matcher reads tables, not records

`src/table.rs` holds the compact columns a detection evaluation reads:
`DetectionTable` (score, area, boxes) and `GroundTruthTable` (area, the ignore
flags each protocol reads, boxes). A table is filled once, from annotation
records or straight from an array, and is never edited: a changed dataset gets
a new table.

- The matcher and box IoU reach annotations only through `detection/dets.rs`
  (`Dets`, `Gts`). Do not add a `get_ann` call to `matching.rs`.
- A field the matcher needs goes in a table, filled in `table.rs`, where the
  rule for reading it off a record lives once (`GtFlags::of`).
- Masks, keypoints, and oriented boxes are still read from records, as are
  TIDE, the confusion matrix, and diagnostics. `plans/ARCH-REVIEW.md` has the
  order they move in.
- A geometry column is its own type (`BoxColumn`), not a field of a row, so
  the next family's table can hold the same column.

## Architecture conformance is enforced

`crates/hotcoco/tests/architecture.rs` fails the build on: a second IoU formula, a
second greedy matcher, a second `MIN_PARALLEL_WORK`, direct access to the
whole-dataset `ious` cache, and layer violations.

**Never weaken these to make a build pass.** The layering checks are *allowlists* on
purpose. Banning `crate::detection` was tried twice and was worthless both times: the
crate re-exports the same types ~25 times at its root, and `super::` written in a
`mod.rs` reaches those too. A banlist must enumerate every path to a thing and a
re-export silently adds one; an allowlist enumerates what a layer is *for*. Widening
one is a decision about what the layer means — make it deliberately, and prove the
check still fails on a real violation before trusting it.

## Conventions with exactly one owner

Each of these was duplicated across 3–5 sites before 1.0. Call the owner; don't
re-derive.

- **`-1.0` means "not computed for this configuration"** — never a low score. It shows
  up for an area range with no ground truth, or a category absent from the split.
  `metrics::mean_or_missing` is the only producer, beside its predicate
  `metrics::is_computed`; each family's `report()` filters on it before emitting a
  per-class metric and `metrics::counts::max_f_beta` skips it.
- **`Params::all_area_idx()`** is the only `"all"` area-range lookup.
- **`params::default_rec_thrs()`** (a free function, not a `Params` method) is the only
  101-point recall grid. A caller using the `metrics` functions directly needs the same
  grid `COCOeval` defaults to, or the two silently disagree about what AP means.
- **Provenance is a property of the whole configuration**, not of an `iou_type`.
  Anything that can make a run incomparable to a reference belongs in
  `COCOeval::deviations_for` — the single predicate behind `reference_deviations()`
  (live `params`, which `Provenance` reads) and `summary_deviations()` (the params
  `accumulate()` recorded, which the `summarize()` warnings read). Adding a condition
  at the `report()` call site instead produces a silent downgrade with no warning,
  which is the exact failure it was written to prevent.
