# hotcoco core crate

## The layered architecture

`crates/hotcoco/src/` splits by **what a function produces**, not by which family calls it:

| Layer | Produces | Contents |
|---|---|---|
| `primitives` | matches and similarities | `sim`, `greedy`, `assign`, `panoptic` |
| `metrics` | numbers from matches | `counts`, `calibration`, `confusion`, `bootstrap`, `panoptic` |
| `report` | the cross-family output contract | `EvalReport`, `Provenance` |
| `detection` | the detection family driver | `COCOeval` + AP/AR, LVIS, Open Images, TIDE |
| `panoptic` | the panoptic family driver | `PanopticEval` + PQ/SQ/RQ, PNG and mask inputs |
| `quality` | dataset introspection | health checks, statistics |

`primitives` and `metrics` are **free functions over flat arrays** — callable with no
evaluator, the way `sklearn.metrics` and `torchmetrics.functional` are. `COCOeval`'s
analysis methods (`calibration`, `confusion_matrix`, `compare`, `f_scores`) are
*adapters*: they decide which detections count, marshal them into arrays, and call the
shared function. **Put metric math in `metrics`, never in a `COCOeval` method** — the
alternative is what 1.0 spent its whole cycle undoing.

Dependencies run one way: `detection` → `metrics` → `primitives`, and `panoptic` →
`metrics` → `primitives` beside it — `tests/architecture.rs` holds `panoptic/` to an
allowlist that excludes `detection`. Tracking and concepts will be further siblings,
composing the same two layers, each with an allowlist entry of its own.

Detection keeps what is genuinely detection-shaped: TIDE's Cls/Loc/Both/Dupe/Bkg
taxonomy is about box localization vs classification, and `image_diagnostics` reports
per-image fields. Those would need redesigning, not moving, to serve another family.

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
  `COCOeval::reference_deviations()` — the single predicate driving both the
  `summarize()` warnings and `Provenance`. Adding a condition at the `report()` call
  site instead produces a silent downgrade with no warning, which is the exact failure
  it was written to prevent.
