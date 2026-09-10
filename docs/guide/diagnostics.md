# Model diagnostics

AP tells you how good a model is; this page is about finding out *why* it isn't better. Confusion matrices show which categories get mixed up, TIDE decomposes errors by type, calibration checks whether confidence scores mean anything, and per-image diagnostics surface the images — and the annotations — worth looking at.

!!! note "Call order"
    Everything on this page except `confusion_matrix()` needs `evaluate()` first, and
    `f_scores()` also needs `accumulate()`. `confusion_matrix()` runs its own matching
    pass and works on a freshly constructed `COCOeval`.

## Confusion matrix

The standard AP pipeline only ever matches detections against ground truth of the **same** category. That means it can't tell you *which* categories your model confuses. `confusion_matrix()` fixes this with a separate cross-category matching pass.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
cm = ev.confusion_matrix(iou_thr=0.5, max_det=100)
```

### Reading the matrix

`cm["matrix"]` is a `(K+1) × (K+1)` numpy uint64 array where `K` is the number of categories (unsigned — cast with `.astype(np.int64)` before subtracting counts). **Rows are ground truth, columns are predicted.** The extra row and column at index `K` represent "background" — unmatched ground truth (missed detections / false negatives) and unmatched detections (false positives) respectively. So `matrix[i][j]` for `i, j < K` counts GT category `i` predicted as category `j` — on-diagonal is a true positive, off-diagonal a class confusion — while `matrix[i][K]` counts false negatives of category `i` and `matrix[K][j]` counts false positives of category `j`. The full return shape is in the [`confusion_matrix` API reference](../api/cocoeval.md#confusion_matrix).

```python
cm = ev.confusion_matrix(iou_thr=0.5)

# Raw counts
matrix = cm["matrix"]          # np.ndarray uint64, shape (K+1, K+1)
cat_ids = cm["cat_ids"]        # list of category IDs for rows/cols 0..K-1

# True positives per category (diagonal, excluding background)
tp_per_cat = matrix.diagonal()[:-1]

# False negatives per category (GT matched to background column)
fn_per_cat = matrix[:-1, -1]

# False positives per category (background row)
fp_per_cat = matrix[-1, :-1]

# Row-normalized version (each row sums to 1.0)
norm = cm["normalized"]
```

### Finding class confusions

The off-diagonal cells (excluding the background row and column) tell you about cross-category confusions:

```python
import numpy as np

matrix = cm["matrix"][:-1, :-1]   # drop background row/col
cat_ids = cm["cat_ids"]

# Zero the diagonal (TPs) to see only confusions
confusion_only = matrix.copy()
np.fill_diagonal(confusion_only, 0)

# Top confusions
flat = confusion_only.flatten()
top_idx = np.argsort(flat)[::-1][:10]
for idx in top_idx:
    if flat[idx] == 0:
        break
    gt_cat = cat_ids[idx // len(cat_ids)]
    pred_cat = cat_ids[idx % len(cat_ids)]
    print(f"GT {gt_cat} predicted as {pred_cat}: {flat[idx]} times")
```

Stricter thresholds and score floors are available — for example `ev.confusion_matrix(iou_thr=0.75, max_det=50, min_score=0.3)`. See the [API reference](../api/cocoeval.md#confusion_matrix) for all parameters and defaults.

## TIDE error analysis

Once AP tells you *how good* your model is, TIDE ([Bolya et al., ECCV 2020](https://arxiv.org/abs/2008.08115), [code](https://github.com/dbolya/tide)) tells you *why* it falls short. `tide_errors()` decomposes every false positive and false negative into one of six mutually exclusive error types and reports the ΔAP — how much AP would improve if each error type were eliminated.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()

result = ev.tide_errors(pos_thr=0.5, bg_thr=0.1)

print(f"Baseline AP: {result['ap_base']:.3f}")
print("\nΔAP by error type (higher = fixing this type gives more AP gain):")
for name in ["Loc", "Bkg", "Miss", "Cls", "Both", "Dupe"]:
    print(f"  {name:4s}: {result['delta_ap'][name]:.4f}")
```

### Error types

Each false-positive detection is assigned exactly one error type (highest-priority match wins):

| Type | Meaning | Priority |
|------|---------|----------|
| `Loc` | Right class, poor localization (`bg_thr` ≤ IoU < `pos_thr`) | 1 |
| `Cls` | Wrong class, good location (cross-class IoU ≥ `pos_thr`) | 2 |
| `Dupe` | Duplicate — correct GT already claimed by a higher-scored TP | 3 |
| `Bkg` | Pure background (IoU < `bg_thr` with all GTs) | 4 |
| `Both` | Wrong class AND poor localization (IoU ∈ [`bg_thr`, `pos_thr`)) | 5 |

Every unmatched (non-ignored) ground-truth annotation that has no correctable FP DT targeting it is counted as `Miss`.

The returned dict contains `delta_ap` and `counts` per error type (plus tidecv's two special oracles, `FP` and `FN`), the baseline AP, and the thresholds used — see the [`tide_errors` API reference](../api/cocoeval.md#tide_errors) for the full shape.

### Prioritizing improvements

The `delta_ap` values rank where to spend engineering effort:

```python
result = ev.tide_errors()

# Sort errors by impact
deltas = [(k, v) for k, v in result["delta_ap"].items()
          if k not in ("FP", "FN")]
deltas.sort(key=lambda x: -x[1])

print("Priority order for improvement:")
for rank, (name, delta) in enumerate(deltas, 1):
    count = result["counts"].get(name, "—")
    print(f"  {rank}. {name:4s}  ΔAP={delta:.4f}  n={count}")
```

### Coming from tidecv

If you have used [tidecv](https://github.com/dbolya/tide), the reference implementation, expect the five false-positive types to agree closely and `Miss` to read higher here. The side-by-side numbers on COCO val2017 are in [Benchmarks](../benchmarks.md#tide). The ranking — which error type is costing you the most AP — is the same, and that is what the metric is for.

The difference is crowd handling, explained under [TIDE parity](../benchmarks.md#tide).

## Confidence calibration

A model that outputs confidence 0.9 should be correct about 90% of the time. `calibration()` measures how well your model's confidence scores align with actual detection accuracy by binning detections by confidence and comparing predicted confidence to the fraction of true positives in each bin.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()

cal = ev.calibration(n_bins=10, iou_threshold=0.5)

print(f"ECE: {cal['ece']:.4f}")   # Expected Calibration Error
print(f"MCE: {cal['mce']:.4f}")   # Maximum Calibration Error
print(f"Detections analyzed: {cal['num_detections']:,}")
```

### Interpreting the results

**ECE (Expected Calibration Error)** is the weighted average of |accuracy - confidence| across all bins. Lower is better — a perfectly calibrated model has ECE = 0. An ECE of 0.05 means confidence scores are off by 5% on average.

**MCE (Maximum Calibration Error)** is the worst-case gap in any single bin. Useful for safety-critical applications where the worst bin matters more than the average.

### Per-bin breakdown

The `bins` list shows the calibration gap per confidence range:

```python
for b in cal["bins"]:
    if b["count"] > 0:
        gap = b["avg_accuracy"] - b["avg_confidence"]
        label = "overconfident" if gap < 0 else "underconfident"
        print(f"  [{b['bin_lower']:.1f}, {b['bin_upper']:.1f}): "
              f"conf={b['avg_confidence']:.3f} acc={b['avg_accuracy']:.3f} "
              f"({label}, n={b['count']})")
```

### Per-category calibration

`per_category` maps each category name to its ECE. Some categories are well calibrated while others are wildly off:

```python
# Top 5 worst-calibrated categories
worst = sorted(cal["per_category"].items(), key=lambda x: -x[1])[:5]
for name, ece in worst:
    print(f"  {name}: ECE={ece:.4f}")
```

### Reliability diagram

Visualize calibration with `plot.reliability_diagram()`:

```python
from hotcoco import plot

with plot.style():
    fig, ax = plot.reliability_diagram(cal)
    # Or pass the COCOeval directly:
    fig, ax = plot.reliability_diagram(ev, n_bins=15, iou_threshold=0.5)
```

See [`calibration`](../api/cocoeval.md#calibration) in the API reference for full parameter details.

!!! tip "Calibrating something that isn't COCO detection"
    `calibration()` is a thin adapter — it decides which detections count, then
    calls [`metrics.calibration_error`](../api/metrics.md#calibration_error).
    That function takes two plain arrays and needs no evaluator:

    ```python
    from hotcoco import metrics

    ece, mce = metrics.calibration_error(scores, matched, n_bins=10)
    ```

    Same for AP and the confusion matrix. See [metrics](../api/metrics.md).

## F-scores

`f_scores()` computes F-beta scores from the precision/recall curves built by `accumulate()`. It finds the confidence threshold that maximizes F-beta for each (IoU, category) combination, then averages — the same summarization strategy as mAP.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

scores = ev.f_scores()
# {"F1": 0.523, "F1_50": 0.712, "F1_75": 0.581}
```

Use `beta` to shift the precision/recall trade-off:

```python
ev.f_scores(beta=0.5)  # precision-weighted  → {"F0.5": ..., "F0.5_50": ..., "F0.5_75": ...}
ev.f_scores(beta=2.0)  # recall-weighted     → {"F2": ..., "F2_50": ..., "F2_75": ...}
```

F-scores complement `get_results()` when you care about a specific operating point rather than area-under-curve. A high AP with a low F1 often signals that performance is concentrated at high recall or high precision, not both simultaneously.

See [`f_scores`](../api/cocoeval.md#f_scores) in the API reference for full parameter details.

## Model comparison

Compare two models on the same dataset with `hotcoco.compare()`. It computes per-metric deltas, per-category AP differences, and optional bootstrap confidence intervals.

```python
import hotcoco

gt = hotcoco.COCO("annotations.json")
dt_a = gt.load_res("model_a.json")
dt_b = gt.load_res("model_b.json")

ev_a = hotcoco.COCOeval(gt, dt_a, "bbox")
ev_a.evaluate()
ev_b = hotcoco.COCOeval(gt, dt_b, "bbox")
ev_b.evaluate()

result = hotcoco.compare(ev_a, ev_b)
# result["deltas"]["AP"]  → 0.033 (B is better by 3.3 AP points)
```

### Bootstrap confidence intervals

Add `n_bootstrap` to get confidence intervals on the metric deltas. This resamples images with replacement and re-accumulates metrics for each sample — parallelized with rayon.

```python
result = hotcoco.compare(ev_a, ev_b, n_bootstrap=1000, confidence=0.95)

ci = result["ci"]["AP"]
print(f"AP delta: {result['deltas']['AP']:+.3f}")
print(f"95% CI:   [{ci['lower']:+.3f}, {ci['upper']:+.3f}]")
print(f"P(B > A): {ci['prob_positive']:.1%}")
```

A CI that excludes zero indicates a statistically significant difference.

### Per-category breakdown

`result["per_category"]` is sorted by delta ascending (worst regressions first):

```python
for cat in result["per_category"][:5]:  # top 5 regressions
    print(f"{cat['cat_name']:<20} {cat['delta']:+.3f}")
```

### Plotting

`plot.comparison_bar()` and `plot.category_deltas()` draw the result — see [Model comparison](plotting.md#model-comparison) in the plotting guide.

See [`compare`](../api/cocoeval.md#compare) in the API reference for the full return shape.

## Per-image diagnostics and label error detection

`image_diagnostics()` gives you per-image F1 and AP scores, per-annotation TP/FP/FN classification, and automatically flags suspected label errors in your ground truth. It's the single call that answers "which images should I look at?" and "are my annotations trustworthy?"

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()

diag = ev.image_diagnostics(iou_thr=0.5, score_thr=0.5)
```

### Per-image scores

Every image gets an F1 score, AP at the selected IoU threshold, TP/FP/FN counts, and an error profile (`perfect`, `fp_heavy`, `fn_heavy`, or `mixed`):

```python
# Find the worst images
worst = sorted(diag["img_summary"].items(), key=lambda x: x[1]["f1"])
for img_id, s in worst[:5]:
    print(f"Image {img_id}: F1={s['f1']:.3f}  TP={s['tp']} FP={s['fp']} FN={s['fn']}")
```

### Label error detection

Two types of GT errors are flagged:

- **Wrong label** — a high-confidence FP detection that overlaps an unmatched GT of a *different* category (bbox IoU ≥ 0.5). The model thinks it's a dog, the annotation says cat, and they're in the same spot.
- **Missing annotation** — a high-confidence FP with no nearby GT at all (max bbox IoU < 0.1). Likely a real object the annotators missed.

```python
for le in diag["label_errors"][:5]:
    if le["type"] == "wrong_label":
        print(f"Image {le['image_id']}: {le['dt_category']}→{le['gt_category']} IoU={le['iou']:.2f}")
    else:
        print(f"Image {le['image_id']}: {le['dt_category']} (score={le['dt_score']:.2f}) — no GT match")
```

Only detections with `score >= score_thr` are considered, so lower the threshold to cast a wider net or raise it to focus on high-confidence candidates.

The result also includes the per-annotation TP/FP/FN status and match maps that power the browse viewer's eval coloring — see the [`image_diagnostics` API reference](../api/cocoeval.md#image_diagnostics) for the full return shape.

## From the CLI

Each diagnostic has a command-line form:

| Diagnostic | Command |
|---|---|
| TIDE | `coco eval --tide` (`--tide-pos-thr`, `--tide-bg-thr`) |
| Calibration | `coco eval --calibration` (`--cal-bins`, `--cal-iou-thr`) |
| Per-image diagnostics | `coco eval --diagnostics` (`--diag-iou-thr`, `--diag-score-thr`) |
| Model comparison | `coco compare --dt-a --dt-b` (`--bootstrap`, `--json`) |

Flags and defaults are in [`coco eval`](../cli.md#coco-eval) and [`coco compare`](../cli.md#coco-compare).
