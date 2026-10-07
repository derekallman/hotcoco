# metrics

Metric functions over flat arrays — no evaluator required.

=== "Python"

    ```python
    from hotcoco import metrics
    ```

=== "Rust"

    ```rust
    use hotcoco::metrics;
    ```

Every function here is pure: arrays in, numbers out — the same shape
`sklearn.metrics` and `torchmetrics.functional` use. `COCOeval`'s analysis methods
call these functions, so the object API and the functional API cannot disagree.
Reach for `metrics` when you have arrays rather than a COCO dataset.

[`primitives`](primitives.md) is the layer below: it decides *which prediction pairs
with which ground truth*; `metrics` turns those matches into numbers.

Float and bool parameters accept lists or 1-D numpy arrays interchangeably —
signatures written `Sequence[float]` also take an ndarray.

```python
from hotcoco import metrics

scores  = [0.95, 0.88, 0.71, 0.40]
matched = [True, True, False, True]

ap = metrics.average_precision(scores, matched, num_gt=5)
ece, mce = metrics.calibration_error(scores, matched)
```

These functions are additive-change-only through 1.x; `COCOeval` and the
pycocotools drop-in surface are frozen.

---

## Functions

### `average_precision`

Average precision from per-prediction scores and match flags.

=== "Python"

    ```python
    average_precision(
        scores: Sequence[float],
        matched: Sequence[bool],
        num_gt: int,
        ignored: Sequence[bool] | None = None,
        rec_thrs: Sequence[float] | None = None,
    ) -> float
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `scores` | `Sequence[float]` | Confidence per prediction, in any order |
    | `matched` | `Sequence[bool]` | Whether each prediction is correct |
    | `num_gt` | `int` | Total ground truths — the recall denominator |
    | `ignored` | `Sequence[bool] \| None` | Predictions counting as neither TP nor FP |
    | `rec_thrs` | `Sequence[float] \| None` | Recall grid; defaults to COCO's 101 points |

=== "Rust"

    ```rust
    metrics::counts::average_precision(
        scores: &[f64],
        matched: &[bool],
        ignored: Option<&[bool]>,
        num_gt: usize,
        rec_thrs: &[f64],
    ) -> f64
    ```

Sorts by score descending, classifies each prediction as TP or FP, and
interpolates precision onto the recall grid using PASCAL VOC interpolation —
the same computation that produces COCO's AP.

```python
>>> round(metrics.average_precision([0.9, 0.8, 0.3], [True, False, True], num_gt=2), 3)
0.835
```

!!! warning "`num_gt` must count ground truths you never predicted"
    It is the recall denominator. Passing only the matched count silently
    overstates recall, and therefore AP.

Returns `0.0` when there are no predictions or no ground truth. If your metric
wants a different answer for the empty case — per-image diagnostics call an
empty image perfect — branch before calling.

---

### `precision_recall_curve`

Precision interpolated onto a recall grid, from cumulative TP/FP counts.

=== "Python"

    ```python
    precision_recall_curve(
        tp_cum: Sequence[float],
        fp_cum: Sequence[float],
        num_gt: int,
        rec_thrs: Sequence[float] | None = None,
    ) -> tuple[float, list[tuple[int, float, int]]]
    ```

=== "Rust"

    ```rust
    metrics::counts::precision_recall_curve(
        tp_cum: &[f64], fp_cum: &[f64], num_gt: usize, rec_thrs: &[f64],
    ) -> (f64, Vec<(usize, f64, usize)>)
    ```

Lower level than `average_precision` — use it when you already hold cumulative
counts, or want the curve rather than the scalar. `tp_cum` and `fp_cum` must
already be prefix-summed over predictions sorted by descending score.

Returns `(final_recall, points)`, where each point is
`(threshold_index, precision, rank)`. Recall thresholds the predictions never
reach are **omitted** rather than reported as zero, so `points` can be shorter
than `rec_thrs`.

---

### `calibration_curve`

Reliability bins: predicted confidence against observed accuracy.

=== "Python"

    ```python
    calibration_curve(
        scores: Sequence[float], matched: Sequence[bool], n_bins: int = 10
    ) -> list[dict]
    ```

=== "Rust"

    ```rust
    metrics::calibration::calibration_curve(
        scores: &[f64], matched: &[bool], n_bins: usize,
    ) -> Vec<CalibrationBin>
    ```

The data behind a reliability diagram. Each dict has `bin_lower`, `bin_upper`,
`avg_confidence`, `avg_accuracy`, and `count`. Empty bins are included with
`count = 0`, so the list always has `n_bins` entries and plots without gaps.

A perfectly calibrated model has `avg_confidence == avg_accuracy` in every bin —
that diagonal is what the diagram compares against.

!!! warning "Scores must be confidences in `[0, 1]`"

    Both calibration functions bucket by `score * n_bins` and clamp the bin
    *index*, not the score, so a raw logit saturates into an end bin and carries
    its magnitude into that bin's mean — an ECE above 1.0 with no other symptom.
    Neither free function validates its input: they are hot-path primitives over
    flat arrays, and the check is a full pass over the scores. `COCOeval.calibration()`
    does validate and raises on out-of-range scores. Apply a sigmoid or softmax
    before calling these directly.

---

### `calibration_error`

Expected and Maximum Calibration Error.

=== "Python"

    ```python
    calibration_error(
        scores: Sequence[float], matched: Sequence[bool], n_bins: int = 10
    ) -> tuple[float, float]
    ```

=== "Rust"

    ```rust
    // Rust splits binning from scoring, so bins can be reused.
    let bins = metrics::calibration::calibration_curve(&scores, &matched, n_bins);
    let (ece, mce) = metrics::calibration::calibration_error(&bins);
    ```

Returns `(ece, mce)`:

- **ECE** — the occupancy-weighted mean gap between confidence and accuracy.
  The headline number.
- **MCE** — the worst single bin's gap, unweighted. Catches a badly calibrated
  region that ECE averages away.

```python
>>> # Always claims 0.9 confidence, right half the time.
>>> ece, mce = metrics.calibration_error([0.9] * 100, [True] * 50 + [False] * 50)
>>> round(ece, 3)
0.4
```

Both are `0.0` for empty input.

---

### `confusion_matrix`

Confusion counts over matched ground-truth/prediction pairs.

=== "Python"

    ```python
    confusion_matrix(
        gt: Sequence[int | None], dt: Sequence[int | None], num_classes: int
    ) -> numpy.ndarray
    ```

=== "Rust"

    ```rust
    metrics::confusion::confusion_matrix(
        gt_labels: &[Option<usize>], dt_labels: &[Option<usize>], num_classes: usize,
    ) -> Vec<u64>
    ```

`sklearn.metrics.confusion_matrix` assumes every sample has both a true and a
predicted label. Detection and tracking don't: a prediction can match nothing,
and a ground truth can go unpredicted. So this takes **optional** labels and
reserves index `num_classes` for background.

| `gt[i]` | `dt[i]` | Meaning | Lands at |
|---|---|---|---|
| `g` | `d` | matched pair (correct when `g == d`) | `[g][d]` |
| `g` | `None` | ground truth with no prediction | `[g][num_classes]` |
| `None` | `d` | prediction matching no ground truth | `[num_classes][d]` |
| `None` | `None` | nothing happened | ignored |

```python
>>> m = metrics.confusion_matrix([0, 1, None], [0, None, 1], num_classes=2)
>>> m[0, 0], m[1, 2], m[2, 1]   # correct, missed, spurious
(1, 1, 1)
```

The entries are one per **match record**, not one per prediction — producing
those records is the caller's job, and it is the only family-specific step.
Counts are integers, so accumulating per-image and summing gives the same answer
as one whole-dataset call. That is what lets you parallelize and reduce.

Class indices outside `range(num_classes)` are dropped rather than raising, so a
stray label can't take down an evaluation run.

### `panoptic_quality`

PQ, SQ, and RQ from match counts.

=== "Python"

    ```python
    panoptic_quality(iou_sum: float, tp: int, fp: int, fn_: int) -> tuple[float, float, float]
    ```

=== "Rust"

    ```rust
    metrics::panoptic::PqCounts { iou, tp, fp, fn_ }.scores_or_missing() -> PqScores
    metrics::panoptic::pq_average(counts: impl IntoIterator<Item = &PqCounts>) -> (PqScores, usize)
    ```

The formulas behind [`panoptic`](panoptic.md), for counts your own matcher
produced: `PQ = Σ IoU / (TP + ½ FP + ½ FN)`, `SQ = Σ IoU / TP`,
`RQ = TP / (TP + ½ FP + ½ FN)`, so `PQ = SQ × RQ`.

```python
>>> metrics.panoptic_quality(1.6, tp=2, fp=1, fn_=1)
(0.5333333333333333, 0.8, 0.6666666666666666)
```

Returns `(-1.0, -1.0, -1.0)` when `tp + fp + fn == 0`: nothing to score, which
panopticapi leaves out of its averages. To reproduce its `All`, `Things`, and
`Stuff` numbers, average the per-category results that `is_computed` accepts.
`fn_` carries a trailing underscore because `fn` is a Rust keyword and the
same counts live there.

### `is_computed` and `is_missing`

```python
is_computed(v: float) -> bool
is_missing(v: float) -> bool
```

`-1.0` in any metric means "not computed for this configuration" — an area range
with no ground truth, or a category absent from the split — never a low score.
`is_missing(v)` is true for that sentinel and `is_computed(v)` is its negation. Use
them when averaging per-class values so a sentinel does not drag the mean down.

---

## Rust-only

**Bootstrap confidence intervals** (`metrics::bootstrap::bootstrap_ci`) take the
statistic as a closure. `compare()` uses them internally and returns the intervals.

**Greedy matching** lives one layer down — see
[Primitives](primitives.md#rust-only).
