# primitives

Matching kernels — deciding what pairs with what.

=== "Python"

    ```python
    from hotcoco import primitives
    ```

=== "Rust"

    ```rust
    use hotcoco::primitives;
    ```

The layer underneath [`metrics`](metrics.md): similarity between two sets, and
the matchers that turn similarity into pairs.

```python
from hotcoco import primitives, metrics

sim = primitives.bbox_iou(detections, ground_truths, iscrowd)
m = primitives.greedy_match(sim, scores, 0.5, iscrowd=iscrowd)   # COCO's matcher
rows, cols = primitives.lsap(sim, maximize=True)                   # optimal assignment
```

Nothing here scores — feed the pairs to [`metrics`](metrics.md) for that. The
matchers are the ones the evaluators run, not copies: `COCOeval` matches with
`greedy_match`'s kernel and `PanopticEval` with `match_segments`'s, so a metric
you build from them gets the evaluators' answers.
[Build your own metric](../guide/custom-metrics.md) builds three.

This module is the strict layer: `bbox_iou` takes boxes and `mask_iou` takes
RLEs, and neither guesses which it was handed. [`hotcoco.mask`](mask.md) is the
pycocotools-shaped layer on top. Its [`iou`](mask.md#iou) accepts either kind
and dispatches on type the way `pycocotools.mask.iou` does, then calls these
kernels, so the two layers cannot disagree on a number. `mask.bbox_iou` is
`primitives.bbox_iou` under a second name.

The IoU kernels are frozen, since pycocotools parity depends on them, and
`lsap` changes only by addition through 1.x. `greedy_match`,
`segment_overlaps`, and `match_segments` are provisional: callable today, with
signatures that freeze once the tracking family has exercised them.

---

## Functions

### `lsap`

Optimal one-to-one assignment on a rectangular cost matrix.

=== "Python"

    ```python
    lsap(
        cost: numpy.ndarray | Sequence[Sequence[float]], maximize: bool = False
    ) -> tuple[numpy.ndarray, numpy.ndarray]
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `cost` | `numpy.ndarray \| Sequence[Sequence[float]]` | 2-D costs; `cost[i][j]` for row `i`, column `j`. A numpy array of any integer or float dtype is read in one pass; anything else is converted element by element |
    | `maximize` | `bool` | Maximize value instead of minimizing cost. Pass `True` for similarities |

=== "Rust"

    ```rust
    primitives::assign::lsap(
        cost: &[f64], nr: usize, nc: usize, maximize: bool,
    ) -> (Vec<usize>, Vec<usize>)
    ```

Solves the linear sum assignment problem — the same thing
`scipy.optimize.linear_sum_assignment` solves, and a semantic port of it, so
results agree including on ties.

Use it when greedy matching is not good enough. Tracking associates detections
to tracks this way, and it is what HOTA and MOTA are defined against. COCO
detection deliberately does *not* use it: pycocotools matches greedily by score
rank, and matching optimally would change the numbers.

Returns `(row_ind, col_ind)`, the matched pairs, with
`len(row_ind) == min(n_rows, n_cols)`.

```python
>>> rows, cols = primitives.lsap([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
>>> list(zip(rows.tolist(), cols.tolist()))   # total cost 1 + 2 + 2 = 5
[(0, 1), (1, 0), (2, 2)]
```

Raises `ValueError` if `cost` is ragged or holds a `NaN` — an unsolvable matrix
is an error, not grounds for an arbitrary assignment.

---

### `bbox_iou`

Pairwise IoU between two sets of boxes.

=== "Python"

    ```python
    bbox_iou(
        dt: numpy.ndarray | Sequence[Sequence[float]],
        gt: numpy.ndarray | Sequence[Sequence[float]],
        iscrowd: Sequence[bool] | numpy.ndarray,
    ) -> numpy.ndarray
    ```

=== "Rust"

    ```rust
    primitives::sim::bbox_iou(dt: &[[f64; 4]], gt: &[[f64; 4]], iscrowd: &[bool]) -> Vec<Vec<f64>>
    primitives::sim::bbox_iou_flat(dt: &[[f64; 4]], gt: &[[f64; 4]], iscrowd: &[bool]) -> Vec<f64>
    ```

    `bbox_iou_flat` is the same matrix as one row-major `D × G` buffer, empty
    when either side is: the shape a dense array wants, with no vector per
    detection.

Boxes are `[x, y, width, height]`: an `(N, 4)` array or a sequence of 4-element
rows. A numpy array of any integer or float dtype is read in one pass; anything
else is converted element by element. Returns shape `(len(dt), len(gt))`.

Where `iscrowd[j]` is true, the denominator is the detection area alone rather
than the union — a detection fully inside a crowd region scores 1.0. That is
pycocotools' convention, and it is why `iscrowd` is required rather than
optional: silently defaulting it would change crowd-region numbers.

[`mask.bbox_iou`](mask.md#bbox_iou) is this same function object.

---

### `mask_iou`

Pairwise IoU between two sets of RLE masks.

=== "Python"

    ```python
    mask_iou(dt: Sequence[dict], gt: Sequence[dict], iscrowd: Sequence[bool] | numpy.ndarray) -> numpy.ndarray
    ```

=== "Rust"

    ```rust
    primitives::sim::mask_iou(dt: &[Rle], gt: &[Rle], iscrowd: &[bool]) -> Vec<Vec<f64>>
    primitives::sim::mask_iou_flat(dt: &[Rle], gt: &[Rle], iscrowd: &[bool]) -> Vec<f64>
    ```

    `mask_iou_flat` is the same matrix as one row-major `D × G` buffer, as
    `bbox_iou_flat` is.

Same crowd convention as `bbox_iou`. Inputs are RLE dicts as produced by
[`mask.encode`](mask.md#encode); boxes are not accepted. To pass either kind with
pycocotools' type dispatch, call [`mask.iou`](mask.md#iou), which forwards RLEs
here.

---

### `greedy_match`

COCO's greedy matching of detections to ground truths, pycocotools-exact.

=== "Python"

    ```python
    greedy_match(
        sim: numpy.ndarray | Sequence[Sequence[float]],
        scores: numpy.ndarray | Sequence[float],
        iou_thrs: float | Sequence[float] | None = None,
        *,
        gt_ignore: Sequence[bool] | numpy.ndarray | None = None,
        iscrowd: Sequence[bool] | numpy.ndarray | None = None,
        dt_ignore: Sequence[bool] | numpy.ndarray | None = None,
    ) -> dict
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `sim` | `numpy.ndarray \| Sequence[Sequence[float]]` | `(D, G)` similarity of each detection to each ground truth — [`bbox_iou`](#bbox_iou) or [`mask_iou`](#mask_iou) output, any integer or float dtype |
    | `scores` | `numpy.ndarray \| Sequence[float]` | `(D,)` detection confidence, which sets the matching order |
    | `iou_thrs` | `float \| Sequence[float] \| None` | One threshold or several. Default: COCO's ten, 0.50 to 0.95 |
    | `gt_ignore` | `Sequence[bool] \| numpy.ndarray \| None` | `(G,)` ground truths that can absorb a detection but never count, such as those outside the area range being scored |
    | `iscrowd` | `Sequence[bool] \| numpy.ndarray \| None` | `(G,)` crowd flags, the ones `sim` was computed with |
    | `dt_ignore` | `Sequence[bool] \| numpy.ndarray \| None` | `(D,)` detections ignored when nothing matches them, such as those outside the area range being scored. A detection that matches takes its ground truth's flag instead |

=== "Rust"

    ```rust
    primitives::greedy::greedy_match_masked(
        iou_flat: &[f64], d: usize, g: usize, num_gt_not_ignored: usize,
        masks: GtMasks<'_>, iou_thrs: &[f64],
    ) -> GreedyMatches
    ```

    The kernel takes its input already ordered: rows score-descending, columns
    non-ignored first, thresholds already clamped with `coco_match_floor`. The
    Python function does that ordering for you, with
    `metrics::counts::score_order` and `params::default_iou_thrs` — the
    ranking and thresholds `COCOeval` uses.

At each threshold, detections take turns from the highest score down, tied
scores in row order as pycocotools' stable sort leaves them. Each takes the
unmatched ground truth it overlaps most, at or above the threshold. Among equal
similarities the later column wins, as pycocotools' `>=` update does. An
ignored ground truth is a candidate only when no other one clears the
threshold, and a crowd ground truth absorbs any number of detections and is
always ignored. A threshold of `1.0` matches as `1 - 1e-10`, pycocotools'
clamp. Give it the boxes, scores, and flags pycocotools sees for one image and
category, and it makes the matches `evalImgs` records.

Returns a dict of three arrays, indexed like `sim`'s rows and columns
whatever order those are in:

| Key | Type | Description |
|-----|------|-------------|
| `dt_matches` | `int64`, `(T, D)` | The column each detection matched, or `-1` |
| `gt_matches` | `int64`, `(T, G)` | The row that matched each ground truth, or `-1`; for a crowd ground truth, the last one to |
| `dt_ignore` | `bool`, `(T, D)` | `evalImgs`' `dtIgnore`: detections matched to an ignored or crowd ground truth, and unmatched ones the `dt_ignore` argument flags — neither true nor false positives |

With a single number for `iou_thrs`, the threshold axis is dropped: `(D,)`,
`(G,)`, `(D,)`, as `numpy.quantile` does for a scalar `q`. The pycocotools
`evalImgs` fields of the same names hold annotation ids instead of indices,
with `0` for no match.

```python
from hotcoco import primitives

sim = primitives.bbox_iou(dt_boxes, gt_boxes, iscrowd)
m = primitives.greedy_match(sim, dt_scores, 0.5, iscrowd=iscrowd)
matched, ignored = m["dt_matches"] >= 0, m["dt_ignore"]
tp = (matched & ~ignored).sum()
fp = (~matched & ~ignored).sum()
```

To score one area range as COCO does, flag the ground truths outside it with
`gt_ignore` and the detections outside it with `dt_ignore`. A flagged ground
truth can still absorb a detection, and a flagged detection that matches a
ground truth in the range is still a true positive.

The matching is greedy, not optimal. That is the protocol, and the numbers
depend on it; for an optimal assignment use [`lsap`](#lsap).

An image with no detections has a `sim` of `(0, G)`, which [`bbox_iou`](#bbox_iou)
returns as such. As a list, `[]` has no columns to count, so the flags, when
given, say how many ground truths there are; without them, pass
`numpy.zeros((0, G))`, or `gt_matches` has no entries to report the misses in.

Raises `ValueError` if `sim`, `scores`, or `iou_thrs` holds a `NaN`, or a
length disagrees with `sim`'s shape.

---

### `segment_overlaps`

Pixel counts of every `(gt_id, pred_id)` pair in two panoptic label maps.

=== "Python"

    ```python
    segment_overlaps(gt: numpy.ndarray, pred: numpy.ndarray) -> dict[tuple[int, int], int]
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `gt` | `numpy.ndarray` | Ground-truth segment ids, `0` for void: an integer or bool array of any shape, or a nested list |
    | `pred` | `numpy.ndarray` | Predicted segment ids, the same shape |

=== "Rust"

    ```rust
    primitives::panoptic::Overlaps::compute(gt: &[u32], pred: &[u32]) -> Overlaps
    primitives::panoptic::Overlaps::from_counts(counts: impl IntoIterator<Item = (u32, u32, u64)>) -> Overlaps
    ```

The similarity step of panoptic quality: everything PQ needs about geometry is
in this table. Returns `{(gt_id, pred_id): pixels}` for every pair that shares
a pixel, void included, ordered by `(gt_id, pred_id)`. It is panopticapi's
`gt_pred_map`, which panopticapi builds with
`np.unique(gt * OFFSET + pred, return_counts=True)`.

```python
>>> gt = np.array([[1, 1, 0], [2, 2, 2]])
>>> pred = np.array([[5, 5, 5], [5, 6, 6]])
>>> primitives.segment_overlaps(gt, pred)
{(0, 5): 1, (1, 5): 2, (2, 5): 1, (2, 6): 2}
```

A COCO panoptic PNG holds colors, not ids: decode it with `R + 256·G + 256²·B`
(panopticapi's `rgb2id`) first. Raises `TypeError` for a map that is not an
integer or bool array, and `ValueError` if the shapes differ or an id is
negative or does not fit in 32 bits.

---

### `match_segments`

Match one image's segments by panoptic quality's rules.

=== "Python"

    ```python
    match_segments(
        gt_segments: list[dict] | tuple[dict, ...],
        pred_segments: list[dict] | tuple[dict, ...],
        overlaps: dict[tuple[int, int], int],
    ) -> dict
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `gt_segments` | `list[dict] \| tuple[dict, ...]` | The image's ground-truth `segments_info`: `id` and `category_id`, optionally `iscrowd` and `area` |
    | `pred_segments` | `list[dict] \| tuple[dict, ...]` | The predicted `segments_info`, the same shape |
    | `overlaps` | `dict[tuple[int, int], int]` | [`segment_overlaps`](#segment_overlaps) output, or panopticapi's own `gt_pred_map` |

=== "Rust"

    ```rust
    primitives::panoptic::match_segments(
        gt: &[Segment], pred: &[Segment], overlaps: &Overlaps,
    ) -> SegmentMatches
    ```

    ```rust
    panoptic::match_image(
        gt: &[(u32, &SegmentInfo)], pred: &[(u32, &SegmentInfo)], overlaps: &Overlaps,
    ) -> Result<SegmentMatches>
    ```

    `match_segments` is the kernel, on segments with their areas resolved;
    `panoptic::match_image` is the Python function's Rust twin, and the
    per-image step `PanopticEval` runs — it applies panopticapi's checks and
    areas around the kernel. Both return indices into the slices they were
    given, and `ImageMatches::from_indices` turns them into the ids the
    Python function returns.

A ground-truth and a predicted segment match when their categories agree, the
ground truth is not crowd, and their IoU is above 0.5, with the prediction's
pixels on void left out of the union. When areas are the pixel counts and no
segment is listed with the void id 0, a match above 0.5 is unique, so nothing
is ranked or assigned. A stated area below a segment's pixels, or the void id
listed as a ground truth, can match a segment twice, and panopticapi counts
both.
An unmatched prediction is ignored, rather than a false
positive, when more than half of it lies on void and on the crowd region of
its own category. These are panopticapi's rules, and
[`PanopticEval`](panoptic.md) runs the same function on every image.

A ground truth's `area` defaults to its pixels in `overlaps`. panopticapi uses
the JSON's when it states one and takes it at its word, and so does this: a
stated area below the pixels found can give an IoU above 1. A prediction's
area is always its pixel count, and a prediction is never crowd. An id listed
twice keeps its last entry at its first position, as panopticapi's dict over
the list does. `bbox` and `segmentation` are checked by the rules a `COCO`
dict's annotations are, and other keys are dropped.

Returns a dict of segment ids by outcome:

| Key | Type | Description |
|-----|------|-------------|
| `matched` | `list[tuple[int, int, float]]` | `(gt_id, pred_id, iou)` per matched pair, ordered by ground-truth id — the order panopticapi sums their IoU in |
| `missed` | `list[int]` | Unmatched ground truths, crowd excluded |
| `false_positives` | `list[int]` | Unmatched predictions that count against the model |
| `ignored` | `list[int]` | Unmatched predictions that count for nothing |

The last three keep `segments_info` order. Per category, the matched pairs are
`tp` and their IoUs summed are `iou_sum`, the false positives are `fp`, and the
misses are `fn` — what [`metrics.panoptic_quality`](metrics.md#panoptic_quality)
takes:

```python
overlaps = primitives.segment_overlaps(gt_map, pred_map)
m = primitives.match_segments(gt_ann["segments_info"], pred_ann["segments_info"], overlaps)
for gt_id, pred_id, iou in m["matched"]:
    ...
```

Raises `ValueError` if a segment dict is malformed; if a predicted id has
pixels in `overlaps` but no entry, or an entry but no pixels, both of which
panopticapi rejects too; or if a matched pair has no union, so its IoU is not a
number, where panopticapi divides by zero. A stated `area` below the segment's
pixels does that, as can the void id `0` listed as a segment, which panopticapi
scores as one. Raises `TypeError` if `overlaps` is not a dict.
