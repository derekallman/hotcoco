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
the assignment that turns similarity into pairs.

```python
from hotcoco import primitives, metrics

sim = primitives.bbox_iou(detections, ground_truths, iscrowd=[False] * len(ground_truths))
rows, cols = primitives.lsap(sim, maximize=True)
```

Nothing here scores — feed the pairs to [`metrics`](metrics.md) for that.

This module is the strict layer: `bbox_iou` takes boxes and `mask_iou` takes
RLEs, and neither guesses which it was handed. [`hotcoco.mask`](mask.md) is the
pycocotools-shaped layer on top. Its [`iou`](mask.md#iou) accepts either kind
and dispatches on type the way `pycocotools.mask.iou` does, then calls these
kernels, so the two layers cannot disagree on a number. `mask.bbox_iou` is
`primitives.bbox_iou` under a second name.

These functions are additive-change-only through 1.x. The IoU kernels are the
exception — they are frozen, since pycocotools parity depends on them.

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
    | `cost` | `numpy.ndarray \| Sequence[Sequence[float]]` | 2-D costs; `cost[i][j]` for row `i`, column `j`. A `float64` array is read in one pass; anything else is converted element by element |
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
    ```

Boxes are `[x, y, width, height]`: an `(N, 4)` array or a sequence of 4-element
rows. A `float64` array is read in one pass; anything else is converted element
by element. Returns shape `(len(dt), len(gt))`.

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
    ```

Same crowd convention as `bbox_iou`. Inputs are RLE dicts as produced by
[`mask.encode`](mask.md#encode); boxes are not accepted. To pass either kind with
pycocotools' type dispatch, call [`mask.iou`](mask.md#iou), which forwards RLEs
here.

---

## Rust-only

**Greedy matching** (`primitives::greedy::greedy_match_masked`) is COCO's
rank-ordered assignment — the reason hotcoco matches pycocotools
detection-for-detection. Its `GtMasks` argument carries the crowd and ignore
semantics that parity depends on. `COCOeval.evaluate()` uses it today; a Python
binding is on the
[roadmap](https://github.com/derekallman/hotcoco/blob/main/ROADMAP.md).
