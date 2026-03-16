# COCOeval

Run COCO evaluation to compute AP/AR metrics.

=== "Python"

    ```python
    from hotcoco import COCO, COCOeval

    coco_gt = COCO("instances_val2017.json")
    coco_dt = coco_gt.load_res("detections.json")

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ```

=== "Rust"

    ```rust
    use hotcoco::{COCO, COCOeval};
    use hotcoco::params::IouType;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("instances_val2017.json"))?;
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

---

## Constructor

=== "Python"

    ```python
    COCOeval(
        coco_gt: COCO,
        coco_dt: COCO,
        iou_type: str,
        *,
        lvis_style: bool = False,
        oid_style: bool = False,
        hierarchy: Hierarchy | None = None,
    )
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `coco_gt` | `COCO` | — | Ground truth COCO object |
    | `coco_dt` | `COCO` | — | Detections COCO object (from `load_res`) |
    | `iou_type` | `str` | — | `"bbox"`, `"segm"`, or `"keypoints"` |
    | `lvis_style` | `bool` | `False` | Enable LVIS federated evaluation mode |
    | `oid_style` | `bool` | `False` | Enable Open Images evaluation mode (IoU=0.5, group-of matching) |
    | `hierarchy` | <code>Hierarchy &#124; None</code> | `None` | Category hierarchy for GT expansion in OID mode |

=== "Rust"

    ```rust
    // Standard COCO
    COCOeval::new(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self

    // LVIS federated
    COCOeval::new_lvis(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self

    // Open Images
    COCOeval::new_oid(coco_gt: COCO, coco_dt: COCO, hierarchy: Option<Hierarchy>) -> Self
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `coco_gt` | `COCO` | Ground truth COCO object |
    | `coco_dt` | `COCO` | Detections COCO object (from `load_res`) |
    | `iou_type` | `IouType` | `IouType::Bbox`, `IouType::Segm`, or `IouType::Keypoints` |
    | `hierarchy` | `Option<Hierarchy>` | Category hierarchy for GT expansion; `None` to skip expansion |

---

## Properties

### `params`

=== "Python"

    ```python
    params: Params
    ```

    Evaluation parameters. Modify before calling `evaluate()`.

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.cat_ids = [1, 2, 3]
    ev.params.max_dets = [1, 10, 100]
    ```

=== "Rust"

    ```rust
    pub params: Params
    ```

    ```rust
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.params.cat_ids = vec![1, 2, 3];
    ev.params.max_dets = vec![1, 10, 100];
    ```

See [Params](params.md) for all configurable fields.

---

### `stats`

=== "Python"

    ```python
    stats: list[float] | None
    ```

    The 12 summary metrics (10 for keypoints), populated after `summarize()`. `None` before `summarize()` is called.

    ```python
    ev.summarize()
    print(f"AP: {ev.stats[0]:.3f}")
    print(f"AP50: {ev.stats[1]:.3f}")
    ```

=== "Rust"

    ```rust
    pub stats: Option<Vec<f64>>
    ```

    ```rust
    ev.summarize();
    if let Some(stats) = &ev.stats {
        println!("AP: {:.3}", stats[0]);
        println!("AP50: {:.3}", stats[1]);
    }
    ```

---

### `eval_imgs`

Per-image evaluation results, populated after `evaluate()`. See [Working with Results](../guide/results.md) for details.

=== "Python"

    ```python
    eval_imgs: list[dict | None]
    ```

=== "Rust"

    ```rust
    pub eval_imgs: Vec<Option<EvalImg>>
    ```

---

### `eval`

Accumulated precision/recall arrays, populated after `accumulate()`. See [Working with Results](../guide/results.md) for details.

=== "Python"

    ```python
    eval: dict | None
    ```

    Contains `"precision"`, `"recall"`, and `"scores"` arrays.

=== "Rust"

    ```rust
    pub eval: Option<AccumulatedEval>
    ```

    Access elements with `precision_idx(t, r, k, a, m)` and `recall_idx(t, k, a, m)`.

---

## Methods

### `evaluate`

```python
evaluate() -> None
```

Run per-image evaluation. Matches detections to ground truth annotations using greedy matching sorted by confidence. Must be called before `accumulate()`.

Populates `eval_imgs`.

---

### `accumulate`

```python
accumulate() -> None
```

Accumulate per-image results into precision/recall curves using interpolated precision at 101 recall thresholds.

Populates `eval`.

---

### `summarize`

```python
summarize() -> None
```

Compute and print the standard COCO metrics. Populates `stats`.

!!! warning "Non-default parameters"
    `summarize()` uses a fixed display format that assumes default `iou_thrs`, `max_dets`, and `area_rng_lbl`. If you've changed any of these, a warning is printed to stderr and some metrics may show `-1.000` (e.g. AP50 when `iou_thrs` doesn't include 0.50). The `stats` array always has 12 entries (10 for keypoints) regardless of your parameters.

Prints 12 lines for bbox/segm (10 for keypoints):

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.584
 ...
```

---

### `run`

```python
run() -> None
```

Run the full pipeline in one call: `evaluate()` → `accumulate()` → `summarize()`. Primarily used with LVIS pipelines (Detectron2, MMDetection) that expect a single `run()` call.

---

### `get_results`

```python
get_results(prefix: str | None = None, per_class: bool = False) -> dict[str, float]
```

Return the summary metrics as a dict. Must be called after `summarize()` (or `run()`). Returns an empty dict if `summarize()` has not been called.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | <code>str &#124; None</code> | `None` | If given, each key is prefixed as `"{prefix}/{metric}"`. |
| `per_class` | `bool` | `False` | If `True`, include per-category AP values keyed as `"AP/{cat_name}"` (or `"{prefix}/AP/{cat_name}"` with a prefix). |

Standard bbox/segm keys: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `AR1`, `AR10`, `AR100`, `ARs`, `ARm`, `ARl`.

Keypoint keys: `AP`, `AP50`, `AP75`, `APm`, `APl`, `AR`, `AR50`, `AR75`, `ARm`, `ARl`.

LVIS keys: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `APr`, `APc`, `APf`, `AR@300`, `ARs@300`, `ARm@300`, `ARl@300`.

```python
ev.run()

# Basic usage (unchanged)
results = ev.get_results()
print(f"AP: {results['AP']:.3f}, AP50: {results['AP50']:.3f}")

# Prefixed keys — ready for any logger
results = ev.get_results(prefix="val/bbox")
# {"val/bbox/AP": 0.578, "val/bbox/AP50": 0.861, ...}

# With per-class AP
results = ev.get_results(prefix="val/bbox", per_class=True)
# {"val/bbox/AP": 0.578, ..., "val/bbox/AP/person": 0.82, ...}
```

---

### `print_results`

```python
print_results() -> None
```

Print a formatted results table to stdout. For LVIS, matches the lvis-api `print_results()` style. Must be called after `summarize()` (or `run()`).

---

### `results`

```python
results(per_class: bool = False) -> dict
```

Return evaluation results as a serializable dict. Must be called after `summarize()` (or `run()`). Raises `RuntimeError` if `summarize()` has not been called.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `per_class` | `bool` | `False` | If `True`, include per-category AP values under the `"per_class"` key. |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"hotcoco_version"` | `str` | hotcoco version that produced these results. |
| `"params"` | `dict` | Evaluation parameters: `iou_type`, `iou_thresholds`, `area_ranges`, `max_dets`, `is_lvis`. |
| `"metrics"` | `dict[str, float]` | Summary metrics keyed by name (same keys as `get_results()`). |
| `"per_class"` | `dict[str, float]` \| absent | Per-category AP values keyed by category name. Only present if `per_class=True`. |

```python
ev.run()
r = ev.results()
print(r["metrics"]["AP"])

# With per-category breakdown
r = ev.results(per_class=True)
print(r["per_class"]["person"])
```

---

### `save_results`

```python
save_results(path: str, per_class: bool = False) -> None
```

Save evaluation results to a JSON file. Must be called after `summarize()` (or `run()`). Raises `RuntimeError` if `summarize()` has not been called, or `IOError` if the file cannot be written.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Output file path. |
| `per_class` | `bool` | `False` | If `True`, include per-category AP values. |

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()
ev.save_results("results.json")

# With per-category AP
ev.save_results("results_per_class.json", per_class=True)
```

The JSON structure matches the dict returned by `results()`.

---

### `confusion_matrix`

```python
confusion_matrix(
    iou_thr: float = 0.5,
    max_det: int | None = None,
    min_score: float | None = None,
) -> dict
```

Compute a per-category confusion matrix. Unlike `evaluate()`, this method compares **all** detections in an image against **all** ground truth boxes regardless of category, enabling cross-category confusion analysis.

This method is **standalone** — no `evaluate()` call is needed first.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iou_thr` | `float` | `0.5` | IoU threshold for a DT↔GT match |
| `max_det` | `int \| None` | last `params.max_dets` value | Max detections per image by score |
| `min_score` | `float \| None` | `None` | Discard detections below this confidence before `max_det` truncation |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"matrix"` | `np.ndarray[int64]` shape `(K+1, K+1)` | Raw confusion counts. Rows = GT category, cols = predicted. Index `K` is background. |
| `"normalized"` | `np.ndarray[float64]` shape `(K+1, K+1)` | Row-normalised version (rows sum to 1.0; zero rows stay zero). |
| `"cat_ids"` | `list[int]` | Category IDs for rows/cols `0..K-1`. |
| `"cat_names"` | `list[str]` | Category names for rows/cols `0..K-1`, in the same order as `cat_ids`. |
| `"num_cats"` | `int` | Number of categories `K`. |
| `"iou_thr"` | `float` | IoU threshold used. |

**Matrix layout** (rows = GT, cols = predicted):

- `matrix[i][j]` where `i ≠ K, j ≠ K` — GT category `i` matched to predicted category `j`. On-diagonal = TP; off-diagonal = class confusion.
- `matrix[i][K]` — GT category `i` unmatched (false negative).
- `matrix[K][j]` — Predicted category `j` unmatched (false positive).

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
cm = ev.confusion_matrix(iou_thr=0.5, max_det=100)

matrix = cm["matrix"]
cat_ids = cm["cat_ids"]

# True positives per category
tp = matrix.diagonal()[:-1]

# False negatives per category
fn = matrix[:-1, -1]

# False positives per category
fp = matrix[-1, :-1]

# Normalised view
print(cm["normalized"])
```

See [Confusion Matrix](../guide/evaluation.md#confusion-matrix) in the evaluation guide for a full walkthrough.

---

### `tide_errors`

```python
tide_errors(
    pos_thr: float = 0.5,
    bg_thr: float = 0.1,
) -> dict
```

Decompose detection errors into six TIDE error types ([Bolya et al., ECCV 2020](https://arxiv.org/abs/2008.08115)) and compute ΔAP — the AP gain from eliminating each error type.

Requires `evaluate()` to have been called first.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pos_thr` | `float` | `0.5` | IoU threshold for TP/FP classification |
| `bg_thr` | `float` | `0.1` | Background IoU threshold for Loc/Both/Bkg discrimination |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"delta_ap"` | `dict[str, float]` | ΔAP for each error type. Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`, `"FP"`, `"FN"`. |
| `"counts"` | `dict[str, int]` | Count of each error type. Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`. |
| `"ap_base"` | `float` | Baseline mean AP at `pos_thr`. |
| `"pos_thr"` | `float` | IoU threshold used. |
| `"bg_thr"` | `float` | Background threshold used. |

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()
result = ev.tide_errors(pos_thr=0.5, bg_thr=0.1)

print(f"ap_base: {result['ap_base']:.3f}")
for k, v in sorted(result["delta_ap"].items(), key=lambda x: -x[1]):
    if k not in ("FP", "FN"):
        print(f"  {k}: ΔAP={v:.4f}  n={result['counts'].get(k, '—')}")
```

See [TIDE Error Analysis](../guide/evaluation.md#tide-error-analysis) in the evaluation guide for a detailed walkthrough.

---

### `f_scores`

```python
f_scores(beta: float = 1.0) -> dict[str, float]
```

Compute F-beta scores after `accumulate()` (or `run()`).

For each (IoU threshold, category), finds the confidence operating point that maximises F-beta, then averages across categories — analogous to how mAP averages precision. Returns three metrics mirroring AP/AP50/AP75.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `1.0` | Trade-off weight. `beta=1` → F1 (equal weight). `beta<1` → weights precision. `beta>1` → weights recall. |

**Returns** a dict with three keys:

| Key | Description |
|-----|-------------|
| `"F1"` | Mean max-F1 across IoU 0.50:0.05:0.95, all categories |
| `"F150"` | Max-F1 at IoU=0.50 |
| `"F175"` | Max-F1 at IoU=0.75 |

Key names reflect `beta`: `"F0.5"`, `"F0.550"`, `"F0.575"` for `beta=0.5`, etc.

Returns an empty dict if `accumulate()` has not been called.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

# F1 (default)
scores = ev.f_scores()
print(f"F1: {scores['F1']:.3f}, F1@50: {scores['F150']:.3f}")

# Precision-weighted
print(ev.f_scores(beta=0.5))   # {"F0.5": ..., "F0.550": ..., "F0.575": ...}

# Recall-weighted
print(ev.f_scores(beta=2.0))   # {"F2.0": ..., "F2.050": ..., "F2.075": ...}
```
