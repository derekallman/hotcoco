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
    ev.evaluate()?;
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
    | `iou_type` | `str` | — | `"bbox"`, `"segm"`, `"keypoints"`, or `"obb"` |
    | `lvis_style` | `bool` | `False` | Enable LVIS federated evaluation mode |
    | `oid_style` | `bool` | `False` | Enable Open Images evaluation mode (IoU=0.5, group-of matching) |
    | `hierarchy` | <code>Hierarchy &#124; None</code> | `None` | Category hierarchy for GT expansion in OID mode |

    !!! note "pycocotools keyword spellings"
        pycocotools spells the constructor keywords `cocoGt`, `cocoDt`, and
        `iouType`, and consumers pass them that way (torchmetrics' pycocotools
        backend calls `COCOeval(gt, dt, iouType=...)`). Both spellings are
        accepted; mixing the two spellings of one argument is an error.

    !!! note "`segm` needs image sizes"
        With `iou_type="segm"`, every image holding a polygon or box annotation
        the evaluation covers — in `params.img_ids` and, with `use_cats` on,
        `params.cat_ids` — must carry `height` and `width`, or `evaluate()`
        raises `ValueError` naming the images. A polygon drawn onto an image of
        unknown size is an empty mask, which would score as AP 0 with nothing
        to say why; pycocotools raises `KeyError` at the same point. RLE
        masks carry their own size and are not affected. Box evaluation never
        reads the fields.

=== "Rust"

    ```rust
    // Standard COCO
    COCOeval::new(coco_gt: impl Into<Arc<COCO>>, coco_dt: impl Into<Arc<COCO>>, iou_type: IouType) -> Self

    // LVIS federated
    COCOeval::new_lvis(coco_gt: impl Into<Arc<COCO>>, coco_dt: impl Into<Arc<COCO>>, iou_type: IouType) -> Self

    // Open Images
    COCOeval::new_oid(coco_gt: impl Into<Arc<COCO>>, coco_dt: impl Into<Arc<COCO>>, hierarchy: Option<Hierarchy>) -> Self
    ```

    Pass a `COCO` to hand the dataset over, or an `Arc<COCO>` to share one
    ground truth across several evaluators. The evaluator holds both datasets
    behind `Arc` and exposes them through `coco_gt()` and `coco_dt()`.

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `coco_gt` | `COCO` or `Arc<COCO>` | Ground truth COCO object |
    | `coco_dt` | `COCO` or `Arc<COCO>` | Detections COCO object (from `load_res`) |
    | `iou_type` | `IouType` | `IouType::Bbox`, `IouType::Segm`, `IouType::Keypoints`, or `IouType::Obb` |
    | `hierarchy` | `Option<Hierarchy>` | Category hierarchy for GT expansion; `None` to skip expansion |

The same classes are also reachable under `hotcoco.detection` (`from hotcoco.detection import COCOeval`), an explicit namespace for the detection metric family. Both spellings return the same objects.

---

## Properties

### `params`

=== "Python"

    ```python
    params: Params
    ```

    Evaluation parameters. Modify before calling `evaluate()`.

=== "Rust"

    ```rust
    pub params: Params
    ```

See [Params](params.md) for every configurable field and an example of changing them.

---

### `stats`

=== "Python"

    ```python
    stats: np.ndarray | list[float]
    ```

    The 12 summary metrics (10 for keypoints) as a `float64` numpy array, populated after `summarize()`. An empty list before `summarize()` is called — both states match pycocotools.

    ```python
    ev.summarize()
    print(f"AP: {ev.stats[0]:.3f}")
    print(f"AP50: {ev.stats[1]:.3f}")
    ```

=== "Rust"

    ```rust
    fn stats(&self) -> Option<&[f64]>
    ```

    ```rust
    ev.summarize();
    if let Some(stats) = ev.stats() {
        println!("AP: {:.3}", stats[0]);
        println!("AP50: {:.3}", stats[1]);
    }
    ```

---

### `eval_imgs`

Per-image, per-category, per-area-range evaluation results, populated after `evaluate()`. Entries are `None` where an image has neither ground truth nor detections for that category, as in pycocotools.

=== "Python"

    ```python
    eval_imgs: list[dict | None]
    ```

=== "Rust"

    ```rust
    fn eval_imgs(&self) -> &[Option<EvalImg>]
    ```

Each entry is a dict whose keys match pycocotools' `evalImgs` — `image_id` and
`category_id` in snake_case, everything else camelCase (there are no snake_case
variants of the camelCase keys):

| Key | Description |
|-----|-------------|
| `image_id` | Image ID |
| `category_id` | Category ID |
| `aRng` | Area range `[min, max]` this entry was evaluated under |
| `maxDet` | Max-detections cap applied |
| `dtIds` | Detection annotation IDs, score-descending |
| `gtIds` | Ground-truth annotation IDs |
| `dtMatches` | Per IoU threshold: matched GT id per detection (0 = unmatched) |
| `gtMatches` | Per IoU threshold: matched DT id per ground truth (0 = unmatched) |
| `dtScores` | Detection confidence scores |
| `dtIgnore` | Per IoU threshold: whether each detection was ignored |
| `gtIgnore` | Whether each GT was ignored (crowd or out of area range) |
| `dtMatched` / `gtMatched` | hotcoco extension: per-threshold boolean match flags |
| `gtInDenominator` | hotcoco extension: whether each GT counts in the recall denominator (differs from `gtIgnore` for Open Images group-of boxes) |

For a worked example, see [Per-image evaluation results](../guide/results.md#per-image-evaluation-results).

---

### `eval`

Accumulated precision/recall arrays, populated after `accumulate()`.

=== "Python"

    ```python
    eval: dict | None
    ```

    Contains `"precision"`, `"recall"`, and `"scores"` arrays.

=== "Rust"

    ```rust
    fn accumulated(&self) -> Option<&AccumulatedEval>
    ```

    Access elements with `precision_idx(t, r, k, a, m)` and `recall_idx(t, k, a, m)`.

`precision` and `scores` have shape `[T x R x K x A x M]`; `recall` has shape `[T x K x A x M]`:

| Dimension | Name | Default size | Description |
|-----------|------|-------------|-------------|
| T | IoU thresholds | 10 | `[0.50, 0.55, ..., 0.95]` |
| R | Recall thresholds | 101 | `[0.00, 0.01, ..., 1.00]` |
| K | Categories | varies | Number of evaluated categories |
| A | Area ranges | 4 | `[all, small, medium, large]` |
| M | Max detections | 3 | `[1, 10, 100]` |

A value of `-1` means no data — for example, no ground truth annotations for that category and area combination.

For worked examples, see [Precision and recall arrays](../guide/results.md#precision-and-recall-arrays) and [Extracting per-category AP](../guide/results.md#extracting-per-category-ap).

---

## Methods

!!! note "Call order"
    `get_results()`, `print_results()`, `summary_lines()`, `report()`, `results()`, and
    `save_results()` read the summary, so call them after `summarize()` (or `run()`).
    `report()`, `results()`, and `save_results()` raise `RuntimeError` otherwise;
    `get_results()` returns an empty dict.

    `metric_keys()`, `metric_defs()`, `provenance()`, `is_benchmark_standard()`, and
    `reference_deviations()` read only the configuration and work before `evaluate()`.

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

The table goes through Python's `sys.stdout`, so `contextlib.redirect_stdout` captures or silences it, and it shows in a notebook cell. To skip printing altogether, call [`summary_lines`](#summary_lines) instead. A library that embeds hotcoco and wants neither the table nor the non-default-parameter warnings should use it, and read what the warnings would have said from [`reference_deviations`](#reference_deviations).

!!! warning "Non-default parameters"
    `summarize()` uses a fixed display format that assumes default `iou_thrs`, `max_dets`, and `area_rng_lbl`. If you've changed any of these, a `UserWarning` is emitted (catchable with `warnings.catch_warnings`, visible in Jupyter) and some metrics might show `-1.000` (for example, AP50 when `iou_thrs` doesn't include 0.50). The `stats` array always has 12 entries (10 for keypoints) regardless of your parameters. `-1.000` always means "not computed for this configuration" — an unknown area label or max-dets value degrades to `-1.0` rather than silently substituting the `"all"` slice.

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

It behaves as those three calls do: the table goes through `sys.stdout`, and the same non-default-parameter `UserWarning`s as [`summarize`](#summarize) are emitted.

---

### `metric_keys`

```python
metric_keys() -> list[str]
```

Return metric names in canonical display order for the current evaluation mode. This is the authoritative ordering — the same list that drives `summarize()` and `get_results()`.

```python
ev = COCOeval(gt, dt, "bbox")
ev.metric_keys()
# ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
```

---

### `metric_defs`

```python
metric_defs() -> list[dict]
```

Return the metric catalog as structured data, one dict per metric in `metric_keys()`
order: `name` (str), `ap` (bool — AP vs AR), `iou_thr` (float or `None` for the full
0.50:0.05:0.95 sweep), `area` (str), `max_det` (int), and `freq_group` (`"rare"` /
`"common"` / `"frequent"` or `None`; LVIS only).

```python
ev.metric_defs()[1]
# {'name': 'AP50', 'ap': True, 'iou_thr': 0.5, 'area': 'all', 'max_det': 100, 'freq_group': None}
```

This exists so renderers read a metric's axes instead of parsing them back out of its
name — `"AR10"` is ambiguous between a detection cap of 10 and an IoU of 0.10, and only
the catalog knows which. hotcoco's own PDF report and dashboard consume it.

---

### `get_results`

```python
get_results(prefix: str | None = None, per_class: bool = False) -> dict[str, float]
```

Return the summary metrics as a dict.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | <code>str &#124; None</code> | `None` | If given, each key is prefixed as `"{prefix}/{metric}"`. |
| `per_class` | `bool` | `False` | If `True`, include per-category AP values keyed as `"AP/{cat_name}"` (or `"{prefix}/AP/{cat_name}"` with a prefix). |

Standard bbox/segm keys: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `AR1`, `AR10`, `AR100`, `ARs`, `ARm`, `ARl`.

Keypoint keys: `AP`, `AP50`, `AP75`, `APm`, `APl`, `AR`, `AR50`, `AR75`, `ARm`, `ARl`.

LVIS keys: see [The 13 LVIS metrics](../guide/lvis-open-images.md#the-13-lvis-metrics).

```python
ev.run()
ev.get_results()   # {"AP": 0.578, "AP50": 0.861, ...}
```

For prefixed and per-class keys and how they feed an experiment tracker, see
[Logging metrics](../guide/results.md#logging-metrics).

---

### `print_results`

```python
print_results() -> None
```

Print a formatted results table to stdout. For LVIS, matches the lvis-api `print_results()` style.

Like [`summarize`](#summarize), it prints through `sys.stdout`, so `contextlib.redirect_stdout`, pytest's `capsys`, and notebook cells capture it. Called before `summarize()`, it prints nothing and emits a `UserWarning`. In Rust, `print_results_lines()` returns the same lines without printing.

---

### `summary_lines`

```python
summary_lines() -> list[str]
```

The same lines `summarize()` prints, returned instead of written to stdout — one string per metric, already formatted. Use it to route the summary into a logger, a report, or a test assertion.

It populates `stats` as `summarize()` does and prints nothing, but it also skips the `UserWarning` about a non-default configuration that `summarize()` emits. That makes it the quiet path for code that embeds hotcoco, and it means nothing tells you the run is not comparable to the reference: call [`reference_deviations`](#reference_deviations), or read `provenance` from [`report`](#report), when that matters.

---

### `virtual_cat_names`

```python
virtual_cat_names: list[str]   # property
```

Category names added by Open Images hierarchy expansion — ancestor categories that exist in the hierarchy but not in the dataset's own taxonomy. Empty when not in OID mode, when no hierarchy expansion occurred, or before `evaluate()`. Use it to distinguish expanded ancestor categories from the model's native classes:

```python
ev = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=h)
ev.evaluate()
ev.virtual_cat_names   # for example, ['Carnivore', 'Mammal'] — no parentheses; it's a property
```

---

### `slice_by`

```python
slice_by(slices: dict[str, list[int]] | Callable[[dict], str]) -> dict[str, Any]
```

Re-accumulate metrics for named subsets of images without recomputing IoU, and return one metrics dict per slice. Pass either an explicit `{name: [image_ids]}` mapping or a function that takes an image dict and returns a slice name.

```python
ev.run()
by_light = ev.slice_by({"day": day_ids, "night": night_ids})
by_light["night"]["AP"]
```

Requires `evaluate()` to have run. Matching is done once and reused for every slice, so slicing a dozen ways costs barely more than slicing one way. See [sliced evaluation](../guide/evaluation.md#sliced-evaluation).

---

### `report`

```python
report() -> dict
```

Return a full evaluation report.

This is the shape every hotcoco metric family reports in, so code that renders a detection report renders a panoptic or tracking one unchanged.

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"task"` | `str` | `"detection"`. |
| `"provenance"` | `str` | `"parity_verified"` or `"extension"` — see [Provenance](#provenance-values). |
| `"metrics"` | `dict[str, float]` | Summary metrics keyed by name. |
| `"per_class"` | `dict[str, dict[str, float]]` | `{class_name: {metric: value}}`. |
| `"per_group"` | `dict[str, dict[str, float]]` | LVIS frequency buckets in LVIS mode; empty otherwise. |
| `"curves"` | `dict[str, list[float]]` | One aggregate precision-recall curve per IoU threshold (`"pr@0.50"`, `"pr@0.55"`, ...), averaged over categories at `area="all"` and the largest `max_dets`, plus the shared `"rec_thrs"` x-axis. |
| `"params"` | `dict` | The evaluation parameters used. |

```python
ev.run()
report = ev.report()

report["metrics"]["AP"]              # 0.377
report["per_class"]["person"]["AP"]  # 0.521
```

#### Provenance values

| Value | Meaning |
|-------|---------|
| `"parity_verified"` | Comparable to a published leaderboard |
| `"extension"` | A real metric or configuration, but not a leaderboard number |

What each value covers and why is in
[Check provenance before you publish a number](../guide/results.md#check-provenance-before-you-publish-a-number).
For plotting `curves`, see
[Plotting precision-recall curves](../guide/results.md#plotting-precision-recall-curves).

---

### `provenance`

```python
provenance() -> str
```

Return the same value as `report()["provenance"]`, read from the configuration alone.
See [Checking before you evaluate](../guide/results.md#checking-before-you-evaluate).

---

### `is_benchmark_standard`

```python
is_benchmark_standard() -> bool
```

Return `True` exactly when `provenance()` is `"parity_verified"`. Any other value —
including a variant added in a future release — reads as `False`, so a renderer that
checks this predicate caveats unknown provenance rather than passing it through. See
[Checking before you evaluate](../guide/results.md#checking-before-you-evaluate).

---

### `reference_deviations`

```python
reference_deviations() -> list[str]
```

Return one human-readable sentence per way this run departs from the reference
configuration, empty exactly when `provenance()` is `"parity_verified"`. See
[Checking before you evaluate](../guide/results.md#checking-before-you-evaluate).

---

### `results`

```python
results(per_class: bool = False) -> dict
```

Return evaluation results as a serializable dict.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `per_class` | `bool` | `False` | If `True`, include per-category AP values under the `"per_class"` key. |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"hotcoco_version"` | `str` | hotcoco version that produced these results. |
| `"provenance"` | `str` | Same value as `report()["provenance"]`. |
| `"params"` | `dict` | Evaluation parameters: `iou_type`, `eval_mode`, `iou_thresholds`, `recall_thresholds`, `area_ranges`, `max_dets`, `use_cats`, `kpt_oks_sigmas`, and `reference_deviations` — enough for a saved run to explain its own provenance. |
| `"metrics"` | `dict[str, float]` | Summary metrics keyed by name (same keys as `get_results()`). |
| `"per_class"` | `dict[str, float]` \| absent | Per-category AP values keyed by category name. Only present if `per_class=True`. |

```python
ev.run()
ev.results(per_class=True)   # {"hotcoco_version": ..., "metrics": {...}, "per_class": {...}}
```

Serialized, the dict looks like this:

```json
{
  "hotcoco_version": "1.0.0",
  "provenance": "parity_verified",
  "params": {
    "iou_type": "bbox",
    "eval_mode": "coco",
    "iou_thresholds": [0.5, 0.55, ...],
    "recall_thresholds": [0.0, 0.01, ...],
    "area_ranges": {"all": [0, 10000000000.0], "small": [0, 1024.0], ...},
    "max_dets": [1, 10, 100],
    "use_cats": true,
    "kpt_oks_sigmas": [0.026, 0.025, ...],
    "reference_deviations": []
  },
  "metrics": {
    "AP": 0.378, "AP50": 0.584, "AP75": 0.412, ...
  },
  "per_class": {
    "person": 0.58, "car": 0.41, ...
  }
}
```

For a worked example, see [Saving results to JSON](../guide/results.md#saving-results-to-json).

---

### `save_results`

```python
save_results(path: str, per_class: bool = False) -> None
```

Save evaluation results to a JSON file. Raises `IOError` if the file cannot be written.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Output file path. |
| `per_class` | `bool` | `False` | If `True`, include per-category AP values. |

```python
ev.run()
ev.save_results("results.json", per_class=True)
```

The file holds the dict returned by [`results()`](#results).

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
| `"matrix"` | `np.ndarray[uint64]` shape `(K+1, K+1)` | Raw confusion counts. Rows = GT category, cols = predicted. Index `K` is background. Unsigned — cast before subtracting counts (for example, `matrix.astype(np.int64)`) to avoid underflow. |
| `"normalized"` | `np.ndarray[float64]` shape `(K+1, K+1)` | Row-normalized version (rows sum to 1.0; zero rows stay zero). |
| `"cat_ids"` | `list[int]` | Category IDs for rows/cols `0..K-1`. |
| `"cat_names"` | `list[str]` | Category names for rows/cols `0..K-1`, in the same order as `cat_ids`. |
| `"num_cats"` | `int` | Number of categories `K`. |
| `"iou_thr"` | `float` | IoU threshold used. |

```python
cm = ev.confusion_matrix(iou_thr=0.5, max_det=100)
cm["matrix"]   # np.ndarray uint64, shape (K+1, K+1)
```

For the cell layout and a full walkthrough, see [Confusion matrix](../guide/diagnostics.md#confusion-matrix) in the diagnostics guide.

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
| `pos_thr` | `float` | `0.5` | IoU threshold for TP/FP classification, snapped to the nearest in `params.iouThrs`. Raises `RuntimeError` if that threshold was not in the grid when `evaluate()` ran. |
| `bg_thr` | `float` | `0.1` | Background IoU threshold for Loc/Both/Bkg discrimination |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"delta_ap"` | `dict[str, float]` | ΔAP for each error type. Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`, plus tidecv's special oracles `"FP"` (suppress every false positive) and `"FN"` (drop every missed GT from the denominator; a superset of `"Miss"`). |
| `"counts"` | `dict[str, int]` | Count of each error type. Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`. |
| `"ap_base"` | `float` | Baseline mean AP at `pos_thr`. |
| `"pos_thr"` | `float` | IoU threshold used. |
| `"bg_thr"` | `float` | Background threshold used. |

```python
result = ev.tide_errors(pos_thr=0.5, bg_thr=0.1)
result["delta_ap"]   # {"Cls": ..., "Loc": ..., "Both": ..., "Dupe": ..., "Bkg": ..., "Miss": ..., "FP": ..., "FN": ...}
```

For the error taxonomy and a full walkthrough, see [TIDE error analysis](../guide/diagnostics.md#tide-error-analysis) in the diagnostics guide.

---

### `calibration`

```python
calibration(
    n_bins: int = 10,
    iou_threshold: float = 0.5,
) -> dict
```

Compute confidence calibration metrics — how well confidence scores predict actual detection accuracy.

Requires `evaluate()` to have been called first. Bins all non-ignored detections by confidence score and compares the mean confidence in each bin to the fraction of true positives.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bins` | `int` | `10` | Number of equal-width confidence bins in [0, 1]. |
| `iou_threshold` | `float` | `0.5` | IoU threshold for TP/FP classification. Must match one of `params.iouThrs`, and have been in it when `evaluate()` ran; otherwise raises `RuntimeError`. |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"ece"` | `float` | Expected Calibration Error — weighted mean of per-bin \|accuracy - confidence\|. |
| `"mce"` | `float` | Maximum Calibration Error — worst per-bin gap. |
| `"bins"` | `list[dict]` | Per-bin breakdown. Each dict has `bin_lower`, `bin_upper`, `avg_confidence`, `avg_accuracy`, `count`. |
| `"per_category"` | `dict[str, float]` | Per-category ECE, keyed by category name. |
| `"iou_threshold"` | `float` | IoU threshold used. |
| `"n_bins"` | `int` | Number of bins. |
| `"num_detections"` | `int` | Total non-ignored detections analyzed. |

```python
cal = ev.calibration(n_bins=10, iou_threshold=0.5)
cal["ece"], cal["mce"]   # (0.0412, 0.1873)
```

For interpretation and a full walkthrough, see [Confidence calibration](../guide/diagnostics.md#confidence-calibration) in the diagnostics guide.

---

### `f_scores`

```python
f_scores(beta: float = 1.0) -> dict[str, float]
```

Compute F-beta scores after `accumulate()` (or `run()`).

For each (IoU threshold, category), finds the confidence operating point that maximizes F-beta, then averages across categories — analogous to how mAP averages precision. Returns three metrics mirroring AP/AP50/AP75.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `1.0` | Trade-off weight. `beta=1` → F1 (equal weight). `beta<1` → weights precision. `beta>1` → weights recall. |

**Returns** a dict with three keys:

| Key | Description |
|-----|-------------|
| `"F1"` | Mean max-F1 across IoU 0.50:0.05:0.95, all categories |
| `"F1_50"` | Max-F1 at IoU=0.50 |
| `"F1_75"` | Max-F1 at IoU=0.75 |

Key names reflect `beta`, formatted with no trailing zeros: `"F0.5"`, `"F0.5_50"`, `"F0.5_75"` for `beta=0.5`; `"F2"`, `"F2_50"`, `"F2_75"` for `beta=2.0`.

Returns an empty dict if `accumulate()` has not been called.

```python
ev.run()
ev.f_scores()   # {"F1": 0.523, "F1_50": 0.712, "F1_75": 0.581}
```

For a full walkthrough, see [F-scores](../guide/diagnostics.md#f-scores) in the diagnostics guide.

---

### `image_diagnostics`

```python
image_diagnostics(
    iou_thr: float = 0.5,
    score_thr: float = 0.5,
) -> dict
```

Per-image diagnostics: annotation TP/FP/FN index, per-image F1 and AP scores, error profiles, and label error candidates.

Requires `evaluate()` to have been called first.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iou_thr` | `float` | `0.5` | IoU threshold for TP/FP classification (snapped to nearest in `params.iouThrs`). Raises `RuntimeError` if that threshold was not in the grid when `evaluate()` ran. |
| `score_thr` | `float` | `0.5` | Minimum detection confidence to consider for label error detection. |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"dt_status"` | `dict[int, str]` | Detection annotation ID → `"tp"` or `"fp"`. |
| `"gt_status"` | `dict[int, str]` | GT annotation ID → `"matched"` or `"fn"`. |
| `"dt_match"` | `dict[int, int]` | TP detection → matched GT annotation ID. |
| `"gt_match"` | `dict[int, int]` | Matched GT → the detection that matched it. |
| `"img_summary"` | `dict[int, dict]` | Per-image summary; see the **image summary** table that follows. |
| `"label_errors"` | `list[dict]` | Suspected label errors, sorted by detection score descending; see the **label error** table that follows. |
| `"iou_thr"` | `float` | Actual IoU threshold used (snapped). |
| `"score_thr"` | `float` | Score threshold used for label error detection. |

Each **image summary** dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `"tp"` | `int` | True positive count. |
| `"fp"` | `int` | False positive count. |
| `"fn"` | `int` | False negative count. |
| `"f1"` | `float` | F1 score: `2*tp / (2*tp + fp + fn)`. 1.0 for empty images. |
| `"ap"` | `float` | AP at the selected IoU threshold (101-point interpolation). |
| `"error_profile"` | `str` | One of `"perfect"`, `"fp_heavy"`, `"fn_heavy"`, `"mixed"`. |

Each **label error** dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `"image_id"` | `int` | Image containing the suspected error. |
| `"dt_id"` | `int` | Detection annotation ID. |
| `"dt_score"` | `float` | Detection confidence. |
| `"dt_category"` | `str` | Detection category name. |
| `"dt_category_id"` | `int` | Detection category ID. |
| `"gt_id"` | `int \| None` | Overlapping GT annotation ID (`None` for missing_annotation). |
| `"gt_category"` | `str \| None` | GT category name. |
| `"gt_category_id"` | `int \| None` | GT category ID. |
| `"iou"` | `float` | Bbox IoU between detection and GT (0.0 for missing_annotation). |
| `"type"` | `str` | `"wrong_label"` or `"missing_annotation"`. |

```python
diag = ev.image_diagnostics(iou_thr=0.5, score_thr=0.5)
diag["img_summary"][42]   # {"tp": 3, "fp": 1, "fn": 0, "f1": 0.857, "ap": 0.75, "error_profile": "fp_heavy"}
```

For a full walkthrough, see [Per-image diagnostics and label error detection](../guide/diagnostics.md#per-image-diagnostics-and-label-error-detection) in the diagnostics guide.

---

## LVIS evaluation

These three names are drop-in replacements for the `lvis-api` package, so Detectron2
and MMDetection pipelines run unchanged. For the federated protocol, see
[LVIS evaluation](../guide/lvis-open-images.md#lvis-evaluation); for redirecting an
existing `from lvis import ...` pipeline, see
[LVIS-based pipelines](../guide/frameworks.md#lvis-based-pipelines).

### `LVISeval`

```python
hotcoco.LVISeval(gt: COCO, dt: COCO, iou_type: str = "segm") -> COCOeval
```

Returns a `COCOeval` configured for federated evaluation — equivalent to
`COCOeval(gt, dt, iou_type, lvis_style=True)`. Supports `run()`,
`print_results()`, and `get_results()`, which is what those pipelines call.
The default `iou_type` is `"segm"`, matching `lvis-api`.

`LVISEval` (capital E) is an alias: that is the spelling `lvis-api` exports and
the one `from lvis import LVISEval` expects.

```python
ev = LVISeval(lvis_gt, lvis_dt, "segm")
ev.run()
ev.get_results()   # {"AP": ..., "APr": ..., "APc": ..., "APf": ..., "AR@300": ..., ...}
```

The metric set is listed in [The 13 LVIS metrics](../guide/lvis-open-images.md#the-13-lvis-metrics).

### `LVIS`

An alias for `COCO`, provided because `lvis-api` names its dataset class `LVIS`.
Loading is identical.

### `LVISResults`

```python
hotcoco.LVISResults(lvis_gt: COCO, results, max_dets: int | None = 300) -> COCO
```

Returns a `COCO` detections object: `lvis_gt.load_res(results)` capped the way
lvis-api's `LVISResults` caps it — each image's `max_dets` highest-scoring
detections across every category, ties in results-file order. `max_dets=-1`
or `None` keeps every detection, and an integral float such as `300.0` is
that integer. The cap is `COCO.cap_detections_per_image(max_dets)`, which you
can call directly (`None` keeps every detection).

`LVISeval` evaluates an `LVISResults` result as is, as lvis-api's `LVISEval`
takes an `LVISResults` object unchanged, so `max_dets=1000` or `-1` holds
through evaluation. Detections loaded with plain `load_res()` get lvis-api's
default instead: `LVISeval` caps them at 300 the same way when constructed,
as lvis-api caps a results path or list it loads itself. The 300 does not
follow `params.max_dets`, which lvis-api's `LVISEval` does not pass to
`LVISResults` either; `StreamingEval` with `lvis_style=True` applies the same
300.

---

## StreamingEval

Evaluate detector batches as they come and get a `COCOeval` back. Each
`update()` call matches its images right away; `finalize()` assembles every
image seen into an evaluator ready for `accumulate()`, `summarize()`, and
`report()`. The results are identical to a batch `COCOeval` over the same
annotations, however the images were batched. For the training loop this is
built for, see
[Streaming evaluation](../guide/evaluation.md#streaming-evaluation).

=== "Python"

    ```python
    StreamingEval(
        categories: list[dict],
        iou_type: str = "bbox",
        lvis_style: bool = False,
        params: Params | None = None,
    )
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `categories` | `list[dict]` | — | Every category the run will see, as COCO category dicts. A category with no annotations keeps a `-1.0` slot instead of vanishing from the per-class results. The list cannot grow later: `update()` raises `KeyError` for an annotation in a category that is not on it. |
    | `iou_type` | `str` | `"bbox"` | `"bbox"`, `"segm"`, `"keypoints"`, or `"obb"` |
    | `lvis_style` | `bool` | `False` | LVIS federated evaluation; sets `max_dets` to `[300]` when `params` is not given |
    | `params` | <code>Params &#124; None</code> | `None` | Evaluation parameters, frozen for the run. `img_ids`, when set, skips images outside it; `cat_ids`, when empty, is filled from `categories`. |

    Raises `ValueError` for Open Images: hierarchy expansion needs the whole
    ground truth before the first image is evaluated.

=== "Rust"

    ```rust
    StreamingEval::new(params: Params, eval_mode: EvalMode, categories: Vec<Category>) -> Result<Self>
    ```

    Returns an error for `EvalMode::OpenImages`.

    `StreamingEval::unknown_category_ids(anns) -> Vec<u64>` returns the ids in
    `anns` that `categories` does not list, sorted and without duplicates —
    the same check `update()` makes, for validating data before streaming it.
    `update()` returns them as `Error::UnknownCategoryIds`. It is empty when
    `params.use_cats` is false.

    `StreamingEval::merge(&mut self, other)`, `to_bytes(&self)`, and
    `from_bytes(&[u8])` match the Python methods in the following sections,
    with errors as `Error::Other`. The Rust `merge` takes `other` by value and
    moves its cells over; `StreamingEval` is `Clone` for a caller that needs
    to keep it.

### `update`

```python
se.update(
    images: list[dict],
    gt_anns: list[dict],
    dt_anns: list[dict] | ndarray,
    *,
    segmentation: list[dict | list[list[float]]] | None = None,
) -> None
```

Match a batch of images' ground truth against their detections now. `images`
are COCO image dicts with at least `id` — plus `neg_category_ids` and
`not_exhaustive_category_ids` in LVIS mode, where they apply. A batch of one
is fine; passing the detector's whole batch amortizes the per-call setup.
`gt_anns` are annotation dicts in the shape `COCO(dict)` accepts; `id` is
assigned and a missing `area` is derived (the mask's pixel count, or the
box's `w × h` for an annotation without a mask), so targets as a data loader
yields them — `image_id`, `category_id`, `bbox`, `iscrowd` — are enough. For `segm`, every image holding
a polygon or box annotation in an evaluated category needs `height` and
`width`, or the call raises `ValueError` naming the images. `dt_anns` are raw predictions in the shape
`load_res()` accepts — `image_id`, `category_id`, `bbox` (or
`segmentation`/`keypoints`), and `score` — and are loaded the same way: ids
assigned, `area` derived, `iscrowd` cleared. Within an image, detections with
tied scores rank in the order given, as they do in a results file, so keep a
batch's predictions in the order the detector emitted them. An image seen
again in a later call replaces its earlier result.

`dt_anns` can also be the numpy array `load_res()` accepts, shape `(N, 7)`
with columns `[image_id, x, y, w, h, score, category_id]`, which skips
building a dict per detection. An `(N, 6)` array has no category column and
puts every row in category 1, as `load_res()` does. An id that is NaN,
negative, or not a whole number raises `ValueError`. For `segm`,
`segmentation` is a list of `N` RLE or polygon entries, one per row; it goes
only with an array, since a dict
carries its own. Every row has a box, so each detection's `area` is the box's
`w × h`, as `load_res()` gives a result that has a `bbox`, and that area
decides the `small`, `medium`, and `large` buckets. For mask area, as
torchmetrics computes it, pass dicts that carry `segmentation` and no `bbox`.

Loading and matching run without the GIL; converting the dicts does not.
The guide has [what `update()` costs](../guide/evaluation.md#what-update-costs).

A ground truth or detection whose `category_id` is not in `categories` raises
`KeyError` naming every such id, as `update_anns` does for an unknown
annotation id. Without it the label would drop out of every metric, because
the category list is fixed when the `StreamingEval` is built. The batch is
rejected whole: the evaluator is as it was before the call, so a caller that
catches the error and carries on gets honest numbers. A batch `COCOeval` does
not check, and drops such an annotation silently. Only the categories listed
count: `params.cat_ids` narrows what is evaluated without making the rest
unknown, and with `use_cats` false no category is checked, because every
annotation pools into one.

A NaN score raises the same `ValueError` as `load_res()`. Calling this after
`finalize()` raises `RuntimeError`.

### `merge`

```python
StreamingEval.merge(evaluators: Iterable[StreamingEval]) -> StreamingEval
```

A new `StreamingEval` holding every image the given evaluators have seen, as
if one evaluator had received all their `update()` calls. This is how a run
split across processes comes back together: each rank streams its shard of
the images, one rank merges them all, and `finalize()` gives exactly what one
stream over every image gives. No matching is redone. `evaluators` is any
iterable, read one at a time, so a generator over `from_bytes` keeps one shard
in memory beside the merged result rather than every rank's at once. An image
seen by more than one evaluator keeps the result of the last, the rule
`update()` applies to an image seen again, so an image that
`DistributedSampler` repeats to even out the shards counts once. The
evaluators passed in are left unchanged and usable.

Every evaluator must be built the same way: the same `categories`,
`iou_type`, `lvis_style`, and `params`, `img_ids` included — leave `img_ids`
empty on every rank rather than setting it to each rank's shard. Categories
are compared by `id`, `name`, and LVIS `frequency`, in any order. Otherwise
`merge` raises `ValueError` naming the first evaluator that differs and the
field it differs in. An empty iterable raises `ValueError`, a single
`StreamingEval` instead of an iterable of them raises `TypeError`, and a
finalized evaluator raises `RuntimeError`.

### `to_bytes` and `from_bytes`

```python
se.to_bytes() -> bytes
StreamingEval.from_bytes(data: bytes) -> StreamingEval
```

Save the state and restore it, in this process or another: for example,
`all_gather` each rank's `to_bytes()`, then `from_bytes()` and `merge()` them on
one rank. Only what `finalize()` reads is saved, so an image replaced by a
later `update()` costs nothing, and the size follows the images seen. The
restored evaluator finalizes to the same numbers, keeps accepting `update()`,
and merges like any other.

`pickle`, `copy.copy`, and `copy.deepcopy` use the same state, so a metric
object that holds a `StreamingEval` can be checkpointed or copied. The bytes
carry a format
version, and `from_bytes` raises `ValueError` for bytes that are truncated,
damaged, or from a version it does not read. Treat them as hotcoco's own
format, not an interchange format: a release that changes it says so in the
[changelog](https://github.com/derekallman/hotcoco/blob/main/CHANGELOG.md).

### `finalize`

```python
se.finalize() -> COCOeval
```

Assemble every image seen so far into a `COCOeval`. Consumes the
`StreamingEval`: `update()` or `finalize()` afterwards raises
`RuntimeError`.

The returned evaluator holds the per-cell match results and the category list,
not the annotations, which is what keeps `finalize()` cheap. It supports
`accumulate()`, `summarize()`, `report()`, `results()`, `get_results()`,
`slice_by()`, and `compare()`. Its `eval_imgs` is empty, and the analyses
that rebuild full per-image records from the datasets — `confusion_matrix()`,
`tide_errors()`, `calibration()`, `f_scores()`, and `image_diagnostics()` —
see no cells. Build a batch `COCOeval` for those.

---

## Module-level functions

### `__version__`

```python
hotcoco.__version__ -> str
```

The version of the compiled extension that is loaded, for example `"1.1.0"`. For an installed wheel it equals `importlib.metadata.version("hotcoco")`. Use it to gate a workaround on a release or to put in a bug report.

---

### `compare`

```python
hotcoco.compare(
    eval_a: COCOeval,
    eval_b: COCOeval,
    n_bootstrap: int = 0,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict
```

Pairwise model comparison. Both evaluators must have had `evaluate()` called and use the same `eval_mode`, `iou_type`, evaluation grid, and category axis — mismatched `iou_thrs`, `rec_thrs`, `max_dets`, area ranges, `use_cats`, or set of `cat_ids` raise `ValueError` rather than summarizing one run under the other's catalog. The same categories in another order compare fine: the per-category table pairs by id. Accumulation and summarization are performed internally on the shared image set.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_a` | `COCOeval` | — | Baseline model evaluation. |
| `eval_b` | `COCOeval` | — | Improved model evaluation. |
| `n_bootstrap` | `int` | `0` | Bootstrap samples for CIs (0 = disabled). |
| `seed` | `int` | `42` | Random seed for reproducibility. |
| `confidence` | `float` | `0.95` | Confidence level, for example 0.95 for a 95% CI. |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `metric_keys` | `list[str]` | Metric names in canonical display order. |
| `metrics_a` | `dict[str, float]` | Summary metrics for model A. |
| `metrics_b` | `dict[str, float]` | Summary metrics for model B. |
| `deltas` | `dict[str, float]` | Per-metric delta (B − A). |
| `ci` | `dict` or `None` | Bootstrap CIs per metric (`lower`, `upper`, `confidence`, `prob_positive`, `std_err`). `None` if `n_bootstrap=0`. |
| `per_category` | `list[dict]` | Per-category AP comparison, sorted by delta ascending. Each entry has `cat_id`, `cat_name`, `ap_a`, `ap_b`, `delta`. |
| `n_bootstrap` | `int` | Number of bootstrap samples used. |
| `num_images` | `int` | Number of shared images. |

```python
result = hotcoco.compare(ev_a, ev_b, n_bootstrap=1000)
result["deltas"]["AP"], result["ci"]["AP"]   # (0.033, {"lower": 0.01, "upper": 0.05, ...})
```

For bootstrap interpretation and a full walkthrough, see [Model comparison](../guide/diagnostics.md#model-comparison) in the diagnostics guide.
