# Evaluation

hotcoco supports three evaluation types: bounding box, segmentation, and keypoints. All three follow the same workflow.

## The three-step pipeline

Every COCO evaluation follows the same pattern:

=== "Python"

    ```python
    from hotcoco import COCO, COCOeval

    coco_gt = COCO("annotations.json")
    coco_dt = coco_gt.load_res("detections.json")

    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()    # Per-image matching
    ev.accumulate()  # Aggregate into precision/recall curves
    ev.summarize()   # Print and compute the 12 summary metrics
    ```

=== "Rust"

    ```rust
    use hotcoco::{COCO, COCOeval};
    use hotcoco::params::IouType;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("annotations.json"))?;
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

    let mut ev = COCOeval::new(coco_gt, coco_dt, iou_type);
    ev.evaluate();    // Per-image matching
    ev.accumulate();  // Aggregate into precision/recall curves
    ev.summarize();   // Print and compute the 12 summary metrics
    ```

The only thing that changes between eval types is the `iou_type` parameter and the format of your detections.

## Bounding box evaluation

Set `iou_type` to `"bbox"` (Python) or `IouType::Bbox` (Rust).

Detection format — each result needs `image_id`, `category_id`, `bbox` as `[x, y, width, height]`, and `score`:

```json
[
  {"image_id": 42, "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.95},
  ...
]
```

IoU is computed as the intersection-over-union of the two bounding boxes.

## Segmentation evaluation

Set `iou_type` to `"segm"` (Python) or `IouType::Segm` (Rust).

Detection format — each result needs `image_id`, `category_id`, `segmentation` as a compressed RLE dict, and `score`:

```json
[
  {
    "image_id": 42,
    "category_id": 1,
    "segmentation": {"counts": "abc123...", "size": [480, 640]},
    "score": 0.95
  },
  ...
]
```

**What is RLE?** Run-Length Encoding is a compact format for binary masks. Instead of storing every pixel, it stores the lengths of alternating runs of 0s and 1s. The `size` field is `[height, width]` and `counts` is a UTF-8 string. To convert a binary mask (numpy array) from your model:

```python
import numpy as np
from hotcoco import mask as mask_utils

binary_mask = ...  # your H×W uint8 array
rle = mask_utils.encode(np.asfortranarray(binary_mask))
if isinstance(rle["counts"], bytes):
    rle["counts"] = rle["counts"].decode("utf-8")
```

See the [Mask Operations guide](masks.md) for more details.

IoU is computed on the binary masks after RLE decoding.

!!! tip
    If your results only have bounding boxes, use bbox evaluation instead. `load_res` generates polygon segmentations from bboxes, but these are axis-aligned rectangles — not instance masks.

## Keypoint evaluation

Set `iou_type` to `"keypoints"` (Python) or `IouType::Keypoints` (Rust).

Detection format — each result needs `image_id`, `category_id`, `keypoints` as a flat list of `[x1, y1, v1, x2, y2, v2, ...]`, and `score`:

```json
[
  {
    "image_id": 42,
    "category_id": 1,
    "keypoints": [x1, y1, v1, x2, y2, v2, ...],
    "score": 0.95
  },
  ...
]
```

Each keypoint has an `(x, y)` position and a visibility flag `v` (0 = not labeled, 1 = labeled but not visible, 2 = labeled and visible).

Similarity is measured using Object Keypoint Similarity ([OKS](https://cocodataset.org/#keypoints-eval)) instead of IoU. OKS uses per-keypoint sigma values that account for annotation noise — keypoints with higher variance (like hips) are weighted less strictly than precise ones (like eyes).

**Differences from bbox/segm:**

- 10 metrics instead of 12 (no small area range — keypoints are only meaningful on medium and large objects)
- Default max detections is `[20]` instead of `[1, 10, 100]`
- Ground truth annotations with `num_keypoints == 0` are automatically ignored

## The 12 COCO metrics

`summarize()` computes and prints these metrics (10 for keypoints). The evaluation protocol is defined in the [COCO detection evaluation](https://cocodataset.org/#detection-eval) specification ([Lin et al., ECCV 2014](https://arxiv.org/abs/1405.0312)):

| Index | Metric | IoU | Area | MaxDets |
|-------|--------|-----|------|---------|
| 0 | **AP** | 0.50:0.95 | all | 100 |
| 1 | AP | 0.50 | all | 100 |
| 2 | AP | 0.75 | all | 100 |
| 3 | AP | 0.50:0.95 | small | 100 |
| 4 | AP | 0.50:0.95 | medium | 100 |
| 5 | AP | 0.50:0.95 | large | 100 |
| 6 | AR | 0.50:0.95 | all | 1 |
| 7 | AR | 0.50:0.95 | all | 10 |
| 8 | AR | 0.50:0.95 | all | 100 |
| 9 | AR | 0.50:0.95 | small | 100 |
| 10 | AR | 0.50:0.95 | medium | 100 |
| 11 | AR | 0.50:0.95 | large | 100 |

### Key concepts

**IoU (Intersection over Union)** measures how well a predicted box (or mask) overlaps with the ground truth. IoU = 1.0 is a perfect match; IoU = 0.0 means no overlap. A detection "counts" only when its IoU with a GT exceeds the threshold. IoU=0.50 is lenient (half the area must overlap); IoU=0.95 is very strict (near-perfect alignment required).

**AP (Average Precision)** measures how precisely your model ranks its detections. It is the area under the precision-recall curve: a model that confidently finds all objects scores AP=1.0; a model that misses many or produces lots of false positives scores lower. The headline **AP** metric averages over 10 IoU thresholds from 0.50 to 0.95 in steps of 0.05 — it rewards both finding objects (recall) and being confident only when correct (precision).

**AR (Average Recall)** measures what fraction of ground-truth objects your model finds, given a cap on detections per image (1, 10, or 100). AR@1 tells you how good your model's single best detection is; AR@100 tells you how much it can find when allowed 100 guesses.

**Area ranges** break results down by object size in the image:

| Range | Pixel area | Intuition |
|-------|-----------|-----------|
| small | < 1,024 px² | Smaller than ~32×32 pixels |
| medium | 1,024–9,216 px² | Roughly 32×32 to 96×96 pixels |
| large | > 9,216 px² | Larger than ~96×96 pixels |

Many models perform differently across sizes — area-range metrics help you identify where improvement is needed.

## Customizing parameters

Modify `ev.params` before calling `evaluate()`:

=== "Python"

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")

    # Evaluate a subset of categories
    ev.params.cat_ids = [1, 3]

    # Evaluate a subset of images
    ev.params.img_ids = [42, 139]

    # Custom IoU thresholds
    ev.params.iou_thrs = [0.5, 0.75, 0.9]

    # Custom max detections
    ev.params.max_dets = [1, 10, 100]

    # Category-agnostic evaluation (pool all categories)
    ev.params.use_cats = False

    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ```

=== "Rust"

    ```rust
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);

    ev.params.cat_ids = vec![1, 3];
    ev.params.img_ids = vec![42, 139];
    ev.params.iou_thrs = vec![0.5, 0.75, 0.9];
    ev.params.max_dets = vec![1, 10, 100];
    ev.params.use_cats = false;

    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

!!! note
    Changing `iou_thrs`, `max_dets`, or `area_ranges` from their defaults affects what `summarize()` can display. The 12-metric output format is fixed — for example, AP50 looks for IoU=0.50 in your thresholds and shows `-1.000` if it's not there. A warning is printed when your parameters don't match the expected defaults. Filtering by `img_ids`, `cat_ids`, or setting `use_cats` is safe and won't trigger warnings.

See [Params](../api/params.md) for the full list of configurable parameters.

## LVIS evaluation

[LVIS](https://www.lvisdataset.org/) ([Gupta et al., ECCV 2019](https://arxiv.org/abs/1908.03195)) is a large-vocabulary instance segmentation dataset with ~1,200 categories. It uses **federated annotation** — each image is only exhaustively labeled for a subset of categories. Running standard COCO eval on LVIS over-penalizes detectors by treating every unannotated category as a missed detection. hotcoco handles this correctly out of the box.

### Drop-in replacement for lvis-api

If your pipeline uses lvis-api (Detectron2, MMDetection, or any code that does `from lvis import LVISEval`), call `init_as_lvis()` once at startup:

```python
from hotcoco import init_as_lvis
init_as_lvis()

# Existing lvis-api code works unchanged
from lvis import LVIS, LVISEval, LVISResults

lvis_gt = LVIS("lvis_v1_val.json")
lvis_dt = LVISResults(lvis_gt, "detections.json")
ev = LVISEval(lvis_gt, lvis_dt, "bbox")
ev.run()
ev.print_results()
results = ev.get_results()
```

### Direct usage

If you're not using lvis-api, use `LVISeval` or pass `lvis_style=True` to `COCOeval`:

```python
from hotcoco import COCO, LVISeval

lvis_gt = COCO("lvis_v1_val.json")
lvis_dt = lvis_gt.load_res("detections.json")

ev = LVISeval(lvis_gt, lvis_dt, "segm")  # lvis_style=True is set automatically
ev.run()
results = ev.get_results()
# {"AP": 0.42, "APr": 0.38, "APc": 0.44, "APf": 0.45, "AR@300": ..., ...}
```

Or equivalently:

```python
from hotcoco import COCO, COCOeval

ev = COCOeval(lvis_gt, lvis_dt, "segm", lvis_style=True)
ev.evaluate()
ev.accumulate()
ev.summarize()
results = ev.get_results()
```

### The 13 LVIS metrics

| Metric | Description |
|--------|-------------|
| AP | mAP @ IoU[0.5:0.05:0.95] |
| AP50 | mAP @ IoU=0.5 |
| AP75 | mAP @ IoU=0.75 |
| APs | AP for small objects (area < 32²) |
| APm | AP for medium objects (32² ≤ area < 96²) |
| APl | AP for large objects (area ≥ 96²) |
| APr | AP for rare categories (1–10 training images) |
| APc | AP for common categories (11–100 training images) |
| APf | AP for frequent categories (100+ training images) |
| AR@300 | Mean recall @ max 300 detections per image |
| ARs@300 | AR for small objects |
| ARm@300 | AR for medium objects |
| ARl@300 | AR for large objects |

The frequency split (rare / common / frequent) is determined by the `frequency` field on each category in the LVIS annotation file (`"r"`, `"c"`, `"f"`). These correspond to the number of training images in which the category appears, as defined in the LVIS paper.

`get_results()` returns all 13 metrics as a dict for programmatic access.

## Open Images evaluation

[Open Images](https://storage.googleapis.com/openimages/web/index.html) uses a different evaluation protocol from COCO: a single AP@IoU=0.5, a category hierarchy for annotation expansion, and `is_group_of` annotations in place of `iscrowd`. hotcoco supports all three.

### Quick start

```python
from hotcoco import COCO, COCOeval

coco_gt = COCO("oid_annotations.json")
coco_dt = coco_gt.load_res("detections.json")

ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True)
ev.run()

result = ev.get_results()
# {"AP": 0.573}
```

`oid_style=True` sets:

- IoU threshold = 0.5 (single threshold, no sweep)
- Area range = "all" only
- Max detections = 100

### Category hierarchy

Open Images categories form a hierarchy — a "Dog" detection also counts as an "Animal" detection if Animal is an ancestor of Dog. Pass a `Hierarchy` to expand GT annotations automatically at evaluation time.

```python
from hotcoco import COCO, COCOeval, Hierarchy

# From the OID hierarchy JSON (bbox_labels_600_hierarchy.json)
label_to_id = {cat["name"]: cat["id"] for cat in coco_gt.dataset["categories"]}
h = Hierarchy.from_file("bbox_labels_600_hierarchy.json", label_to_id=label_to_id)

ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True, hierarchy=h)
ev.run()
```

Without a hierarchy, `oid_style=True` still uses OID matching semantics (group-of handling, single IoU threshold) — the hierarchy only affects annotation expansion.

If you don't have a hierarchy JSON, you can derive a hierarchy from `supercategory` fields in your annotation file:

```python
from hotcoco import Hierarchy

# Derives parent→child relationships from Category.supercategory
h = Hierarchy.from_categories(coco_gt.dataset["categories"])
```

Or build one manually from a parent map:

```python
h = Hierarchy.from_parent_map({
    3: 1,   # cat 3's parent is cat 1
    4: 1,   # cat 4's parent is cat 1
    5: 2,   # cat 5's parent is cat 2
})
```

### Detection expansion

By default only GT annotations are expanded up the hierarchy. To also expand detections (so a "Dog" detection also counts as an "Animal" detection):

```python
ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True, hierarchy=h)
ev.params.expand_dt = True
ev.run()
```

### Group-of annotations

OID uses `is_group_of: true` on annotations that represent a cluster of objects rather than a single instance. These are handled differently from `iscrowd`:

- **Ignored for false negatives** — a group-of GT that goes undetected does not count as a miss.
- **Multiple detections can match** — if two detections both overlap a group-of GT at IoU ≥ 0.5, both are genuine TPs (no duplicate penalty).

Your annotations need `"is_group_of": true` in the JSON for this to take effect. Standard annotations without this field default to `false`.

### The OID metric

`summarize()` reports a single metric:

| Metric | IoU | Area | MaxDets |
|--------|-----|------|---------|
| **AP** | 0.50 | all | 100 |

`get_results()` returns `{"AP": <float>}`.

See [Hierarchy](../api/hierarchy.md) in the API reference for full construction and query methods.

---

## Confusion matrix

The standard AP pipeline only ever matches detections against ground truth of the **same** category. That means it can't tell you *which* categories your model confuses. `confusion_matrix()` fixes this with a separate cross-category matching pass.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
cm = ev.confusion_matrix(iou_thr=0.5, max_det=100)
```

No `evaluate()` call is needed first — `confusion_matrix()` is fully standalone.

### Reading the matrix

`cm["matrix"]` is a `(K+1) × (K+1)` numpy int64 array where `K` is the number of categories. **Rows are ground truth, columns are predicted.** The extra row and column at index `K` represent "background" — unmatched ground truth (missed detections / false negatives) and unmatched detections (false positives) respectively.

| | pred cat A | pred cat B | … | background |
|---|---|---|---|---|
| **gt cat A** | TP (same cat) | confusion | … | FN |
| **gt cat B** | confusion | TP | … | FN |
| **background** | FP | FP | … | 0 |

```python
cm = ev.confusion_matrix(iou_thr=0.5)

# Raw counts
matrix = cm["matrix"]          # np.ndarray int64, shape (K+1, K+1)
cat_ids = cm["cat_ids"]        # list of category IDs for rows/cols 0..K-1

# True positives per category (diagonal, excluding background)
tp_per_cat = matrix.diagonal()[:-1]

# False negatives per category (GT matched to background column)
fn_per_cat = matrix[:-1, -1]

# False positives per category (background row)
fp_per_cat = matrix[-1, :-1]

# Row-normalised version (each row sums to 1.0)
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

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iou_thr` | `0.5` | IoU threshold for a DT↔GT match |
| `max_det` | last `params.max_dets` value | Max detections per image, sorted by score |
| `min_score` | `None` (keep all) | Drop detections below this confidence before `max_det` truncation |

```python
# Stricter threshold, limit to top 50 dets, ignore low-confidence dets
cm = ev.confusion_matrix(iou_thr=0.75, max_det=50, min_score=0.3)
```

---

## TIDE error analysis

Once AP tells you *how good* your model is, TIDE ([Bolya et al., ECCV 2020](https://arxiv.org/abs/2008.08115), [code](https://github.com/dbolya/tide)) tells you *why* it falls short. `tide_errors()` decomposes every false positive and false negative into one of six mutually exclusive error types and reports the ΔAP — how much AP would improve if each error type were eliminated.

`evaluate()` must be called before `tide_errors()`.

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

### Return value

`tide_errors()` returns a dict:

| Key | Type | Description |
|-----|------|-------------|
| `"delta_ap"` | `dict[str, float]` | ΔAP for each error type. Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`, `"FP"` (all FP types combined), `"FN"` (same as `"Miss"`). |
| `"counts"` | `dict[str, int]` | Count of each error type. Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`. |
| `"ap_base"` | `float` | Baseline mean AP at `pos_thr`. |
| `"pos_thr"` | `float` | IoU threshold for TP/FP classification (default `0.5`). |
| `"bg_thr"` | `float` | Background IoU threshold (default `0.1`). |

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

### From the CLI

Pass `--tide` to `coco eval` to print the TIDE table after the standard metrics:

```bash
coco eval --gt instances_val2017.json --dt bbox_results.json --tide

# Custom thresholds
coco eval --gt instances_val2017.json --dt bbox_results.json \
    --tide --tide-pos-thr 0.75 --tide-bg-thr 0.2
```

## Confidence calibration

A model that outputs confidence 0.9 should be correct about 90% of the time. `calibration()` measures how well your model's confidence scores align with actual detection accuracy by binning detections by confidence and comparing predicted confidence to the fraction of true positives in each bin.

Requires `evaluate()` to have been called first.

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

`per_category` maps each category name to its ECE. Some categories may be well-calibrated while others are wildly off:

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

### From the CLI

Pass `--calibration` to `coco eval`:

```bash
coco eval --gt annotations.json --dt detections.json --calibration

# Custom bins and IoU threshold
coco eval --gt annotations.json --dt detections.json \
    --calibration --cal-bins 15 --cal-iou-thr 0.75
```

See [`calibration`](../api/cocoeval.md#calibration) in the API reference for full parameter details.

---

## F-scores

`f_scores()` computes F-beta scores from the precision/recall curves built by `accumulate()`. It finds the confidence threshold that maximises F-beta for each (IoU, category) combination, then averages — the same summarisation strategy as mAP.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

scores = ev.f_scores()
# {"F1": 0.523, "F150": 0.712, "F175": 0.581}
```

Use `beta` to shift the precision/recall trade-off:

```python
ev.f_scores(beta=0.5)  # precision-weighted  → {"F0.5": ..., "F0.550": ..., "F0.575": ...}
ev.f_scores(beta=2.0)  # recall-weighted     → {"F2.0": ..., "F2.050": ..., "F2.075": ...}
```

F-scores complement `get_results()` when you care about a specific operating point rather than area-under-curve. A high AP with a low F1 often signals that performance is concentrated at high recall or high precision, not both simultaneously.

See [`f_scores`](../api/cocoeval.md#f_scores) in the API reference for full parameter details.

## Sliced evaluation

`slice_by()` re-computes all summary metrics for named subsets of images — without re-running IoU computation. This is useful for comparing model performance across data splits (e.g., indoor vs outdoor, day vs night, small images vs large images).

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()

# Define slices as {name: [img_ids]}
slices = {
    "indoor": [42, 139, 203, ...],
    "outdoor": [78, 412, 901, ...],
}

results = ev.slice_by(slices)
```

Each slice in `results` contains the full set of summary metrics plus a delta vs the overall baseline:

```python
for name, sr in results.items():
    if name == "_overall":
        continue
    print(f"{name}: AP={sr['AP']:.3f} (Δ{sr['delta']['AP']:+.3f})")
```

You can also pass a callable instead of a dict — it receives each image dict and returns a slice name (or `None` to skip):

```python
results = ev.slice_by(lambda img: "large" if img["width"] > 1000 else "small")
```

### From the CLI

Pass `--slices <path>` to `coco eval` with a JSON file mapping slice names to image ID lists:

```bash
coco eval --gt annotations.json --dt detections.json --slices slices.json
```

---

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

### From the CLI

```bash
coco compare --gt ann.json --dt-a baseline.json --dt-b improved.json
coco compare --gt ann.json --dt-a a.json --dt-b b.json --bootstrap 1000
coco compare --gt ann.json --dt-a a.json --dt-b b.json --json  # CI/CD
```

### Plotting

```python
from hotcoco import plot

result = hotcoco.compare(ev_a, ev_b, n_bootstrap=1000)
plot.comparison_bar(result, save_path="comparison.png")
plot.category_deltas(result, top_k=10, save_path="deltas.png")
```

See [`compare`](../api/cocoeval.md#compare) and [`comparison_bar`](../api/plot.md) in the API reference.

---

## Per-image diagnostics & label error detection

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

### Annotation index

The result also includes the per-annotation TP/FP/FN index that powers the browse viewer's eval coloring:

```python
diag["dt_status"]  # {ann_id: "tp" | "fp"}
diag["gt_status"]  # {ann_id: "matched" | "fn"}
diag["dt_match"]   # {dt_id: gt_id}  (TP pairs)
diag["gt_match"]   # {gt_id: dt_id}  (reverse)
```

### From the CLI

```bash
coco eval --gt annotations.json --dt detections.json --diagnostics

# Custom thresholds
coco eval --gt ann.json --dt det.json --diagnostics --diag-iou-thr 0.75 --diag-score-thr 0.3
```

See [`image_diagnostics`](../api/cocoeval.md#image_diagnostics) in the API reference for full parameter details.

---

## Saving results to JSON

`results()` and `save_results()` serialize the full evaluation output — parameters, summary metrics, and optionally per-category AP — to a JSON dict or file. Both require `summarize()` (or `run()`) first.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

# Get results as a dict
r = ev.results()
print(r["metrics"]["AP"])      # 0.378
print(r["params"]["iou_type"]) # "bbox"

# Save to a file
ev.save_results("results.json")

# Include per-category AP
ev.save_results("results.json", per_class=True)
```

The JSON structure:

```json
{
  "params": {
    "iou_type": "bbox",
    "iou_thresholds": [0.5, 0.55, ...],
    "area_ranges": {"all": [0, 10000000000.0], "small": [0, 1024.0], ...},
    "max_dets": [1, 10, 100],
    "is_lvis": false
  },
  "metrics": {
    "AP": 0.378, "AP50": 0.584, "AP75": 0.412, ...
  },
  "per_class": {
    "person": 0.58, "car": 0.41, ...
  }
}
```

From the `coco-eval` CLI, pass `--output <path>` to write the same JSON automatically (always includes per-category AP):

```bash
coco-eval --gt instances_val2017.json --dt bbox_results.json --output results.json
```

---

## Logging metrics

`get_results()` accepts an optional `prefix` and `per_class` flag, returning a flat `dict[str, float]` that plugs directly into any experiment tracker.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

metrics = ev.get_results(prefix="val/bbox", per_class=True)
# {"val/bbox/AP": 0.578, ..., "val/bbox/AP/person": 0.82, "val/bbox/AP/car": 0.71, ...}
```

### Weights & Biases

```python
import wandb
wandb.log(ev.get_results(prefix="val/bbox", per_class=True), step=epoch)
```

### MLflow

```python
import mlflow
mlflow.log_metrics(ev.get_results(prefix="val/bbox"), step=epoch)
```

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
for k, v in ev.get_results(prefix="val/bbox").items():
    writer.add_scalar(k, v, global_step=epoch)
```

See [`get_results`](../api/cocoeval.md#get_results) in the API reference for full parameter details.
