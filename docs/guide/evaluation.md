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

Detection format — each result needs `image_id`, `category_id`, `segmentation` as an RLE dict, and `score`:

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

Similarity is measured using Object Keypoint Similarity (OKS) instead of IoU. OKS uses per-keypoint sigma values that account for annotation noise — keypoints with higher variance (like hips) are weighted less strictly than precise ones (like eyes).

**Differences from bbox/segm:**

- 10 metrics instead of 12 (no small area range — keypoints are only meaningful on medium and large objects)
- Default max detections is `[20]` instead of `[1, 10, 100]`
- Ground truth annotations with `num_keypoints == 0` are automatically ignored

## The 12 COCO metrics

`summarize()` computes and prints these metrics (10 for keypoints):

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

- **AP** (Average Precision) is the area under the precision-recall curve, averaged across IoU thresholds.
- **AR** (Average Recall) is the maximum recall at a fixed number of detections per image, averaged across IoU thresholds.
- **Area ranges**: small (0-32²), medium (32²-96²), large (96²+) pixels.

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
    Changing `iou_thrs`, `max_dets`, or `area_rng_lbl` from their defaults affects what `summarize()` can display. The 12-metric output format is fixed — for example, AP50 looks for IoU=0.50 in your thresholds and shows `-1.000` if it's not there. A warning is printed when your parameters don't match the expected defaults. Filtering by `img_ids`, `cat_ids`, or setting `use_cats` is safe and won't trigger warnings.

See [Params](../api/params.md) for the full list of configurable parameters.

## LVIS evaluation

[LVIS](https://www.lvisdataset.org/) is a large-vocabulary instance segmentation dataset with ~1,200 categories. It uses **federated annotation** — each image is only exhaustively labeled for a subset of categories. Running standard COCO eval on LVIS over-penalizes detectors by treating every unannotated category as a missed detection. hotcoco handles this correctly out of the box.

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
| APr | AP for rare categories (1–10 instances) |
| APc | AP for common categories (11–100 instances) |
| APf | AP for frequent categories (100+ instances) |
| AR@300 | Mean recall @ max 300 detections per image |
| ARs@300 | AR for small objects |
| ARm@300 | AR for medium objects |
| ARl@300 | AR for large objects |

The frequency split (rare / common / frequent) is determined by the `frequency` field on each category in the LVIS annotation file (`"r"`, `"c"`, `"f"`).

`get_results()` returns all 13 metrics as a dict for programmatic access.

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

Once AP tells you *how good* your model is, TIDE ([Bolya et al., ECCV 2020](https://github.com/dbolya/tide)) tells you *why* it falls short. `tide_errors()` decomposes every false positive and false negative into one of six mutually exclusive error types and reports the ΔAP — how much AP would improve if each error type were eliminated.

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
