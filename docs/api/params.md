# Params

Evaluation parameters. Created automatically by `COCOeval`, but can be modified before calling `evaluate()`.

=== "Python"

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.max_dets = [1, 10, 100]
    ev.params.area_rng = [[0, 10000000000]]
    ev.params.area_rng_lbl = ["all"]
    ```

=== "Rust"

    ```rust
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.params.max_dets = vec![1, 10, 100];
    ev.params.area_rng = vec![[0.0, 1e10]];
    ev.params.area_rng_lbl = vec!["all".to_string()];
    ```

---

## Constructor

=== "Python"

    ```python
    Params(iou_type: str = "bbox")
    ```

=== "Rust"

    ```rust
    Params::new(iou_type: IouType) -> Self
    ```

You rarely need to construct `Params` directly — `COCOeval` creates one automatically.

---

## Properties

### `iou_type`

Evaluation type.

| | Python | Rust |
|---|---|---|
| **Type** | `str` | `IouType` |
| **Default** | `"bbox"` | `IouType::Bbox` |
| **Values** | `"bbox"`, `"segm"`, `"keypoints"` | `Bbox`, `Segm`, `Keypoints` |

!!! note "camelCase alias"
    Also available as `iouType` in Python.

---

### `img_ids`

Image IDs to evaluate. Empty list means all images.

| | Python | Rust |
|---|---|---|
| **Type** | `list[int]` | `Vec<u64>` |
| **Default** | `[]` | `vec![]` |

!!! note "camelCase alias"
    Also available as `imgIds` in Python.

---

### `cat_ids`

Category IDs to evaluate. Empty list means all categories.

| | Python | Rust |
|---|---|---|
| **Type** | `list[int]` | `Vec<u64>` |
| **Default** | `[]` | `vec![]` |

!!! note "camelCase alias"
    Also available as `catIds` in Python.

---

### `iou_thrs`

IoU thresholds for evaluation.

| | Python | Rust |
|---|---|---|
| **Type** | `list[float]` | `Vec<f64>` |
| **Default** | `[0.5, 0.55, 0.6, ..., 0.95]` (10 values) | Same |

!!! note "camelCase alias"
    Also available as `iouThrs` in Python.

---

### `rec_thrs`

Recall thresholds for precision interpolation.

| | Python | Rust |
|---|---|---|
| **Type** | `list[float]` | `Vec<f64>` |
| **Default** | `[0.0, 0.01, 0.02, ..., 1.0]` (101 values) | Same |

!!! note "camelCase alias"
    Also available as `recThrs` in Python.

---

### `max_dets`

Maximum detections per image. The summary metrics report results at each of these thresholds.

| | Python | Rust |
|---|---|---|
| **Type** | `list[int]` | `Vec<usize>` |
| **Default (bbox/segm)** | `[1, 10, 100]` | Same |
| **Default (keypoints)** | `[20]` | Same |

!!! note "camelCase alias"
    Also available as `maxDets` in Python.

---

### `area_rng`

Area ranges for size-based evaluation. Each range is `[min_area, max_area]` in square pixels (pixel area).

| | Python | Rust |
|---|---|---|
| **Type** | `list[list[float]]` | `Vec<[f64; 2]>` |
| **Default (bbox/segm)** | `[[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]` | Same |
| **Default (keypoints)** | `[[0, 1e10], [1024, 9216], [9216, 1e10]]` | Same |

The defaults correspond to: all, small (area < 32² px²), medium (32² ≤ area < 96² px²), large (area ≥ 96² px²). Keypoints skip the small range.

!!! note "camelCase alias"
    Also available as `areaRng` in Python.

---

### `area_rng_lbl`

Labels for the area ranges.

| | Python | Rust |
|---|---|---|
| **Type** | `list[str]` | `Vec<String>` |
| **Default (bbox/segm)** | `["all", "small", "medium", "large"]` | Same |
| **Default (keypoints)** | `["all", "medium", "large"]` | Same |

!!! note "camelCase alias"
    Also available as `areaRngLbl` in Python.

---

### `use_cats`

Whether to evaluate per-category. When `False`, all detections and ground truth annotations are pooled regardless of category label.

| | Python | Rust |
|---|---|---|
| **Type** | `bool` | `bool` |
| **Default** | `True` | `true` |

!!! note "camelCase alias"
    Also available as `useCats` in Python.

---

### `kpt_oks_sigmas`

Per-keypoint OKS sigma values. Controls how strictly each keypoint is evaluated — higher sigma means more tolerance.

| | Python | Rust |
|---|---|---|
| **Type** | `list[float]` | `Vec<f64>` |
| **Default** | 17 COCO keypoint sigmas | Same |

Default values (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles):

```python
[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
```

!!! note "camelCase alias"
    Also available as `kptOksSigmas` in Python.
