# Params

Evaluation parameters. Created automatically by `COCOeval`, but can be modified before calling `evaluate()`.

Every property below also accepts its pycocotools camelCase spelling — `maxDets` for `max_dets`, `iouThrs` for `iou_thrs`, and so on. See the [alias table](../getting-started/migration.md#method-naming).

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
| **Values** | `"bbox"`, `"segm"`, `"keypoints"`, `"obb"` | `Bbox`, `Segm`, `Keypoints`, `Obb` |

`"obb"` evaluates oriented boxes with a rotated IoU kernel — see [OBB evaluation](../guide/evaluation.md#oriented-bounding-box-obb-evaluation).

---

### `img_ids`

Image IDs to evaluate. Empty list means all images.

| | Python | Rust |
|---|---|---|
| **Type** | `list[int]` | `Vec<u64>` |
| **Default** | `[]` | `vec![]` |

---

### `cat_ids`

Category IDs to evaluate. Empty list means all categories.

| | Python | Rust |
|---|---|---|
| **Type** | `list[int]` | `Vec<u64>` |
| **Default** | `[]` | `vec![]` |

---

### `iou_thrs`

IoU thresholds for evaluation.

| | Python | Rust |
|---|---|---|
| **Type** | `list[float]` | `Vec<f64>` |
| **Default** | `[0.5, 0.55, 0.6, ..., 0.95]` (10 values) | Same |

Assigning a grid that is the default rounded through `float32` stores the default grid exactly. That is what `torch.linspace(...).tolist()` returns: every point sits within about 4e-8 of the default. The rule is a grid of the same length with every point within 1e-6 of the default; any other grid is stored as given. It snaps instead of tolerating the difference because the difference is not harmless. Recall `k / n` lands exactly on a recall-grid point, and a point one ulp higher excludes it, so a `float32` recall grid changes which precision some cells pick up (on a category with 20 ground truths, up to 0.33 in a cell). The snapped run gets the default grid's numbers and no `iou_thrs differ` warning. Reading the property back returns the default grid, not the exact floats you set. In Rust, `Params::set_iou_thrs` snaps; assigning the field stores the grid as given.

---

### `rec_thrs`

Recall thresholds for precision interpolation.

| | Python | Rust |
|---|---|---|
| **Type** | `list[float]` | `Vec<f64>` |
| **Default** | `[0.0, 0.01, 0.02, ..., 1.0]` (101 values) | Same |

Snaps a `float32`-rounded copy of the default grid to the default, by the same rule as [`iou_thrs`](#iou_thrs). In Rust, `Params::set_rec_thrs` snaps.

---

### `max_dets`

Maximum detections per image. The summary metrics report results at each of these thresholds.

| | Python | Rust |
|---|---|---|
| **Type** | `list[int]` | `Vec<usize>` |
| **Default (bbox/segm)** | `[1, 10, 100]` | Same |
| **Default (keypoints)** | `[20]` | Same |

---

### `area_rng`

Area ranges for size-based evaluation. Each range is `[min_area, max_area]` in square pixels (pixel area).

| | Python | Rust |
|---|---|---|
| **Type** | `list[list[float]]` | `Vec<[f64; 2]>` |
| **Default (bbox/segm)** | `[[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]` | Same |
| **Default (keypoints)** | `[[0, 1e10], [1024, 9216], [9216, 1e10]]` | Same |

The defaults correspond to: all, small (area < 32² px²), medium (32² ≤ area < 96² px²), large (area ≥ 96² px²). Keypoints skip the small range.

---

### `area_rng_lbl`

Labels for the area ranges.

| | Python | Rust |
|---|---|---|
| **Type** | `list[str]` | `Vec<String>` |
| **Default (bbox/segm)** | `["all", "small", "medium", "large"]` | Same |
| **Default (keypoints)** | `["all", "medium", "large"]` | Same |

---

### `use_cats`

Whether to evaluate per-category. When `False`, all detections and ground truth annotations are pooled regardless of category label.

| | Python | Rust |
|---|---|---|
| **Type** | `bool` | `bool` |
| **Default** | `True` | `true` |

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

---

### `expand_dt`

Whether to expand detection annotations up the category hierarchy in Open Images mode. When `True`, a "Dog" detection is also propagated as an "Animal" detection (if Animal is an ancestor of Dog).

Only has an effect when `oid_style=True` and a `Hierarchy` is attached to the evaluator. Default is `False` — only GT annotations are expanded.

| | Python | Rust |
|---|---|---|
| **Type** | `bool` | `bool` |
| **Default** | `False` | `false` |

```python
ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True, hierarchy=h)
ev.params.expand_dt = True
ev.run()
```
