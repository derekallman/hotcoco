# Troubleshooting

Common issues and how to fix them.

## Import errors

### `ModuleNotFoundError: No module named 'hotcoco'`

The package isn't installed in the active environment. Run:

```bash
pip install hotcoco
```

If you're in a virtual environment, make sure it's activated before installing.

---

### `ImportError` after upgrading numpy

hotcoco ships prebuilt wheels that bundle a compiled Rust extension. If you upgrade numpy to a major version that changes the C ABI (e.g. numpy 1.x → 2.x), you may need to reinstall hotcoco to pick up a compatible wheel:

```bash
pip install --upgrade hotcoco
```

---

### Name conflict with pycocotools

If you have both `hotcoco` and `pycocotools` installed and import `COCO` from `pycocotools` by accident:

```python
# Wrong — gets pycocotools
from pycocotools.coco import COCO

# Right — gets hotcoco
from hotcoco import COCO
```

To use hotcoco as a drop-in without changing any imports, call `init_as_pycocotools()` once at the start of your script:

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()

# All pycocotools imports now route through hotcoco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
```

See [Migrating from pycocotools](migration.md) for more details.

---

## Detection format errors

### `load_res` raises a `KeyError` or returns empty results

Your detection file is missing a required field. Every detection dict must have:

| Field | Required for |
|-------|-------------|
| `image_id` | All types |
| `category_id` | All types |
| `score` | All types |
| `bbox` | `"bbox"` type — `[x, y, width, height]` |
| `segmentation` | `"segm"` type — RLE dict with `counts` and `size` |
| `keypoints` | `"keypoints"` type — flat list `[x1, y1, v1, x2, y2, v2, ...]` |

A minimal valid bbox detection:

```python
{"image_id": 42, "category_id": 1, "bbox": [10.0, 20.0, 100.0, 80.0], "score": 0.95}
```

---

### Bbox format: `[x1, y1, x2, y2]` vs `[x, y, w, h]`

COCO uses `[x, y, width, height]` (top-left corner + size) in **pixel coordinates**. Two common pitfalls:

- Many model outputs use `[x1, y1, x2, y2]` (two corners) instead of `[x, y, w, h]`
- Some formats (e.g. YOLO) use normalized coordinates in `[0, 1]` — COCO always expects pixel values

Convert before passing to `load_res`:

```python
# Convert XYXY → XYWH
detections = [
    {**d, "bbox": [d["bbox"][0], d["bbox"][1],
                   d["bbox"][2] - d["bbox"][0],
                   d["bbox"][3] - d["bbox"][1]]}
    for d in raw_detections
]
```

---

### `image_id` not found in ground truth

If a detection's `image_id` doesn't exist in the ground-truth dataset, it won't raise an error — `load_res` accepts it, but the evaluator silently ignores it since there's no matching GT image to evaluate against. No warning is emitted, so mismatches can be hard to spot.

Verify your image IDs match:

```python
gt_img_ids = set(coco_gt.get_img_ids())
dt_img_ids = {d["image_id"] for d in detections}
missing = dt_img_ids - gt_img_ids
if missing:
    print(f"Detections reference {len(missing)} unknown image IDs: {list(missing)[:5]} ...")
```

---

### All metrics are `-1.000`

This usually means `evaluate()` found no matching (image_id, category_id) pairs between GT and DT. Common causes:

- Wrong `iou_type` — e.g. passing segmentation results to `COCOeval(..., "bbox")`
- `category_id` mismatch — model uses 0-indexed classes but COCO uses 1-indexed IDs
- All detections were dropped by `load_res` (see `image_id` issue above)

Check that categories align:

```python
gt_cats = {c["id"]: c["name"] for c in coco_gt.load_cats(coco_gt.get_cat_ids())}
dt_cat_ids = {d["category_id"] for d in detections}
missing_cats = dt_cat_ids - set(gt_cats)
if missing_cats:
    print(f"Unknown category IDs in detections: {missing_cats}")
```

---

## Segmentation issues

### RLE `counts` field: bytes vs string

The RLE `counts` field should be a UTF-8 string, not raw bytes. Some libraries return `bytes`:

```python
import hotcoco.mask as mask_utils
import numpy as np

rle = mask_utils.encode(np.asfortranarray(binary_mask))

# Fix bytes → str
if isinstance(rle["counts"], bytes):
    rle["counts"] = rle["counts"].decode("utf-8")
```

---

### `size` field order: `[height, width]`

COCO RLE uses `[height, width]` order, not `[width, height]`. If your masks look wrong or you get shape mismatches, check that `size` matches the image dimensions in `[H, W]` order.

---

## Evaluation results

### Metrics differ slightly from pycocotools

hotcoco is verified to match pycocotools within floating-point tolerance (bbox ≤ 0.0001, segm ≤ 0.0002, keypoints exact). Differences within these tolerances are expected.

If you see differences larger than these tolerances, the most common cause is mismatched `iou_thrs` or `area_ranges` — double-check that `ev.params` matches your expected configuration.

---

### `summarize()` prints `-1.000` for some metrics

A metric shows `-1.000` when its required IoU threshold or area range isn't in `ev.params`. For example, AP50 requires `0.50` to be in `ev.params.iou_thrs`. If you've customized `iou_thrs`, the standard 12-metric output format will show `-1.000` for thresholds not in your list. This is expected — use `ev.get_results()` to access only the metrics that were actually computed.

---

### `tide_errors()` raises `RuntimeError`

`tide_errors()` requires `evaluate()` to have been called first:

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()          # required before tide_errors
result = ev.tide_errors()
```

---

## Getting help

If your issue isn't covered here, open an issue on [GitHub](https://github.com/derekallman/hotcoco/issues) with a minimal reproducer.
