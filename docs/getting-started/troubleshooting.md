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

hotcoco ships prebuilt wheels that bundle a compiled Rust extension. If you upgrade numpy to a major version that changes the C ABI (numpy 1.x → 2.x, for example), you might need to reinstall hotcoco to pick up a compatible wheel:

```bash
pip install --upgrade hotcoco
```

The `coco` CLI distinguishes this case: when the package is present but its
compiled extension fails to import (broken wheel, wrong platform), the error says
so and suggests reinstalling rather than claiming hotcoco is not installed.

---

### Name conflict with pycocotools

If you have both `hotcoco` and `pycocotools` installed and import `COCO` from `pycocotools` by accident:

```python
# Wrong — gets pycocotools
from pycocotools.coco import COCO

# Right — gets hotcoco
from hotcoco import COCO
```

To use hotcoco as a drop-in without changing any imports, call `init_as_pycocotools()` once at the start of your script — see [Framework integrations](../guide/frameworks.md) for the details and per-framework notes.

---

## Detection format errors

### `load_res` raises a `KeyError` or returns empty results

Your detection file is missing a required field. The fields each `iou_type` needs, with a minimal example, are in [Bounding box evaluation](../guide/evaluation.md#bounding-box-evaluation) and the segmentation and keypoint sections of the same page; if the boxes load but the numbers look wrong, check [Bbox format](#bbox-format-x1-y1-x2-y2-vs-x-y-w-h).

---

### Bbox format: `[x1, y1, x2, y2]` vs `[x, y, w, h]`

COCO uses `[x, y, width, height]` (top-left corner + size) in **pixel coordinates**. Two common pitfalls:

- Many model outputs use `[x1, y1, x2, y2]` (two corners) instead of `[x, y, w, h]`
- Some formats, such as YOLO, use normalized coordinates in `[0, 1]` — COCO always expects pixel values

`load_res` checks for the first mistake: when most detection boxes run past their
image when read as `[x, y, w, h]` but fit it when read as `[x1, y1, x2, y2]`, it
emits a `DatasetWarning` that points here. A box or two past the edge is a dataset
quirk and says nothing; a majority is a format. The check needs the images'
`width` and `height`, so it is silent on a ground truth without them. Normalized
coordinates have no such signature and go unflagged.

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

If a detection's `image_id` doesn't exist in the ground-truth dataset, it won't raise an error — `load_res` accepts it and emits one `DatasetWarning` naming the first such id, and the evaluator ignores those detections since there's no matching GT image to evaluate against.

Verify your image IDs match:

```python
gt_img_ids = set(coco_gt.get_img_ids())
dt_img_ids = {d["image_id"] for d in detections}
missing = dt_img_ids - gt_img_ids
if missing:
    print(f"Detections reference {len(missing)} unknown image IDs: {list(missing)[:5]} ...")
```

---

## Segmentation issues

### RLE `counts` field: bytes vs string

`mask.encode` returns `counts` as `bytes`, and JSON cannot hold bytes — the conversion is in [RLE `counts` is bytes, not a string](../guide/masks.md#rle-counts-is-bytes-not-a-string).

---

### `size` field order: `[height, width]`

COCO RLE uses `[height, width]` order, not `[width, height]`. If your masks look wrong or you get shape mismatches, check that `size` matches the image dimensions in `[H, W]` order.

---

## Evaluation results

### Metrics differ slightly from pycocotools

First check that `ev.params` matches the configuration you expect — mismatched `iou_thrs` or `area_rng` is the usual cause. With default parameters, hotcoco matches pycocotools bit for bit (see [Benchmarks](../benchmarks.md#metric-parity)); anything above about 1e-12 is a bug worth reporting.

---

### `summarize()` prints `-1.000`

**Every metric is `-1.000`.** `evaluate()` found no matching (image_id, category_id) pairs between GT and DT. Common causes:

- Wrong `iou_type` — for example, passing segmentation results to `COCOeval(..., "bbox")`
- `category_id` mismatch — model uses 0-indexed classes but COCO uses 1-indexed IDs
- All detections were dropped by `load_res` — see [`image_id` not found in ground truth](#image_id-not-found-in-ground-truth)

Check that categories align:

```python
gt_cats = {c["id"]: c["name"] for c in coco_gt.load_cats(coco_gt.get_cat_ids())}
dt_cat_ids = {d["category_id"] for d in detections}
missing_cats = dt_cat_ids - set(gt_cats)
if missing_cats:
    print(f"Unknown category IDs in detections: {missing_cats}")
```

**Some metrics are `-1.000`.** You customized `iou_thrs`, `max_dets`, or `area_rng`, and the fixed 12-line display has no value for that slot — AP50 needs `0.50` in `ev.params.iou_thrs`, for example. This is expected; `ev.get_results()` returns only the metrics that were computed. See [`summarize`](../api/cocoeval.md#summarize).

---

### `tide_errors()` raises `RuntimeError`

`tide_errors()` needs `evaluate()` to have run first — see [`tide_errors`](../api/cocoeval.md#tide_errors).

---

## Getting help

If your issue isn't covered here, open an issue on [GitHub](https://github.com/derekallman/hotcoco/issues) with a minimal reproducer.
