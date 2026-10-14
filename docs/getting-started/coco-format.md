# The COCO format

The ground-truth file is a single JSON object with three required top-level keys —
`images`, `annotations`, and `categories` — plus optional `info` and `licenses`
metadata. This page documents what hotcoco requires in each, what is optional, and
what happens to fields it doesn't recognize.

Detection *results* are a different, flatter format (a JSON array of detection
dicts) — see [Working with results](../guide/results.md#loading-results).

## Minimal example

```json
{
  "images": [
    {"id": 1, "width": 640, "height": 480, "file_name": "000000000001.jpg"}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 18,
     "bbox": [258.2, 41.3, 348.3, 243.5], "area": 84810.0, "iscrowd": 0}
  ],
  "categories": [
    {"id": 18, "name": "dog", "supercategory": "animal"}
  ]
}
```

## `images`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `id` | int | **yes** | Unique per image. 0-based IDs are accepted, and so is an integral float such as `1.0`, which is how a numpy-backed JSON writer spells an int. |
| `width`, `height` | int | **yes** | Pixel dimensions. Needed for mask decoding, YOLO/Open Images export, and the browser. |
| `file_name` | str | no | Defaults to `""`. Needed by `browse()` and image-dir workflows. |
| `license`, `coco_url`, `flickr_url`, `date_captured` | — | no | Carried through untouched. |
| `neg_category_ids`, `not_exhaustive_category_ids` | list[int] | no | LVIS federated-annotation fields — see [LVIS and Open Images](../guide/lvis-open-images.md). |

## `annotations`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `image_id` | int | **yes** | Must reference an entry in `images`. |
| `category_id` | int | **yes** | Must reference an entry in `categories`. |
| `id` | int | no | Defaults to 0. Duplicate ids are tolerated but reported in [`load_warnings`](../api/coco.md#load_warnings). |
| `bbox` | `[x, y, w, h]` | for bbox eval | Pixel coordinates, top-left corner + size — not `[x1, y1, x2, y2]`. |
| `area` | float | no | Computed by `load_res` when missing. Annotations without `area` are excluded from explicit area-range queries (`get_ann_ids(area_rng=...)`, `filter(area_rng=...)`). |
| `segmentation` | polygon(s) or RLE | for segm eval | A list of flat polygon coordinate lists, an uncompressed RLE dict (`counts` as a list of ints), or a compressed RLE dict (`counts` as a string, or as the `bytes` that `mask.encode` and pycocotools produce). See [Mask operations](../guide/masks.md). |
| `iscrowd` | 0/1 or bool | no | Defaults to 0. Crowd regions match by IoA and are ignored rather than scored. |
| `keypoints` | flat list | for keypoint eval | `[x1, y1, v1, x2, y2, v2, ...]` with visibility flags. |
| `num_keypoints` | int | no | GT annotations with `num_keypoints == 0` are ignored in keypoint eval. When absent, it is derived from `keypoints` (the count with visibility `> 0`). |
| `obb` | `[cx, cy, w, h, angle]` | for OBB eval | hotcoco extension; angle in radians. See [OBB evaluation](../guide/evaluation.md#oriented-bounding-box-obb-evaluation). |
| `is_group_of` | bool | no | Open Images group-of flag — distinct matching semantics from `iscrowd`. |
| `score` | float | no | Present only in detection results, not ground truth. |

## `categories`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `id` | int | **yes** | Unique per category. |
| `name` | str | no | Used in per-class reports and format converters. A category without one gets the display name `cat_<id>`, and the load is flagged in `load_warnings`. |
| `supercategory` | str | no | Used for `get_cat_ids(sup_nms=...)` queries and Open Images hierarchy derivation. |
| `keypoints`, `skeleton` | — | no | Keypoint names and connectivity, as in the COCO person file. |
| `frequency` | `"r"`/`"c"`/`"f"` | no | LVIS frequency bucket, drives APr/APc/APf. |

## Unknown keys are preserved

Any key not listed above — custom metadata on images, annotations, or categories —
survives `load → filter/split/merge → save` verbatim, appears in the dicts returned
by `load_imgs`/`load_anns`/`load_cats`, and is visible to `slice_by` callables.
This matches pycocotools, which stores raw dicts.

```python
coco = COCO("annotations.json")   # images carry a custom "weather" key
by_weather = ev.slice_by(lambda img: img["weather"])
```

## Loading quirks worth knowing

- **Non-finite values.** Bare `NaN`/`Infinity` tokens (which Python's `json`
  module happily writes) are normalized to `null` on load, matching pycocotools.
  They surface as `None` on fields such as `area` and `score`, and the load
  notice reports how many were normalized. The exception is a `NaN` detection
  score in a file passed to `load_res()`: it raises `ValueError` naming the
  line, as a `NaN` score in a list of dicts does, rather than loading a
  detection with no score.
- **`load_warnings`.** Anything the loader tolerated but flagged — duplicate
  annotation ids, sanitized non-finite values, orphaned result ids — is collected
  on [`coco.load_warnings`](../api/coco.md#load_warnings) as well as printed.
- **Malformed geometry errors.** Corrupt RLE strings, out-of-range polygon
  coordinates, and impossible mask dimensions raise an error instead of
  producing garbage — hotcoco treats annotation files as untrusted input.
- **Deeper validation** is a separate step: run
  [`coco.healthcheck()`](../guide/datasets.md#healthcheck) to catch structural
  errors, degenerate boxes, and distribution problems before training on a file.
