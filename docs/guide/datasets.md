# Dataset Operations

hotcoco can do more than evaluate — it can reshape your datasets before evaluation starts.

All operations return a new `COCO` object and leave the original unchanged. They compose naturally,
so you can chain filter → split → sample in a single expression.

!!! tip
    Run `coco.stats()` first to understand your dataset before reshaping it. See the
    [`stats` API reference](../api/coco.md#stats) for the full return structure.

---

## filter

Subset a dataset to a specific set of categories, images, or annotation sizes.
Returns a new `COCO` with matching annotations and — by default — only the images
that have at least one match.

```python
from hotcoco import COCO

coco = COCO("instances_val2017.json")

# Keep only "person" annotations
person_id = coco.get_cat_ids(cat_nms=["person"])[0]
people = coco.filter(cat_ids=[person_id])

print(len(people.dataset["images"]))       # 2693
print(len(people.dataset["annotations"])) # 10777
```

Pass `drop_empty_images=False` to keep all images even if they have no matching
annotations — useful when you need consistent image IDs across filtered splits.

```python
# Same annotations, all 5000 images preserved
people_all_imgs = coco.filter(cat_ids=[person_id], drop_empty_images=False)
```

Filter by annotation area to focus on a size range:

```python
# Medium objects only (32² – 96² px²)
medium = coco.filter(area_rng=[1024.0, 9216.0])
```

Filters compose — all criteria are ANDed:

```python
medium_people = coco.filter(cat_ids=[person_id], area_rng=[1024.0, 9216.0])
```

---

## split

Split a dataset into train/val (or train/val/test) subsets. Images are shuffled
deterministically and partitioned by fraction. Annotations follow their images;
all splits share the full category list.

```python
# 80/20 train/val split
train, val = coco.split(val_frac=0.2, seed=42)

print(len(train.dataset["images"]))  # 4000
print(len(val.dataset["images"]))    # 1000
```

Add a test set with a second fraction:

```python
train, val, test = coco.split(val_frac=0.15, test_frac=0.15, seed=42)
# train ~70%, val ~15%, test ~15%
```

The same `seed` always produces the same split — important for reproducibility
across experiments:

```python
# These are identical
train_a, val_a = coco.split(val_frac=0.2, seed=42)
train_b, val_b = coco.split(val_frac=0.2, seed=42)
```

A typical eval workflow — filter first, then split:

```python
people = coco.filter(cat_ids=[person_id])
train, val = people.split(val_frac=0.2, seed=42)
```

---

## sample

Draw a random subset of images (with their annotations). Useful for quick
iteration during development without running full-dataset evaluation.

```python
# Sample 500 images
subset = coco.sample(n=500, seed=0)

# Or by fraction
subset = coco.sample(frac=0.1, seed=0)
```

Like `split`, the sample is deterministic for the same seed:

```python
# Always the same 500 images
a = coco.sample(n=500, seed=0)
b = coco.sample(n=500, seed=0)
```

---

## merge

Combine multiple annotation files into one. Common when annotations arrive in
separate batches or from separate labeling jobs.

All datasets must share the same category taxonomy (same names and supercategories).
Image and annotation IDs are remapped automatically to be globally unique.

```python
batch1 = COCO("batch1.json")
batch2 = COCO("batch2.json")

combined = COCO.merge([batch1, batch2])

print(len(combined.dataset["images"]))
# len(batch1.images) + len(batch2.images)
```

Merging a dataset with itself doubles the image and annotation count — useful
for stress-testing:

```python
doubled = COCO.merge([coco, coco])
```

`merge` raises `ValueError` if the datasets have different category sets:

```python
# Raises ValueError: category 'horse' not found in first dataset
COCO.merge([coco_animals, coco_vehicles])
```

---

## save

Write any `COCO` object back to a JSON file. The output format is
standard COCO JSON, readable by any tool that accepts COCO annotations.

```python
merged = COCO.merge([batch1, batch2])
merged.save("combined.json")
```

`save` works at any point in a pipeline:

```python
coco.filter(cat_ids=[person_id]).sample(n=1000, seed=0).save("person_sample.json")
```

---

## convert

Convert between COCO and other annotation formats. Supported formats:

| Format | Direction | Method |
|--------|-----------|--------|
| YOLO | COCO ↔ YOLO | `to_yolo()` / `from_yolo()` |
| Pascal VOC | COCO ↔ VOC | `to_voc()` / `from_voc()` |
| CVAT | COCO ↔ CVAT | `to_cvat()` / `from_cvat()` |
| DOTA | COCO ↔ DOTA | `to_dota()` / `from_dota()` |

### COCO → YOLO

```python
from hotcoco import COCO

coco = COCO("instances_val2017.json")
stats = coco.to_yolo("labels/val2017/")
print(stats)
# {'images': 5000, 'annotations': 36781, 'skipped_crowd': 12, 'missing_bbox': 0}
```

`to_yolo` creates `labels/val2017/` (if it doesn't exist) and writes:

- One `<stem>.txt` per image, where each line is `class_idx cx cy w h`
  — all coordinates normalized to `[0, 1]` by image dimensions.
- An empty `<stem>.txt` for images with no annotations (YOLO convention).
- `data.yaml` with `nc` (category count) and an ordered `names` list.

Category IDs are sorted numerically and assigned 0-indexed YOLO class IDs in
that order: COCO ID 1 → class 0, ID 3 → class 1, ID 7 → class 2, etc.

Crowd annotations and annotations without a bounding box are silently skipped
and counted in the returned stats dict.

### YOLO → COCO

```python
# Without image dimensions (width/height stored as 0)
coco = COCO.from_yolo("labels/val2017/")

# With image dimensions read from disk via Pillow
coco = COCO.from_yolo("labels/val2017/", images_dir="images/val2017/")
coco.save("reconstructed.json")
print(f"{len(coco.dataset['images'])} images, {len(coco.dataset['annotations'])} annotations")
```

`from_yolo` reads `data.yaml` for the category list, then parses every `.txt`
file in the directory. If `images_dir` is given, hotcoco uses Pillow to read
each image's `(width, height)` — install it with `pip install Pillow` if needed.

Without `images_dir`, bounding boxes are still parsed but stored relative to a
`0×0` canvas. This is fine for inspection or re-evaluation, but tools that need
pixel-space coordinates (visualization, `ann_to_mask`) will need real dims.

### Round-trip

COCO bbox values round-trip within floating-point precision (less than 0.0001 px
error for typical image sizes):

```python
coco     = COCO("instances_val2017.json")
stats    = coco.to_yolo("labels/")
coco2    = COCO.from_yolo("labels/", images_dir="images/val2017/")
coco2.save("reconstructed.json")
```

### COCO → Pascal VOC

```python
from hotcoco import COCO

coco = COCO("instances_val2017.json")
stats = coco.to_voc("voc_output/")
print(stats)
# {'images': 5000, 'annotations': 36781, 'crowd_as_difficult': 12, 'missing_bbox': 0}
```

`to_voc` creates `voc_output/Annotations/` and writes one `<stem>.xml` per image
in standard Pascal VOC format (`<annotation>/<object>/<bndbox>` with absolute
pixel coordinates). Also writes `labels.txt` listing category names sorted by
COCO ID.

Field mapping:

- COCO `bbox [x, y, w, h]` → VOC `<xmin>/<ymin>/<xmax>/<ymax>` (rounded to integers)
- COCO `iscrowd` → VOC `<difficult>1</difficult>` (approximate mapping)
- Segmentation and keypoints are not exported (bbox-only)

### Pascal VOC → COCO

```python
coco = COCO.from_voc("VOCdevkit/VOC2012/")
coco.save("voc2012_as_coco.json")
print(f"{len(coco.dataset['images'])} images, {len(coco.dataset['annotations'])} annotations")
```

`from_voc` scans for `*.xml` files in `voc_dir/Annotations/` (falls back to the
root directory). Image dimensions are read from each XML's `<size>` element — no
Pillow needed.

If `labels.txt` is present (as written by `to_voc`), it determines category
ordering. Otherwise, categories are sorted alphabetically with IDs starting at 1.

VOC `<difficult>` and `<truncated>` fields are dropped — `difficult` is not
equivalent to COCO `iscrowd` (different concepts).

### Round-trip precision

VOC uses integer pixel coordinates, so COCO→VOC→COCO round-trip error is bounded
at ≤1 pixel per coordinate due to rounding:

```python
coco  = COCO("instances_val2017.json")
coco.to_voc("voc_output/")
coco2 = COCO.from_voc("voc_output/")
coco2.save("reconstructed.json")
```

### COCO → CVAT

```python
from hotcoco import COCO

coco = COCO("instances_val2017.json")
stats = coco.to_cvat("annotations.xml")
print(stats)
# {'images': 5000, 'boxes': 36781, 'polygons': 0, 'skipped_no_geometry': 0}
```

`to_cvat` writes a single CVAT for Images 1.1 XML file. Bounding boxes become
`<box>` elements; polygon segmentations become `<polygon>` elements with
semicolon-separated point pairs.

### CVAT → COCO

```python
coco = COCO.from_cvat("annotations.xml")
coco.save("cvat_as_coco.json")
print(f"{len(coco.dataset['images'])} images, {len(coco.dataset['annotations'])} annotations")
```

`from_cvat` reads a single CVAT XML file. Category ordering comes from the
`<meta><task><labels>` block. Supports `<box>` and `<polygon>` elements;
`<polyline>`, `<points>`, and `<cuboid>` are skipped.

Polygon area is computed via the shoelace formula; bounding boxes are derived
from polygon vertex extents.

### COCO → DOTA {#dota}

```python
coco = COCO("annotations.json")
stats = coco.to_dota("dota_labels/")
print(f"{stats['images']} images, {stats['annotations']} annotations")
```

Exports oriented bounding box annotations to DOTA text format. Each image gets
a `.txt` file with one line per annotation: `x1 y1 x2 y2 x3 y3 x4 y4 category difficulty`.
Annotations without an `obb` field are skipped.

### DOTA → COCO

```python
coco = COCO.from_dota("dota_labels/", image_dir="images/")
coco.save("dota_as_coco.json")
```

Reads DOTA text files and converts 8-point polygon coordinates to the
`[cx, cy, w, h, angle]` OBB representation. Categories are auto-discovered
from the label files. Image dimensions are read from the `image_dir` if provided.

---

## healthcheck

Validate a dataset for common issues before training or evaluation. The healthcheck
runs four layers of checks, each catching progressively subtler problems:

1. **Structural** — duplicate IDs, orphaned annotation references (errors)
2. **Quality** — degenerate bboxes, zero-area annotations, out-of-bounds, extreme aspect ratios, near-duplicates (warnings)
3. **Distribution** — category imbalance, low/zero-instance categories (warnings)
4. **Compatibility** — GT/DT image/category mismatches (requires detections)

```python
from hotcoco import COCO

coco = COCO("annotations.json")
report = coco.healthcheck()

for f in report["errors"]:
    print(f"ERROR [{f['code']}] {f['message']}")
for f in report["warnings"]:
    print(f"WARN  [{f['code']}] {f['message']}")

print(f"Images: {report['summary']['num_images']}")
print(f"Imbalance: {report['summary']['imbalance_ratio']:.1f}x")
```

Pass detections to also run GT/DT compatibility checks:

```python
dt = coco.load_res("detections.json")
report = coco.healthcheck(dt)
```

### From the CLI

```bash
# Dataset only
coco healthcheck annotations.json

# With detections
coco healthcheck annotations.json --dt detections.json

# As a pre-flight check before evaluation
coco eval --gt annotations.json --dt detections.json --healthcheck
```

---

## CLI

All operations are available as `coco` subcommands — no Python required
beyond the initial install. See the [CLI reference](../cli.md) for full flag
documentation.

```bash
coco filter  instances_val2017.json --cat-ids 1 -o person.json
coco split   person.json --val-frac 0.2 -o splits/person
coco sample  person.json --n 500 --seed 0 -o person_sample.json
coco merge   batch1.json batch2.json -o combined.json
coco convert --from coco --to yolo --input instances_val2017.json --output labels/
coco convert --from yolo --to coco --input labels/ --output reconstructed.json
```
