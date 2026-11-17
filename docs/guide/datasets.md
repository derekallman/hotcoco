# Dataset operations

hotcoco can do more than evaluate — it can reshape your datasets before evaluation starts.

All operations return a new `COCO` object and leave the original unchanged. They compose naturally,
so you can chain filter → split → sample in a single expression.

!!! tip
    Run `coco.stats()` first to understand your dataset before reshaping it. See the
    [`stats` API reference](../api/coco.md#stats) for the full return structure.

Files that are not quite valid JSON — bare `NaN`/`Infinity` from Python's `json`
module — still load; see
[Loading quirks worth knowing](../getting-started/coco-format.md#loading-quirks-worth-knowing).

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
across experiments. The guarantee holds per installed hotcoco version (not across
`rand` upgrades or 32-bit targets), so pin a split for a paper by saving the split
files or image IDs, not just the seed.

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

Convert between COCO and other annotation formats:

| Format | Direction | Method |
|--------|-----------|--------|
| YOLO | COCO ↔ YOLO | `to_yolo()` / `from_yolo()` |
| Pascal VOC | COCO ↔ VOC | `to_voc()` / `from_voc()` |
| CVAT | COCO ↔ CVAT | `to_cvat()` / `from_cvat()` |
| DOTA | COCO ↔ DOTA | `to_dota()` / `from_dota()` |
| Open Images | COCO ↔ CSV | `to_oid()` / `from_oid()` |

Each `to_*` method returns a stats dict recording what was written and what was
skipped, by reason (`skipped_crowd`, `skipped_no_bbox`, …) — records the target
format cannot express are counted, never silently dropped, while malformed
*input* raises an error naming the file and position. Full field mappings,
category-ordering rules, return shapes, and the shared converter contract are in
the [`convert` API reference](../api/coco.md#convert).

### YOLO

```python
from hotcoco import COCO

coco = COCO("instances_val2017.json")
coco.to_yolo("labels/val2017/")   # one .txt per image + data.yaml

# Round-trip — images_dir lets Pillow restore real image dimensions
coco2 = COCO.from_yolo("labels/val2017/", images_dir="images/val2017/")
coco2.save("reconstructed.json")
```

Bbox values round-trip to the six decimals YOLO labels are written with: within
0.001 px at typical image sizes. YOLO coordinates are normalized to the image size, so both
directions need real dimensions — `images_dir` is how the import gets them. What
[`from_yolo`](../api/coco.md#from_yolo) accepts in `data.yaml` and how it names
images is in the API reference.

### Pascal VOC

```python
coco.to_voc("voc_output/")        # Annotations/<stem>.xml + labels.txt
coco2 = COCO.from_voc("voc_output/")
```

VOC is bbox-only and writes integer pixel coordinates in the devkit's 1-based
inclusive convention, which hotcoco applies in both directions, so a
COCO→VOC→COCO round-trip is bounded only by the integer rounding on export. The
coordinate formula and the `iscrowd`/`<difficult>` mapping are in
[`to_voc`](../api/coco.md#to_voc) and [`from_voc`](../api/coco.md#from_voc).

### CVAT

```python
coco.to_cvat("annotations.xml")   # CVAT for Images 1.1, single XML file
coco2 = COCO.from_cvat("annotations.xml")
```

Bounding boxes and polygon segmentations convert in both directions; shapes
COCO cannot express are skipped and counted — [`from_cvat`](../api/coco.md#from_cvat)
lists which.

### DOTA {#dota}

DOTA is the aerial-detection benchmark's label format: one text file per image,
each line holding the 8 corner coordinates of a rotated box, then the category
name and a difficulty flag. It is the usual source of data for
[OBB evaluation](evaluation.md).

```python
coco.to_dota("labelTxt/")         # one .txt per image
coco2 = COCO.from_dota("labelTxt/", images_dir="images/")
```

Corner coordinates are written to one decimal place, which bounds a
COCO→DOTA→COCO round-trip at ≤0.1 px per coordinate. An imported dataset
evaluates under either `iou_type`, because [`from_dota`](../api/coco.md#from_dota)
fills in the axis-aligned envelope as well.

Categories are numbered in the order the label files first use them, reading
the files in name order. Pass
`categories=[...]` to fix the numbering instead — two splits of one dataset
otherwise disagree on IDs whenever a class is missing from one of them.

### Open Images {#open-images}

Open Images ships CSV rather than JSON, with coordinates normalized to `[0, 1]`:

```python
gt = COCO.from_oid(
    "challenge-2019-validation-detection-bbox.csv",
    class_descriptions="class-descriptions-boxable.csv",
)
dt = gt.load_res_oid("predictions.csv")

ev = COCOeval(gt, dt, "bbox", oid_style=True)
ev.run()
```

`IsGroupOf` boxes become `is_group_of` annotations, which
[Open Images evaluation](lvis-open-images.md) matches by IoA rather than IoU.

`class_descriptions` is optional. Pass the same file to `load_res_oid` that you
passed to `from_oid`, so detections resolve to the same categories.

[`load_res_oid`](../api/coco.md#load_res_oid) aligns detections onto the ground
truth's image and category IDs, and treats a detection it cannot place as an
error — silently discarding detections moves recall, and nothing downstream
would show it.

!!! note "Image dimensions are optional, with one consequence"

    Open Images CSVs don't record pixel sizes. Without `images_dir`, boxes stay
    in `[0, 1]` against a 1×1 image. IoU and IoA are ratios of areas scaled
    identically on both axes, so Open Images AP is unaffected — but absolute
    areas, and therefore the small/medium/large ranges, are meaningless in that
    mode. Pass `images_dir` if you need them.

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
coco healthcheck annotations.json --dt detections.json
coco eval --gt annotations.json --dt detections.json --healthcheck   # pre-flight check
```

`coco healthcheck` has an exit status a CI step can gate on — see
[`coco healthcheck`](../cli.md#coco-healthcheck).
