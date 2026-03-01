# CLI

hotcoco ships two CLI tools:

- **`coco`** — Python CLI. Installed with `pip install hotcoco`. Covers dataset management (filter, merge, split, sample, stats) and is the primary tool for most workflows.
- **`coco-eval`** — Rust CLI. Installed with `cargo install hotcoco-cli`. Evaluation only, no Python required.

---

## `coco` — Python CLI

```bash
pip install hotcoco
```

### `coco eval`

Evaluate detections against ground truth annotations. Prints the standard COCO metrics table.

```bash
coco eval --gt <gt.json> --dt <dt.json> [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--gt <path>` | Ground truth annotations JSON | *required* |
| `--dt <path>` | Detection results JSON | *required* |
| `--iou-type` | `bbox`, `segm`, or `keypoints` | `bbox` |
| `--img-ids 1,2,3` | Evaluate only these image IDs | all |
| `--cat-ids 1,2,3` | Evaluate only these category IDs | all |
| `--no-cats` | Pool all categories (class-agnostic evaluation) | off |
| `--tide` | Print TIDE error decomposition after standard metrics | off |
| `--tide-pos-thr` | IoU threshold for TP/FP classification in TIDE | `0.5` |
| `--tide-bg-thr` | Minimum IoU with any GT for Loc/Both/Bkg distinction | `0.1` |

```bash
# Bounding box evaluation
coco eval --gt instances_val2017.json --dt bbox_results.json

# Segmentation
coco eval --gt instances_val2017.json --dt segm_results.json --iou-type segm

# Keypoints
coco eval --gt person_keypoints_val2017.json --dt kpt_results.json --iou-type keypoints

# With TIDE error decomposition
coco eval --gt instances_val2017.json --dt bbox_results.json --tide

# TIDE at a stricter localization threshold
coco eval --gt instances_val2017.json --dt bbox_results.json --tide --tide-pos-thr 0.75
```

### `coco stats`

Print a health-check summary of a dataset: image and annotation counts, per-category
breakdown, image dimensions, and annotation area distribution.

```bash
coco stats instances_val2017.json
coco stats instances_val2017.json --all-cats  # show all categories, not just top 20
```

### `coco filter`

Subset a dataset by category, image ID, or annotation area.

```bash
coco filter <file> -o <output> [options]
```

| Flag | Description |
|------|-------------|
| `--cat-ids 1,2,3` | Keep only these category IDs |
| `--img-ids 1,2,3` | Keep only these image IDs |
| `--area-rng MIN,MAX` | Keep annotations within this area range (inclusive) |
| `--keep-empty-images` | Preserve images with no matching annotations |
| `-o / --output` | Output JSON path *(required)* |

```bash
# Keep only "person" (category 1)
coco filter instances_val2017.json --cat-ids 1 -o person.json

# Medium-sized objects only
coco filter instances_val2017.json --area-rng 1024,9216 -o medium.json
```

### `coco split`

Split a dataset into train/val (or train/val/test) subsets. Writes separate JSON
files for each split.

```bash
coco split <file> -o <prefix> [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--val-frac` | Fraction of images for validation | `0.2` |
| `--test-frac` | Fraction for a test set (omit for two-way split) | — |
| `--seed` | Random seed for reproducibility | `42` |
| `-o / --output` | Output prefix | *(required)* |

Writes `<prefix>_train.json`, `<prefix>_val.json`, and optionally `<prefix>_test.json`.

```bash
# 80/20 split
coco split person.json -o splits/person --val-frac 0.2

# 70/15/15 split
coco split person.json -o splits/person --val-frac 0.15 --test-frac 0.15
```

### `coco merge`

Combine multiple annotation files into one. All files must share the same category
taxonomy.

```bash
coco merge <file1> <file2> [<file3> ...] -o <output>
```

```bash
coco merge batch1.json batch2.json batch3.json -o combined.json
```

### `coco sample`

Draw a random subset of images (with their annotations).

```bash
coco sample <file> -o <output> [options]
```

| Flag | Description |
|------|-------------|
| `--n N` | Number of images to sample |
| `--frac F` | Fraction of images to sample |
| `--seed` | Random seed (default `42`) |
| `-o / --output` | Output JSON path *(required)* |

```bash
# Sample 500 images
coco sample instances_val2017.json --n 500 --seed 0 -o sample.json

# Sample 10% of the dataset
coco sample instances_val2017.json --frac 0.1 -o sample.json
```

---

## `coco-eval` — Rust CLI

Evaluation only. No Python required — useful in environments where installing
a Python package isn't practical.

```bash
cargo install hotcoco-cli
```

### Usage

```bash
coco-eval --gt annotations.json --dt detections.json --iou-type bbox
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--gt <path>` | Path to ground truth annotations JSON file | *required* |
| `--dt <path>` | Path to detection results JSON file | *required* |
| `--iou-type <type>` | Evaluation type: `bbox`, `segm`, or `keypoints` | `bbox` |
| `--img-ids <ids>` | Filter to specific image IDs (comma-separated) | all images |
| `--cat-ids <ids>` | Filter to specific category IDs (comma-separated) | all categories |
| `--no-cats` | Pool all categories (disable per-category evaluation) | off |

### Examples

```bash
# Bounding box evaluation
coco-eval --gt instances_val2017.json --dt bbox_results.json --iou-type bbox

# Segmentation evaluation
coco-eval --gt instances_val2017.json --dt segm_results.json --iou-type segm

# Keypoint evaluation
coco-eval --gt person_keypoints_val2017.json --dt kpt_results.json --iou-type keypoints

# Filter to specific categories
coco-eval --gt instances_val2017.json --dt results.json --cat-ids 1,3

# Category-agnostic evaluation
coco-eval --gt instances_val2017.json --dt results.json --no-cats
```

### Output

The standard 12 COCO metrics (10 for keypoints):

```
 Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.783
 Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.971
 Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.849
 Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.621
 Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.893
 Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.988
 Average Recall (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.502
 Average Recall (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.835
 Average Recall (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.854
 Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.701
 Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.935
 Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.997
```
