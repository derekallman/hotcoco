# CLI

hotcoco ships two CLI tools:

- **`coco`** — Python CLI. Installed with `pip install hotcoco`. Covers dataset management (filter, merge, split, sample, stats) and is the primary tool for most workflows.
- **`coco-eval`** — Rust CLI. Installed with `cargo install hotcoco-cli`.

---

## `coco` — Python CLI

```bash
pip install hotcoco
```

`coco --version` prints the installed version. Every subcommand takes `--help`.

### JSON output mode

Every subcommand except `coco explore` accepts a `--json` flag that writes a single JSON object to stdout
instead of human-readable text. stderr (progress, warnings, errors) is untouched.

```bash
coco eval --gt ann.json --dt det.json --json
coco stats ann.json --json
coco healthcheck ann.json --json
```

This is designed for CI/CD pipelines, dashboards, and shell scripts that need to
gate on metric values without parsing human output:

```bash
# Gate a CI step on AP ≥ 0.50
AP=$(coco eval --gt ann.json --dt det.json --json | jq '.metrics.AP')
python -c "import sys; sys.exit(0 if $AP >= 0.50 else 1)"
```

`coco eval --json` also carries `provenance` and `reference_deviations`, so a pipeline
can refuse to publish numbers that are not leaderboard-comparable (see
[provenance](guide/results.md#check-provenance-before-you-publish-a-number)). The
warnings `summarize()` prints go to stderr and do not survive a pipe — these do:

```bash
# Fail if the run is not comparable to the reference implementation
coco eval --gt ann.json --dt det.json --json \
  | jq -e '.provenance == "parity_verified"' > /dev/null \
  || { echo "not benchmark-standard"; exit 1; }
```

When `--json` is set and an error occurs, the exit code is still 1 and the error
is also JSON:

```json
{"error": "No such file or directory (os error 2)"}
```

---

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
| `--lvis` | LVIS-style evaluation (max 300 dets, frequency-group AP) | off |
| `--img-ids 1,2,3` | Evaluate only these image IDs | all |
| `--cat-ids 1,2,3` | Evaluate only these category IDs | all |
| `--no-cats` | Pool all categories (class-agnostic evaluation) | off |
| `--tide` | Print TIDE error decomposition after standard metrics | off |
| `--tide-pos-thr` | IoU threshold for TP/FP classification in TIDE | `0.5` |
| `--tide-bg-thr` | Minimum IoU with any GT for Loc/Both/Bkg distinction | `0.1` |
| `--diagnostics` | Per-image diagnostics: worst images by F1 and label error candidates | off |
| `--diag-iou-thr` | IoU threshold for diagnostics TP/FP classification | `0.5` |
| `--diag-score-thr` | Min detection score for label error candidates | `0.5` |
| `--report <path>` | Save a PDF evaluation report to this path (requires `hotcoco[plot]`) | off |
| `--title` | Report title shown in the header | derived from eval mode |
| `--slices <path>` | JSON file with named image ID groups for sliced evaluation | off |
| `--healthcheck` | Run dataset healthcheck before evaluation (warnings to stderr) | off |
| `--calibration` | Compute confidence calibration (ECE/MCE) after standard metrics | off |
| `--cal-bins` | Number of calibration bins | `10` |
| `--cal-iou-thr` | IoU threshold for calibration TP/FP classification | `0.5` |
| `--json` | Results as JSON | off |

```bash
# Bounding box evaluation
coco eval --gt instances_val2017.json --dt bbox_results.json

# Segmentation
coco eval --gt instances_val2017.json --dt segm_results.json --iou-type segm

# Keypoints
coco eval --gt person_keypoints_val2017.json --dt kpt_results.json --iou-type keypoints

# LVIS-style evaluation
coco eval --gt lvis_val.json --dt lvis_results.json --lvis

# With TIDE error decomposition
coco eval --gt instances_val2017.json --dt bbox_results.json --tide

# TIDE at a stricter localization threshold
coco eval --gt instances_val2017.json --dt bbox_results.json --tide --tide-pos-thr 0.75

# Save a PDF evaluation report
coco eval --gt instances_val2017.json --dt bbox_results.json --report report.pdf

# PDF report with custom title and LVIS-style evaluation
coco eval --gt lvis_val.json --dt lvis_results.json --lvis --report lvis_report.pdf --title "LVIS Evaluation"

# Sliced evaluation (compare metrics across image subsets)
coco eval --gt instances_val2017.json --dt bbox_results.json --slices slices.json

# Pre-flight healthcheck before evaluation
coco eval --gt instances_val2017.json --dt bbox_results.json --healthcheck

# JSON output for CI/CD pipelines
coco eval --gt instances_val2017.json --dt bbox_results.json --json

# JSON with TIDE and slices combined
coco eval --gt instances_val2017.json --dt bbox_results.json --tide --slices slices.json --json
```

**JSON output shape:**

```json
{
  "hotcoco_version": "1.0.0",
  "provenance": "parity_verified",
  "params": { "iou_type": "bbox", "iou_thresholds": [...], "area_ranges": {...}, ... },
  "metrics": { "AP": 0.578, "AP50": 0.861, "AP75": 0.600, "APs": 0.327, ... },
  "reference_deviations": [],
  "tide": { "delta_ap": {...}, "counts": {...}, "ap_base": 0.578, ... },
  "slices": { "daytime": { "AP": 0.61, ... }, "_overall": { ... } },
  "healthcheck": { "errors": [], "warnings": [] }
}
```

`tide`, `slices`, and `healthcheck` keys are only present when the corresponding
flag is passed.

!!! note "OBB is Python-only"
    `--iou-type` covers `bbox`, `segm`, and `keypoints`. Oriented bounding box
    evaluation is available through the Python API — see
    [OBB evaluation](guide/evaluation.md#oriented-bounding-box-obb-evaluation).

### `coco healthcheck`

Validate a dataset for structural errors, quality warnings, and distribution issues.

```bash
coco healthcheck <annotation_file> [--dt <detections.json>]
```

| Flag | Description |
|------|-------------|
| `--dt <path>` | Detection results JSON — enables GT/DT compatibility checks |
| `--json` | Results as JSON |

**Exit status:** exits `1` when any ERROR-level finding is present (in both human
and `--json` modes), so it can gate a CI step. Warnings alone exit `0`.

```bash
# Dataset only
coco healthcheck instances_val2017.json

# With detections (also checks GT/DT compatibility)
coco healthcheck instances_val2017.json --dt bbox_results.json

# JSON output (full errors/warnings list + summary)
coco healthcheck instances_val2017.json --json
```

### `coco stats`

Print a health-check summary of a dataset: image and annotation counts, per-category
breakdown, image dimensions, and annotation area distribution.

```bash
coco stats instances_val2017.json
coco stats instances_val2017.json --all-cats  # show all categories, not just top 20
coco stats instances_val2017.json --json       # machine-readable output
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
| `--json` | Before/after counts as JSON |

```bash
# Keep only "person" (category 1)
coco filter instances_val2017.json --cat-ids 1 -o person.json

# Medium-sized objects only
coco filter instances_val2017.json --area-rng 1024,9216 -o medium.json

# JSON output: {"before": {"images": 5000, ...}, "after": {...}, "output": "..."}
coco filter instances_val2017.json --cat-ids 1 -o person.json --json
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
| `--test-frac` | Fraction for a test set (omit for two-way split; `0.0` gives a three-way split with an empty test file) | — |
| `--seed` | Random seed for reproducibility | `42` |
| `-o / --output` | Output prefix | *(required)* |
| `--json` | Per-split counts as JSON | off |

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

# JSON output: input list with per-file counts + output counts
coco merge batch1.json batch2.json -o combined.json --json
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
| `--json` | Before/after counts as JSON |

```bash
# Sample 500 images
coco sample instances_val2017.json --n 500 --seed 0 -o sample.json

# Sample 10% of the dataset
coco sample instances_val2017.json --frac 0.1 -o sample.json
```

### `coco explore`

Launch a local dataset browser to explore a dataset interactively. Requires
`pip install hotcoco[browse]`.

```bash
coco explore --gt <annotations.json> --images <images_dir/> [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--gt <path>` | Ground truth annotation JSON | *required* |
| `--images <dir>` | Directory containing image files | *required* |
| `--dt <path>` | Detection results JSON (enables detection overlay) | off |
| `--iou-type` | Evaluation type for TP/FP/FN coloring: `bbox`, `segm`, or `keypoints` | `bbox` |
| `--iou-thr THR` | Initial IoU threshold for TP/FP classification; sets the UI slider's starting position (snapped to 0.50–0.95 in steps of 0.05) | `0.5` |
| `--no-eval` | Disable automatic evaluation — show detections without TP/FP/FN coloring | off |
| `--slices <path>` | JSON file mapping slice names to image ID lists, for example `{"daytime": [1, 2, 3]}` | off |
| `--batch-size N` | Images loaded per batch | `12` |
| `--port N` | Local server port | `7860` |

```bash
coco explore --gt instances_val2017.json --images /data/coco/val2017/

# With detection overlay
coco explore --gt instances_val2017.json --images /data/images/ --dt results.json

# Detection overlay without TP/FP/FN coloring
coco explore --gt instances_val2017.json --images /data/images/ --dt results.json --no-eval

# Segmentation-based eval coloring at a stricter threshold
coco explore --gt gt.json --images imgs/ --dt results.json --iou-type segm --iou-thr 0.75

# Custom port
coco explore --gt instances_val2017.json --images /data/images/ --port 7861
```

The [Dataset browser guide](guide/browse.md) describes the UI.

---

### `coco compare`

Compare two model evaluations on the same dataset with per-metric deltas, per-category breakdown, and optional bootstrap confidence intervals.

```bash
coco compare --gt <annotations.json> --dt-a <model_a.json> --dt-b <model_b.json> [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--gt` | Ground truth annotations (COCO JSON) | *required* |
| `--dt-a` | Detections from model A | *required* |
| `--dt-b` | Detections from model B | *required* |
| `--iou-type` | `bbox`, `segm`, or `keypoints` | `bbox` |
| `--lvis` | LVIS-style federated evaluation | off |
| `--bootstrap N` | Bootstrap samples for confidence intervals | `0` (disabled) |
| `--seed` | Random seed for bootstrap | `42` |
| `--confidence` | Confidence level for CIs | `0.95` |
| `--name-a` | Display name for model A | `Model A` |
| `--name-b` | Display name for model B | `Model B` |
| `--json` | Comparison as JSON | off |

```bash
# Basic comparison
coco compare --gt ann.json --dt-a baseline.json --dt-b improved.json

# With bootstrap CIs
coco compare --gt ann.json --dt-a a.json --dt-b b.json --bootstrap 1000

# JSON output for CI/CD
coco compare --gt ann.json --dt-a a.json --dt-b b.json --bootstrap 1000 --json
```

---

### `coco panoptic eval`

Panoptic quality — PQ, SQ, RQ for All, Things, and Stuff — from two COCO panoptic
JSON files and their PNG folders, as panopticapi's `pq_compute` computes it. See the
[panoptic guide](guide/panoptic.md).

```bash
coco panoptic eval --gt <gt.json> --pred <pred.json> [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--gt` | Ground truth panoptic JSON | *required* |
| `--pred` | Predicted panoptic JSON | *required* |
| `--gt-folder DIR` | Ground truth PNG folder | `--gt` without `.json` |
| `--pred-folder DIR` | Prediction PNG folder | `--pred` without `.json` |
| `--json` | `results()` plus the `report` as JSON | off |

```bash
coco panoptic eval --gt panoptic_val2017.json --pred predictions.json
coco panoptic eval --gt gt.json --pred pred.json --gt-folder gt_png/ --pred-folder pred_png/ --json
```

---

### `coco convert`

Convert between annotation formats. Supports COCO JSON ↔ YOLO labels, Pascal VOC XML, CVAT for Images XML, DOTA oriented-box labels, and Open Images CSV.

**COCO → YOLO:**

```bash
coco convert --from coco --to yolo --input <annotations.json> --output <labels_dir/>
```

**YOLO → COCO:**

```bash
coco convert --from yolo --to coco --input <labels_dir/> --output <annotations.json> [--images-dir <images/>]
```

**COCO → Pascal VOC:**

```bash
coco convert --from coco --to voc --input <annotations.json> --output <voc_dir/>
```

**Pascal VOC → COCO:**

```bash
coco convert --from voc --to coco --input <voc_dir/> --output <annotations.json>
```

**COCO → CVAT:**

```bash
coco convert --from coco --to cvat --input <annotations.json> --output <annotations.xml>
```

**CVAT → COCO:**

```bash
coco convert --from cvat --to coco --input <annotations.xml> --output <annotations.json>
```

**COCO → DOTA:**

```bash
coco convert --from coco --to dota --input <annotations.json> --output <labelTxt_dir/>
```

**DOTA → COCO:**

```bash
coco convert --from dota --to coco --input <labelTxt_dir/> --output <annotations.json> [--images-dir <images/>]
```

**COCO → Open Images:**

```bash
coco convert --from coco --to oid --input <annotations.json> --output <boxes.csv>
```

**Open Images → COCO:**

```bash
coco convert --from oid --to coco --input <boxes.csv> --output <annotations.json> [--class-descriptions <descriptions.csv>]
```

| Flag | Description |
|------|-------------|
| `--from` | Source format: `coco`, `yolo`, `voc`, `cvat`, `dota`, or `oid` |
| `--to` | Target format: `coco`, `yolo`, `voc`, `cvat`, `dota`, or `oid` |
| `--input` | Input path — JSON file (COCO), CSV file (Open Images), XML file (CVAT), or label directory (YOLO, VOC, DOTA) |
| `--output` | Output path — JSON file (COCO), CSV file (Open Images), XML file (CVAT), or label directory (YOLO, VOC, DOTA) |
| `--images-dir` | *(YOLO, DOTA, Open Images → COCO)* Directory of source images, read by Pillow for `width`/`height` — which formats need it and why is in [Format conversion](api/coco.md#convert). Requires `pip install Pillow`. |
| `--class-descriptions` | *(Open Images → COCO only)* Path to `class-descriptions-boxable.csv`; see [`from_oid`](api/coco.md#from_oid). |
| `--json` | Conversion stats as JSON |

```bash
# Export val2017 to YOLO labels
coco convert --from coco --to yolo \
    --input instances_val2017.json \
    --output labels/val2017/

# Import YOLO labels back (with image dims)
coco convert --from yolo --to coco \
    --input labels/val2017/ \
    --output reconstructed.json \
    --images-dir images/val2017/

# Export to Pascal VOC
coco convert --from coco --to voc \
    --input instances_val2017.json \
    --output voc_output/

# Import Pascal VOC
coco convert --from voc --to coco \
    --input VOCdevkit/VOC2012/ \
    --output voc2012_as_coco.json

# Export to CVAT
coco convert --from coco --to cvat \
    --input instances_val2017.json \
    --output annotations.xml

# Import CVAT
coco convert --from cvat --to coco \
    --input annotations.xml \
    --output cvat_as_coco.json

# Import DOTA oriented boxes
coco convert --from dota --to coco \
    --input labelTxt/ \
    --output dota_as_coco.json \
    --images-dir images/

# Import Open Images, resolving MIDs to readable category names
coco convert --from oid --to coco \
    --input challenge-2019-validation-detection-bbox.csv \
    --output oid_val_as_coco.json \
    --class-descriptions class-descriptions-boxable.csv
```

Detections are a Python-side step: the CLI converts annotation files, but
pairing an Open Images predictions CSV with its ground truth needs
[`load_res_oid`](api/coco.md#load_res_oid).

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

Evaluation is the default action, so the flags can be passed bare, as in the preceding example. The
same run can be written explicitly with the `eval` subcommand — useful in scripts
where the intent should be obvious:

```bash
coco-eval eval --gt annotations.json --dt detections.json --iou-type bbox
```

### Options

These apply to `eval` and to the bare form. The other subcommands are `panoptic`,
below, and `completions`, covered under [Shell completions](#shell-completions).

| Flag | Description | Default |
|------|-------------|---------|
| `--gt <path>` | Path to ground truth annotations JSON file | *required* |
| `--dt <path>` | Path to detection results JSON file | *required* |
| `--iou-type <type>` | Evaluation type: `bbox`, `segm`, or `keypoints` | `bbox` |
| `--img-ids <ids>` | Filter to specific image IDs (comma-separated) | all images |
| `--cat-ids <ids>` | Filter to specific category IDs (comma-separated) | all categories |
| `--no-cats` | Pool all categories (disable per-category evaluation) | off |
| `-o / --output <path>` | Write evaluation results to a JSON file | off |

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

# Save results as JSON (includes per-category AP)
coco-eval --gt instances_val2017.json --dt bbox_results.json --output results.json
```

### Output

The standard 12 COCO metrics (10 for keypoints), in pycocotools' table layout:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.783
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.971
 ...
```

A full run is shown in the [quick start](getting-started/quickstart.md#4-run-evaluation).

### `coco-eval panoptic`

The panoptic family, with the same flags as [`coco panoptic eval`](#coco-panoptic-eval):

```bash
coco-eval panoptic --gt panoptic_val2017.json --pred predictions.json
coco-eval panoptic --gt gt.json --pred pred.json --gt-folder gt_png/ --pred-folder pred_png/ -o report.json
```

Prints the panopticapi table and a `stats:` line with the nine headline values;
`-o` writes the results — the same content as `results()` in Python — as JSON.

---

## Shell completions

Both CLIs support tab completion for flags, subcommands, and values.

??? note "Setting up completions"

    **`coco` (Python)** — install `argcomplete`, then register it for your shell:

    ```bash
    pip install "hotcoco[completions]"
    ```

    === "bash"

        Add to `~/.bashrc`:

        ```bash
        eval "$(register-python-argcomplete coco)"
        ```

    === "zsh"

        Add to `~/.zshrc`:

        ```zsh
        autoload -U bashcompinit && bashcompinit
        eval "$(register-python-argcomplete coco)"
        ```

    === "fish"

        ```fish
        register-python-argcomplete --shell fish coco | source
        ```

    **`coco-eval` (Rust)** — `coco-eval completions <SHELL>` prints a script to
    stdout; write it where your shell looks for completions:

    === "bash"

        ```bash
        coco-eval completions bash > ~/.bash_completion.d/coco-eval
        source ~/.bash_completion.d/coco-eval
        ```

    === "zsh"

        ```zsh
        mkdir -p ~/.zsh/completions
        coco-eval completions zsh > ~/.zsh/completions/_coco-eval
        # ~/.zshrc must have ~/.zsh/completions on fpath, then `autoload -U compinit && compinit`
        ```

    === "fish"

        ```fish
        coco-eval completions fish > ~/.config/fish/completions/coco-eval.fish
        ```

    Supported shells: `bash`, `zsh`, `fish`, `elvish`, `powershell`. Restart your
    shell afterwards.
