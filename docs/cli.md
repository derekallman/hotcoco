# CLI

hotcoco ships two CLI tools:

- **`coco`** — Python CLI. Installed with `pip install hotcoco`. Covers dataset management (filter, merge, split, sample, stats) and is the primary tool for most workflows.
- **`coco-eval`** — Rust CLI. Installed with `cargo install hotcoco-cli`. Evaluation only, no Python required.

---

## `coco` — Python CLI

```bash
pip install hotcoco
```

### JSON output mode

Every subcommand accepts a `--json` flag that writes a single JSON object to stdout
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
| `--report <path>` | Save a PDF evaluation report to this path (requires `hotcoco[plot]`) | off |
| `--title` | Report title shown in the header | `COCO Evaluation Report` |
| `--slices <path>` | JSON file with named image ID groups for sliced evaluation | off |
| `--healthcheck` | Run dataset healthcheck before evaluation (warnings to stderr) | off |
| `--json` | Write results as JSON to stdout instead of human-readable text | off |

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
  "hotcoco_version": "0.3.0",
  "params": { "iou_type": "Bbox", "iou_thresholds": [...], "area_ranges": {...}, ... },
  "metrics": { "AP": 0.578, "AP50": 0.861, "AP75": 0.600, "APs": 0.327, ... },
  "tide": { "delta_ap": {...}, "counts": {...}, "ap_base": 0.578, ... },
  "slices": { "daytime": { "AP": 0.61, ... }, "_overall": { ... } },
  "healthcheck": { "errors": [], "warnings": [] }
}
```

`tide`, `slices`, and `healthcheck` keys are only present when the corresponding
flag is passed.

### `coco healthcheck`

Validate a dataset for structural errors, quality warnings, and distribution issues.

```bash
coco healthcheck <annotation_file> [--dt <detections.json>]
```

| Flag | Description |
|------|-------------|
| `--dt <path>` | Detection results JSON — enables GT/DT compatibility checks |
| `--json` | Write results as JSON to stdout |

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
| `--json` | Write before/after counts as JSON to stdout |

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
| `--test-frac` | Fraction for a test set (omit for two-way split) | — |
| `--seed` | Random seed for reproducibility | `42` |
| `-o / --output` | Output prefix | *(required)* |
| `--json` | Write per-split counts as JSON to stdout | off |

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
| `--json` | Write before/after counts as JSON to stdout | |

```bash
# Sample 500 images
coco sample instances_val2017.json --n 500 --seed 0 -o sample.json

# Sample 10% of the dataset
coco sample instances_val2017.json --frac 0.1 -o sample.json
```

### `coco convert`

Convert between annotation formats. Currently supports COCO JSON ↔ YOLO labels.

**COCO → YOLO:**

```bash
coco convert --from coco --to yolo --input <annotations.json> --output <labels_dir/>
```

**YOLO → COCO:**

```bash
coco convert --from yolo --to coco --input <labels_dir/> --output <annotations.json> [--images-dir <images/>]
```

| Flag | Description |
|------|-------------|
| `--from` | Source format: `coco` or `yolo` |
| `--to` | Target format: `coco` or `yolo` |
| `--input` | Input path — JSON file (COCO) or label directory (YOLO) |
| `--output` | Output path — label directory (YOLO) or JSON file (COCO) |
| `--images-dir` | *(YOLO → COCO only)* Directory of source images; used by Pillow to populate `width`/`height` on each image record. Requires `pip install Pillow`. |
| `--json` | Write conversion stats as JSON to stdout | |

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

The standard 12 COCO metrics (10 for keypoints):

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.783
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.971
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.849
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.621
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.893
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.988
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.835
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.854
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.701
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.935
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.997
```

---

## Shell completions

Both CLIs support tab completion for flags, subcommands, and values.

### `coco` (Python)

Install `argcomplete`:

```bash
pip install "hotcoco[completions]"
```

Then register the completion for your shell. The one-time setup depends on your shell:

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

After restarting your shell (or sourcing the config), `coco <TAB>` completes subcommands and `coco eval --<TAB>` completes flags.

### `coco-eval` (Rust)

`coco-eval --completions <SHELL>` prints a completion script to stdout. Pipe it to the right location for your shell:

=== "bash"

    ```bash
    coco-eval --completions bash > ~/.bash_completion.d/coco-eval
    # or for system-wide:
    coco-eval --completions bash | sudo tee /etc/bash_completion.d/coco-eval
    ```

    Then add to `~/.bashrc` if not already sourcing `~/.bash_completion.d/`:

    ```bash
    source ~/.bash_completion.d/coco-eval
    ```

=== "zsh"

    ```zsh
    mkdir -p ~/.zsh/completions
    coco-eval --completions zsh > ~/.zsh/completions/_coco-eval
    ```

    Make sure `~/.zsh/completions` is on your `fpath` in `~/.zshrc`:

    ```zsh
    fpath=(~/.zsh/completions $fpath)
    autoload -U compinit && compinit
    ```

=== "fish"

    ```fish
    coco-eval --completions fish > ~/.config/fish/completions/coco-eval.fish
    ```

Supported shells: `bash`, `zsh`, `fish`, `elvish`, `powershell`.
