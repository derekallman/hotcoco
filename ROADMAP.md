# Roadmap

Planned features and improvements, organized by priority.

## Architecture

All core logic lives in the Rust library. The Python package and Rust CLI are thin wrappers.

```
                       ┌──→ PyO3 ──→ hotcoco (Python library + CLI)
Rust Core (all logic) ─┤
                       └──→ hotcoco-cli (Rust CLI, eval only)
```

- **Rust core** — types, masks, eval, dataset ops, format conversion, streaming
- **Python CLI** (primary) — all subcommands: `$ coco eval`, `$ coco stats`, `$ coco merge`, `$ coco plot`, etc. Rich formatting, plots via matplotlib/plotly.
- **Rust CLI** (`hotcoco-cli`) — evaluation only. JSON/CSV/markdown output, no plots, no Python. New features added on request.

| Registry | Package | Contents |
|----------|---------|----------|
| PyPI | `hotcoco` | Python library + Python CLI + compiled Rust core |
| crates.io | `hotcoco` | Rust library |
| crates.io | `hotcoco-cli` | Rust CLI binary (eval only, no Python) |

---

## Shipped

### Python CLI Evaluation

`coco eval` — full COCO evaluation from the Python CLI, matching all flags of the Rust `coco-eval` binary: `--gt`, `--dt`, `--iou-type`, `--img-ids`, `--cat-ids`, `--no-cats`.

### Dataset Statistics

Quick health check for any COCO dataset — annotation counts per category, image size distributions, area distributions, crowd/iscrowd breakdown. Available as `coco.stats()` in Python and `coco stats` in the Python CLI.

### Dataset Operations

Split, merge, filter, and sample COCO datasets:

- ~~**Filter** — subset by category, image ID, area range, or custom predicate~~
- ~~**Merge** — combine multiple annotation files (e.g., separate labeling batches)~~
- ~~**Split** — reproducible train/val/test split with deterministic shuffle~~
- ~~**Sample** — random or deterministic subset for quick iteration~~

All implemented in Rust core, exposed via Python CLI and Python API.

### CI/CD

- ~~`cargo test` / `cargo clippy` / `cargo fmt --check`~~
- ~~Cross-platform matrix (Linux/macOS/Windows)~~
- ~~Python smoke test (`maturin develop` + inline import/eval check)~~
- ~~Automated release publishing to crates.io and PyPI~~

---

## Tier 1 — Next

### Objects365 Verification

Standard COCO evaluation protocol over 365 categories and ~2M images. Likely works today since the format is standard COCO — needs verification on a real O365 annotation file and explicit documentation. The main story is scale: at O365 size, hotcoco's speed advantage is most dramatic.

### LVIS

**Shipped.**

~~1,200 category long-tail dataset requiring **federated AP** evaluation — per-category results are computed independently across the subset of images that contain each category, rather than globally. Using standard pycocotools on LVIS gives subtly wrong numbers. Increasingly common in foundation model benchmarking (SAM, DINO, CLIP-based detectors). faster-coco-eval already supports this and is winning users there.~~

### Confusion Matrices

**Shipped.**

~~Per-category confusion matrix generation to identify systematic misclassifications. Self-contained, high value for practitioners debugging model failures, and the natural starting point for the visualization story.~~

---

## Tier 2 — Medium Term

### Format Conversion

COCO ↔ YOLO first (most requested), then COCO ↔ Pascal VOC and COCO ↔ CVAT. Everyone has a slightly broken converter script — a correct, well-tested one has real value. Scope tightly to YOLO before expanding.

### TIDE Error Analysis

**Shipped.**

~~Error decomposition following [TIDE](https://github.com/dbolya/tide) — classification, localization, duplicate, background, and missed errors. Builds naturally on confusion matrices and is a meaningful differentiator from faster-coco-eval.~~

### Streaming Evaluation

Evaluate datasets that don't fit in memory. Process annotations in chunks without loading the full ground truth and detection sets upfront. Needed at O365/LVIS scale in production but not blocking anyone today — slot in once real users hit memory limits.

---

## Tier 3 — Later

### CrowdPose

Keypoint dataset for crowded scenes. Uses a modified OKS matching algorithm with a crowd factor. Niche audience and significant custom eval logic — not worth prioritising before LVIS.

### Hierarchical Evaluation

Open Images-style evaluation with category hierarchies, where a detection of a parent category is not penalized against a child. Small, specialised audience.

### Video Sequence Analysis

Lightweight per-sequence metric breakdowns for video object detection, surfacing high-level trends like which clips perform worst. Track AP (used by TAO, BURST, YouTube-VIS) is a natural extension of the existing COCO AP pipeline and the most likely entry point here.

Longer term, full multi-object tracking metrics — MOTA, HOTA, IDF1 — are worth exploring as a Phase 2 effort. TrackEval (the de-facto standard) is effectively unmaintained and slow; there's a real opening for a fast Rust alternative. This would be a meaningful scope expansion and is not planned for the near term, but the direction is intentional.
