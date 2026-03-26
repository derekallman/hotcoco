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

### Plotting & PDF Report

**Shipped.**

~~Publication-quality matplotlib plots (`pr_curve`, `confusion_matrix`, `top_confusions`, `per_category_ap`, `tide_errors`) and a single-page PDF evaluation report (`report()`). Mode-aware metrics table adapts rows for bbox/segm, keypoints, and LVIS. Available via Python API and `coco report` CLI subcommand.~~

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

### Objects365

**Shipped.**

~~Standard COCO evaluation protocol over 365 categories and ~2M images.~~ Verified working on real O365 annotation data. Benchmark numbers published: 39× vs pycocotools, 14× vs faster-coco-eval on 80k images / 365 categories / 1.2M detections, using 8 GB committed vs 24–30 GB for alternatives.

### LVIS

**Shipped.**

~~1,200 category long-tail dataset requiring **federated AP** evaluation — per-category results are computed independently across the subset of images that contain each category, rather than globally. Using standard pycocotools on LVIS gives subtly wrong numbers. Increasingly common in foundation model benchmarking (SAM, DINO, CLIP-based detectors). faster-coco-eval already supports this and is winning users there.~~

### Confusion Matrices

**Shipped.**

~~Per-category confusion matrix generation to identify systematic misclassifications. Self-contained, high value for practitioners debugging model failures, and the natural starting point for the visualization story.~~

### Confidence Calibration

**Shipped.**

~~Expected Calibration Error (ECE), Maximum Calibration Error (MCE), per-bin accuracy vs confidence breakdown, and per-category ECE. `COCOeval.calibration()` in Python, `calibration()` in Rust, reliability diagram plot, `coco eval --calibration` CLI flag.~~

### Model Comparison & A/B Testing

**Shipped.**

~~Compare two or more models on the same dataset — per-category AP deltas, per-image score comparison, and statistical significance via bootstrap confidence intervals.~~

- ~~`coco.compare(eval_a, eval_b)` — side-by-side metrics table with deltas and significance~~
- ~~Per-category AP delta sorted by largest change; worst regressions and best improvements~~
- In browse: toggle Model A / Model B predictions on the same image (future)
- ~~Bootstrap resampling for confidence intervals on AP ("Model B is better by 0.3 AP, 95% CI [0.1, 0.5]")~~

### Per-Image Diagnostics & Failure Mining

**Shipped.**

~~Find the hardest images, understand failure modes, sort by difficulty.~~

- ~~Per-image AP / F1 — sortable in gallery ("worst images first")~~
- ~~Error profiles: perfect / FP-heavy / FN-heavy / mixed~~
- Failure clustering — group images by dominant TIDE error type (future)
- Quick filters: "Images with only FP", "Images with only FN" (browse UI — future)
- Click-through from confusion matrix cell to browse filtered to those misclassifications (browse UI — future)

### Label Error Detection

**Shipped.**

~~Automatically surface likely annotation mistakes using model predictions as a proxy.~~

- ~~"Wrong label" — high-confidence predictions that disagree with GT labels~~
- ~~Missing annotation detector — FPs that might be GT errors~~
- Integration with healthcheck: flag likely label errors alongside structural issues (future)

### Interactive Eval Dashboard

Unified page showing all evaluation results with click-through to browse.

- `/dashboard` route in browse server with embedded Plotly charts (via `fig.to_html()`)
- Click confusion matrix cell → filtered gallery
- Click PR curve point → browse images at that threshold
- Click per-category AP bar → browse that category's predictions
- Drill-down: aggregate → category → image → annotation

---

## Tier 2 — Medium Term

### Format Conversion

**Shipped (COCO ↔ YOLO, COCO ↔ Pascal VOC, COCO ↔ CVAT).**

~~COCO ↔ YOLO first (most requested). Everyone has a slightly broken converter script — a correct, well-tested one has real value.~~

~~Pascal VOC~~ — bbox-only, bidirectional. VOC `<difficult>` mapped from COCO `iscrowd` on export, dropped on import. Integer-pixel round-trip within ≤1px.

~~CVAT for Images 1.1~~ — single XML file, boxes + polygon segmentation. Polygon area via shoelace, bbox from vertex extents. `<polyline>`, `<points>`, `<cuboid>` skipped. Video deferred.

### Oriented Bounding Boxes (OBB)

Evaluate rotated/oriented detections for aerial imagery, document analysis, and scene text. The rotated IoU kernel is the only new Rust code (~150 lines); everything else (AP accumulation, TIDE, confusion matrix) works unchanged with the different IoU function.

- Rotated IoU computation (polygon intersection of two rotated rectangles)
- OBB parameterized as `(cx, cy, w, h, angle)` — DOTA format
- DOTA format import/export
- OBB visualization in browse

### Panoptic Segmentation (PQ)

Panoptic Quality metric (Kirillov et al.) for unified "stuff" + "things" evaluation. Simpler than AP — no confidence scores, no accumulation over IoU thresholds.

- PQ = SQ (Segmentation Quality) × RQ (Recognition Quality)
- Per-class PQ, PQ_things, PQ_stuff breakdown
- Panoptic PNG format support (category_id × 256 + instance_id encoding)

### TIDE Error Analysis

**Shipped.**

~~Error decomposition following [TIDE](https://github.com/dbolya/tide) — classification, localization, duplicate, background, and missed errors. Builds naturally on confusion matrices and is a meaningful differentiator from faster-coco-eval.~~

### PyTorch Ecosystem Integrations

**Shipped.**

~~Drop-in ``CocoEvaluator`` and ``CocoDetection`` classes for PyTorch training loops. No torchvision or pycocotools dependency required. ``CocoEvaluator`` supports distributed multi-GPU evaluation with ``synchronize_between_processes()``. ``CocoDetection`` is a lightweight COCO dataset class compatible with ``DataLoader``.~~

---

## Tier 3 — Later

### Streaming Evaluation

Evaluate datasets that don't fit in memory. Process annotations in chunks without loading the full ground truth and detection sets upfront. Needed at O365/LVIS scale in production but not blocking anyone today — slot in once real users hit memory limits.

### CrowdPose

Keypoint dataset for crowded scenes. Uses a modified OKS matching algorithm with a crowd factor. Niche audience and significant custom eval logic — not worth prioritising before LVIS.

### Hierarchical Evaluation

**Shipped.**

~~Open Images-style evaluation with category hierarchies, where a detection of a parent category is not penalized against a child. Small, specialised audience.~~ `COCOeval(oid_style=True, hierarchy=h)` with `Hierarchy.from_file()` / `from_categories()` / `from_parent_map()`; group-of matching semantics, GT expansion, optional DT expansion via `params.expand_dt`.

### Dataset Healthcheck

**Shipped.**

~~4-layer validation (structural, quality, distribution, GT/DT compatibility) for COCO annotation files. `coco.healthcheck()` and `coco.healthcheck(dt)` in Python; `healthcheck()` / `healthcheck_compatibility()` in Rust; `coco healthcheck` CLI subcommand. `--healthcheck` flag on `coco eval` runs validation before evaluation.~~

### Sliced Evaluation

**Shipped.**

~~`COCOeval.slice_by(slices)` re-accumulates metrics for named image-ID subsets (e.g. indoor/outdoor, day/night) without recomputing IoU. `--slices <json>` flag on `coco eval` CLI.~~

### Shell Completions

**Shipped.**

~~Tab completion for both CLIs: `coco-eval --completions <bash|zsh|fish|elvish|powershell>` (Rust, powered by `clap_complete`); `pip install "hotcoco[completions]"` enables `coco` tab completion via `argcomplete`.~~

### Viewer Migration: FastAPI + HTMX

**Shipped.**

~~The current Gradio-based viewer hits a hard ceiling: no click/hover events, no interactive zoom, ~50MB dependency weight. Replaced with FastAPI + HTMX + Jinja2 + Canvas:~~
~~- Gallery browsing and filtering via HTMX (server-rendered, no build toolchain)~~
~~- Annotation detail panel as Canvas overlay with vanilla JS hover/zoom/click~~
~~- Same `coco.browse()` / `coco explore` API surface — drop-in replacement~~
~~- ~5MB dependency weight vs ~50MB for Gradio~~
~~- Jupyter embed via IFrame; responsive from 400px to 1400px+~~

### Multi-Object Tracking Metrics (HOTA/MOTA/IDF1)

Full tracking evaluation — HOTA (Higher Order Tracking Accuracy), MOTA, IDF1. TrackEval (the de-facto standard) is pure Python, effectively unmaintained, and the Ultralytics community is actively requesting tracking metrics. A fast Rust implementation would be a strong differentiator. HOTA decomposes into DetA × AssA, evaluated across IoU thresholds; requires cross-frame global alignment. Track AP (TAO) is the most natural entry point since it reuses the AP accumulation pipeline.

This is a significant scope expansion — commit fully or not at all. A partial MOT implementation has no value.

### Video Sequence Analysis

Lightweight per-sequence metric breakdowns for video object detection, surfacing high-level trends like which clips perform worst. Track AP (used by TAO, BURST, YouTube-VIS) is a natural extension of the existing COCO AP pipeline and the most likely entry point here.

### FiftyOne Evaluation Backend

Custom evaluation backend for [FiftyOne](https://github.com/voxel51/fiftyone) (Voxel51). FiftyOne's built-in COCO eval is slow and surfaces only the 12 standard metrics. A hotcoco backend would bring the speed advantage and expose TIDE error breakdowns and confusion matrices directly in the FiftyOne UI — metrics that don't exist in any other FiftyOne backend today. `init_as_pycocotools()` may already work as a zero-code path; the full backend adds discoverability and UI integration.

Longer term, full multi-object tracking metrics — MOTA, HOTA, IDF1 — are worth exploring as a Phase 2 effort. TrackEval (the de-facto standard) is effectively unmaintained and slow; there's a real opening for a fast Rust alternative. This would be a meaningful scope expansion and is not planned for the near term, but the direction is intentional.

### torchmetrics Backend

`MeanAveragePrecision(backend="hotcoco")` via a setuptools entry point that torchmetrics discovers at runtime. One-word change in training code; no other modifications needed.

### Experiment Tracking Integrations — **Shipped.**

~~Logging COCO metrics to experiment trackers is boilerplate every practitioner rewrites.~~
~~A `hotcoco.loggers` submodule with a `log_metrics(eval, logger, step=None)` helper would~~
~~eliminate this, providing sensible default metric names (`eval/AP`, `eval/AP50`, etc.) and~~
~~handling the flat dict format each platform expects.~~

Implemented as `get_results(prefix, per_class)` — returns a flat `dict[str, float]` with
prefixed keys and optional per-category AP, ready for `wandb.log()`, `mlflow.log_metrics()`,
or any tracker. No framework-specific wrappers needed.

### Hugging Face Integration

Hugging Face's [`evaluate`](https://github.com/huggingface/evaluate) library has a metric
backend concept — publish a `hotcoco` metric module and it shows up in hub search, gets
discovered by anyone browsing COCO-related metrics, and slots into any HF training pipeline
(Trainer, Accelerate, etc.) with a one-liner. COCO detection and segmentation metrics are
among the most searched in `evaluate`; a faster, drop-in implementation is a natural fit.

### Kaggle

No formal plugin system, but Kaggle is high-traffic for object detection competitions that
use COCO-format annotations. Two practical plays:

- **Notebooks** — a public notebook on a popular COCO-format competition demonstrating
  hotcoco eval gets surfaced in Kaggle search and "related notebooks," reaching a targeted
  audience of practitioners who are already running COCO evaluation.

### Data Viewer / Explorer

**Shipped.**

~~Lightweight dataset browsing without pulling in FiftyOne or other heavy tools. Two layers:~~

~~- **Python API** — `coco.browse(category="person", limit=20)` opens a simple grid view of
  images with overlaid annotations. Enough to sanity-check a dataset, spot labeling errors,
  and verify detections at a glance. Renders inline in Jupyter notebooks via IPython display;
  falls back to a temp HTML file outside notebooks.~~

~~- **Streamlit app** — `coco explore` CLI command launches a local Streamlit dashboard with
  filtering by category, image ID, area range, and confidence score. Covers the 90% use case
  of "let me look at my data" without any setup. Includes a one-click "Open in FiftyOne"
  button for users who outgrow the built-in viewer — generates the FiftyOne dataset import
  snippet and copies it to the clipboard.~~

~~Intentionally minimal — not trying to compete with FiftyOne's full feature set. The goal is
zero-dependency quick looks (Python API) and a lightweight local dashboard (Streamlit) that
bridge the gap between "stare at JSON" and "install a full data platform."~~
