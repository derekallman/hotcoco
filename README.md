# hotcoco

[![CI](https://github.com/derekallman/hotcoco/actions/workflows/ci.yml/badge.svg)](https://github.com/derekallman/hotcoco/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hotcoco)](https://pypi.org/project/hotcoco/)
[![Crates.io](https://img.shields.io/crates/v/hotcoco)](https://crates.io/crates/hotcoco)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**hotcoco is a perception evaluation toolkit, written in Rust with Python bindings.** It's a drop-in replacement for [pycocotools](https://github.com/ppwwyyxx/cocoapi) — same numbers to double precision, up to 84× faster — plus the analysis that usually lives in separate tools: TIDE error analysis, confusion matrices, confidence calibration, model comparison, label-error detection, and a dataset browser.

It covers detection today — boxes, masks, keypoints, and oriented boxes on the COCO, LVIS, and Open Images protocols. Panoptic and tracking are planned, on the same engine.

Pure Rust, available as a **Python package**, **CLI tool**, and **Rust library**.

**[Documentation](https://derekallman.github.io/hotcoco/)** | **[Changelog](CHANGELOG.md)** | **[Roadmap](ROADMAP.md)**

## Performance

Bbox evaluation on COCO val2017 runs in **0.06s** against 4.91s for pycocotools; segm is ~81× faster and keypoints ~50×. Two other Rust evaluators, ultrafast-pycocotools and vernier, are in the same speed class; the benchmarks page compares all five on time and memory. The precision, recall, and scores arrays are bit-identical to pycocotools', and every summary metric matches to the limit of double precision, so your AP scores don't change.

Full tables, hardware, the Objects365 scale run, phase breakdowns, and parity verification: [Benchmarks](https://derekallman.github.io/hotcoco/benchmarks/).

## Get started

```bash
pip install hotcoco
```

No Cython, no C compiler, no Microsoft Build Tools. Prebuilt wheels for Linux, macOS, and Windows.

Already using pycocotools? One line:

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()
```

Or use it directly — the API is identical:

```python
from hotcoco import COCO, COCOeval

coco_gt = COCO("instances_val2017.json")
coco_dt = coco_gt.load_res("detections.json")

ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()
```

## What's included

### Evaluate

- **COCO, LVIS, and Open Images evaluation** — bbox, segmentation, keypoints, and oriented bounding box (OBB); all standard metrics plus LVIS federated eval (APr/APc/APf) and Open Images hierarchy-aware eval (group-of matching, GT expansion). OBB evaluation uses rotated IoU via polygon clipping for aerial imagery, document analysis, and scene text. See the [evaluation guide](https://derekallman.github.io/hotcoco/guide/evaluation/) and [LVIS and Open Images](https://derekallman.github.io/hotcoco/guide/lvis-open-images/).
- **Evaluation reports** — `ev.report()` returns metrics, per-class and per-group breakdowns, plottable PR curves, and a `provenance` field that says whether each number is comparable to a published leaderboard or is a hotcoco extension. See [the evaluation report](https://derekallman.github.io/hotcoco/guide/results/#the-evaluation-report).

### Diagnose

- **TIDE error analysis** — classifies every false positive and false negative into one of six error types and reports how much AP each type costs. See [TIDE errors](https://derekallman.github.io/hotcoco/guide/diagnostics/#tide-error-analysis).
- **Confusion matrix** — cross-category matching with per-class breakdowns. See [confusion matrix](https://derekallman.github.io/hotcoco/guide/diagnostics/#confusion-matrix).
- **Confidence calibration** — ECE/MCE metrics and reliability diagrams measure whether your model's confidence scores are meaningful. See [calibration](https://derekallman.github.io/hotcoco/guide/diagnostics/#confidence-calibration).
- **Per-image diagnostics and label errors** — per-image F1/AP scores, automatic detection of wrong labels and missing annotations in your ground truth. See [diagnostics](https://derekallman.github.io/hotcoco/guide/diagnostics/#per-image-diagnostics-and-label-error-detection).
- **Model comparison** — `hotcoco.compare(eval_a, eval_b)` with per-metric deltas, per-category AP breakdown, and bootstrap confidence intervals for statistical significance. See [model comparison](https://derekallman.github.io/hotcoco/guide/diagnostics/#model-comparison).
- **F-scores** — F-beta averaging over precision/recall curves, analogous to mAP. See [F-scores](https://derekallman.github.io/hotcoco/guide/diagnostics/#f-scores).
- **Sliced evaluation** — re-accumulate metrics for named image subsets (indoor/outdoor, day/night) without recomputing IoU. See [sliced evaluation](https://derekallman.github.io/hotcoco/guide/evaluation/#sliced-evaluation).
- **Plotting** — publication-quality PR curves, per-category AP, confusion matrices, and TIDE error breakdowns. Light and dark themes (`cyanotype`, `cyanotype-dark`) with `paper_mode` for LaTeX/PowerPoint embedding. `report()` generates a single-page PDF summary. `pip install hotcoco[plot]`. See [plotting](https://derekallman.github.io/hotcoco/guide/plotting/).

### Explore your data

- **Dataset browser** — `coco.browse()` / `coco explore` opens a local browser with category filter, annotation overlays (bbox/segm/keypoints/OBB), hover-to-highlight, zoom/pan, and detection comparison. Pass `eval=` to enable an interactive eval dashboard with PR curves, confusion matrix, TIDE errors, calibration, and per-image F1. `pip install hotcoco[browse]`. See [Dataset browser](https://derekallman.github.io/hotcoco/guide/browse/).
- **Dataset healthcheck** — 4-layer validation (structural, quality, distribution, GT/DT compatibility) catches duplicate IDs, degenerate bboxes, category imbalance, and more. See [healthcheck](https://derekallman.github.io/hotcoco/guide/datasets/#healthcheck).
- **Format conversion** — COCO ↔ YOLO, Pascal VOC, CVAT, DOTA (oriented boxes), and Open Images CSV, from Python or the CLI. See [format conversion](https://derekallman.github.io/hotcoco/guide/datasets/#convert).

### Compose and integrate

- **Metric functions on plain arrays** — `hotcoco.metrics` and `hotcoco.primitives` expose the engine as free functions, the way `sklearn.metrics` and `torchmetrics.functional` do. No evaluator, no dataset, no COCO JSON: `metrics.average_precision(scores, matched, num_gt=...)`, `primitives.lsap(cost)`. `COCOeval` calls the same functions, so the numbers cannot diverge. See [metrics](https://derekallman.github.io/hotcoco/api/metrics/) and [primitives](https://derekallman.github.io/hotcoco/api/primitives/).
- **PyTorch integrations** — `CocoDetection` and `CocoEvaluator` drop-in replacements for torchvision's detection classes; no torchvision or pycocotools dependency required. See [PyTorch integration](https://derekallman.github.io/hotcoco/guide/pytorch/).
- **Experiment tracker integration** — `get_results(prefix="val/bbox", per_class=True)` returns a flat dict ready for W&B, MLflow, or any logger. See [logging metrics](https://derekallman.github.io/hotcoco/guide/results/#logging-metrics).
- **Python CLI** (`coco`) — included with `pip install hotcoco`; `eval`, `healthcheck`, `stats`, `filter`, `merge`, `split`, `sample`, `convert`, `compare`, and `explore` subcommands. See [CLI reference](https://derekallman.github.io/hotcoco/cli/).
- **Rust CLI** (`coco-eval`) — lightweight eval-only binary; `cargo install hotcoco-cli`. See [CLI reference](https://derekallman.github.io/hotcoco/cli/).
- **Type stubs** — ships with `.pyi` stubs and `py.typed` marker for full autocomplete and type checking in VS Code, PyCharm, and other IDEs.
- **Rust library** — use hotcoco directly in your Rust projects via `cargo add hotcoco`. See [Rust API](https://docs.rs/hotcoco).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the architecture overview, build and test workflow, and pre-commit checks.

Parity with pycocotools is a hard requirement for any change to evaluation logic; `CONTRIBUTING.md` describes how to verify it.

## License

MIT
