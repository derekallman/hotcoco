# hotcoco

[![CI](https://github.com/derekallman/hotcoco/actions/workflows/ci.yml/badge.svg)](https://github.com/derekallman/hotcoco/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hotcoco)](https://pypi.org/project/hotcoco/)
[![Crates.io](https://img.shields.io/crates/v/hotcoco)](https://crates.io/crates/hotcoco)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast enough for every epoch, lean enough for every dataset. A drop-in replacement for [pycocotools](https://github.com/ppwwyyxx/cocoapi) that doesn't become the bottleneck — in your training loop or at foundation model scale. Up to 23× faster on standard COCO, 39× faster on Objects365, and fits comfortably in memory where alternatives run out.

Available as a **Python package**, **CLI tool**, and **Rust library**. Pure Rust — no Cython, no C compiler, no Microsoft Build Tools. Prebuilt wheels for Linux, macOS, and Windows.

Beyond raw speed, hotcoco ships a diagnostic toolkit that pycocotools and faster-coco-eval don't have: TIDE error breakdown, cross-category confusion matrix, per-category AP, F-scores, confidence calibration (ECE/MCE), per-image diagnostics with label error detection, sliced evaluation, dataset healthcheck, and publication-quality plots with a one-call PDF report. Same pip install, no extra config.

**[Documentation](https://derekallman.github.io/hotcoco/)** | **[Changelog](CHANGELOG.md)** | **[Roadmap](ROADMAP.md)**

## Performance

Benchmarked on COCO val2017 (5,000 images, 36,781 synthetic detections), Apple M1 MacBook Air:

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|-----------|
| bbox      | 9.46s | 2.45s (3.9×) | **0.41s (23.0×)** |
| segm      | 9.16s | 4.36s (2.1×) | **0.49s (18.6×)** |
| keypoints | 2.62s | 1.78s (1.5×) | **0.21s (12.7×)** |

Speedups in parentheses are vs pycocotools. Results verified against pycocotools on COCO val2017 with a 10,000+ case parity test suite — your AP scores won't change.

At scale (Objects365 val — 80k images, 365 categories, 1.2M detections), hotcoco completes in **18s** vs 721s for pycocotools (**39×**) and 251s for faster-coco-eval (**14×**) — while using half the memory. See the [full benchmarks](https://derekallman.github.io/hotcoco/benchmarks/).

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

- **COCO, LVIS & Open Images evaluation** — bbox, segmentation, and keypoints; all standard metrics plus LVIS federated eval (APr/APc/APf) and Open Images hierarchy-aware eval (group-of matching, GT expansion). See the [evaluation guide](https://derekallman.github.io/hotcoco/guide/evaluation/).
- **Confidence calibration** — ECE/MCE metrics and reliability diagrams measure whether your model's confidence scores are meaningful. See [calibration](https://derekallman.github.io/hotcoco/guide/evaluation/#confidence-calibration).
- **Model comparison** — `hotcoco.compare(eval_a, eval_b)` with per-metric deltas, per-category AP breakdown, and bootstrap confidence intervals for statistical significance. See [model comparison](https://derekallman.github.io/hotcoco/guide/evaluation/#model-comparison).
- **Per-image diagnostics & label errors** — per-image F1/AP scores, automatic detection of wrong labels and missing annotations in your ground truth. See [diagnostics](https://derekallman.github.io/hotcoco/guide/evaluation/#per-image-diagnostics-label-error-detection).
- **TIDE error analysis** — breaks down every FP and FN into six error types so you know *why* your model falls short, not just by how much. See [TIDE errors](https://derekallman.github.io/hotcoco/guide/tide/).
- **Confusion matrix** — cross-category matching with per-class breakdowns. See [confusion matrix](https://derekallman.github.io/hotcoco/guide/confusion-matrix/).
- **F-scores** — F-beta averaging over precision/recall curves, analogous to mAP. See [F-scores](https://derekallman.github.io/hotcoco/guide/f-scores/).
- **Plotting** — publication-quality PR curves, per-category AP, confusion matrices, and TIDE error breakdowns. Three built-in themes (`warm-slate`, `scientific-blue`, `ember`) with `paper_mode` for LaTeX/PowerPoint embedding. `report()` generates a single-page PDF summary. `pip install hotcoco[plot]`. See [plotting](https://derekallman.github.io/hotcoco/guide/plotting/).
- **Sliced evaluation** — re-accumulate metrics for named image subsets (indoor/outdoor, day/night) without recomputing IoU. See [sliced evaluation](https://derekallman.github.io/hotcoco/guide/evaluation/#sliced-evaluation).
- **Dataset healthcheck** — 4-layer validation (structural, quality, distribution, GT/DT compatibility) catches duplicate IDs, degenerate bboxes, category imbalance, and more. See [healthcheck](https://derekallman.github.io/hotcoco/guide/datasets/#healthcheck).
- **Format conversion** — COCO ↔ YOLO, COCO ↔ Pascal VOC, and COCO ↔ CVAT in either direction, from Python or the CLI. See [format conversion](https://derekallman.github.io/hotcoco/guide/conversion/).
- **PyTorch integrations** — `CocoDetection` and `CocoEvaluator` drop-in replacements for torchvision's detection classes; no torchvision or pycocotools dependency required. See [PyTorch integration](https://derekallman.github.io/hotcoco/integrations/).
- **Experiment tracker integration** — `get_results(prefix="val/bbox", per_class=True)` returns a flat dict ready for W&B, MLflow, or any logger. See [logging metrics](https://derekallman.github.io/hotcoco/guide/logging/).
- **Dataset browser** — `coco.browse()` / `coco explore` opens a local browser with category filter, annotation overlays (bbox/segm/keypoints), hover-to-highlight, zoom/pan, and detection comparison. `pip install hotcoco[browse]`. See [Dataset Browser](https://derekallman.github.io/hotcoco/guide/browse/).
- **Python CLI** (`coco`) — included with `pip install hotcoco`; `eval`, `healthcheck`, `stats`, `filter`, `merge`, `split`, `sample`, `convert`, and `explore` subcommands. See [CLI reference](https://derekallman.github.io/hotcoco/cli/).
- **Rust CLI** (`coco-eval`) — lightweight eval-only binary; `cargo install hotcoco-cli`. See [CLI reference](https://derekallman.github.io/hotcoco/cli/).
- **Rust library** — use hotcoco directly in your Rust projects via `cargo add hotcoco`. See [Rust API](https://docs.rs/hotcoco).

See the [documentation](https://derekallman.github.io/hotcoco/) for full API reference and examples.

## Contributing

Contributions are welcome. The core library is pure Rust in `crates/hotcoco/` — if you're new to Rust but comfortable with Python and the COCO spec, the PyO3 bindings in `crates/hotcoco-pyo3/` are a gentler entry point.

Before submitting a PR, run the pre-commit checks locally:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test
```

Parity with pycocotools is a hard requirement — if your change touches evaluation logic, verify metrics haven't shifted with `just parity` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## License

MIT
