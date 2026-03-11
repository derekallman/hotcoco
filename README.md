# hotcoco

[![CI](https://github.com/derekallman/hotcoco/actions/workflows/ci.yml/badge.svg)](https://github.com/derekallman/hotcoco/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hotcoco)](https://pypi.org/project/hotcoco/)
[![Crates.io](https://img.shields.io/crates/v/hotcoco)](https://crates.io/crates/hotcoco)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

COCO evaluation shouldn't be the slowest part of your training loop.

hotcoco is a Rust rewrite of pycocotools — 11–26× faster, with metric-for-metric parity and a drop-in API that works with Detectron2, mmdetection, RF-DETR, and any pycocotools-based pipeline.

**[Documentation](https://derekallman.github.io/hotcoco/)** | **[Changelog](CHANGELOG.md)** | **[Roadmap](ROADMAP.md)**

## Performance

Benchmarked on COCO val2017 (5,000 images, ~43,700 detections), Apple M1 MacBook Air:

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|---------|
| bbox      | 11.79s | 3.47s (3.4×) | **0.74s (15.9×)** |
| segm      | 19.49s | 10.52s (1.9×) | **1.58s (12.3×)** |
| keypoints | 4.79s  | 3.08s (1.6×) | **0.19s (25.0×)** |

All 12 metrics verified against pycocotools on COCO val2017 with a 10,000+ case parity test suite.

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

- **COCO & LVIS evaluation** — bbox, segmentation, and keypoints; all standard metrics plus LVIS federated eval (APr/APc/APf)
- **TIDE error analysis** — breaks down every FP and FN into six error types so you know *why* your model falls short, not just by how much
- **Confusion matrix** — cross-category matching with per-class breakdowns
- **F-scores** — F-beta averaging over precision/recall curves, analogous to mAP
- **Format conversion** — COCO ↔ YOLO in either direction, from Python or the CLI
- **PyTorch integrations** — `CocoDetection` and `CocoEvaluator` drop-in replacements for torchvision's detection classes; no torchvision or pycocotools dependency required
- **Experiment tracker integration** — `get_results(prefix="val/bbox", per_class=True)` returns a flat dict ready for W&B, MLflow, or any logger
- **Python CLI** (`coco`) — included with `pip install hotcoco`; `eval`, `stats`, `filter`, `merge`, `split`, `sample`, and `convert` subcommands
- **Rust CLI** (`coco-eval`) — lightweight eval-only binary; `cargo install hotcoco-cli`
- **Rust library** — use hotcoco directly in your Rust projects via `cargo add hotcoco`

See the [documentation](https://derekallman.github.io/hotcoco/) for full API reference and examples.

## Contributing

Contributions are welcome. The core library is pure Rust in `crates/hotcoco/` — if you're new to Rust but comfortable with Python and the COCO spec, the PyO3 bindings in `crates/hotcoco-pyo3/` are a gentler entry point.

Before submitting a PR, run the pre-commit checks locally:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test
```

Parity with pycocotools is a hard requirement — if your change touches evaluation logic, verify metrics haven't shifted with `python data/parity.py` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## License

MIT
