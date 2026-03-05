# hotcoco

11-26x faster COCO evaluation — a drop-in replacement for [pycocotools](https://github.com/ppwwyyxx/cocoapi) that works with Ultralytics YOLO, Detectron2, mmdetection, RF-DETR, and any pycocotools-based pipeline.

Available as a **Python package**, **CLI tool**, and **Rust library**. Pure Rust — no Cython, no C compiler, no Microsoft Build Tools. Prebuilt wheels for Linux, macOS, and Windows.

**[Documentation](https://derekallman.github.io/hotcoco/)** | **[Changelog](CHANGELOG.md)** | **[Roadmap](ROADMAP.md)**

## Performance

Benchmarked on COCO val2017 (5,000 images, 36,781 ground truth annotations, ~43,700 detections), Apple M1 MacBook Air:

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|-----------|
| bbox      | 11.79s      | 3.47s (3.4x)    | 0.74s (15.9x) |
| segm      | 19.49s      | 10.52s (1.9x)   | 1.58s (12.3x) |
| keypoints | 4.79s       | 3.08s (1.6x)    | 0.19s (25.0x) |

Speedups in parentheses are vs pycocotools. Results verified against pycocotools on COCO val2017 with a 10,000+ case parity test suite — your AP scores won't change.

## Quick Start

### Python

```bash
pip install hotcoco
```

```python
from hotcoco import COCO, COCOeval

coco_gt = COCO("instances_val2017.json")
coco_dt = coco_gt.load_res("detections.json")

ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()
```

#### Drop-in replacement for pycocotools

If you use Detectron2, Ultralytics YOLO, mmdetection, or any other pycocotools-based pipeline, call `init_as_pycocotools()` once at startup — no other code changes needed:

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()

# Existing code works unchanged
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
```

#### LVIS evaluation

hotcoco supports [LVIS](https://www.lvisdataset.org/) federated evaluation with all 13 metrics (AP, APr, APc, APf, AR@300, and more). Use `LVISeval` directly or call `init_as_lvis()` to drop into any existing lvis-api pipeline:

```python
from hotcoco import COCO, LVISeval

lvis_gt = COCO("lvis_v1_val.json")
lvis_dt = lvis_gt.load_res("detections.json")

ev = LVISeval(lvis_gt, lvis_dt, "segm")
ev.run()
print(ev.get_results())  # {"AP": ..., "APr": ..., "APc": ..., "APf": ..., "AR@300": ...}
```

```python
# Or as a drop-in for Detectron2 / MMDetection lvis-api pipelines
from hotcoco import init_as_lvis
init_as_lvis()

from lvis import LVIS, LVISEval, LVISResults  # resolves to hotcoco
```

#### Format conversion

Convert between COCO JSON and YOLO label format in either direction:

```python
from hotcoco import COCO

# COCO → YOLO
coco = COCO("instances_val2017.json")
stats = coco.to_yolo("labels/val2017/")
print(stats)  # {'images': 5000, 'annotations': 36781, 'skipped_crowd': 12, 'missing_bbox': 0}

# YOLO → COCO (with Pillow to read image dims)
coco2 = COCO.from_yolo("labels/val2017/", images_dir="images/val2017/")
coco2.save("reconstructed.json")
```

Or from the CLI:

```bash
coco convert --from coco --to yolo --input annotations.json --output labels/
coco convert --from yolo --to coco --input labels/ --output annotations.json --images-dir images/
```

#### F-scores

`f_scores()` computes F-beta scores from the precision/recall curves. For each IoU threshold and category it finds the operating point that maximises F-beta, then averages — analogous to mAP:

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

ev.f_scores()          # {"F1": 0.523, "F150": 0.712, "F175": 0.581}
ev.f_scores(beta=0.5)  # precision-weighted F-score
ev.f_scores(beta=2.0)  # recall-weighted F-score
```

#### Logging metrics

`get_results()` accepts an optional prefix and per-class flag, returning a flat dict that plugs directly into any experiment tracker:

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

import wandb
wandb.log(ev.get_results(prefix="val/bbox", per_class=True), step=epoch)
# {"val/bbox/AP": 0.578, ..., "val/bbox/AP/person": 0.82, "val/bbox/AP/car": 0.71, ...}
```

#### TIDE error analysis

`tide_errors()` decomposes every false positive and false negative into six error types — Localization, Classification, Duplicate, Background, Both, and Miss — and reports the ΔAP for each. Use it to understand *why* your model falls short, not just how much:

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()

result = ev.tide_errors()
for name, delta in sorted(result["delta_ap"].items(), key=lambda x: -x[1]):
    if name not in ("FP", "FN"):
        print(f"  {name}: ΔAP={delta:.4f}  n={result['counts'].get(name, '—')}")
```

Or from the CLI:

```bash
coco eval --gt instances_val2017.json --dt bbox_results.json --tide
```

### CLI

```bash
cargo install hotcoco-cli
coco-eval --gt annotations.json --dt detections.json --iou-type bbox
```

### Rust

```bash
cargo add hotcoco
```

```rust
use hotcoco::{COCO, COCOeval};
use hotcoco::params::IouType;
use std::path::Path;

let coco_gt = COCO::new(Path::new("annotations.json"))?;
let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

let mut eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
eval.evaluate();
eval.accumulate();
eval.summarize();
```

## License

MIT
