# hotcoco

A drop-in replacement for [pycocotools](https://github.com/ppwwyyxx/cocoapi) — 11-26x faster COCO evaluation for object detection, segmentation, and keypoints.

Available as a **Python package**, **CLI tool**, and **Rust library**.

**[Documentation](https://derekallman.github.io/hotcoco/)** | **[Changelog](CHANGELOG.md)** | **[Roadmap](ROADMAP.md)**

## Performance

Benchmarked on COCO val2017 (5,000 images, 36,781 ground truth annotations, ~43,700 detections), Apple M1 MacBook Air:

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|-----------|
| bbox      | 11.79s      | 3.47s (3.4x)    | 0.74s (15.9x) |
| segm      | 19.49s      | 10.52s (1.9x)   | 1.58s (12.3x) |
| keypoints | 4.79s       | 3.08s (1.6x)    | 0.19s (25.0x) |

Speedups in parentheses are vs pycocotools. All metrics match pycocotools within 0.003 (many are exact).

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

If you have existing code that imports from `pycocotools` and don't want to change every import, call `init_as_pycocotools()` once at startup:

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()

# Existing code works unchanged
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
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
