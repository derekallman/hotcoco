# hotcoco

[![CI](https://github.com/derekallman/hotcoco/actions/workflows/ci.yml/badge.svg)](https://github.com/derekallman/hotcoco/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hotcoco)](https://pypi.org/project/hotcoco/)
[![Crates.io](https://img.shields.io/crates/v/hotcoco)](https://crates.io/crates/hotcoco)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A drop-in replacement for [pycocotools](https://github.com/ppwwyyxx/cocoapi), written in Rust.** Same numbers, bit for bit. Up to 113× faster. No compiler to install.

```bash
pip install hotcoco
```

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()

# Every pycocotools import now resolves to hotcoco. Nothing else changes.
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
```

Or import it directly; the API is the same:

```python
from hotcoco import COCO, COCOeval

coco_gt = COCO("instances_val2017.json")
coco_dt = coco_gt.load_res("detections.json")

ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()
```

**[Documentation](https://derekallman.github.io/hotcoco/)** | **[Changelog](CHANGELOG.md)** | **[Roadmap](ROADMAP.md)**

## Why switch

- **The numbers do not change.** The precision, recall, and scores arrays and every summary metric are bit-identical to pycocotools' on COCO val2017, for bbox, segm, and keypoints. Parity is a hard requirement for every change; [CONTRIBUTING.md](CONTRIBUTING.md) describes how it is verified.
- **It is fast.** The [benchmarks](https://derekallman.github.io/hotcoco/benchmarks/) compare hotcoco, pycocotools, and three other evaluators on time and peak memory for bbox, segm, and keypoints on COCO val2017, with the hardware and the method.
- **It installs anywhere.** Prebuilt wheels for Linux, macOS, and Windows. No Cython, no C compiler, no Microsoft Build Tools.

## Beyond pycocotools

The same engine covers more than COCO detection. Each item links to its guide.

- [LVIS and Open Images](https://derekallman.github.io/hotcoco/guide/lvis-open-images/) protocols, and [oriented boxes](https://derekallman.github.io/hotcoco/guide/evaluation/).
- [Panoptic quality](https://derekallman.github.io/hotcoco/guide/panoptic/), matching panopticapi on val2017, and [semantic segmentation](https://derekallman.github.io/hotcoco/guide/semantic/), matching mmsegmentation.
- [Streaming evaluation](https://derekallman.github.io/hotcoco/guide/evaluation/#streaming-evaluation) inside a validation loop, straight from the detector's prediction array, with shards that merge across ranks.
- [TIDE error analysis](https://derekallman.github.io/hotcoco/guide/diagnostics/#tide-error-analysis), [confusion matrices](https://derekallman.github.io/hotcoco/guide/diagnostics/#confusion-matrix), [calibration](https://derekallman.github.io/hotcoco/guide/diagnostics/#confidence-calibration), [label-error detection](https://derekallman.github.io/hotcoco/guide/diagnostics/#per-image-diagnostics-and-label-error-detection), and [model comparison](https://derekallman.github.io/hotcoco/guide/diagnostics/#model-comparison) with bootstrap confidence intervals.
- [Metric functions on plain arrays](https://derekallman.github.io/hotcoco/api/metrics/), the way `sklearn.metrics` works, for [building your own metric](https://derekallman.github.io/hotcoco/guide/custom-metrics/).
- A [dataset browser](https://derekallman.github.io/hotcoco/guide/browse/), a [healthcheck](https://derekallman.github.io/hotcoco/guide/datasets/#healthcheck), and [format conversion](https://derekallman.github.io/hotcoco/guide/datasets/#convert) to and from YOLO, Pascal VOC, CVAT, DOTA, and Open Images CSV.
- [PyTorch integration](https://derekallman.github.io/hotcoco/guide/pytorch/), [plotting](https://derekallman.github.io/hotcoco/guide/plotting/), a [CLI](https://derekallman.github.io/hotcoco/cli/), type stubs, and a [Rust library](https://docs.rs/hotcoco).

The full list is on the [documentation site](https://derekallman.github.io/hotcoco/).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the architecture overview, build and test workflow, and pre-commit checks.

## License

MIT
