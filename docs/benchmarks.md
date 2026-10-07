# Benchmarks

## Feature comparison

Five libraries evaluate COCO-format detections from Python; three of them are
Rust engines. Versions: pycocotools 2.0.11, faster-coco-eval 1.8.0,
ultrafast-pycocotools 0.1.13, vernier 0.5.4, hotcoco `main` (1.2 development).

| Feature | pycocotools | faster-coco-eval | ultrafast-pycocotools | vernier | hotcoco |
|---------|-------------|------------------|-----------------------|---------|---------|
| **Installation** | Prebuilt wheels | Prebuilt wheels | Prebuilt wheels | Prebuilt wheels | Prebuilt wheels — `pip install` just works |
| **Metric parity** | Reference | Exact | `precision`/`recall`/`scores` bit-identical on x86-64; segm differs on arm64 (see [Segmentation](#segmentation)) | Bit-identical in `parity_mode="strict"` on x86-64; segm differs on arm64 | `precision`/`recall`/`scores` bit-identical on x86-64 and arm64; summary metrics bit-identical (measured on arm64) |
| **LVIS evaluation** | No | Yes — via `lvis_style=True` flag | Yes — via `lvis_style=True` flag | Yes — federated AP | Yes — 13 metrics, `LVISeval` class, `init_as_lvis()` |
| **Open Images, oriented boxes** | No | No | No | No | Yes — Challenge protocol with group-of; OBB IoU |
| **Panoptic, semantic** | No | No | No | Yes — PQ, mIoU | Planned |
| **TIDE error analysis** | No | No | No | Yes | Yes — 6 error types, ΔAP per type |
| **Confusion matrix** | No | No | Yes | Yes | Yes — cross-category, configurable threshold |
| **Calibration** | No | No | No | Yes — ECE/MCE | Yes — ECE/MCE, reliability curve |
| **F-scores** | No | No | No | No | Yes — F-beta at any β |
| **Per-class AP** | Manual only | Yes — via `extended_metrics` | Yes — `per_category_stats()` | Yes — `per_class` table | Built-in via `get_results(per_class=True)` |
| **Model comparison** | No | No | No | No | Yes — bootstrap CIs per metric |
| **Dataset operations** | No | No | No | No | Yes — filter, merge, split, sample, stats |
| **Format conversion** | No | No | No | No | Yes — COCO ↔ YOLO, VOC, CVAT, DOTA, Open Images CSV |
| **Dataset browser** | No | No | No | No | Yes — `coco explore` |
| **PyTorch integration** | Via torchvision | Yes — TorchVision compatible | `init_as_pycocotools()` shim | DLPack tensors; `patch_pycocotools()` | Yes — `CocoDetection`, `CocoEvaluator` |
| **Rust API** | No | No | No | Yes — crate on crates.io | Yes — native crate on crates.io |
| **CLI** | No | No | No | Yes — `vernier` binary | Yes — `coco` (Python) + `coco-eval` (Rust) |
| **Results export** | No | No | `stats_as_dict` | JSON via CLI | Yes — JSON with params + metrics + per-class |
| **Memory at scale** | Exceeds physical RAM on O365 | Exceeds physical RAM on O365 | Completes (their measurement) | Completes (their measurement) | Completes within physical RAM ([details](#objects365-scale-benchmark)) |
| **Python versions** | 3.9+ | 3.10+ | 3.8+ | 3.10+ | 3.10+ |
| **License** | BSD | Apache 2.0 | BSD-2 | MIT or Apache 2.0 | MIT |

## Speed benchmarks

**Hardware:** Apple M1 MacBook Air — 8 cores (4 performance + 4 efficiency), 8 GB RAM
**Dataset:** COCO val2017 — 5,000 images
**Detections:** 36,781 synthetic — see [Methodology](#methodology)
**Timing:** Wall clock time — per-cell median of 3 runs, each in a fresh process, at
both 1× and 10×, captured back to back in one session (2026-10-07). Absolute times
drift between captures on this machine, so compare them only within a table; the
speedup ratios are not affected. Peak memory is each process's own resident set.
**Versions:** pycocotools 2.0.11, faster-coco-eval 1.8.0, ultrafast-pycocotools 0.1.13,
vernier 0.5.4, hotcoco 1.2.0.

### Results (1x detections)

<figure markdown>
![Grouped bar chart of evaluation wall clock for bbox, segm and keypoints across the five libraries](assets/benchmark-speed.png#only-light)
![Grouped bar chart of evaluation wall clock for bbox, segm and keypoints across the five libraries](assets/benchmark-speed-dark.png#only-dark)
<figcaption>The axis is linear, not logarithmic — hotcoco's bar really is that small next
to pycocotools'. Chart drawn with <code>hotcoco.plot</code>.</figcaption>
</figure>

| Eval Type | pycocotools | faster-coco-eval | ultrafast-pycocotools | vernier | hotcoco |
|-----------|-------------|------------------|-----------------------|---------|---------|
| bbox      | 5.11s | 0.88s (5.8×) | 0.07s (74.9×) | 0.22s (22.9×) | **0.05s (113.3×)** |
| segm      | 6.25s | 2.63s (2.4×) | 0.10s (63.4×) | 0.58s (10.8×) | **0.06s (110.5×)** |
| keypoints | 2.32s | 0.95s (2.5×) | 0.07s (32.2×) | 0.17s (13.7×) | **0.04s (55.3×)** |

Speedups in parentheses are vs pycocotools.

Peak resident memory, same runs (MB):

| Eval Type | pycocotools | faster-coco-eval | ultrafast-pycocotools | vernier | hotcoco |
|-----------|-------------|------------------|-----------------------|---------|---------|
| bbox      | 659 | 629 | 94 | 173 | 114 |
| segm      | 608 | 657 | 117 | 200 | 132 |
| keypoints | 266 | 287 | 106 | 146 | 97 |

The three Rust engines are 11–113× ahead of pycocotools, and hotcoco is
fastest on every row. On memory it sits below vernier and within about 1.2×
of ultrafast-pycocotools, which parses detections straight into columnar
arrays while hotcoco materializes one annotation record per detection.

### Results (10x detections)

Scaling detections by 10x (~368,000) to test behavior under higher load:

| Eval Type | pycocotools | faster-coco-eval | ultrafast-pycocotools | vernier | hotcoco |
|-----------|-------------|------------------|-----------------------|---------|---------|
| bbox      | 24.81s | 3.40s (7.3×) | 0.15s (160.7×) | 0.89s (27.8×) | **0.12s (199.5×)** |
| segm      | 25.24s | 7.06s (3.6×) | 0.27s (93.5×) | 1.73s (14.6×) | **0.23s (108.0×)** |
| keypoints | 9.42s | 5.99s (1.6×) | 0.31s (29.9×) | 0.92s (10.2×) | **0.24s (39.2×)** |

Peak resident memory, same runs (MB):

| Eval Type | pycocotools | faster-coco-eval | ultrafast-pycocotools | vernier | hotcoco |
|-----------|-------------|------------------|-----------------------|---------|---------|
| bbox      | 1519 | 1080 | 187 | 558 | 216 |
| segm      | 1626 | 1437 | 375 | 860 | 381 |
| keypoints | 1173 | 1256 | 463 | 704 | 331 |

The results file is read a few megabytes at a time, so at this scale the peak
is set by the evaluation, not the load: hotcoco holds about 210 bytes per
detection with no slack around it, the evaluator shares the datasets rather
than copying them, and the peak is within 1.2× of ultrafast-pycocotools' on
bbox and segm and below it on keypoints. On keypoints, the 325 MB results file
no longer sits in memory beside its records.

Absolute times stay under 0.3s at 368,000 detections, and hotcoco is fastest
on every row.

`StreamingEval` holds less still. On Objects365 val with 1M synthetic detections,
matching each batch as it arrives and keeping only the lean per-pair records
peaks at 270 MB above the loaded ground truth, against 739 MB for collecting
every prediction and evaluating at the end. The numbers are identical either
way, whatever the batch size.

### Where the time goes

The preceding end-to-end numbers blend two very different phases: **load** (JSON
parsing and index building — `COCO()` + `loadRes`) and **eval** (`evaluate` +
`accumulate` + `summarize`). Splitting them shows where each library spends its
time (same protocol, same synthetic detections as the 1× table):

| Eval type | Phase | pycocotools | faster-coco-eval | ultrafast-pycocotools | vernier | hotcoco |
|-----------|-------|-------------|------------------|-----------------------|---------|---------|
| bbox      | load  | 0.31s | 0.31s (1.0×) | 0.05s (6.1×) | 0.09s (3.6×) | **0.03s (11.5×)** |
|           | eval  | 4.66s | 0.59s (8.0×) | 0.02s (229.0×) | 0.12s (39.5×) | **0.01s (378.0×)** |
| segm      | load  | 0.33s | 0.33s (1.0×) | 0.05s (6.1×) | 0.09s (3.8×) | **0.03s (10.5×)** |
|           | eval  | 5.38s | 2.19s (2.5×) | 0.05s (115.5×) | 0.48s (11.1×) | **0.02s (244.7×)** |
| keypoints | load  | 0.48s | 0.48s (1.0×) | 0.04s (13.2×) | 0.05s (10.5×) | **0.03s (15.2×)** |
|           | eval  | 1.79s | 0.44s (4.0×) | 0.03s (52.8×) | 0.12s (15.6×) | **0.01s (186.2×)** |
| bbox, bbox-only GT | load | 0.16s | 0.15s (1.0×) | 0.03s (5.6×) | 0.02s (7.1×) | **0.02s (9.3×)** |
|           | eval  | 4.44s | 0.52s (8.5×) | 0.02s (221.1×) | 0.12s (36.7×) | **0.01s (369.6×)** |

hotcoco's evaluation phase is 186–378× faster than pycocotools'; the
end-to-end headline is lower because loading, which no library can speed up by
much more than 10×, is a large share of hotcoco's total.

The *bbox-only GT* row strips the polygon segmentation the official instances
files carry on every annotation — about two-thirds of the file bytes, which bbox
evaluation never reads. It represents datasets that never had masks (custom bbox
datasets, YOLO conversions, Objects365), where the load column is the one you'll
actually see.

### Objects365 scale benchmark

**Hardware:** Windows 11, AMD Ryzen 5 5600X — 6 cores / 12 threads, 16 GB RAM + swap
**Dataset:** Objects365 val — 80,000 images, 1.2M annotations, 365 categories
**Detections:** ~1.2M synthetic bbox (capped at 100/image, seed=42)
**Timing:** Wall clock time, single run
**Versions:** pycocotools 2.0.11, faster-coco-eval 1.7.2, hotcoco 0.3.0 — this run
predates the 1.0 loading work, so the hotcoco figure is conservative

| Library | Time | Peak RAM | Committed | Speedup |
|---------|------|----------|-----------|---------|
| pycocotools | 721.18s | 14.34 GB | 23.71 GB | baseline |
| faster-coco-eval | 250.90s | 14.57 GB | 29.96 GB | 2.9x |
| **hotcoco** | **18.32s** | **7.47 GB** | **8.11 GB** | **39.4x** |

Peak RAM is the peak working set (physical memory). Committed includes swap — both pycocotools and faster-coco-eval exceeded physical RAM and relied heavily on the pagefile, which significantly inflated their wall clock times. hotcoco completed within physical memory with minimal swap.

## Metric parity

**Reference:** pycocotools 2.0.11, numpy 2.4.3.
**Ground truth:** COCO val2017 — 5,000 images.
**Detections:** the published `instances_val2017_fake*_results.json` files from
[ppwwyyxx/cocoapi](https://github.com/ppwwyyxx/cocoapi), the same inputs pycocotools uses in its own
tests.

**The `precision`, `recall`, and `scores` arrays are bit-identical to pycocotools'**
on all three evaluation types — every cell, checked with an integer compare of the
raw `f64` bits. The summary metrics below are bit-identical too: they are means over
those identical arrays, and hotcoco sums them in the order numpy's `mean` does —
pairwise, over the same flattened elements. The diffs in the following tables are
raw measured differences, not rounded.

### Bounding box

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|---------|------|
| AP     | 0.57793065 | 0.57793065 | 0.00e+00 |
| AP50   | 0.86052720 | 0.86052720 | 0.00e+00 |
| AP75   | 0.60003745 | 0.60003745 | 0.00e+00 |
| APs    | 0.32723763 | 0.32723763 | 0.00e+00 |
| APm    | 0.70684507 | 0.70684507 | 0.00e+00 |
| APl    | 0.91751661 | 0.91751661 | 0.00e+00 |
| AR1    | 0.42708926 | 0.42708926 | 0.00e+00 |
| AR10   | 0.68690535 | 0.68690535 | 0.00e+00 |
| AR100  | 0.70127765 | 0.70127765 | 0.00e+00 |
| ARs    | 0.43712612 | 0.43712612 | 0.00e+00 |
| ARm    | 0.80637778 | 0.80637778 | 0.00e+00 |
| ARl    | 0.95956720 | 0.95956720 | 0.00e+00 |

The threshold grids are constructed to match `numpy.linspace` bit-for-bit, which
removed the last systematic source of divergence here.

### Segmentation

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|---------|------|
| AP     | 0.65763117 | 0.65763117 | 0.00e+00 |
| AP50   | 0.92315461 | 0.92315461 | 0.00e+00 |
| AP75   | 0.70141134 | 0.70141134 | 0.00e+00 |
| APs    | 0.46056290 | 0.46056290 | 0.00e+00 |
| APm    | 0.77182113 | 0.77182113 | 0.00e+00 |
| APl    | 0.93431919 | 0.93431919 | 0.00e+00 |
| AR1    | 0.45457448 | 0.45457448 | 0.00e+00 |
| AR10   | 0.74556101 | 0.74556101 | 0.00e+00 |
| AR100  | 0.76167772 | 0.76167772 | 0.00e+00 |
| ARs    | 0.54570783 | 0.54570783 | 0.00e+00 |
| ARm    | 0.85891625 | 0.85891625 | 0.00e+00 |
| ARl    | 0.98103170 | 0.98103170 | 0.00e+00 |

Exact. The residual segmentation once carried (AP ~1e-5) came from polygon
rasterization, where the reference's C compiler contracts `s*t+ys` into a single
fused multiply-add on arm64 but not on x86-64, whose PyPI wheels target a baseline
without the instruction. hotcoco mirrors that choice per architecture, so the
comparison is exact on either kind of machine. Masks can differ by a boundary pixel
between the two architectures — as pycocotools' own do. The other two Rust
evaluators do not mirror the contraction: on Apple Silicon, ultrafast-pycocotools
0.1.11 and vernier 0.5.4 differ from the arm64 pycocotools wheel in 283 cells of the
segm precision tensor on these detections, by up to 0.72 per cell and 1e-5 in AP
(measured 2026-09-25); on x86-64 all three agree.

### Keypoints

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|---------|------|
| AP     | 0.41255451 | 0.41255451 | 0.00e+00 |
| AP50   | 0.60631206 | 0.60631206 | 0.00e+00 |
| AP75   | 0.42916428 | 0.42916428 | 0.00e+00 |
| APm    | 0.40337197 | 0.40337197 | 0.00e+00 |
| APl    | 0.88304294 | 0.88304294 | 0.00e+00 |
| AR     | 0.76642003 | 0.76642003 | 0.00e+00 |
| AR50   | 0.97481108 | 0.97481108 | 0.00e+00 |
| AR75   | 0.80636020 | 0.80636020 | 0.00e+00 |
| ARm    | 0.62190658 | 0.62190658 | 0.00e+00 |
| ARl    | 0.96335935 | 0.96335935 | 0.00e+00 |

Keypoint metrics are exact. Keypoint evaluation reports 10 metrics — see
[Keypoint evaluation](guide/evaluation.md#keypoint-evaluation).

### TIDE

**Reference:** [tidecv](https://github.com/dbolya/tide), on COCO val2017 at `pos_thr=0.5`.
On the 4,589 images without a crowd region, every ΔAP agrees within ±0.005 and
every error count matches. On all 5,000 images, `Loc` reads about 0.01 lower here
and `Miss` about 0.007 higher; the other types still agree within ±0.005:

| Error | hotcoco ΔAP | tidecv ΔAP | hotcoco count | tidecv count | hotcoco ΔAP, crowd-free | tidecv ΔAP, crowd-free |
|---|---|---|---|---|---|---|
| Cls | 0.0000 | 0.0000 | 13 | 13 | 0.0000 | 0.0000 |
| Loc | 0.1036 | 0.1135 | 3,121 | 3,738 | 0.0972 | 0.0974 |
| Both | 0.0007 | 0.0001 | 726 | 766 | 0.0006 | 0.0001 |
| Dupe | 0.0001 | 0.0000 | 27 | 37 | 0.0001 | 0.0000 |
| Bkg | 0.0109 | 0.0105 | 6,039 | 6,414 | 0.0109 | 0.0104 |
| Miss | 0.0148 | 0.0075 | 1,102 | 529 | 0.0065 | 0.0059 |

The ranking — which error type is costing you the most AP — is the same, and that is
what the metric is for. The difference comes from crowd handling: hotcoco builds TIDE
on the same COCO-convention matching as its AP (so `tide_errors()` and `ev.stats`
always agree about which detections exist), while tidecv removes crowd regions from
matching entirely — its false-positive counts run higher and its `Miss` runs lower as
a result. `scripts/parity_tide.py` runs both comparisons.

### Open Images

**Reference:** the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) —
the implementation the official protocol page points to.

Open Images evaluation is compared over 70 cases covering group-of absorption, IoA
containment at and around the 0.5 boundary, undetected group-of boxes, overlapping
group-of boxes, and randomized multi-class scenes. Both mAP and per-class AP are
compared, and **every case agrees to within one ulp** (worst difference 1.11e-16). It
runs in CI on every commit.

Two things that comparison does **not** cover, and why `provenance` still reports
`"extension"` for Open Images runs:

- **Non-exhaustive image-level labels.** The challenge ignores detections of a class
  not verified on an image, and counts detections of a negatively-labeled class as
  false positives. hotcoco does not implement this — it needs per-image label data
  that COCO-format JSON cannot carry. A real challenge submission would score
  differently.
- **Hierarchy expansion** is applied to annotations before evaluation rather than
  inside it, so it sits outside the compared surface.

### Verify it yourself

You do not have to take these numbers on faith, and you should not have to clone
the repo to check them. Install both libraries and run your own ground truth and
detections through each:

```python
import contextlib, io
import numpy as np
from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval
import hotcoco

GT, DT, IOU_TYPE = "instances_val2017.json", "my_detections.json", "bbox"

def run(coco_cls, eval_cls):
    with contextlib.redirect_stdout(io.StringIO()):   # both print a lot
        gt = coco_cls(GT)
        dt = gt.loadRes(DT)
        e = eval_cls(gt, dt, IOU_TYPE)
        e.evaluate(); e.accumulate(); e.summarize()
    return np.asarray(e.stats)

ref = run(PyCOCO, PyCOCOeval)
got = run(hotcoco.COCO, hotcoco.COCOeval)

for i, (a, b) in enumerate(zip(ref, got)):
    print(f"[{i:2}] pycocotools={a:.8f}  hotcoco={b:.8f}  diff={abs(a - b):.2e}")
print("max diff:", np.abs(ref - got).max())
```

On val2017 the project's own parity gate requires exact equality, so any
difference at all is a bug. On your own data, anything above ~1e-12 is worth
[opening an issue](https://github.com/derekallman/hotcoco/issues).

The same shape works for `hotcoco.mask` against `pycocotools.mask`, operation by
operation.

Beyond val2017, a hypothesis-based fuzzer checks ~10,000 generated datasets —
including degenerate zero-area boxes and other edge cases hand-written tests miss —
against pycocotools at a 1e-10 tolerance.

## Methodology

- **Wall clock time** includes file I/O, evaluation, and accumulation. Excludes Python import time.
- **Core count affects the ratio.** hotcoco evaluates in parallel; pycocotools is
  single-threaded. Speedups therefore scale with the cores available, and the numbers
  here come from an 8-core machine — a 4-core laptop sees less, a 32-core server
  more. Run the suite on your own hardware for a figure that describes it.
- **Detections are synthetic** — generated from GT annotations with a fixed seed (`seed=42`), so AP scores are meaningless but detection count and format are representative of real model output. Fixed seed means results are identical across runs.
- **Only detections are scaled** for the 10x benchmark — ground truth annotations are unchanged.

## Reproducing the benchmarks

These run from a repo checkout — see [CONTRIBUTING](https://github.com/derekallman/hotcoco/blob/main/CONTRIBUTING.md)
for the build. One command fetches the annotations and generates the synthetic
detection files:

```bash
just download-coco   # ~240 MB — val2017 annotations + parity result files
```

That produces:

```
data/
├── annotations/
│   ├── instances_val2017.json
│   └── person_keypoints_val2017.json
├── bbox_val2017_results.json
├── segm_val2017_results.json
└── kpt_val2017_results.json
```

With that in place:

```bash
just bench                                  # speed benchmark (1x)
uv run python scripts/bench.py --phases     # load/eval phase breakdown
uv run python scripts/bench.py --scale 10   # 10x stress test
just parity                                 # metric parity vs pycocotools
just parity-tide                            # TIDE vs tidecv
uv run pytest tests/test_mask_parity.py     # hotcoco.mask vs pycocotools.mask, bit for bit
uv run pytest tests/test_parity_oid.py      # Open Images vs the TF Object Detection API
just fuzz                                   # hypothesis fuzzer, ~10,000 generated datasets
```

The Objects365 benchmark needs a separate download:

```bash
uv pip install polars
just download-o365                              # ~220 MB — O365 val annotations
uv run python scripts/bench_objects365.py
```
