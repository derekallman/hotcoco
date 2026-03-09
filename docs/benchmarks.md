# Benchmarks

## Setup

- **Hardware:** Apple M1 MacBook Air, 16 GB RAM
- **Dataset:** COCO val2017 — 5,000 images, 36,781 ground truth annotations
- **Detections:** ~43,700 detections (1x scale)
- **Timing:** Wall clock time, best of 3 runs
- **Versions:** pycocotools 2.0.8, faster-coco-eval 1.6.5, hotcoco 0.1.0

## Results (1x detections)

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|-----------|
| bbox      | 11.79s      | 3.47s (3.4x)     | 0.74s (15.9x) |
| segm      | 19.49s      | 10.52s (1.9x)    | 1.58s (12.3x) |
| keypoints | 4.79s       | 3.08s (1.6x)     | 0.19s (25.0x) |

Speedups in parentheses are vs pycocotools.

## Results (10x detections)

Synthetic benchmark scaling detections by 10x (~437,000 detections) to test behavior at scale:

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|-----------|
| bbox      | 106.27s     | 27.68s (3.8x)    | 4.07s (26.1x) |
| segm      | 184.35s     | 99.73s (1.8x)    | 10.84s (17.0x) |
| keypoints | 42.60s      | 26.54s (1.6x)    | 0.93s (45.8x) |

hotcoco scales better at higher detection counts due to multi-threaded evaluation.

## Metric parity

All 34 metrics accurate to within 1e-4 of pycocotools. Verified on COCO val2017:

### Bounding box

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|-----------|------|
| AP     | 0.382       | 0.382     | 0.000 |
| AP50   | 0.584       | 0.584     | 0.000 |
| AP75   | 0.412       | 0.412     | 0.000 |
| APs    | 0.209       | 0.209     | 0.000 |
| APm    | 0.420       | 0.420     | 0.000 |
| APl    | 0.529       | 0.529     | 0.000 |
| AR1    | 0.323       | 0.323     | 0.000 |
| AR10   | 0.498       | 0.498     | 0.000 |
| AR100  | 0.520       | 0.520     | 0.000 |
| ARs    | 0.308       | 0.308     | 0.000 |
| ARm    | 0.562       | 0.562     | 0.000 |
| ARl    | 0.680       | 0.680     | 0.000 |

7 of 12 metrics are exact; the remaining 5 differ by less than 1e-4.

### Segmentation

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|-----------|------|
| AP     | 0.355       | 0.355     | 0.000 |
| AP50   | 0.568       | 0.568     | 0.000 |
| AP75   | 0.377       | 0.377     | 0.000 |
| APs    | 0.163       | 0.163     | 0.000 |
| APm    | 0.384       | 0.384     | 0.000 |
| APl    | 0.531       | 0.531     | 0.000 |
| AR1    | 0.303       | 0.303     | 0.000 |
| AR10   | 0.462       | 0.462     | 0.000 |
| AR100  | 0.482       | 0.482     | 0.000 |
| ARs    | 0.259       | 0.259     | 0.000 |
| ARm    | 0.521       | 0.521     | 0.000 |
| ARl    | 0.672       | 0.672     | 0.000 |

All metrics accurate to within 2e-4 (shown rounded to 3 decimal places).

### Keypoints

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|-----------|------|
| AP     | 0.669       | 0.669     | 0.000 |
| AP50   | 0.873       | 0.873     | 0.000 |
| AP75   | 0.730       | 0.730     | 0.000 |
| APm    | 0.635       | 0.635     | 0.000 |
| APl    | 0.732       | 0.732     | 0.000 |
| AR1    | 0.291       | 0.291     | 0.000 |
| AR10   | 0.707       | 0.707     | 0.000 |
| AR100  | 0.739       | 0.739     | 0.000 |
| ARm    | 0.685       | 0.685     | 0.000 |
| ARl    | 0.815       | 0.815     | 0.000 |

Keypoint metrics are exact.

## Methodology

- **Wall clock time** includes file I/O, evaluation, and accumulation. Excludes Python import time.
- **Only detections are scaled** for the 10x benchmark — ground truth annotations are unchanged.
- All three tools were verified to produce identical metrics before timing.
- Benchmark scripts are in the repository under `crates/hotcoco-pyo3/data/`.

## Reproducing the benchmarks

### Prerequisites

You'll need:

- Rust (stable, 1.70+) and [uv](https://docs.astral.sh/uv/)
- COCO val2017 annotation and results files — see below

### 1. Build hotcoco

```bash
git clone https://github.com/derekallman/hotcoco.git
cd hotcoco/crates/hotcoco-pyo3
uv venv
uv pip install maturin ".[dev]"
uv run maturin develop --release
```

### 2. Get the data

All required files should be placed in `crates/hotcoco-pyo3/data/`:

```
data/
  annotations/
    instances_val2017.json
    person_keypoints_val2017.json
  bbox_val2017_results.json
  segm_val2017_results.json
  kpt_val2017_results.json
```

Download COCO val2017 annotations from the [COCO dataset page](https://cocodataset.org/#download). For detection results, use any COCO-format model output or generate synthetic ones with the bench script's `--scale` flag.

### 3. Run the speed benchmark

```bash
cd crates/hotcoco-pyo3
uv run python data/bench.py
```

Options:

```bash
uv run python data/bench.py --scale 10      # 10x detections
uv run python data/bench.py --types bbox    # bbox only
uv run python data/bench.py --types bbox segm keypoints
```

### 4. Verify metric parity

```bash
uv run python data/parity.py
```

This runs hotcoco and pycocotools on the same data and prints a diff for all 34 metrics. Expected tolerances: bbox ≤ 1e-4, segm ≤ 2e-4, keypoints exact.

### 5. Large-scale benchmark (Objects365)

To reproduce the O365 memory and speed numbers, you'll additionally need the Objects365 val annotations. The script auto-generates synthetic detections and caches them:

```bash
uv run python data/bench_objects365.py --dt data/objects365_val_synth_det_100per.json
```

Peak memory is measured via subprocess RSS polling — run on a machine with at least 32 GB RAM for faster-coco-eval to complete.
