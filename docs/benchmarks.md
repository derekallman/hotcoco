# Benchmarks

## Feature comparison

| Feature | pycocotools | faster-coco-eval | hotcoco |
|---------|-------------|------------------|---------|
| **Installation** | Requires C compiler | Requires C compiler | Prebuilt wheels — `pip install` just works |
| **Metric parity** | Reference | Within 1e-3 | Within 1e-4 (keypoints exact) |
| **LVIS evaluation** | No | No | Yes — 13 metrics, federated annotation |
| **TIDE error analysis** | No | No | Yes — 6 error types, ΔAP per type |
| **Confusion matrix** | No | No | Yes — cross-category, configurable threshold |
| **F-scores** | No | No | Yes — F-beta at any β |
| **Per-class AP** | Manual only | Manual only | Built-in via `get_results(per_class=True)` |
| **Dataset operations** | No | No | Yes — filter, merge, split, sample, stats |
| **Format conversion** | No | No | Yes — COCO ↔ YOLO |
| **PyTorch integration** | Via torchvision | No | Yes — `CocoDetection`, `CocoEvaluator` |
| **Rust API** | No | No | Yes — native crate on crates.io |
| **CLI** | No | No | Yes — `coco` (Python) + `coco-eval` (Rust) |
| **Results export** | No | No | Yes — JSON with params + metrics + per-class |
| **Memory at scale** | OOM on O365 | ~25 GB on O365 | Sparse eval, completes cleanly |
| **Python versions** | 3.7+ | 3.8+ | 3.9+ |
| **License** | BSD | BSD | MIT |

## Speed benchmarks

**Hardware:** Apple M1 MacBook Air, 16 GB RAM
**Dataset:** COCO val2017 — 5,000 images
**Detections:** 36,781 synthetic (seed=42; AP scores are not meaningful)
**Timing:** Wall clock time, single run
**Versions:** pycocotools 2.0.11, faster-coco-eval 1.7.2, hotcoco 0.1.0

### Results (1x detections)

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|-----------|
| bbox      | 9.46s | 2.45s (3.9×) | **0.41s (23.0×)** |
| segm      | 9.16s | 4.36s (2.1×) | **0.49s (18.6×)** |
| keypoints | 2.62s | 1.78s (1.5×) | **0.21s (12.7×)** |

Speedups in parentheses are vs pycocotools.

### Results (10x detections)

Scaling detections by 10x (~368,000) to test behavior under higher load:

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|-----------|
| bbox      | 34.53s | 5.72s (6.0×) | **1.91s (18.0×)** |
| segm      | 39.91s | 11.91s (3.4×) | **3.42s (11.7×)** |
| keypoints | 16.93s | 16.28s (1.0×) | **1.76s (9.6×)** |

hotcoco scales better at higher detection counts due to multi-threaded evaluation.

## Metric parity

**Dataset:** COCO val2017 — 5,000 images, synthetic detections (included in repository)

All 34 metrics accurate to within 1e-4 of pycocotools:

### Bounding box

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|---------|------|
| AP     | 0.578       | 0.578   | 0.000 |
| AP50   | 0.861       | 0.861   | 0.000 |
| AP75   | 0.600       | 0.600   | 0.000 |
| APs    | 0.327       | 0.327   | 0.000 |
| APm    | 0.707       | 0.707   | 0.000 |
| APl    | 0.918       | 0.918   | 0.000 |
| AR1    | 0.427       | 0.427   | 0.000 |
| AR10   | 0.687       | 0.687   | 0.000 |
| AR100  | 0.701       | 0.701   | 0.000 |
| ARs    | 0.437       | 0.437   | 0.000 |
| ARm    | 0.806       | 0.806   | 0.000 |
| ARl    | 0.960       | 0.960   | 0.000 |

7 of 12 metrics are exact; the remaining 5 differ by less than 1e-4.

### Segmentation

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|---------|------|
| AP     | 0.658       | 0.658   | 0.000 |
| AP50   | 0.923       | 0.923   | 0.000 |
| AP75   | 0.701       | 0.701   | 0.000 |
| APs    | 0.461       | 0.461   | 0.000 |
| APm    | 0.772       | 0.772   | 0.000 |
| APl    | 0.934       | 0.934   | 0.000 |
| AR1    | 0.455       | 0.455   | 0.000 |
| AR10   | 0.746       | 0.746   | 0.000 |
| AR100  | 0.762       | 0.762   | 0.000 |
| ARs    | 0.546       | 0.546   | 0.000 |
| ARm    | 0.859       | 0.859   | 0.000 |
| ARl    | 0.981       | 0.981   | 0.000 |

All metrics accurate to within 2e-4 (shown rounded to 3 decimal places).

### Keypoints

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|---------|------|
| AP     | 0.413       | 0.413   | 0.000 |
| AP50   | 0.606       | 0.606   | 0.000 |
| AP75   | 0.429       | 0.429   | 0.000 |
| APm    | 0.403       | 0.403   | 0.000 |
| APl    | 0.883       | 0.883   | 0.000 |
| AR1    | 0.766       | 0.766   | 0.000 |
| AR10   | 0.975       | 0.975   | 0.000 |
| AR100  | 0.806       | 0.806   | 0.000 |
| ARm    | 0.622       | 0.622   | 0.000 |
| ARl    | 0.963       | 0.963   | 0.000 |

Keypoint metrics are exact.

## Methodology

- **Wall clock time** includes file I/O, evaluation, and accumulation. Excludes Python import time.
- **Detections are synthetic** — generated from GT annotations with a fixed seed (`seed=42`), so AP scores are meaningless but detection count and format are representative of real model output. Fixed seed means results are identical across runs.
- **Only detections are scaled** for the 10x benchmark — ground truth annotations are unchanged.
- Benchmark scripts are in the repository under `crates/hotcoco-pyo3/data/`.

## Reproducing the benchmarks

All benchmark scripts are in `crates/hotcoco-pyo3/data/`. You'll need the COCO val2017 annotation files and a working hotcoco build — see the [installation page](getting-started/installation.md) for setup. Then:

```bash
uv run python data/bench.py              # speed benchmark (1x)
uv run python data/bench.py --scale 10  # 10x stress test
uv run python data/parity.py            # metric parity vs pycocotools
uv run python data/bench_objects365.py  # O365 scale (requires O365 annotations)
```
