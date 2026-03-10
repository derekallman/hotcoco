# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Shell completions for `coco-eval` (Rust) — `coco-eval --completions <bash|zsh|fish|elvish|powershell>` prints a completion script to stdout; powered by `clap_complete`
- Shell completions for `coco` (Python) — `pip install "hotcoco[completions]"` enables tab completion via `argcomplete`; `# PYTHON_ARGCOMPLETE_OK` magic comment added to CLI entrypoint
- `docs/getting-started/troubleshooting.md` — covers import conflicts with pycocotools, numpy version issues, detection format mistakes (XYXY vs XYWH, missing fields, unknown image IDs), RLE pitfalls, and all-`-1` metric diagnosis
- `docs/guide/pytorch.md` — full guide for `CocoDetection` and `CocoEvaluator`: transforms, distributed training, multi-iou-type evaluation, migration from torchvision
- `docs/guide/frameworks.md` — Detectron2, MMDetection, RF-DETR integration via `init_as_pycocotools()`; Ultralytics `save_json` workflow; LVIS-based pipeline drop-in via `init_as_lvis()`
- Feature comparison table in `docs/benchmarks.md` — hotcoco vs pycocotools vs faster-coco-eval across installation, parity, LVIS, TIDE, confusion matrix, dataset ops, PyTorch integration, CLI, memory, and license
- Sample data section in `docs/getting-started/installation.md` — downloads COCO val2014 annotations and synthetic results from cocoapi, with a working sanity-check snippet
- Rust examples: `crates/hotcoco/examples/basic_eval.rs` and `custom_params.rs` — runnable end-to-end evaluation examples with `cargo run --example`
- Notebook link surfaced in quickstart "Next steps" and index hero actions
- "Troubleshooting", "PyTorch Integration", and "Framework Integrations" added to `zensical.toml` nav

### Fixed

- README removed incorrect claim that hotcoco works as a drop-in for Ultralytics YOLO — Ultralytics implements its own internal metrics and does not use pycocotools or faster-coco-eval

### Added (previously)

- `ConfusionMatrix.cat_names` / `confusion_matrix()` dict now includes `"cat_names"` — category names parallel to `cat_ids`, eliminating a manual `load_cats` lookup after computing a confusion matrix
- `EvalResults.hotcoco_version` — records the library version that produced the results file; included in the `results()` dict and saved JSON
- `TideErrors` now derives `Serialize` (Rust) — can be serialized directly with `serde_json`

### Changed

- `EvalResults::to_json_string()` renamed to `to_json()` for consistency with Rust naming conventions

## [0.2.0] - 2026-03-11

### Added

- Objects365 benchmark results (80k images, 365 categories, ~1.2M detections): hotcoco **39×** vs pycocotools and **14×** vs faster-coco-eval; peak committed RAM 8 GB vs 24–30 GB for alternatives
- `bench_objects365.py` now includes pycocotools as a third runner; Windows support (`peak_wset` + pagefile for memory measurement, `.exe` binary name); `_bench_python_runner` shared helper; process-tree memory tracking via psutil

### Changed

- Feature comparison table in `docs/benchmarks.md` corrected: faster-coco-eval installation (prebuilt wheels available), metric parity (exact vs pycocotools), LVIS support (`lvis_style=True`), per-class AP (`extended_metrics`), Python version floor (3.7+)
- Parity tolerance claim updated from flat "≤1e-4" to per-type breakdown: bbox ≤1e-4, segm ≤2e-4, keypoints exact
- Benchmark numbers in `README.md` and `docs/index.md` synced to current bench.py output (bbox 0.41s 23×, segm 0.49s 18.6×, kpts 0.21s 12.7×); corrected detection count from ~43,700 to 36,781
- Documentation: added paper citations for COCO eval (Lin et al. ECCV 2014), OKS (cocodataset.org), LVIS (Gupta et al. ECCV 2019), and TIDE (Bolya et al. ECCV 2020 arxiv); area range notation clarified to square pixels (px²); LVIS frequency definition corrected from instance count to training image count

### Added

- `COCOeval.results(per_class=False)` — return serializable evaluation results as a dict; `save_results(path, per_class=False)` writes the same structure as pretty-printed JSON
- `coco-eval --output / -o <path>` — CLI flag to write evaluation results JSON after evaluation (always includes per-category AP)
- `AreaRange` struct in `hotcoco::params` (re-exported from crate root) — replaces the two parallel `area_rng` / `area_rng_lbl` vecs in `Params` with a single `Vec<AreaRange { label, range }>`
- `Params::area_range_idx(label) -> Option<usize>` — label-based lookup helper; eliminates all positional `unwrap_or(0)` fallbacks
- `FreqGroup` enum (`Rare` / `Common` / `Frequent`) and `FreqGroups` struct in `hotcoco::eval::types` — named fields replace the implicit `[Vec<usize>; 3]` index convention for LVIS frequency groups
- `MetricDef.name` field — `metric_keys()` is now derived from the same `Vec<MetricDef>` that drives `summarize()`, eliminating the parallel-list sync risk; `metrics_lvis()` brings LVIS into the unified `MetricDef` path
- `EvalShape` re-exported from the crate root for Rust users who need to index into `AccumulatedEval.precision`/`recall` arrays directly
- `CONTRIBUTING.md` — contributor guide covering build setup, pre-commit hook, parity workflow, and PR process
- `CODE_OF_CONDUCT.md` — Contributor Covenant
- `SECURITY.md` — vulnerability disclosure policy
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates
- `.github/pull_request_template.md` — PR checklist with parity output section
- `examples/coco_evaluation_101.ipynb` — Jupyter notebook: quickstart, per-class AP, F-scores, TIDE error analysis, drop-in replacement, and experiment logging
- `docs/benchmarks.md` — "Reproducing the benchmarks" section with step-by-step clone, build, data setup, and benchmark commands
- CI, PyPI, Crates.io, and MIT license badges in `README.md`

### Changed

- Pre-commit hook relocated from `hooks/pre-commit` to `.github/hooks/pre-commit` (standard location)
- `crates/hotcoco-pyo3/README.md` converted to a symlink to root `README.md` — always in sync, no manual copy needed
- `.gitignore` tightened: `data/` blanket exclusion replaced with targeted patterns so benchmark scripts and test fixtures are now tracked; `examples/*.ipynb` exempted from `*.ipynb` exclusion
- Deleted stale investigation and one-off run scripts from `data/`

- `COCO(dict)` — constructor now accepts an in-memory dataset dict in addition to a file path or `None`
- `COCOeval.f_scores(beta=1.0)` — compute F-beta scores after `accumulate()`; for each (IoU threshold, category) finds the confidence operating point that maximises F-beta, then averages across categories; returns `{"F1": ..., "F150": ..., "F175": ...}` (key prefix reflects beta value); supports arbitrary beta for precision/recall trade-off weighting
- `get_results(prefix, per_class)` — optional `prefix` parameter prepends a path to all metric keys (e.g. `"val/bbox/AP"`), and `per_class=True` adds per-category AP entries keyed as `"AP/{cat_name}"`; returns a flat dict ready for `wandb.log()`, `mlflow.log_metrics()`, or any experiment tracker
- `IouType` now implements `Display` and `FromStr` traits

### Fixed

- `mask.area()` PyO3 binding now returns native `u64` instead of truncating to `u32`
- `get_results(per_class=True)` index misalignment when a category ID is missing from the GT dataset

### Changed

- Simplified Rust internals: extracted shared helpers (`cross_category_iou`, `subset_by_img_ids`, `per_cat_ap`, `metric_keys`, `format_metric`), pre-sized HashMap allocations, pre-computed GT bbox coordinates in `bbox_iou` hot path

### Removed

- `hotcoco.loggers` module (`log_wandb`, `log_mlflow`, `log_tensorboard`) — replaced by the `prefix`/`per_class` parameters on `get_results()`, which produce logger-ready dicts without framework-specific wrappers

### Changed

- `lvis` moved from runtime dependency to `dev` optional dependency; hotcoco implements the lvis-api interface natively and never imports `lvis` at runtime

### Added

- `mask.frPyObjects(seg, h, w)` — pycocotools-compatible unified entry point: accepts a list of polygon coord lists, a single uncompressed RLE dict, or a list of uncompressed RLE dicts; returns the same type as input (single dict or list of dicts)
- `mask.encode` now accepts 3-D `(H, W, N)` arrays and returns a list of N RLE dicts (pycocotools batch encoding)
- `mask.decode` now accepts a list of RLE dicts and returns a `(H, W, N)` Fortran-order array (pycocotools batch decoding)
- `mask.area` and `mask.to_bbox` / `mask.toBbox` now accept a single dict or a list of dicts, matching pycocotools batch semantics
- camelCase aliases `frPoly`, `frBbox`, `toBbox` in `hotcoco.mask` matching pycocotools naming
- `mask.iou` now returns a numpy float64 ndarray instead of a nested list

### Changed

- `mask.encode` signature changed: `h` and `w` parameters removed; dimensions are inferred from the array shape. Accepts both Fortran-order (pycocotools convention) and C-order arrays.
- All RLE-returning mask functions (`encode`, `decode`, `merge`, `fr_poly`, `fr_bbox`, `rle_from_string`) now return pycocotools format `{"size": [h, w], "counts": b"..."}` instead of the previous internal format `{"h": h, "w": w, "counts": [ints]}`
- `py_to_rle` now accepts `bytes` counts (pycocotools format) in addition to `str` and `list[int]`
- `integrations.py` segm path simplified — no longer manually converts RLE format; `mask.encode` now returns coco format directly

### Changed

- Eval internals: split `eval.rs` (2500 lines) into 8 focused submodules — `accumulate`, `evaluate`, `iou`, `summarize`, `tide`, `confusion`, `types`, `mod`; no API change
- Eval performance: greedy matching now uses a linear scan instead of pre-sorted index vectors, eliminating 2×D `Vec` allocations per (image, category) pair; faster for typical COCO (≤5 GTs/cat); `precision_recall_curve` extracted as a shared kernel reused by both `accumulate` and `tide_errors`
- Eval performance: flat IoU matrix, OKS single-pass accumulation, direct index tracking (no HashMaps), area_rng HashMap in accumulate — 4–26% faster depending on dataset scale
- Mask performance: rayon sequential fallback for small D×G (`MIN_PARALLEL_WORK = 1024`), intersection_area early exit, fr_poly allocation reduction — biggest impact on segm (10% on val2017)
- PyO3 error handling: `.unwrap()` → proper `PyValueError` with descriptive messages in convert.rs and mask.rs
- PyO3 safety: mask decode/encode use safe numpy array construction (no unsafe `PyArray2::new()`)
- `tide_errors()` returns `Result<TideErrors, String>` instead of panicking on precondition failure

### Added

- `COCO.to_yolo(output_dir)` — export a COCO dataset to YOLO label format; writes one `<stem>.txt` per image with normalized `class_idx cx cy w h` lines plus `data.yaml`; crowd and no-bbox annotations are skipped; returns a stats dict with `images`, `annotations`, `skipped_crowd`, `missing_bbox`
- `COCO.from_yolo(yolo_dir, images_dir=None)` — load a YOLO label directory as a COCO dataset; reads `data.yaml` for the category list; if `images_dir` is given, Pillow reads image dimensions from disk (requires `pip install Pillow`)
- `hotcoco::convert::coco_to_yolo` / `yolo_to_coco` — Rust functions backing the above; `YoloStats` and `ConvertError` types re-exported from crate root
- `coco convert --from coco --to yolo --input <json> --output <dir>` / `--from yolo --to coco --input <dir> --output <json> [--images-dir <dir>]` — CLI subcommand for format conversion
- `coco eval --tide` — print TIDE error decomposition after standard metrics; `--tide-pos-thr` and `--tide-bg-thr` control the IoU thresholds (defaults: 0.5 and 0.1)
- `COCOeval.tide_errors(pos_thr=0.5, bg_thr=0.1)` — TIDE error decomposition (Bolya et al., ECCV 2020); classifies every FP into six mutually exclusive types (Loc, Cls, Dupe, Bkg, Both, Miss) and reports ΔAP — the AP gain from eliminating each type; requires `evaluate()` first; priority order matches tidecv (Loc > Cls > Dupe > Bkg > Both); Bkg/Both/Dupe ΔAP uses suppression (not flip-to-TP) for correct curve behaviour
- `TideErrors` Rust type with `delta_ap`, `counts`, `ap_base`, `pos_thr`, `bg_thr` fields
- `COCO.load_res()` now accepts three input formats: file path (`str`), list of annotation dicts (`list[dict]`), or a numpy float64 array of shape `(N, 6)` or `(N, 7)` with columns `[image_id, x, y, w, h, score[, category_id]]` — matches pycocotools `loadNumpyAnnotations` convention
- `COCO::load_res_anns(Vec<Annotation>)` — new Rust method for in-memory result loading without a filesystem round-trip
- `COCOeval.confusion_matrix(iou_thr=0.5, max_det=None, min_score=None)` — per-category confusion matrix with cross-category greedy matching; returns `(K+1)×(K+1)` numpy int64 array (rows = GT, cols = predicted, index K = background); standalone, no `evaluate()` needed; parallelised with rayon
- `ConfusionMatrix` Rust type with `.get(gt_idx, pred_idx)` and `.normalized()` methods
- LVIS federated evaluation — `COCOeval(..., lvis_style=True)` and `LVISeval` drop-in replacement for lvis-api `LVISEval`; 13 metrics (AP, AP50, AP75, APs/m/l, APr/c/f, AR@300, ARs/m/l@300); federated FP filtering via `neg_category_ids` / `not_exhaustive_category_ids`
- `init_as_lvis()` — `sys.modules` patch so `from lvis import LVIS, LVISEval, LVISResults` transparently resolves to hotcoco; enables drop-in use in Detectron2 and MMDetection LVIS pipelines
- `LVISResults`, `LVIS` Python aliases matching lvis-api conventions
- `COCOeval.run()`, `.get_results()`, `.print_results()` methods (used by lvis-api-style pipelines)
- `COCO.stats()` — dataset health-check statistics: annotation counts, image dimensions, area distributions, per-category breakdowns
- Dataset operations on `COCO`: `filter`, `merge` (classmethod), `split`, `sample`, `save`
- Python CLI (`coco`) with subcommands: `eval`, `stats`, `filter`, `merge`, `split`, `sample`

## [0.1.0] - 2025-06-15

### Added

- Pure Rust COCO API — dataset loading, indexing, querying (bbox, segmentation, keypoints)
- Full evaluation pipeline with all 12 AP/AR metrics (10 for keypoints)
- Pure Rust RLE encoding/decoding (no C FFI)
- Rayon-based parallel evaluation
- CLI tool (`hotcoco-cli`) with `--no-cats` flag
- PyO3 Python bindings (`hotcoco` package) with numpy interop
- `init_as_pycocotools()` drop-in replacement via `sys.modules` patching
- camelCase aliases for pycocotools API compatibility
- `eval_imgs` and `eval` properties on COCOeval
- MkDocs documentation site with GitHub Actions deployment
- Performance optimizations: fused intersection, analytical `fr_bbox`, pre-computed indexing, in-place precision interpolation (11-26x faster than pycocotools)

### Fixed

- Zero-length RLE run handling in `intersection_area` and `merge_two`
- `iscrowd` vs `gt_ignore` matching bug in evaluation
- RLE string delta encoding parity with maskApi.c
- Segmentation and keypoints metric parity with pycocotools
