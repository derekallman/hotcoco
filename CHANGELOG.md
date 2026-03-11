# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `hotcoco.integrations` — `CocoDetection` and `CocoEvaluator` drop-in replacements for torchvision's detection reference classes; no torchvision or pycocotools dependency required; `CocoEvaluator` supports distributed multi-GPU evaluation via `synchronize_between_processes()`
- GitHub Release job in release workflow — extracts changelog notes for the released version and creates a GitHub Release automatically on each `v*` tag
- `EvalShape` re-exported from the crate root for Rust users who need to index into `AccumulatedEval.precision`/`recall` arrays directly
- `CONTRIBUTING.md` — contributor guide covering build setup, pre-commit hook, parity workflow, and PR process
- `CODE_OF_CONDUCT.md` — Contributor Covenant
- `SECURITY.md` — vulnerability disclosure policy
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates
- `.github/pull_request_template.md` — PR checklist with parity output section
- `examples/coco_evaluation_101.ipynb` — Jupyter notebook: quickstart, per-class AP, F-scores, TIDE error analysis, drop-in replacement, and experiment logging
- `docs/benchmarks.md` — "Reproducing the benchmarks" section with step-by-step clone, build, data setup, and benchmark commands
- CI, PyPI, Crates.io, and MIT license badges in `README.md`
- `COCO(dict)` — constructor now accepts an in-memory dataset dict in addition to a file path or `None`
- `COCOeval.f_scores(beta=1.0)` — compute F-beta scores after `accumulate()`; for each (IoU threshold, category) finds the confidence operating point that maximises F-beta, then averages across categories; returns `{"F1": ..., "F150": ..., "F175": ...}` (key prefix reflects beta value)
- `IouType` now implements `Display` and `FromStr` traits
- `mask.frPyObjects(seg, h, w)` — pycocotools-compatible unified entry point: accepts polygon coord lists, a single uncompressed RLE dict, or a list of uncompressed RLE dicts
- `mask.encode` now accepts 3-D `(H, W, N)` arrays and returns a list of N RLE dicts (batch encoding)
- `mask.decode` now accepts a list of RLE dicts and returns a `(H, W, N)` Fortran-order array (batch decoding)
- `mask.area` and `mask.to_bbox` / `mask.toBbox` now accept a single dict or a list of dicts, matching pycocotools batch semantics
- camelCase aliases `frPoly`, `frBbox`, `toBbox` in `hotcoco.mask` matching pycocotools naming
- `COCO.to_yolo(output_dir)` — export to YOLO label format; writes one `<stem>.txt` per image with normalized `class_idx cx cy w h` lines plus `data.yaml`; returns a stats dict
- `COCO.from_yolo(yolo_dir, images_dir=None)` — load a YOLO label directory as a COCO dataset; reads `data.yaml` for the category list; Pillow reads image dimensions if `images_dir` is given
- `hotcoco::convert::coco_to_yolo` / `yolo_to_coco` — Rust functions backing the above; `YoloStats` and `ConvertError` re-exported from crate root
- `coco convert --from coco --to yolo / --from yolo --to coco` — CLI format conversion subcommand
- `coco eval --tide` — print TIDE error decomposition after standard metrics; `--tide-pos-thr` and `--tide-bg-thr` control IoU thresholds
- `COCOeval.tide_errors(pos_thr=0.5, bg_thr=0.1)` — TIDE error decomposition (Bolya et al., ECCV 2020); classifies every FP into Loc, Cls, Dupe, Bkg, Both, Miss and reports ΔAP; requires `evaluate()` first
- `TideErrors` Rust type with `delta_ap`, `counts`, `ap_base`, `pos_thr`, `bg_thr` fields
- `COCO.load_res()` now accepts a numpy float64 array of shape `(N, 6)` or `(N, 7)` with columns `[image_id, x, y, w, h, score[, category_id]]` — matches pycocotools `loadNumpyAnnotations`
- `COCO::load_res_anns(Vec<Annotation>)` — Rust method for in-memory result loading without a filesystem round-trip
- `COCOeval.confusion_matrix(iou_thr=0.5, max_det=None, min_score=None)` — per-category confusion matrix with cross-category greedy matching; returns `(K+1)×(K+1)` numpy int64 array; no `evaluate()` needed; parallelised with rayon
- `ConfusionMatrix` Rust type with `.get(gt_idx, pred_idx)` and `.normalized()` methods
- LVIS federated evaluation — `COCOeval(..., lvis_style=True)` and `LVISeval` drop-in replacement for lvis-api `LVISEval`; 13 metrics (AP, AP50, AP75, APs/m/l, APr/c/f, AR@300, ARs/m/l@300)
- `init_as_lvis()` — `sys.modules` patch so `from lvis import LVIS, LVISEval, LVISResults` resolves to hotcoco; enables drop-in use in Detectron2 and MMDetection LVIS pipelines
- `LVISResults`, `LVIS` Python aliases matching lvis-api conventions
- `COCOeval.run()`, `.print_results()` methods (lvis-api-style pipeline compatibility)
- `COCO.stats()` — dataset health-check: annotation counts, image dimensions, area distributions, per-category breakdowns
- Dataset operations on `COCO`: `filter`, `merge` (classmethod), `split`, `sample`, `save`
- Python CLI (`coco`) with subcommands: `eval`, `stats`, `filter`, `merge`, `split`, `sample`

### Changed

- `load_res()` now warns on stderr when any annotation references an `image_id` or `category_id` not present in the GT dataset — catches wrong GT split mistakes before they silently produce low metrics
- `accumulate()`, `summarize()`, and `f_scores()` now print a descriptive warning on stderr when called out of order (before `evaluate()` or `accumulate()` respectively)
- `get_results(prefix, per_class)` — optional `prefix` prepends a path to all metric keys (e.g. `"val/bbox/AP"`); `per_class=True` adds per-category AP entries keyed as `"AP/{cat_name}"`; returns a flat dict ready for `wandb.log()`, `mlflow.log_metrics()`, or any experiment tracker
- Pre-commit hook relocated from `hooks/pre-commit` to `.github/hooks/pre-commit` (standard location)
- `crates/hotcoco-pyo3/README.md` converted to a symlink to root `README.md` — always in sync, no manual copy needed
- `.gitignore` tightened: `data/` blanket exclusion replaced with targeted patterns so benchmark scripts and test fixtures are tracked; `examples/*.ipynb` exempted from `*.ipynb` exclusion
- `data/bench_lvis_parity.py` → `data/parity_lvis.py`; `data/bench_tide_parity.py` → `data/parity_tide.py` — consistent `parity_*` naming
- Deleted stale investigation and one-off run scripts from `data/`
- `mask.encode` signature: `h` and `w` parameters removed; dimensions inferred from array shape; accepts Fortran-order (pycocotools) and C-order arrays
- All RLE-returning mask functions now return pycocotools format `{"size": [h, w], "counts": b"..."}` instead of the previous internal format
- `py_to_rle` now accepts `bytes` counts (pycocotools format) in addition to `str` and `list[int]`
- `mask.iou` now returns a numpy float64 ndarray instead of a nested list
- `tide_errors()` returns `Result<TideErrors, String>` instead of panicking on precondition failure
- `lvis` moved from runtime dependency to `dev` optional dependency; hotcoco implements the lvis-api interface natively
- Eval internals: split `eval.rs` (2500 lines) into 8 focused submodules; no API change
- Eval performance: linear scan matching, flat IoU matrix, OKS single-pass, direct index tracking — 4–26% faster depending on dataset scale
- Mask performance: rayon sequential fallback for small D×G, `intersection_area` early exit, `fr_poly` allocation reduction
- PyO3 error handling: `.unwrap()` → `PyValueError` with descriptive messages; safe numpy array construction (no `unsafe PyArray2::new()`)

### Fixed

- `mask.area()` PyO3 binding now returns native `u64` instead of truncating to `u32`
- `get_results(per_class=True)` index misalignment when a category ID is missing from the GT dataset

### Removed

- `hotcoco.loggers` module (`log_wandb`, `log_mlflow`, `log_tensorboard`) — replaced by the `prefix`/`per_class` parameters on `get_results()`, which produce logger-ready dicts without framework-specific wrappers

## [0.1.0] - 2026-02-23

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
