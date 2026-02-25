# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
