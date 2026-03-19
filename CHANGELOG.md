# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `coco.browse()` dataset browser rewritten: replaced Gradio with FastAPI + HTMX + Jinja2 + vanilla JS Canvas; sidebar with multi-select category filter and shuffle; infinite-scroll thumbnail grid with server-side annotated thumbnails; lightbox with full-resolution canvas overlay for bbox/segmentation/keypoint annotations; hover-to-highlight syncs canvas and annotation sidebar; scroll-to-zoom and drag-to-pan; keyboard navigation (arrow keys, Escape); responsive layout adapts from 400px to 1400px+ viewports; works inline in Jupyter IFrames
- `coco.browse(dt=...)` — detection overlay: GT solid bboxes, DT dashed bboxes, confidence scores on labels; Sources toggle (GT/DT) and Min Score slider for filtering detections
- `coco explore --dt <results.json>` CLI flag — enables detection overlay from the command line
- `coco.browse(port=7860)` — new `port` parameter for custom server port
- `python/hotcoco/server.py` — new FastAPI server module with `create_app()`, `run_server()`, and `start_server_background()` for Jupyter
- `python/hotcoco/static/` — new static assets: `style.css` (responsive dark theme), `overlay.js` (Canvas annotation renderer), `htmx.min.js` (vendored HTMX 2.0.4)
- `python/hotcoco/templates/` — new Jinja2 templates: `base.html`, `index.html`, `partials/gallery.html`, `partials/detail.html`
- `COCO(annotation_file, image_dir=...)` — new `image_dir` constructor arg and settable attribute; propagated through `filter`, `split`, `sample`, and `load_res`
- `--json` flag on every `coco` subcommand (`eval`, `stats`, `healthcheck`, `filter`, `merge`, `split`, `sample`, `convert`) — writes a single JSON object to stdout; intended for CI/CD pipelines, dashboards, and shell scripts; stderr and exit codes are unchanged; errors also emit JSON when the flag is active
- `coco eval --json` suppresses the Rust-side metrics table (via fd-level stdout redirect) and returns `{metrics, params, tide?, slices?, healthcheck?}` — optional keys only present when their flags are passed
- `docs/cli.md` — new "JSON output mode" section with CI gating example and JSON error format; `--json` row added to every subcommand flags table; JSON output shape documented for `eval`

### Fixed

- `ann_to_mask` returned striped/diagonal artifacts for non-rectangular polygon segmentations due to swapped `h`/`w` arguments in the column-major → row-major transpose; masks now match pycocotools exactly

### Changed

- `pip install hotcoco[browse]` optional extra — now pulls in `fastapi>=0.100`, `uvicorn>=0.20`, `jinja2>=3.1`, `Pillow>=8.0` (previously required `gradio>=4.0`)
- `coco explore` CLI — removed `--share` flag (Gradio-specific); added `--dt` flag
- `coco.browse()` return type changed from `gr.Blocks` to `None`; use `create_app()` from `hotcoco.server` for advanced control
- `python/hotcoco/browse.py` — removed Gradio-specific code (`build_app`, `render_annotated_image`, `_require_gradio`, `_build_theme`, `_CSS`); added `prepare_annotation_data()` for client-side canvas rendering
- `python/hotcoco/cli.py` — extracted `_load_res()` helper to deduplicate error handling
- Documentation updated: `docs/guide/browse.md`, `docs/api/coco.md`, `docs/cli.md`, `README.md` — all Gradio references removed
- Internal: removed `Box<dyn Iterator>` in `COCO::get_ann_ids` — extracted filter closure, eliminated heap allocation and dynamic dispatch
- Internal: removed unnecessary `Vec::clone()` in `tide_errors()` (borrowed slices) and `confusion_matrix()` (`Cow<[u64]>` avoids allocation when params are already set)
- Internal: added `IouMatrix` type alias for `Vec<Vec<f64>>` in eval module, removed `#[allow(clippy::type_complexity)]`
- Internal: pre-allocated `Vec`s in `accumulate()` with capacity hints based on total detection count, eliminating repeated reallocations
- Internal: added `#[inline]` to hot mask functions (`area`, `to_bbox`, `intersection_area`)
- `coco --help` and subcommand help: new description and epilog examples on top-level parser and `eval`/`healthcheck` subparsers; `--gt`/`--dt`/`--slices` help text improved; `stats` one-liner updated
- `scripts/test_parity.py` renamed to `scripts/fuzz_parity.py` — clarifies that this is the slow hypothesis-based fuzzer (`just fuzz`), distinct from `scripts/test_parity.py` (the fast CI regression suite, `just test`)
- `scripts/fixtures/adversarial/` added to `.gitignore` and removed from tracking — hypothesis-generated fixtures are ephemeral outputs, not source files; the directory is recreated locally by running `just fuzz`
- `docs/stylesheets/extra.css` — full docs theme redesign: custom CSS variable palettes for light (stone-cream) and dark (cool charcoal) modes; all 12 `--md-code-hl-*` syntax token colors set to a warm editorial palette (dusty steel blue keywords, sage strings, clay numbers, plum functions); admonition type overrides (note/info/warning/tip) with flat tinted backgrounds and no title-bar box artifact; hero pill buttons, feature card lift-on-hover, warm-tinted shadows throughout
- `zensical.toml` — docs theme: `primary`/`accent` palette entries switched to `"custom"`; `navigation.tabs` added to features (top-level sections move to tab bar, freeing sidebar width); `[project.theme.font]` added with `text = "Nunito"` and `code = "IBM Plex Mono"`; color palette shifted from saturated warm-brown to desaturated gray-brown (`#4A4540`) with dusty slate blue accent (`#6B7E9A`) for a cooler, less heavy feel; logo icon updated to `lucide/coffee`

## [0.3.0] - 2026-03-16

### Changed

- Internal: replaced `is_lvis: bool` with `eval_mode: EvalMode` enum (`Coco | Lvis | OpenImages`) across all evaluation branch points; `EvalParams.is_lvis` serialized field renamed to `eval_mode` (string: `"coco"`, `"lvis"`, `"openimages"`); no behavior change — prepares for Open Images evaluation support
- `hotcoco.plot` internal refactor: new `PlotData` dataclass (`python/hotcoco/plot/data.py`) centralises eval extraction from `COCOeval`, exposes `area_idx`, `max_det_idx`, `nearest_iou_idx` helpers and cat-name lookup; all plot functions now consume `PlotData` instead of reaching into `COCOeval` internals directly
- `hotcoco.plot` figure saving: increased output DPI from 150 to 200; added `bbox_inches="tight"` to prevent label clipping on save
- `hotcoco.plot.confusion_matrix`: colorbar now uses `make_axes_locatable` for proportional sizing; normalized matrix clamped to `[0, 1]` with `vmax=1.0`; PR-curve plots switched to `layout="compressed"` for tighter axis packing
- `_annotate_bars` internal helper removed in favour of native `ax.bar_label`

### Added

- Open Images evaluation mode (`oid_style=True` / `COCOeval::new_oid()`): single AP@IoU=0.5, `is_group_of` ignore semantics (no FN penalty), group-of second-pass multi-match, iscrowd re-matching disabled
- `Hierarchy` type — category hierarchy for GT/DT expansion with three construction methods: `from_parent_map`, `from_categories` (supercategory fields), `from_oid_json` / `from_file` / `from_dict` (OID JSON format)
- `Annotation.is_group_of: Option<bool>` field (`#[serde(default)]`) — Open Images group-of flag
- `Params.expand_dt: bool` flag — opt-in DT expansion up the hierarchy (default `false`; GT is always expanded)
- `Hierarchy` Python class with `from_file`, `from_dict`, `from_parent_map`, `ancestors`, `children`, `parent` methods
- `docs/guide/evaluation.md` — "Open Images evaluation" section covering hierarchy, group-of, detection expansion, and the single AP metric
- `docs/api/hierarchy.md` — API reference for the `Hierarchy` class
- `docs/api/cocoeval.md` — updated constructor docs with `oid_style` and `hierarchy` parameters; Rust `new_oid()` constructor
- `docs/api/params.md` — `expand_dt` parameter documented
- `docs/api/plot.md` — documented `pr_curve_iou_sweep`, `pr_curve_by_category`, and `pr_curve_top_n`; these three functions were in `__all__` and importable but had no API reference entries; `pr_curve` section updated to describe it as a convenience dispatcher and to prefer calling the named functions directly
- `docs/guide/masks.md` — warning admonition: `mask.encode()` returns `counts` as `bytes`; must decode to UTF-8 string before passing to `load_res()` or storing in a COCO JSON file
- `docs/getting-started/quickstart.md` — bbox format warning: COCO uses `[x, y, width, height]`, not `[x1, y1, x2, y2]`; silent failure if wrong format is passed; includes conversion snippet
- `docs/guide/evaluation.md` — "Key concepts" subsection before the metrics table with plain-language definitions of AP, AR, IoU, and area ranges (with pixel-scale reference); RLE format explanation and conversion snippet in the Segmentation section
- `docs/guide/pytorch.md` — tip noting that standard torchvision models (Faster R-CNN, RetinaNet, FCOS, etc.) output XYXY boxes and that `CocoEvaluator` converts to XYWH automatically
- `docs/api/cocoeval.md` — `per_class=True` example output showing `"AP/person"`, `"AP/car"` keys in `results()` entry
- `docs/api/coco.md` — Pillow dependency note in `from_yolo()` parameters table
- `docs/cli.md` — `--output / -o` flag documented in `coco-eval` section
- `docs/getting-started/troubleshooting.md` — category ID mismatch diagnostic: how to detect and fix mismatched IDs between GT and DT files
- `hotcoco.plot.report()` — single-page PDF evaluation report: run context block, mode-aware metrics table (correct rows for bbox/segm, keypoints, and LVIS), PR curves at IoU 0.50/0.75/mean, F1 peak tile, and per-category AP chart; "hotcoco" brand mark in header
- `coco eval --report <path>` — saves a PDF evaluation report as a side-output of eval; `--lvis` and `--title` flags added to `coco eval`; requires `pip install hotcoco[plot]`
- `hotcoco.plot` module — publication-quality matplotlib plots for evaluation results: `pr_curve`, `confusion_matrix`, `top_confusions`, `per_category_ap`, `tide_errors`
- `theme` and `paper_mode` parameters on all plot functions — `theme` selects one of three built-in palettes (`"warm-slate"`, `"scientific-blue"`, `"ember"`); `paper_mode=True` forces white figure/axes backgrounds for LaTeX or PowerPoint embedding
- `hotcoco.plot.style(theme, paper_mode)` context manager — apply any theme to custom matplotlib code outside of hotcoco plot functions
- Bundled Inter font (Medium + Bold) in `python/hotcoco/_fonts/` for consistent typography across platforms
- `plot` optional dependency group: `pip install hotcoco[plot]` (matplotlib >= 3.5)
- `docs/guide/plotting.md` — user guide with examples for all 5 plot types, unstyled mode, and subplot composition
- `docs/api/plot.md` — API reference for all plot functions and color palette constants
- Shell completions for `coco-eval` (Rust) — `coco-eval --completions <bash|zsh|fish|elvish|powershell>` prints a completion script to stdout; powered by `clap_complete`
- Shell completions for `coco` (Python) — `pip install "hotcoco[completions]"` enables tab completion via `argcomplete`; `# PYTHON_ARGCOMPLETE_OK` magic comment added to CLI entrypoint
- `docs/getting-started/troubleshooting.md` — covers import conflicts with pycocotools, numpy version issues, detection format mistakes (XYXY vs XYWH, missing fields, unknown image IDs), RLE pitfalls, and all-`-1` metric diagnosis
- `docs/guide/pytorch.md` — full guide for `CocoDetection` and `CocoEvaluator`: transforms, distributed training, multi-iou-type evaluation, migration from torchvision
- `docs/guide/frameworks.md` — Detectron2, MMDetection, RF-DETR integration via `init_as_pycocotools()`; Ultralytics `save_json` workflow; LVIS-based pipeline drop-in via `init_as_lvis()`
- Feature comparison table in `docs/benchmarks.md` — hotcoco vs pycocotools vs faster-coco-eval across installation, parity, LVIS, TIDE, confusion matrix, dataset ops, PyTorch integration, CLI, memory, and license
- `scripts/download_coco.py` — downloads COCO val2017 annotations and generates deterministic parity result files; replaces the old untracked `data/gen_*.py` scripts
- `scripts/download_o365.py` — downloads Objects365 validation annotations from HuggingFace (moved from gitignored `data/`, now tracked)
- `just download-coco`, `just download-o365`, `just download-all` recipes in `Justfile`
- Benchmark data section in `docs/getting-started/installation.md` — one-command setup for COCO val2017 and Objects365 benchmark data via `just download-coco` / `just download-o365`
- Rust examples: `crates/hotcoco/examples/basic_eval.rs` and `custom_params.rs` — runnable end-to-end evaluation examples with `cargo run --example`
- Notebook link surfaced in quickstart "Next steps" and index hero actions
- "Troubleshooting", "PyTorch Integration", and "Framework Integrations" added to `zensical.toml` nav
- `ConfusionMatrix.cat_names` / `confusion_matrix()` dict now includes `"cat_names"` — category names parallel to `cat_ids`, eliminating a manual `load_cats` lookup after computing a confusion matrix
- `EvalResults.hotcoco_version` — records the library version that produced the results file; included in the `results()` dict and saved JSON
- `TideErrors` now derives `Serialize` (Rust) — can be serialized directly with `serde_json`
- Dataset healthcheck — 4-layer validation (structural, quality, distribution, GT/DT compatibility) for COCO annotation files; `coco.healthcheck()` and `coco.healthcheck(dt)` in Python, `healthcheck()` / `healthcheck_compatibility()` in Rust, `coco healthcheck` CLI subcommand
- `--healthcheck` flag on `coco eval` — runs healthcheck before evaluation and prints errors/warnings to stderr
- Sliced evaluation — `COCOeval.slice_by(slices)` re-accumulates metrics for named image-ID subsets (indoor/outdoor, day/night) without recomputing IoU; `--slices <json>` flag on `coco eval` CLI

### Fixed

- `hotcoco.plot.report()`: table caption underline was too far below the caption text; moved from `y=0.0` to `y=0.3` (axes coordinates)
- `hotcoco.plot.report()`: floating-point values in the metrics table, per-category AP table, and PR-curve legend were right-aligned; now left-aligned
- `hotcoco.plot.report()`: PR-curve legend labels now lead with the numeric value (e.g. `0.456  AP50`) so stacked values align correctly regardless of label width
- `_annotate_f1_peak`: guard against all-NaN precision arrays that caused `ValueError` from `nanargmax`
- `evaluate_img_static` (eval/evaluate.rs): detection-side area-ignore flags were not applied when a (image, category) pair had detections but no GT annotations — `dt_ignore_flags` was initialized to all-`false` and only populated inside the `if let Some(iou_mat)` branch, so DTs with area outside the area range were silently treated as false positives instead of being ignored; fixed by initializing `dt_ignore_flags` from `dt_area_ignore` unconditionally; affected APm/APl/APs for images with zero GT for a given category
- `docs/benchmarks.md` feature comparison table: four inaccurate cells corrected — pycocotools Installation changed from "Requires C compiler" to "Prebuilt wheels available (Python 3.9+)"; pycocotools Python versions changed from "3.7+" to "3.9+"; faster-coco-eval License changed from "BSD" to "Apache 2.0"; faster-coco-eval PyTorch changed from "No" to "Yes — TorchVision compatible"
- `docs/guide/results.md` per-category AP Python example: was indexing `ev.stats[0]` (the scalar overall AP) for every category in the loop, printing the same number for every class; fixed to index the precision array by category (`precision[:, :, i, 0, 2]`); promoted `get_results(per_class=True)` as the recommended approach
- `docs/guide/datasets.md` area range comment: `area_rng=[1024.0, 9216.0]` covers medium objects only (32²–96² px²), not "medium-to-large"
- README removed incorrect claim that hotcoco works as a drop-in for Ultralytics YOLO — Ultralytics implements its own internal metrics and does not use pycocotools or faster-coco-eval

### Changed

- Summary table alignment widened from 18 to 22 characters so "Average Precision (AP)" and "Average Recall (AR)" align at the `@` sign across all rows
- Sliced evaluation table uses fixed-width columns (14 chars) with `_overall` values aligned to the integer part of slice metric values for vertical readability
- Healthcheck imbalance label now shows actual category names and counts (e.g., `person: 11,004 / toaster: 9`) instead of a bare ratio
- `accumulate_impl` and `summarize_impl` extracted as `pub(super)` pure functions; `summarize_impl` now accepts `&[MetricDef]` to avoid redundant `build_metric_defs` calls across `summarize()`, `slice_by()`, and `metric_keys()`
- `docs/index.md` feature card updated from "Just pip install / No Cython, no compiler" to "More than a metric / TIDE error breakdown, confusion matrix, per-category AP, and publication-quality plots" — installation ease is no longer a unique differentiator since pycocotools now ships prebuilt wheels; analysis toolkit is the clearer differentiator
- `README.md` opening expanded with a paragraph calling out the diagnostic toolkit (TIDE error breakdown, confusion matrix, per-category AP, F-scores, publication-quality plots with PDF report) as features pycocotools and faster-coco-eval don't have
- Consolidated repo layout: single root `pyproject.toml` (maturin `manifest-path` pattern); Python package source moved from `crates/hotcoco-pyo3/python/` to root `python/`; all scripts moved from `crates/hotcoco-pyo3/data/` to root `scripts/`
- `Justfile` added at repo root with `build`, `test`, `parity`, `bench`, `lint`, `fmt`, `fmt-check`, `download-coco`, `download-o365`, `download-all` recipes — replaces ad-hoc `uv run python ...` invocations
- `EvalResults::to_json_string()` renamed to `to_json()` for consistency with Rust naming conventions

## [0.2.0] - 2026-03-11

### Added

- Objects365 benchmark results (80k images, 365 categories, ~1.2M detections): hotcoco **39×** vs pycocotools and **14×** vs faster-coco-eval; peak committed RAM 8 GB vs 24–30 GB for alternatives
- `bench_objects365.py` now includes pycocotools as a third runner; Windows support (`peak_wset` + pagefile for memory measurement, `.exe` binary name); `_bench_python_runner` shared helper; process-tree memory tracking via psutil
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
- `COCO(dict)` — constructor now accepts an in-memory dataset dict in addition to a file path or `None`
- `COCOeval.f_scores(beta=1.0)` — compute F-beta scores after `accumulate()`; for each (IoU threshold, category) finds the confidence operating point that maximises F-beta, then averages across categories; returns `{"F1": ..., "F150": ..., "F175": ...}` (key prefix reflects beta value); supports arbitrary beta for precision/recall trade-off weighting
- `get_results(prefix, per_class)` — optional `prefix` parameter prepends a path to all metric keys (e.g. `"val/bbox/AP"`), and `per_class=True` adds per-category AP entries keyed as `"AP/{cat_name}"`; returns a flat dict ready for `wandb.log()`, `mlflow.log_metrics()`, or any experiment tracker
- `IouType` now implements `Display` and `FromStr` traits
- `mask.frPyObjects(seg, h, w)` — pycocotools-compatible unified entry point: accepts a list of polygon coord lists, a single uncompressed RLE dict, or a list of uncompressed RLE dicts; returns the same type as input (single dict or list of dicts)
- `mask.encode` now accepts 3-D `(H, W, N)` arrays and returns a list of N RLE dicts (pycocotools batch encoding)
- `mask.decode` now accepts a list of RLE dicts and returns a `(H, W, N)` Fortran-order array (pycocotools batch decoding)
- `mask.area` and `mask.to_bbox` / `mask.toBbox` now accept a single dict or a list of dicts, matching pycocotools batch semantics
- camelCase aliases `frPoly`, `frBbox`, `toBbox` in `hotcoco.mask` matching pycocotools naming
- `mask.iou` now returns a numpy float64 ndarray instead of a nested list
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

### Fixed

- `mask.area()` PyO3 binding now returns native `u64` instead of truncating to `u32`
- `get_results(per_class=True)` index misalignment when a category ID is missing from the GT dataset

### Changed

- Feature comparison table in `docs/benchmarks.md` corrected: faster-coco-eval installation (prebuilt wheels available), metric parity (exact vs pycocotools), LVIS support (`lvis_style=True`), per-class AP (`extended_metrics`), Python version floor (3.7+)
- Parity tolerance claim updated from flat "≤1e-4" to per-type breakdown: bbox ≤1e-4, segm ≤2e-4, keypoints exact
- Benchmark numbers in `README.md` and `docs/index.md` synced to current bench.py output (bbox 0.41s 23×, segm 0.49s 18.6×, kpts 0.21s 12.7×); corrected detection count from ~43,700 to 36,781
- Documentation: added paper citations for COCO eval (Lin et al. ECCV 2014), OKS (cocodataset.org), LVIS (Gupta et al. ECCV 2019), and TIDE (Bolya et al. ECCV 2020 arxiv); area range notation clarified to square pixels (px²); LVIS frequency definition corrected from instance count to training image count
- Pre-commit hook relocated from `hooks/pre-commit` to `.github/hooks/pre-commit` (standard location)
- `crates/hotcoco-pyo3/README.md` converted to a symlink to root `README.md` — always in sync, no manual copy needed
- `.gitignore` tightened: `data/` blanket exclusion replaced with targeted patterns so benchmark scripts and test fixtures are now tracked; `examples/*.ipynb` exempted from `*.ipynb` exclusion
- Deleted stale investigation and one-off run scripts from `data/`
- Simplified Rust internals: extracted shared helpers (`cross_category_iou`, `subset_by_img_ids`, `per_cat_ap`, `metric_keys`, `format_metric`), pre-sized HashMap allocations, pre-computed GT bbox coordinates in `bbox_iou` hot path
- `lvis` moved from runtime dependency to `dev` optional dependency; hotcoco implements the lvis-api interface natively and never imports `lvis` at runtime
- `mask.encode` signature changed: `h` and `w` parameters removed; dimensions are inferred from the array shape. Accepts both Fortran-order (pycocotools convention) and C-order arrays.
- All RLE-returning mask functions (`encode`, `decode`, `merge`, `fr_poly`, `fr_bbox`, `rle_from_string`) now return pycocotools format `{"size": [h, w], "counts": b"..."}` instead of the previous internal format `{"h": h, "w": w, "counts": [ints]}`
- `py_to_rle` now accepts `bytes` counts (pycocotools format) in addition to `str` and `list[int]`
- `integrations.py` segm path simplified — no longer manually converts RLE format; `mask.encode` now returns coco format directly
- Eval internals: split `eval.rs` (2500 lines) into 8 focused submodules — `accumulate`, `evaluate`, `iou`, `summarize`, `tide`, `confusion`, `types`, `mod`; no API change
- Eval performance: greedy matching now uses a linear scan instead of pre-sorted index vectors, eliminating 2×D `Vec` allocations per (image, category) pair; faster for typical COCO (≤5 GTs/cat); `precision_recall_curve` extracted as a shared kernel reused by both `accumulate` and `tide_errors`
- Eval performance: flat IoU matrix, OKS single-pass accumulation, direct index tracking (no HashMaps), area_rng HashMap in accumulate — 4–26% faster depending on dataset scale
- Mask performance: rayon sequential fallback for small D×G (`MIN_PARALLEL_WORK = 1024`), intersection_area early exit, fr_poly allocation reduction — biggest impact on segm (10% on val2017)
- PyO3 error handling: `.unwrap()` → proper `PyValueError` with descriptive messages in convert.rs and mask.rs
- PyO3 safety: mask decode/encode use safe numpy array construction (no unsafe `PyArray2::new()`)
- `tide_errors()` returns `Result<TideErrors, String>` instead of panicking on precondition failure

### Removed

- `hotcoco.loggers` module (`log_wandb`, `log_mlflow`, `log_tensorboard`) — replaced by the `prefix`/`per_class` parameters on `get_results()`, which produce logger-ready dicts without framework-specific wrappers

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
