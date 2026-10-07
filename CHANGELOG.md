# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Column-form inputs, with no Python dict per annotation.** Building dicts
  had become the dominant cost on the caller's side: about two thirds of
  `StreamingEval.update()`. Three additions, all additive:
  - `StreamingEval.update(images, gt_anns, dt_anns, *, segmentation=None)`
    takes the `(N, 7)` float array `load_res()` accepts as `dt_anns`, with an
    optional list of `N` RLE or polygon entries for `segm`. On 3,000 images of
    300 detections, `update()` takes about a third as long with arrays as with
    dicts. `load_res()` and `update()` read the array through one parser, so an
    `(N, 6)` array means category 1 in both. Every row has a box, so a
    detection's area is the box's, as `load_res()` gives a result with a
    `bbox`, even for `segm`; for mask-area size buckets, pass dicts.
  - `COCO.from_arrays(images, categories, image_ids, category_ids, boxes, *,
    ids, area, iscrowd, segmentation)` builds a dataset equal to `COCO(dict)`
    over the same annotations. On 300,000 annotations it takes 0.015 s where
    building the dicts and calling `COCO(dict)` takes 0.294 s. `categories` is
    a required argument and `area` defaults to each box's `w * h`;
    `segmentation` needs an explicit `area`.
  - `COCO.update_anns(ids=..., area=...)` writes a column of areas by id. On
    300,000 annotations it takes 0.009 s where the dict form takes 0.091 s
    with the dicts built. The `anns` argument of `update_anns` is now
    optional; positional calls are unchanged.

  Based on [#26](https://github.com/derekallman/hotcoco/pull/26) by Jirka
  Borovec.

- **`load_res(array, segmentation=rles)` takes masks with the detection
  array,** one RLE or polygon per row, as `StreamingEval.update()` does, so
  segm results load with no dict per detection. Each row has a box, so its
  `area` is the box's, as for a result dict with a `bbox`; the column form of
  `update_anns` writes mask areas in its place. On 500 COCO val2017 images
  with 46,000 mask detections, an RF-DETR-style bbox+segm evaluation from
  arrays takes about 92 ms where the same evaluation from torchmetrics' dicts
  takes about 310 ms, with identical metrics. *Rust API:* `mask::RleRef`
  borrows an RLE in either spelling, with `area()`, `to_bbox()`,
  `area_and_bbox()`, and `to_rle()`; `Segmentation::rle_ref()`;
  `mask::areas` and `mask::bboxes` for batches; `COCO::update_ann_areas`
  and `COCO::check_ann_ids`.
- **`StreamingEval.merge()`, `to_bytes()`, and `from_bytes()`.**
  A run split across processes can now stream each shard on its own rank,
  `all_gather` the bytes, and merge them on one rank; `finalize()` gives
  exactly what one stream over every image gives.
  `StreamingEval.merge(evaluators)` returns a new evaluator built from the
  evaluators' stored results without matching anything again, as
  `COCO.merge` returns a new dataset. It reads any iterable one evaluator at
  a time, so a generator over `from_bytes` holds one shard beside the result.
  An image seen by more than one keeps the last one's result, and a mismatch
  in categories, mode, or params raises `ValueError` naming the evaluator and
  the field. The state is versioned, only
  what `finalize()` reads is saved, and damaged bytes raise `ValueError`.
  *Rust API:* `StreamingEval::merge`, `to_bytes`, and `from_bytes`;
  `StreamingEval` is now `Clone`.

  Based on [#24](https://github.com/derekallman/hotcoco/pull/24) by Jirka
  Borovec.

- **`just fuzz-torchmetrics`** runs `scripts/fuzz_torchmetrics.py`:
  torchmetrics' `MeanAveragePrecision` with hotcoco swapped in, the way
  RF-DETR evaluates, against its pycocotools backend, requiring every output
  to match exactly. It found the `float32` grid divergence described under
  Fixed. Its dependencies are a new `torchmetrics` dependency group, which
  is not published and which `just setup` does not install.
- *Rust API:* `StreamingEval::unknown_category_ids` returns the category ids in
  a set of annotations that the evaluator was not built with, the same check
  `update()` makes.
- *Rust API:* `hotcoco::mask::area_from_string(s, h, w)` returns the area
  of the RLE a compressed `counts` string encodes without building its run
  list, and fails on exactly the strings `rle_from_string` rejects.
- *Rust API:* `hotcoco::mask::check_counts(counts, h, w)` rejects a run
  list that sums past `h * w`, with the message `rle_from_string` gives for
  a compressed string that does.

### Changed

- **hotcoco requires Python 3.10 or later.** Under the 3.10 stable ABI, the
  bindings read ASCII text in place, which covers COCO keys and RLE `counts`;
  3.9's made two copies of every string. Non-ASCII text gets a UTF-8 copy,
  which CPython keeps on the string object for as long as the string lives.
  Python 3.9 reached end of life in October 2025. The wheel is `cp310-abi3`,
  and pip on Python 3.9 installs 1.1.x.
- **`load_res()` takes an array of any integer or float dtype, not only
  `float64`,** cast once on the way in; so does `StreamingEval.update()`.
  Detectors emit `float32`, and every caller had to convert first.
- **Coding-agent instructions live in `AGENTS.md`,** which Codex and Claude
  Code both read; `CLAUDE.md` is a one-line import of it. The core-crate
  architecture rules moved to `crates/hotcoco/AGENTS.md`, and the docs owner
  map and voice rules moved into `STYLE.md`. Two Claude Code skills are now
  tracked under `.claude/skills/`: `ship` (the pre-commit checklist) and
  `bench` (the benchmark-table procedure). Neither file ships in the crate.
- **`hotcoco.primitives` is the strict layer; `hotcoco.mask` is the
  pycocotools-shaped one.** `primitives.mask_iou` takes RLEs only and
  `primitives.bbox_iou` takes an `(N, 4)` array or 4-element rows;
  both are typed that way in the stubs. `hotcoco.mask.iou` is a lenient
  wrapper over them with pycocotools' type dispatch, and `mask.bbox_iou` is
  the same object as `primitives.bbox_iou`. Before, `primitives.mask_iou` was
  an alias of `mask.iou`, so the two layers could not differ. *Rust API:*
  `COCOeval::print_results_lines()` returns the lines `print_results()`
  prints.
- **`evaluate()` and `accumulate()` are faster, most of all on large result
  sets.** On the benchmark's 10× COCO val2017 detections (368,000), a bbox
  evaluation end to end takes 0.14 s where it took 0.22 s, and segm 0.24 s
  where it took 0.33 s; on the 1× set, bbox takes 0.04 s where it took
  0.06 s. Peak memory at 10× is 208 MB where it was 241 MB for bbox, and
  372 MB where it was 404 MB for segm. Measured on an M1 the same day,
  before and after the change; `docs/benchmarks.md` has every cell. The metrics are unchanged: val2017
  parity and the four fuzzers agree with pycocotools as before.
  - An (image, category) pair with detections but no ground truth, or ground
    truth but no detections, skips the matcher, since nothing in it can
    match. That is 96% of the pairs at 10×, where `evaluate()` takes about
    27 ms instead of 87 ms.
  - `evaluate()` lists the pairs to visit by merging each image's
    categories from the two annotation indexes, already in order, instead
    of collecting them in a hash set and sorting it: about 2 ms instead of
    30 ms on one thread at 10×.
  - `accumulate()` groups the evaluated cells by category with one counting
    sort, stores each detection's matched and ignore flags for every IoU
    threshold side by side so it reads them in one go, and sorts
    detections on integer keys: about 34 ms instead of 49 ms at 10×.
- **A detection array loads faster, and large result sets are freed off the
  caller's thread.** On RF-DETR's `compute()` benchmark — 5,000 images of 300
  box detections, `max_dets` [1, 10, 500], handed over as one array as
  RF-DETR's #1584 does — `compute()` takes about 0.54 s where it took about
  0.68 s, within about 5% of ultrafast-pycocotools in the same runs. Measured
  on an M1, alternating the two:
  - `load_res` converts the array's rows to records in parallel, validates
    them and assigns ids in parallel passes, and its index reads each record
    once: about 70 ms where it took about 96 ms for 1.5 million rows.
  - `accumulate()` ranks a category's detections once for all its area ranges
    instead of once per area range: about 72 ms where it took about 103 ms.
  - `evaluate()` matches each (image, category, area range) cell in buffers
    it reuses rather than about a dozen new vectors, stops scanning a
    threshold's detections once no ground truth is left to match, and keeps
    each cell's IoU matrix in one buffer rather than one vector per
    detection: about 65 ms where it took about 105 ms, and about 30 MB less
    memory.
  - Dropping a `COCO` with 100,000 annotations or more, or a `COCOeval` with
    100,000 evaluated pairs or more, frees their large buffers on the thread
    pool rather than the calling thread, where Python held the GIL for about
    25 ms per 1.5 million records. The memory comes back just after the drop
    rather than during it.

  *Rust API:* `COCO::detections_from_rows` is the row conversion. `COCO` and
  `COCOeval` implement `Drop`, so a field can no longer be moved out of one;
  `std::mem::take(&mut coco.dataset)` takes the dataset.
- **`mask.encode`, `mask.area`, and reading and writing `COCO` records are
  faster on the calls TorchMetrics makes for every validation batch.**
  Measured on 46,000 full-image masks and their detections from COCO val2017
  on an M1:
  - `mask.encode` of one Fortran-order mask, TorchMetrics' `update()` call,
    takes about 8 µs where it took about 220 µs; pycocotools takes about
    120 µs. A Fortran-order mask is read in place, runs are found 32 and 8
    bytes at a time, and a C-order mask goes through an 8×8-block transpose,
    which `COCO.ann_to_mask` now uses as well. The dtype is checked without
    calling into Python, so an 8×8 mask takes about 0.3 µs a call where it
    took about 1.5 µs.
  - `mask.area` of one RLE dict takes about 0.7 µs where it took about
    1.5 µs. It sums the runs as the `counts` string decodes, and every other
    reader of a compressed `counts` string (`decode`, `toBbox`, `iou`,
    `merge`, `frPyObjects`, segm evaluation) shares the faster decoder.
  - `COCO(dict)` takes about 32 ms where it took about 130 ms on
    TorchMetrics' 46,000 prediction dicts, and about 37 ms where it took
    about 50 ms on COCO val2017's ground truth loaded with `json.load`
    (36,781 annotations). A custom value converts in Rust when it is a
    `None`, `bool`, `int`, `float`, or `str` (a subclass such as
    `numpy.float64` or an `IntEnum` member included), or a `list`, `tuple`,
    or `str`-keyed `dict` of those, up to 32 levels deep: 46,000 synthetic
    box annotations, each with an `attributes` dict, load in about 40 ms
    instead of about 120 ms. A record with any other custom value goes
    through `json.dumps` as before, so what is stored, and any error raised,
    is unchanged.
  - Reading records back (`dataset`, `load_anns`, `anns`, and the image and
    category forms) builds custom values directly instead of through
    `json.loads`, and reuses one key object per field: `load_anns` of the
    46,000 annotations with two custom keys takes about 75 ms where it took
    about 140 ms.
  - Image and category dicts are read in one pass over their keys, as
    annotation dicts are: 50,000 COCO val2017 image dicts load in about
    19 ms where they took about 39 ms, and 50,000 bare `{"id": n}` images,
    as TorchMetrics builds them, in about 5 ms where they took about 9 ms.
- **`mask.toBbox`, and `load_res` of mask results without a `bbox`, read
  the box straight off a compressed `counts` string.** The area and the box
  come out of one pass as the string decodes, with no run list built. On
  COCO val2017's 43,691 segm results as compressed RLEs, `toBbox` takes about
  27 ms where it took about 67 ms, and `load_res` without boxes about 19 ms
  where it took about 27 ms. *Rust API:* `mask::area_and_bbox_from_string`.
- **Every other array argument reads any integer or float dtype in one pass
  too:** the boxes of `COCO.from_arrays`, `mask.iou`, `primitives.bbox_iou`,
  and `mask.frPyObjects`, the cost matrix of `primitives.lsap`, the `area`
  column, and the scores and curves the `hotcoco.metrics` functions take.
  numpy casts the array to `float64` once. Before, any array but `float64`,
  `float32` straight from a detector included, went through Python one
  element at a time: `COCO.from_arrays` with 100,000 `float32` boxes takes
  about 5 ms where it took about 45 ms. The cast is exact for `float32` and
  for integers up to 2^53, so the results are unchanged. The id columns of
  `COCO.from_arrays` and `update_anns` and the `iscrowd` flags read any
  integer dtype in one pass too, an unsigned one as `uint64`, so a `uint64`
  id above 2^63 is kept exactly. *Rust API:*
  `primitives::sim::{mask_iou_flat, bbox_iou_flat}` are the IoU kernels as
  one row-major buffer, what `mask.iou` now hands numpy with no copy.
- **`mask.area` and `mask.toBbox` decode a list of 1,024 RLEs or more across
  threads,** and `update_anns(ids=, area=)` writes the areas in place instead
  of copying each annotation, segmentation included. On 46,000 detection
  masks from COCO val2017, `mask.area` takes about 12 ms where it took about
  27 ms, and `update_anns` of their areas about 0.4 ms where it took about
  6 ms.
- **One reader for every RLE dict.** A record's `segmentation` and the
  `hotcoco.mask` functions read RLE dicts through the same code, so they take
  the same spellings and raise the same errors: a record now also takes
  `{"h": h, "w": w, "counts": ...}`, from a dict or a JSON file, a missing
  key raises `ValueError: RLE dict missing 'size'` from either, and a
  `counts` that is not a string or a list of ints raises the same
  `TypeError` from either.
- **`mask.encode` holds the GIL while it reads the numpy buffer.** Releasing
  it for a scan of about 10 µs cost about 2.5 ms per mask whenever another
  Python thread was busy, waiting out the switch interval to get it back;
  pycocotools holds it too. An `(H, W, N)` stack that takes longer releases
  it once the masks are copied into hotcoco's own buffer: a C-order stack of
  16 MiB or more after its transpose, and a Fortran-order stack after about
  5 ms of encoding in place. A 1024×1024×100 stack of noise masks held the
  GIL for about 80 ms instead of about 320 ms in C order, and about 11 ms
  instead of about 255 ms in Fortran order.
- **`mask.encode` names the type of an array-like that is not a numpy
  array,** such as a pandas `Series`, instead of an error about its numpy
  dtype or dimensions.

### Fixed

Each entry has a regression test that fails on the old code. Most came out of
a whole-project review on 2026-10-02, which looked for places where hotcoco
produced a plausible number instead of the right one, or instead of an error.

- **`StreamingEval.update()` raises `KeyError` for a category it was not built
  with.** A ground truth or detection whose `category_id` is not in
  `categories` used to drop out of every metric without a trace, so an
  off-by-one class map or a background id looked like a model that never
  predicted that class. The error names every unknown id, rejects the batch
  whole, and leaves the evaluator as it was. A batch `COCOeval` still drops
  such an annotation silently. A category that is listed but excluded by
  `params.cat_ids` is not an error, and with `use_cats` false nothing is
  checked. Based on [#23](https://github.com/derekallman/hotcoco/pull/23) by
  Jirka Borovec. *Rust API:* `StreamingEval::update` returns
  `Error::UnknownCategoryIds` for it, the way `COCO::update_anns` returns
  `Error::UnknownAnnIds`.
- **`load_res()` raises `ValueError` for an array row whose `image_id` or
  `category_id` is NaN or negative.** The float was cast to an integer that
  saturated at 0, so such a row became image or category 0, which is a real id
  in some datasets. pycocotools' `int()` raises on NaN as well.
- **A `float32` threshold grid gets one warning that says what it is.**
  torchmetrics builds both grids with `torch.linspace`, and read back as
  `float64` they sit up to 4e-8 from the default. `summarize()` gave two
  warnings for them, and one was wrong: it said the AP50 and AP75 lines might
  show -1.000, but 0.5 and 0.75 are exact in `float32`. It now gives one
  warning, saying the grid is the default rounded through `float32`, is
  evaluated as given as pycocotools evaluates it, and should be `float64` for
  the reference numbers. The rounding is not only noise: on a category with
  20 ground truths a `float32` recall grid moves 240 of 12,120 precision
  cells, by up to 0.33, because recall `k / n` lands exactly on a grid point
  that is one ulp too high to include it. A grid counts as rounded when it
  has the default's length and every point is within 1e-6 of the default.
  Based on [#23](https://github.com/derekallman/hotcoco/pull/23) by Jirka
  Borovec.
- **`pickle`, `copy.copy`, and `copy.deepcopy` work on a `StreamingEval`.**
  Each raised `TypeError`. They now go through the `to_bytes()` state, and
  `StreamingEval.__module__` is `hotcoco`, which pickle needs to find the
  class again. Based on [#24](https://github.com/derekallman/hotcoco/pull/24)
  by Jirka Borovec.
- **`hotcoco.__version__` exists.** It was missing, so
  `hasattr(hotcoco, "__version__")` was `False` and the install check in
  CONTRIBUTING.md raised `AttributeError`. It names the compiled extension
  that is loaded, and equals `importlib.metadata.version("hotcoco")` for an
  installed wheel. Based on
  [#23](https://github.com/derekallman/hotcoco/pull/23) by Jirka Borovec.
- **An RLE whose counts stop short of `h × w` no longer scores IoU above 1.**
  pycocotools accepts such a mask (`decode` fills the missing tail with
  background), but hotcoco's run-stream walkers kept the last run's value over
  the tail, so a detection `{"size": [10, 1], "counts": [0, 2]}` against a
  full-column ground truth returned IoU 5.0 and cleared every threshold.
  `mask.iou`, `mask.merge`, and the segm evaluation now read the tail as
  background, as `decode` does.
- **`mask.iou` returns `-1` for masks of different sizes,** as pycocotools'
  `rleIou` does, instead of a plausible overlap from walking two run streams
  of different geometry.
- **A box rasterizes to the same mask as its corner polygon.** `fr_bbox`
  filled `floor(x)..ceil(x + w)` analytically while pycocotools' `rleFrBbox`
  draws the four corners through `rleFrPoly`; a fractional box such as
  `[0.5, 0.5, 2, 2]` got area 9 instead of 4. Inside a segm evaluation a
  box-only ground truth took the analytic path while its detection took the
  polygon path, so identical fractional boxes scored IoU 0.44 against each
  other and missed at 0.5. `fr_bbox` now routes through `fr_poly`; integer
  boxes are unchanged.
- **A segm `COCOeval.evaluate()` raises `ValueError` when an image whose
  annotations are polygons or boxes has no `height` and `width`.** The fields
  are optional on an image record and defaulted to 0, so every polygon
  rasterized onto a 0×0 canvas and segm AP came out 0.000 with no warning.
  pycocotools raises `KeyError` at the same point. RLE carries its own size
  and box evaluation never reads the fields, so neither is affected. Only the
  annotations the evaluation covers are checked, the ones pycocotools
  rasterizes: an annotation whose `image_id` has no image record, on an image
  outside `params.img_ids`, or in a category outside `params.cat_ids` (with
  `use_cats` on) is ignored as before.
  `StreamingEval.update()` and `coco eval` run the same check in segm mode.
  *Rust API:* `COCOeval::check_inputs` (call it before `evaluate()`),
  `COCO::check_mask_dims(img_ids, cat_ids)`, and `COCO::ann_to_rle` returns
  `None` rather than an empty mask for such an annotation.
- **`StreamingEval.update()` assigns ground-truth ids and derives a missing
  `area`.** Targets in the shape a data loader yields carry neither. Without
  ids, every ground truth in a batch shared id 0 and the id index resolved all
  of them to the batch's last annotation — possibly in another image and
  category — so matching ran against the wrong box. Without `area`, the
  matcher read 0 and put every object in `small`, so APm and APl were `-1` or
  counted every detection as a false positive while APs counted everything.
  Ids are now assigned per batch (nothing after `update()` reads them), and a
  missing `area` is derived by the rule in the next entry, starting from the
  mask's pixel count, COCO's definition of instance area; an authored `area`
  is kept. In segm mode each such polygon is rasterized once and the mask
  reused for matching. *Rust API:* `COCO::fill_missing_areas`.
- **With `use_cats` off, tied scores rank as pycocotools ranks them.**
  pycocotools lists an image's annotations category by category, in
  `cat_ids` order, before its stable sort by score, and leaves out a category
  not in `cat_ids`; hotcoco listed them in load order and kept every
  category. A true positive and a false positive at the same score in two
  categories could swap ranks: AP 1.0 where pycocotools gives 0.5. Ground
  truths are listed the same way, which decides which of two equal-IoU
  matches wins.
- **`-0.0` and `0.0` tie in `metrics.average_precision` and in TIDE,** as in
  `COCOeval` and pycocotools. `average_precision` ranked `0.0` first, and
  TIDE's ranking was not a total order with `NaN` present.
- **`COCOeval` derives a missing `area`, as `StreamingEval` does.** An
  annotation with no `area` read as 0, so every such object landed in
  `small`, while `StreamingEval` derived it: the same annotations gave two
  answers. Both evaluators now fill in their own copy of the datasets, and
  the dataset you pass keeps its missing areas. A ground truth takes its
  mask's pixel count, then its rotated box's `w × h`, then its box's, then
  the extent of its labeled keypoints; a mask of no pixels, such as
  `"segmentation": []`, does not count. A detection takes the order
  `load_res` derives a result's area in: box, mask, keypoints, rotated box.
  pycocotools raises `KeyError` here instead.
- **LVIS frequency-group AP (`APr`/`APc`/`APf`) and the per-class table read
  the K axis the accumulation was built on.** Both used `params.cat_ids` as
  it stood at `summarize()` time, which disagrees with `accumulate()`'s axis
  whenever `cat_ids` is narrowed or reordered between the two calls, and
  whenever `use_cats` is false (one pooled slot). The frequency groups then
  averaged another category's precision or indexed past the end — a panic for
  a streaming LVIS run with `use_cats=False` — and `get_results(per_class=True)`
  / `results(per_class=True)` reported the pooled class-agnostic AP under the
  first category's name. `AccumulatedEval` now records its K axis and both
  consumers read it; a pooled run has no per-class entries. *Rust API:*
  `AccumulatedEval::cat_ids`.
- **`coco convert --from coco --to yolo|voc|oid` no longer exits 1 after
  writing its output.** The summary table read a `missing_bbox` stat the
  converters never return, so every non-`--json` run ended with
  `error: 'missing_bbox'`. It now reads the real `skipped_no_bbox` count, and
  the CVAT summary also reports `skipped_degenerate` polygons.
- **`COCO.from_cvat` keeps a label's own name.** Any `<name>` inside a
  `<label>` overwrote the label name, including `<attribute><name>`, so a
  label with attributes was recorded under its last attribute's name — a
  spurious category, the real one moved out of declared order, and every id
  shifted. Every CVAT export with label attributes hit this.
- **`COCO.from_voc` raises `ValueError` naming the file and position for an
  `<object>` with an empty or missing `<name>`, no `<bndbox>`, a missing
  coordinate, or an inverted box.** A missing coordinate defaulted to 0, so a
  missing `ymax` produced the box `[9, 9, 41, -9]`.
- **`healthcheck` flags `bbox_out_of_bounds` for a box past the top or left
  edge** (negative `x` or `y`), not only the right or bottom edge.
- **`accumulate()` after changing `params.iou_thrs` reads each threshold's own
  matches.** The T axis was sized from the current grid while each cell holds
  one row per evaluate-time threshold, so a grid reordered after `evaluate()`
  reported the 0.5 row as AP75, and a longer grid read past the rows (a panic
  in debug builds). Each threshold now resolves by value to the row
  `evaluate()` matched it on, as area ranges already did, and a threshold that
  was never evaluated reports `-1.0`; run `evaluate()` again to compute it.
  `calibration()`, `tide_errors()`, and `image_diagnostics()` read the same
  rows the same way — each read its row by position in the current grid —
  and raise `RuntimeError` for a threshold that was never evaluated.
- **`compare()` raises `ValueError` when the two evaluators differ in
  `use_cats` or in their set of `cat_ids`,** instead of differencing and
  bootstrapping a one-category AP against an 80-category mAP. The same
  categories in another order still compare: the per-category table pairs by
  id, and the means do not depend on order.
- **`calibration()` omits a category with no counted detections from
  `per_category`** instead of reporting ECE `0.0` for it. An empty set of
  detections has no calibration error to report; `0.0` read as perfect.
- **Open Images hierarchy expansion keeps each detection's own score on its
  ancestor copies.** The copy-deduplication key — there so that a second
  `evaluate()` does not expand again — omitted the score and the mask, so with
  `expand_dt` two children on one box (cat@0.3, dog@0.9) yielded a single
  ancestor copy at whichever score came first in the file, and
  segmentation-only annotations without a bbox all collapsed into one copy per
  image. Each annotation now expands on its own, as the TF Object Detection
  API expands each box, so two ground truths on one box in sibling classes
  are two ancestor ground truths, not one. A copy is skipped only when the
  input already holds an identical annotation — score, `iscrowd`, and
  `is_group_of` included — so a pre-expanded dataset and a second
  `evaluate()` stay as they are.
- **Polygons reaching far outside the image rasterize as pycocotools does.**
  Vertices were clamped to one image extent past the edges — there to keep
  untrusted coordinates like `±1e9` from overflowing the edge walk — but the
  clamp bent every edge that crossed it, so `[0, 0, 300, 100, 0, 100]` on a
  100×100 image got area 7530 instead of 8350. Vertices within about ±4
  million pixels now rasterize bit-identically, and only coordinates beyond
  that are clamped. The boundary walk skips the stretches that cannot reach
  the image, so its cost stays bounded by the image size: a vertex at
  `x = 1e6` on a 640×480 image takes 0.1 ms and no extra memory.
- **`compare()` bootstrap confidence intervals use `numpy.quantile`'s default
  linear percentile.** The old indices put the upper bound one rank too high,
  so the interval was asymmetric around the point estimate. Bounds are now
  interpolated and bit-identical to numpy's.
- **`mask.frPyObjects` decides box-versus-polygon once, from the first entry,
  as pycocotools does.** It decided per entry, so a 4-value entry after a
  polygon was a box where pycocotools makes it a degenerate polygon (area 0).
  Lists of boxes are still accepted (pycocotools' box path needs an array).
- **`tide_errors()` matches tidecv's Loc, Cls, and Miss ΔAP definitions.**
  Loc and Cls follow tidecv's best-GT-match rule: an error aimed at an
  already-matched ground truth is suppressed instead of becoming a TP, only
  the highest-scoring error per missed ground truth is promoted, and a fixed
  Cls error counts as a TP in the ground truth's category, not the
  detection's. Every such error used to become a TP, which could push recall
  past 1.0. Fixing a Miss now removes the ground truth from the recall
  denominator, as tidecv does, instead of injecting a top-scoring TP. On
  val2017: Loc 0.1115 → 0.1036, Cls 0.0002 → 0.0000, Miss 0.0242 → 0.0148.
  On images without crowd regions every ΔAP now matches tidecv within ±0.005;
  the remaining full-set Loc and Miss gaps are the documented crowd-handling
  difference. The old full-set Loc agreement was two errors cancelling —
  over-promotion against tidecv's crowd-region Loc errors — so
  `scripts/parity_tide.py` now gates the crowd-free comparison at ±0.005 and
  bounds the full set.
- **`CocoEvaluator.synchronize_between_processes()` works under NCCL.** It
  gathered pickled results as CPU byte tensors with `all_gather`, which NCCL
  rejects, so every multi-GPU run failed there. It now uses
  `torch.distributed.all_gather_object` and returns at once when the world
  size is 1.
- **The browse dashboard no longer returns 500 on an evaluator that has only
  been evaluated.** `coco.browse(dt=...)`, `coco explore`, and `browse(eval=ev)`
  all construct one that way; `/dashboard` now accumulates and computes the
  summary on first request, without printing, in a worker thread so the
  gallery keeps being served, and once even when two first requests overlap.
- **The browse server escapes JSON it inlines into pages.** `categories` and
  `slice` query values and category names reached an inline `<script>` block
  unescaped, so a crafted URL could inject script on the local server. Inlined
  JSON now escapes `<`, `>`, `&`, U+2028, and U+2029, and navigation query
  strings are URL-encoded.
- **Drop-in gaps against pycocotools, all reproduced side by side:**
  `COCO()` accepts any `os.PathLike` such as `pathlib.Path`; `getAnnIds`
  accepts `iscrowd=0`/`1` (and numpy ints or bools) and `areaRng=[]` for no
  area filter; `mask.merge(rles, 1)` accepts an int `intersect`; the id-list
  arguments of `getAnnIds`, `getCatIds`, `getImgIds`, `loadAnns`, `loadCats`,
  and `loadImgs` (and their snake_case forms) accept any iterable of ints,
  such as a `set` or `dict.keys()`. Each raised `TypeError` or `ValueError`
  where pycocotools works. The snake_case `get_ann_ids` keeps its typed
  `bool`/two-element forms.
- **`mask.iou` accepts boxes, as `pycocotools.mask.iou` does:** an `(N, 4)`
  array of any numeric dtype or a list of `[x, y, w, h]` rows, dispatching the
  way pycocotools does; mixing boxes and RLEs raises `TypeError`. Custom
  `COCOeval` subclasses that call `maskUtils.iou` from `computeIoU` work again
  under `init_as_pycocotools()`.
- **`COCOeval.run()` and `print_results()` print through `sys.stdout`,** so
  `contextlib.redirect_stdout`, pytest's `capsys`, and notebook cells capture
  the table; both wrote with Rust `println!` to the process's fd 1, which none
  of those see. `run()` now emits the same comparability `UserWarning`s as
  `summarize()`, and `print_results()` before `summarize()` emits a
  `UserWarning` instead of writing to fd 2.
- **LVIS evaluation caps detections at 300 per image across all categories,**
  as lvis-api's `LVISResults` does, instead of 300 per `(image, category)`
  cell. Ties keep results-file order. Numbers change only for images with more
  than 300 detections; `tests/test_parity_lvis.py` now has one. `LVISeval`
  applies the cap when constructed, and `LVISResults` now honors its
  `max_dets` argument instead of ignoring it. As in lvis-api, `LVISeval`
  takes an `LVISResults` result as is, so `max_dets=1000` or `-1` holds
  through evaluation; a plain `load_res()` result gets the 300 cap. The
  cap is 300 whatever `params.max_dets` says, as `LVISEval` wraps raw results
  in `LVISResults` without passing its own `max_dets`; `StreamingEval` in
  LVIS mode applies the same 300. `LVISResults` takes `max_dets=None` (no
  cap, like `-1`) and an integral float such as `300.0`. New
  `COCO.cap_detections_per_image(max_det)` (Rust:
  `COCO::cap_detections_per_image`, and `cap_detections_per_image_shared` for
  a set behind an `Arc`) exposes the cap; `None` keeps every detection. It
  copies the dataset only when the cap changes what LVIS evaluates.
- **RLE `counts` given as a `bytearray` or `memoryview` raises `TypeError`.**
  Every `mask` function and `COCO(dict)` read it as a list of ints, so the
  bytes of the compressed string became run lengths: `mask.area` returned 308
  for a 12-pixel mask, with no error. pycocotools raises `TypeError` too.
- **An RLE dict spelled `{"h": h, "w": w, "counts": ...}` decodes compressed
  `counts`.** `bytes` counts in that spelling were read as a list of ints,
  the same silently wrong mask, and `str` counts raised `TypeError`. Both now
  decode as they do in the `{"size": [h, w], "counts": ...}` spelling.
- **A run list (`counts` as a list of ints) that sums past `h × w` raises
  `ValueError` in every `mask` function,** as the same runs as a compressed
  string already did. `mask.area({"size": [2, 2], "counts": [0, 100]})`
  returned 100 for a 4-pixel mask. pycocotools has no consistent answer:
  its `frPyObjects` accepts the list, `area` and `toBbox` then return
  numbers, `decode` raises, and `iou` and `merge` loop forever. hotcoco's
  `frPyObjects` raises instead. Runs that stop short of `h × w` are still
  accepted.
- **`None` in an optional field reads as absent in `COCO(dict)`,
  `load_res()`, and `update_anns()`,** as `null` does when the same JSON is
  loaded from its path. These are `license`, `coco_url`, `flickr_url`, and
  `date_captured` on an image; `supercategory`, `skeleton`, `keypoints`, and
  `frequency` on a category; and `bbox`, `area`, `segmentation`, `keypoints`,
  `num_keypoints`, `obb`, `score`, and `is_group_of` on an annotation. A
  `json.load`-ed file with `null` in one raised `TypeError`; pycocotools
  takes it. In `update_anns()`, `None` clears the field.

## [1.1.0] - 2026-10-01

### Added

- **`COCO.update_anns`** edits annotations that are already loaded, without
  rebuilding the dataset: each dict is merged into the annotation with the
  same `id`, so `{"id": 1, "area": 5000.0}` is a one-field edit, and an unknown
  id raises `KeyError` instead of being skipped. `coco.dataset` returns a copy,
  so an edit through it never landed, and assigning the whole dataset back was
  the only way — too coarse for evaluating one dataset under several IoU
  types, where each annotation's `area` follows the box or the mask. A key
  outside the COCO schema is a custom key; adding one an annotation does not
  carry yet needs `create=True`, so a misspelled schema field
  (`"Area"`, `"iscrowed"`) raises instead of landing quietly beside the field
  you meant to change. The API reference has the contract. *Rust API:*
  `COCO::update_anns` and `Error::UnknownAnnIds`. Based on
  [#8](https://github.com/derekallman/hotcoco/pull/8) by Jirka Borovec.

- **`StreamingEval`** evaluates as detections are produced. `update(images,
  gt_anns, dt_anns)` matches a detector batch as soon as its predictions
  exist — between training steps, overlapped with postprocessing — and
  `finalize()` returns an ordinary `COCOeval` for `accumulate()`,
  `summarize()`, and `report()`, so loading a results file and `evaluate()`
  leave the end of the epoch. Each batch runs through the same `evaluate()`
  the whole-dataset path runs, and only its lean cells are kept, so the
  numbers are identical to a batch run over the same annotations and memory
  is about 20 bytes per detection: on Objects365 val with 1M detections,
  peak memory above the loaded ground truth drops from 739 MB to 270 MB.
  The evaluation guide has the loop and the API reference has what the
  finalized evaluator supports. *Rust API:* `hotcoco::StreamingEval` and
  `EvalMode::default_params`. Based on
  [#22](https://github.com/derekallman/hotcoco/pull/22) by Jirka Borovec.

- **`metrics::counts::precision_recall_curve_of_order_into`** — the interpolated
  precision-recall curve straight from ranked match flags, without the
  cumulative TP/FP arrays `precision_recall_curve_into` reads. Same values, same
  emission order; `accumulate()` now runs on it (below).

### Changed

- **`summarize()` stats are bit-identical to pycocotools.** Each mean now
  visits the selected precision or recall cells in the order numpy flattens
  them — `[T, R, K]` with the category fastest, where hotcoco walked
  `[T, K, R]` — and sums them with numpy's pairwise summation instead of
  left to right. Before, 11 of 12 bbox stats on val2017 differed in the last
  digit (up to 4e-14); now all 12 match, as do segm and keypoints. LVIS
  APr, APc, and APf are now one mean over the bucket's precision cells, as
  lvis-api takes them, rather than a mean of per-category APs; the value is
  the same, and the last bit now follows lvis-api.

- **`accumulate()` reuses one scratch per thread across (category, area)
  items.** Each item gathered its detections' scores, ranks, sort order, and
  matched and ignore rows into fresh buffers that grew by doubling, with the
  matched and ignore rows for every IoU threshold live at once. A thread now
  keeps one set of buffers, sized exactly from the item's detection count,
  and sweeps the thresholds one at a time. On ten times val2017's bbox
  detections (437,000) the RSS growth across `accumulate()` drops from about
  46 MB to 36 MB and its allocations from 132,000 to 62,000. The output
  arrays are bit-identical.

- **`StreamingEval.update()` keeps small batches off the thread pool.** Each
  batch runs `evaluate()`, and its passes — building the annotation index,
  deriving result geometry, computing IoUs, matching — fanned out to rayon
  however little work the batch held, so a one-image batch spent more on
  thread handoff than on evaluation. Those passes now run on the calling
  thread below the shared parallel-work threshold, the one the IoU kernels
  already used. Per-image cost at batch size 1 drops about 3×, from 112 µs
  to 36 µs on val2017 and from 160 µs to 61 µs on Objects365; batch 32 and
  whole-dataset runs are unchanged. Values are unchanged.

- **`evaluate()` keeps its per-pair records in flat arenas.** Each (image,
  category) pair `evaluate()` visited was one record with two heap vectors —
  its scores and one 32-byte entry per area range — built as one object per
  pair. The pairs now live in one arena: a 24-byte header each, then every
  pair's scores, ground-truth counts, and matched and ignore bits packed back
  to back, sized exactly from a pass over the index before any pair is
  matched and written in parallel runs into disjoint windows, so nothing is
  copied afterwards. The grouping `accumulate()` builds holds a 4-byte pair
  index and area position instead of a pointer and a `usize`, which halves
  its entries. On ten times val2017's detections (246,000 pairs)
  `evaluate()` keeps 23 MB instead of 55 MB with no peak above that, and the
  smaller grouping takes `accumulate()`'s peak from 56 MB to 31 MB over it;
  segm keeps 30 MB less as well. Values are unchanged.

- **The loader reads a file in blocks instead of whole.** `COCO()` and
  `load_res()` read the entire file into memory and parsed it from there, so
  the peak while loading was the file's bytes plus its records — and a
  keypoints results file, which spells out 17 keypoints per detection in
  text, outweighs its own records. The file is now read 4 MB at a time: the
  annotations in each block are parsed in parallel while the next block is
  read, moved into the output vector as the block finishes, and the record a
  block ends inside waits for the next. The output vector is reserved from
  the first block's record density, grown only if that falls short, and
  shrunk to what it holds; `images` and `categories` stream the same way,
  record by record, and come out exactly sized too. Peak heap over the live
  records while loading the benchmark's 10× results (367,810 detections)
  falls from 56 MB to 4 MB for bbox, 175 MB to 4 MB for segm, and 316 MB to
  3 MB for keypoints — a 325 MB keypoints file loads with 237 MB — and from
  17 MB to 8 MB for val2017's ground truth; the results load is also 5–10%
  faster, since the
  read overlaps the parse and a buffer reused across blocks is not faulted in
  page by page like a fresh one. A block whose records carry `},{` inside
  them — a nested list of objects, as panoptic `segments_info` is — parses
  serially instead of in parallel runs, still a block at a time. A file with
  `NaN` or `Infinity` tokens is read whole and sanitized, then streamed
  from memory, so its peak is its bytes plus its records; a malformed file
  is read whole so that serde can report where it fails. Values are
  unchanged.

- **Unknown keys cost a small vector, not a B-tree.** `Annotation`, `Image`,
  and `Category` keep keys outside the COCO schema in `extra`, which was a
  `serde_json::Map`: a B-tree whose first entry allocates a node of about
  600 bytes, three times the record itself, and detector outputs often carry
  one unknown key on every detection. `extra` is now `hotcoco::Extra`, a
  small map over a boxed slice in file order with the same `get`, `insert`,
  `remove`, `iter`, and `is_empty` calls and conversions to and from
  `serde_json::Map`. Ten times val2017's bbox results with one unknown key
  per detection keep 156 MB instead of 448 MB; a clean file keeps 4 MB less
  from the smaller record, now 200 bytes. Saved files list unknown keys in
  the order they were read, not sorted. Values are unchanged.

- **The file's bytes are gone before its annotations are joined.** The
  parallel parse produces one vector of records per run and then joins them
  into one; the file bytes stayed alive through the join, and each run's
  vector kept the slack it grew by. Runs now shrink to what they hold as they
  finish and are joined only after the bytes are dropped, so the bytes, the
  runs, and the joined vector are never alive together. Peak heap while
  loading ten times val2017's bbox results falls from 292 MB to 208 MB, segm
  from 486 MB to 378 MB, and the ground truth itself from 59 MB to 47 MB.
  Values are unchanged.

- **Loaded vectors are sized to what they hold.** serde cannot size a JSON
  array before reading it, so every `Vec` grew by doubling and kept the
  slack: a 51-value keypoint list held 64 slots, val2017's polygons carried
  8 MB of slack on 14 MB of coordinates plus a four-slot list around nearly
  every single polygon, and the annotations vector itself, assembled from the
  parallel parse runs, kept up to a run's worth of slack per doubling. Polygons
  and keypoints are now read through a per-thread buffer and stored at exact
  size, and the annotations vector is sized from the runs before they are
  joined. `COCO()` on val2017 keeps 30 MB instead of 46 MB and parses faster
  with fewer reallocations; `load_res` keeps 13 MB instead of 22 MB on
  val2017's bbox results, 125 MB instead of 229 MB at ten times that, and
  224 MB instead of 307 MB on ten times the segm results. Values are
  unchanged.

- **`accumulate()` writes its output arrays in place.** Each (category, area
  range) work item staged its writes as index-value lists sized for the whole
  M × T × R slab, about 96 KB per item, and a serial merge copied them into
  the arrays afterwards: on COCO's 320 items that peaked at 48 MB for 16 MB
  of output. Items now stage on a per-thread buffer and apply their slab under
  a lock taken once per item. The peak is 19 MB, the phase is a few
  milliseconds faster, and every bootstrap resample in `compare()` and every
  `slice_by()` call pays the smaller cost too. Values are unchanged.

- **A box result's segmentation is the box.** `load_res` gives every box
  result that came without a segmentation the four-corner polygon pycocotools'
  `loadRes` builds; it was stored as two heap vectors per detection. The new
  `Segmentation::Rect([x, y, w, h])` holds it inline and reads as that polygon
  everywhere: JSON, Python dicts, CVAT export, and rasterization, where the RLE
  is identical. `mask::fr_polys` now rasterizes a single polygon directly
  instead of through a one-element merge that copied the result, which every
  single-polygon ground truth paid. `Annotation::obb` is boxed (`Option<Box<[f64; 5]>>`), which
  takes the record from 248 to 208 bytes. Beyond the ground truth, `load_res`
  keeps 22 MB instead of 29 MB on val2017's bbox results and 229 MB instead
  of 313 MB at ten times that count. *Rust API:* `Segmentation` is
  `#[non_exhaustive]`, so a `match` on it needs a `_` arm; later families add
  formats without another break. Code matching `Segmentation::Polygon` to
  read a result's polygon should call `Segmentation::polygons()`, which
  returns the list for either variant.

- **The annotation index is flat.** `COCO` answered "which annotations does
  this image, or this (image, category) pair, hold" from hash maps holding
  one heap vector per key — about 285k vectors for a 500k-detection results
  file, allocated one by one. The index is now two flat id lists with small
  range tables: the per-image grouping is built in two counting passes and
  the per-pair grouping by sorting each image's slice by category in
  parallel; a lookup is one hash probe on the image plus a binary search
  over its categories. Results files, whose ids are always `1..=n`, also
  skip the id-to-position map entirely. Building the index for 500k
  detections takes about 17 ms instead of 35 ms, and `load_res` keeps about
  60 MB instead of 103 MB beyond the records themselves. Query results and
  their order are unchanged.

- **Ground-truth files load without a pass to find the annotations array.**
  The loader walks a dataset object by hand and parses the annotations array in
  place, in parallel, with the run that reaches the closing bracket reporting
  where the array ends; before, serde_json tokenized the whole array once to
  find that end before any parallel work began, about 40% of `COCO()` time.
  `COCO()` on val2017 takes 0.02s instead of 0.06s and on val2014 0.18s
  instead of 0.42s (M1 Air, best of seven). Results files in object form
  benefit the same way. A shape the walk does not expect — a duplicate key,
  `"annotations": null` — falls back to the serde derive, which also reports
  the error for a malformed file.

- **`COCOeval` shares its datasets instead of copying them.** The constructor
  cloned both `COCO` objects (about 250 MB and 0.1s on 500k detections). Both
  now sit behind an `Arc` that the Python `COCO` object and the evaluator
  share, so constructing an evaluator or reading `ev.coco_gt` costs nothing.
  Python behavior is unchanged. Together with the loader change below, the
  peak resident memory of a full run is about half of 1.0.1's (320/363/355 MB
  for bbox/segm/keypoints on val2017, 856/1184/2092 MB at 10× detections);
  `docs/benchmarks.md` carries the current tables. *Rust API:* `COCOeval::coco_gt` and `coco_dt`
  are accessors returning `&Arc<COCO>` rather than public fields, and the
  constructors take `impl Into<Arc<COCO>>`, so existing calls compile
  unchanged. A Rust-visible break shipped in a minor on purpose: the crate has
  no dependents outside this repository.

- **JSON loading streams into the records and parses annotations on all
  cores; peak memory is the file plus the records.** `COCO()` and `load_res()`
  read with serde_json (its `float_roundtrip` parser) in place of simd-json,
  whose tape peaked at about twelve times the file size before a single record
  existed: +940 MB for a 79 MB bbox results file (500k detections, 134 MB of
  records), +1.3 GB for a 115 MB keypoint results file, +389 MB for the 19 MB
  val2017 ground truth. The annotations array is cut at record boundaries and
  the pieces parsed in parallel, with one serial pass as the fallback when a
  cut lands inside a string. Every field parses bit-identically to before —
  checked on val2017 ground truth (polygons) and on bbox, segm (RLE), and
  keypoint results — and the precision, recall, and score arrays stay
  bit-identical to pycocotools on all six parity datasets. `load_res()` also
  derives each result's area and box in parallel, which for segm results is
  an RLE decode per mask (0.91s → 0.40s on 500k masks). On the 500k-detection
  bbox file `load_res()` takes 0.14s instead of 0.33s, and `COCO()` on val2017
  0.06s instead of 0.08s (M1 Air), and the peak resident memory of a full run
  fell by about a third before the shared-dataset change above took it further.
  Parse failures are `Error::Json`.

- **Python tests live in `tests/`.** Every pytest file — the regression suites
  from `scripts/`, the drop-in and mask suites from `crates/hotcoco-pyo3/tests/`,
  and the three fuzzers — now sits in one root `tests/` directory, and
  `pyproject.toml` points bare `pytest` at it. `scripts/` keeps only tools you
  run by hand: real-data parity, benchmarks, downloads, fixture generators.
  The LVIS, Open Images, and mask parity scripts became pytest files
  (`test_parity_lvis.py`, `test_parity_oid.py`, and a 200-case randomized
  section in `test_mask_parity.py`, which replaces the separate mask script),
  so `just test` and CI run one `pytest` and the same set. The `parity-lvis`,
  `parity-oid`, `parity-mask`, and `adversarial-all` recipes are gone: each was
  one pytest file, so run the file. The browse tests, which no runner had
  listed, run again, and `test_adversarial.py` calls the harness in process
  instead of spawning it once per fixture.

- **`scripts/bench.py` benchmarks five libraries, one process per cell.**
  ultrafast-pycocotools and vernier join pycocotools and faster-coco-eval as
  baselines (both are dev extras now; a missing one leaves its column blank).
  Every (library, eval type) cell runs in a fresh subprocess and reports its own
  wall-clock load/eval times and peak resident memory, with the table showing
  the median of `--reps` runs (default 3). `docs/benchmarks.md` carries the
  memory table beside each timing table and a five-column feature comparison.

- **`evaluate()` keeps a lean record per cell; `evalImgs` are built on first
  access.** `accumulate()` reads exactly three things per detection — its
  score, and a matched bit and an ignore bit per IoU threshold — so that is
  what `evaluate()` now stores: one score list per (image, category) pair and
  two bit-packed matrices per area range, about 20 bytes per detection. The
  full per-image records (`evalImgs`: every id, both sides of the match, one
  copy of the scores per area range, about 460 bytes per detection) are no
  longer built during `evaluate()`. They are materialized, once, the first
  time something reads them — the `evalImgs` attribute, `tide_errors()`,
  `calibration()`, `image_diagnostics()` — from a snapshot of the inputs that
  `evaluate()` saw, so they describe that run whatever `params` was set to
  afterwards. Nothing in the API changes and no flag is needed; the cost moves
  to a second matching pass paid only by callers that read the records, and
  the in-crate analyses (`tide_errors()`, `calibration()`,
  `image_diagnostics()`) materialize only the `"all"` range they read. On a
  500,000-detection val2017 bbox run, `evaluate()` goes from 0.37 s to
  0.15 s and `accumulate()` from 0.13 s to 0.08 s; peak resident memory for
  load + evaluate + accumulate + summarize drops from 1.1 GB to 0.9 GB. Every
  `precision`, `recall`, and `scores` value is bit-identical to before.

- **`accumulate()` sorts each (category, area range) once, not once per
  `maxDets` entry.** pycocotools concatenates every image's `dtScores[0:maxDet]`
  and mergesorts the result for each cap. The three sorted sequences are the same
  sequence: a stable sort of the concatenation truncated at the largest cap,
  filtered to detections whose rank inside their image is below the smaller cap,
  is the stable sort of the smaller-cap concatenation — filtering keeps relative
  order, and ties break on concatenation position, which the filter also keeps.
  So the gather and the sort run once per (category, area range) and each
  `maxDets` slot is a filter of that order. On a 1.5M-detection RF-DETR-shaped
  workload (300 detections per image, `maxDets=[1, 10, 300]`) `accumulate()` is
  22–24% faster at 1, 2, and 16 threads, and end-to-end evaluate + accumulate +
  summarize 10–14%. Every `precision`, `recall`, and `scores` value is
  bit-identical to before, including under cross-image score ties, an unsorted
  `maxDets`, and `maxDets` lowered between `evaluate()` and `accumulate()`; a new
  Rust test pins each of those against fresh single-cap runs.
- **`accumulate()` computes each precision-recall curve in one pass over the
  ranked detections instead of four array round trips.** The previous kernel
  wrote cumulative TP and FP arrays, then recall and precision arrays, then
  read them back for the envelope and the recall-threshold scan. The new one
  keeps integer counters, computes precision only at true-positive ranks — the
  only ranks the VOC envelope can take its maximum from, and the only ranks a
  recall threshold can first be met at — and samples the thresholds while
  scanning. `accumulate()` is 34–37% faster and `compare()` (one accumulation
  per bootstrap resample) 37% faster on a 1.5M-detection, 300-per-image
  workload at 1 and 2 threads; end-to-end 18–25%. Every `precision`, `recall`,
  and `scores` value is bit-identical to before; Open Images keeps the
  cumulative arrays, which its all-points AP needs. The identity is checked by
  `metrics::counts::tests::fused_curve_matches_cumulative_then_interpolate_bit_for_bit`
  against the previous two-function path, including unsorted, duplicated, and
  `NaN` recall thresholds.
- **`accumulate()` buckets its evaluated cells in parallel, with one hash lookup
  per (image, category) pair instead of three per cell.** Before the
  precision-recall work starts, `accumulate()` groups every evaluated cell by
  category and area range. That walk was sequential and did three hash lookups
  per cell — category id, area-range key, image id — over ~1.5M 360-byte cells
  on an RF-DETR-shaped run (5,000 images × 300 detections, `maxDets=[1, 10,
  300]`), which made it 69 ms of the 189 ms `accumulate()` took on 16 threads:
  the one serial step left. The walk is now cut into a few runs per thread that
  are concatenated in order, the area range is matched by a bit-exact scan of
  the handful of ranges, and the category and image lookups are memoized on the
  id of the previous cell, which `evaluate()`'s layout makes a hit on almost
  every cell. Same workload, same machine: grouping 69 → 32 / 18 / 6 ms and
  `accumulate()` 1.49 → 1.45 s (≈ −3%, noise-level) / 0.80 → 0.75 s (−7%) /
  0.207 → 0.136 s (**−34%**) at 1 / 2 / 16 threads. `precision`, `recall`,
  `scores`, and `stats` are bit-identical to before on ten configurations,
  including `maxDets` reassigned between `evaluate()` and `accumulate()`. A
  `params` reconfigured between the two — categories or area ranges dropped,
  reordered, or listed twice — resolves every cell exactly as the sequential
  walk did; `detection::accumulate::tests` checks that against the old walk
  under several run boundaries, and
  `accumulate_arrays_are_independent_of_thread_count` checks every output
  array and a `slice_by` re-accumulation bitwise across 1 to 16 threads on a
  dataset with tied scores across images.
- **`COCO::create_index` reserves its (image, category) index for the number of
  distinct pairs it will hold, not the number of annotations, and derives the
  category-to-images index from those pairs instead of pushing once per
  annotation.** `img_cat_to_anns` holds one entry per distinct `(img, cat)`
  pair — on a 1.5M-annotation, 300-per-image RF-DETR-shaped workload that is
  ~400K pairs, so reserving for the annotation count left most of the table's
  capacity unused. `cat_to_imgs` is now built from those already-unique pair
  keys after the annotation loop (~400K pushes) instead of once per annotation
  (~1.5M), then sorted into the same shape as before. All six index maps
  (`anns`, `imgs`, `cats`, `img_to_anns`, `cat_to_imgs`, `img_cat_to_anns`) —
  all private — also switched from the standard library's `HashMap` (SipHash)
  to `rustc_hash::FxHashMap`, which is faster on the integer and integer-pair
  keys these indices use throughout. `rustc-hash` was already in the dependency
  graph transitively (via `numpy`); this makes it a direct dependency of
  `hotcoco` (MIT/Apache-2.0). FxHash is not resistant to adversarially chosen
  keys, an accepted trade-off for a local library indexing ids the caller
  already chose to load — map iteration order was confirmed to never reach any
  observable output before making the swap (every iteration site feeds a sort).
  On the same RF-DETR-shaped workload, `gt.loadRes(ndarray)` is 34% faster and
  the `COCOeval` constructor 49% faster (both call `create_index`); end-to-end
  20% faster. `precision`, `recall`, `scores`, and `stats` are bit-identical to
  before across ten configurations, including `maxDets` reassigned between
  `evaluate()` and `accumulate()`. `coco::tests::test_cat_to_imgs_derived_from_pair_keys`
  pins `cat_to_imgs`'s membership and deduplication and `get_ann_ids_for_img_cat`'s
  dataset-order contract, and fails if the two index maps are conflated or the
  order guarantee is dropped.
- **`COCOeval`'s constructor copies an already-indexed `COCO` instead of rebuilding the index
  from scratch, for both the ground-truth and detection side.** `PyCOCOeval::new` used to clone
  only the raw dataset and call `COCO::from_dataset`, which rehashes every annotation, image, and
  category id into fresh index maps — even though the `COCO` object passed in already carries a
  built index that is always kept in sync with its dataset (every write path — the `dataset`
  setter, dataset-derived constructors — rebuilds the whole object, so the cached index can never
  go stale). `COCO` now derives `Clone`, and the constructor copies it directly. On a
  1.5M-annotation RF-DETR-shaped workload the `COCOeval` constructor is 74% faster and the
  evaluator's whole `compute()`-shaped call sequence 11% faster end-to-end; peak RSS is unchanged
  — the copy is still a full one, so nothing is saved on memory, only on the redundant rehash.
  One side effect: `create_index()` prints non-fatal warnings (duplicate annotation ids, unnamed
  categories) to stderr as it runs, so the previous rebuild-per-constructor re-printed a source
  dataset's warnings on every `COCOeval()` call; the copy carries the already-collected warnings
  instead of regenerating them, so the reprint is gone. `precision`, `recall`, `scores`, and
  `stats` are bit-identical to before across ten configurations, including `maxDets` reassigned
  between `evaluate()` and `accumulate()`. `coco::tests::test_clone_is_a_faithful_reindex` pins
  that the clone's six index maps and warnings match a fresh rebuild and that the copy is an
  independent snapshot, and fails if a future hand-written `Clone` impl drops a field.
- **`load_res_anns` validates, derives geometry, and assigns ids in one pass
  over the detections instead of five, and checks GT membership against the
  index this `COCO` already built instead of rebuilding a `HashSet` of GT ids
  on every call.** The image-id and category-id mismatch checks used to scan
  the whole detection list independently of the NaN-score check and the
  geometry/id loops that follow; they now run together, still warning at most
  once per mismatch kind (naming the first offender) and still rejecting a NaN
  score outright. `derive_from_bbox`'s rectangular polygon, the unconditional
  detection ids, and the mismatch warnings are all unchanged in content — only
  how many times the detection list is walked to produce them. Both the
  in-memory dict path and the numpy `loadRes(ndarray)` fast path share this
  function, so both benefit from one fix. On a 1.5M-detection RF-DETR-shaped
  workload `loadRes` itself is 16–19% faster, a reproducible win (the two
  builds' measured ranges across five repeats don't overlap); the end-to-end
  effect is not distinguishable from run-to-run noise on this workload, since
  `loadRes` is a smaller share of the total than the constructor and evaluation
  phases. `precision`, `recall`, `scores`, and `stats` are bit-identical to
  before across ten configurations. One behavior note for direct Rust callers
  only (the Python binding is unaffected): the mismatch checks now read the
  same index that every other query method already depends on being current,
  rather than the raw dataset — a caller that mutates `.dataset` in place
  without calling `create_index()` was already getting a stale index from
  every other method, and now gets it here too, which brings this method in
  line with the rest of the type. Two new tests,
  `load_res_anns_warns_once_per_mismatch_kind` and
  `load_res_anns_skips_category_check_when_gt_has_no_categories`, pin the
  once-per-kind warning behavior and the no-categories edge case, and fail if
  either guard is dropped.
- **Segmentation `evaluate()` converts masks to RLE only for detections and
  ground truth that share a category with something on the other side.**
  `SegmRles::prepare` used to rasterize every mask in scope up front, mirroring
  pycocotools' `_prepare`; `evaluate()` already skips computing an IoU matrix
  for an `(image, category)` cell with only ground truth or only detections
  (that cell's IoU is empty by construction), so those masks were converted for
  nothing. A DETR-shaped result set spreads detections across every category
  per image while each image's ground truth covers only a handful, so most of
  that conversion was waste: on a 1.5M-detection, 80-category RF-DETR-shaped
  workload, 90.7% of detections sit in a category with no matching ground truth
  in their image. `evaluate()` on a segmentation run is 58–60% faster at both 2
  and 16 threads (peak RSS during that phase down ~2.5 GB), with
  `precision`/`recall`/`scores`/`stats` bit-identical to before on 4
  configurations spanning two workload sizes and `maxDets` settings, and the
  existing 10-configuration bbox baseline unaffected (bbox never builds this
  cache). `confusion_matrix()`/`tide()` still read the same cache after
  `evaluate()`; a detection this change excludes from it now falls back to
  converting its mask on the spot instead of hitting a pre-built entry — a cost
  that moves from `evaluate()` to whichever of those calls needs it, computed
  at most once per image per call, and repeated on a second such call, since
  neither extends the cache. Bounding-box evaluation is untouched — this cache
  is built only for `iouType="segm"`. A new test,
  `segm_rle_cache_skips_dt_only_cells`, pins that a detection with no matching
  ground-truth category is excluded from the cache and fails if that gate is
  dropped.
- **Decoding a Python dict into a dataset interns its field-name keys instead of
  allocating a new string per key per record.** `COCO(dict)`, `loadRes(list of
  dicts)`, and the RLE/segmentation decoders looked up each field with a bare
  string literal, and `PyDict.get_item` allocates a fresh `PyString` for that on
  every call; decoding an RF-DETR-shaped annotation list calls this once per
  field per detection — millions of throwaway strings for a fixed set of ~10
  field names. Those lookups now go through `pyo3::intern!`, which builds each
  literal's `PyString` once per process and reuses it. `COCO(dict)` construction
  is 18–20% faster across three shapes (a small dict, a 1.5M-annotation
  segmentation dict, and a bounding-box-only dict), with no numeric or
  structural change to the decoded dataset. A new test,
  `TestKnownKeysRoundTrip`, round-trips every known field of an annotation,
  image, and category through `COCO(dict)` and fails if any interned key
  literal drifts from the field it names, whether the field is required
  (raises instead of decoding) or optional (silently dropped instead of
  decoding) — both failure shapes are pinned by injection. A separate
  reprofile found that `extra`-field extraction, not covered by this change,
  now costs roughly half of what remains of dict decoding on the same
  workloads; that cost is unaddressed here.

### Fixed

- **`hotcoco.browse` runs on Python 3.9 again.** The two FastAPI route handlers
  in `server.py` spelled their optional query parameters `str | None`. Every
  other annotation in the package is a string under `from __future__ import
  annotations`, but FastAPI evaluates route signatures at runtime, and 3.9
  cannot evaluate a PEP 604 union, so `create_app()` raised `TypeError` on the
  floor interpreter the wheel is built for. They are `Optional[str]` now. CI
  runs the browse tests on 3.9, which is how this surfaced.

- **`eval["precision"]`, `eval["recall"]`, and `eval["scores"]` are bit-identical
  to pycocotools' arrays.** They agreed to an ulp before; two small things kept
  them from being the same bits, and neither ever moved a headline metric.
  - `precision` carried `tp / (tp + fp)` where pycocotools computes
    `tp / (fp + tp + np.spacing(1))`. The guard term only matters at
    `tp + fp == 1`, where it puts a lone leading true positive at `1 - 2^-52`
    instead of `1.0`; on COCO val2017 bbox that is ~7,600 cells of the
    precision tensor, each one ulp off. `metrics::counts` now keeps the term,
    so a single perfect match reports AP an ulp or two under 1.0 — the value
    pycocotools reports for it.
  - `evaluate()` dropped an (image, category, area range) cell that had
    detections but no ground truth when the area range ignored every
    detection. pycocotools keeps that cell: its `evaluateImg` skips only when
    the raw ground-truth and detection lists are both empty. The dropped
    detections match nothing and move no counter, but they still occupy ranks
    in the score order `accumulate()` samples `scores` from, so the score
    reported at a recall threshold could come from a later detection than the
    reference's. The cell is now kept, `evalImgs` carries it, and
    `test_rank_filler_and_epsilon_guard_match_pycocotools_bit_for_bit` pins
    both fixes against pycocotools on a two-image dataset built to show them.

  The three arrays are checked bit-for-bit against pycocotools on COCO
  val2017 (bbox, segm, keypoints) and on a 500,000-detection synthetic run.
  The summary `stats` still differ from pycocotools by up to 3.6e-14: numpy's
  `mean` is a pairwise sum, and that reduction order is not reproduced yet.

### Removed

- **`Error::JsonParse` and the simd-json dependency.** The loader reports
  every parse failure as `Error::Json`, so the variant could never be
  constructed; dropping it also drops the ten crates that only simd-json
  pulled in. Rust code that matched on `JsonParse` matches `Json` instead. A
  Rust-visible break in a minor, under the same policy as the shared-dataset
  change above.

- **`scripts/fixture_builder.py` and `scripts/bench_fiftyone.py`.** The first
  had no entry point and no importer; the adversarial corpus it described is
  tracked in `tests/fixtures/adversarial/`. The second imported a package the
  project never declared and had no recipe.

## [1.0.1] - 2026-09-12

### Added

- **`coco --version`** prints the installed hotcoco version. `coco-eval --version`
  already did.
- **`scripts/fuzz_dropin.py`, run as `just fuzz-dropin`** — a drop-in spelling
  fuzzer. `fuzz_parity.py` round-trips every dataset through JSON files, so it
  can never see a `bytes` RLE `counts`, a numpy scalar id, a tuple bbox, or a
  missing optional key. This one builds a dataset in memory, evaluates it under
  57 spellings pycocotools treats as equivalent, and fails on any spelling that
  changes the stats or the `precision`/`recall`/`scores` arrays. Loud gaps and
  places where both implementations drift together are summarized, not failed.
- **`crates/hotcoco-pyo3/tests` runs in `just test` and in CI.** The binding
  regression tests (drop-in gaps, mask parity, integrations) gated nothing
  before; only `scripts/` was collected.

### Changed

- Python formatting and lint recipes use the configured pre-commit Ruff hooks.
  The git hook runs every pre-commit hook on the staged files of each commit, and
  CI runs them over the whole tree. The Ruff version is pinned only in
  `.pre-commit-config.yaml`; `pre-commit` replaces `ruff` in the dev extra.
  `just py-fmt-check` is removed — it had become identical to `just py-fmt`.
- **A category without `name` loads.** pycocotools tolerates the omission and
  TorchMetrics emits bare `{"id": i}` records. The record gets the display name
  `cat_<id>` — the same placeholder `COCO::cat_name` already used for an unknown
  id — and the load is flagged in `load_warnings`.
- **Integral floats are accepted for integer fields**, from a JSON file and from
  a dict alike — `image_id: 1.0` is how a JSON written through pandas or numpy
  reads back, and pycocotools accepts it because `1.0 == 1`. A fractional or
  negative value still raises, naming the value; one that does not fit the
  field is an `OverflowError`. Image `height`/`width` are optional in a file, as
  they already were in a dict.
- **`iscrowd`, `is_group_of`, `params.useCats`, and the mask functions'
  `iscrowd` argument share one flag reader**, from a JSON file and from a dict
  alike: `bool`, any int, or an integral float. pycocotools' own default is the
  int `useCats = 1`, Open Images spells `IsGroupOf` as `0`/`1`, and each site
  previously accepted a different subset.
- **`keypoints` may be an `(N, 3)` array** as well as the flat COCO list.

### Fixed

- **`mask.encode` accepts `bool` masks.** Any one-byte integer or boolean dtype
  is read as `uint8` instead of rejected, so the arrays torch-side code stores
  (TorchMetrics keeps masks as `bool`) encode without a cast, and any memory
  layout works — C-order, Fortran-order, or a sliced view. A wider dtype, or a
  non-array input, now raises a `TypeError` naming the dtype and the fix rather
  than `'ndarray' object is not an instance of 'ndarray'`. Reported in
  [#5](https://github.com/derekallman/hotcoco/issues/5), fixed in
  [#7](https://github.com/derekallman/hotcoco/pull/7) by @Borda.
- **RLE `counts` as `bytes` loads correctly.** `mask.encode()` returns `counts`
  as a `bytes` object, matching pycocotools, but the dataset loader decoded only
  the `str` form. A `bytes` object is a Python sequence of ints, so it extracted
  as a list of run lengths and became an uncompressed RLE holding the compressed
  string's byte values — a mask that decodes to nothing. Nothing raised, so the
  failure surfaced as segmentation AP of exactly `0.000`. `bytes` is now accepted
  wherever a `segmentation` dict is parsed, in the `COCO` constructor and in
  `load_res()` alike, and a `counts` value that is not `str`, `bytes`, or a list
  of ints raises `TypeError` naming the type it got instead of producing an empty
  mask. Reported in [#5](https://github.com/derekallman/hotcoco/issues/5),
  fixed in [#6](https://github.com/derekallman/hotcoco/pull/6) by @Borda.
- **`COCOeval.summarize()` prints its table again.** The 1.0 binding computed
  the lines and dropped them; `stats` was populated and every documented
  example showed a table that never appeared. The lines now go through Python's
  `sys.stdout`, so `contextlib.redirect_stdout` and notebook capture work.
- **A keypoint ground truth without `num_keypoints` is scored, not ignored.**
  The missing field was read as 0, which is the "no labeled keypoints" ignore
  rule, so a file that omits it evaluated as a dataset with nothing to match.
  `num_keypoints` is now derived from the visibility flags when absent (the
  Rust `Annotation::num_visible_keypoints` is the one reader). Found by the new
  `scripts/fuzz_dropin.py`, which evaluates one dataset under many in-memory
  spellings — bytes counts, numpy scalars, tuples, missing optional keys — and
  fails on any spelling that changes the numbers.
- **The non-default-parameter warning fires once, not twice.** `summarize()`
  raised it as a Python `UserWarning` and the Rust core also wrote it to fd 2.
  The Rust `COCOeval::summarize_lines` now prints nothing; `summarize()` owns
  the stderr copy.

## [1.0.0] - 2026-09-02

### Added

- **`primitives.lsap` reads numpy arrays natively.** A `float64` ndarray — any
  strides, so C-order, Fortran-order, a transposed or sliced view — is read in a
  single pass instead of one boxed extraction per element; other dtypes and
  nested sequences take the element-wise path as before, with the same results
  and the same `ValueError` on NaN. The stub and the API page now spell the
  parameter `numpy.ndarray | Sequence[Sequence[float]]`.
- **`convert::IMAGE_EXTENSIONS`** (Rust) — the one list of image file
  extensions the converters and the Python `read_image_dims` fallback try when
  a bare stem has no direct hit.
- **`STYLE.md`** — the documentation style authority. hotcoco now follows the
  [Google developer documentation style guide](https://developers.google.com/style);
  `STYLE.md` records the rules that come up most (sentence-case headings, no
  `e.g.`/`i.e.`/`etc.`, `might` for possibility and `may` only for permission, no
  "see below", descriptive link text) and the deliberate deviations — spaced em
  dashes stay, and `//` implementation comments are out of scope. Referenced from
  `CONTRIBUTING.md`, `CLAUDE.md`, and the `docs` skill, which gained a diff-grep
  for the words-to-avoid list. No linter enforces it: Vale was evaluated and
  removed as too noisy for a single-contributor repo.
- **The Cyanotype design system**, replacing Cold Brew across every visual
  surface — browse UI, docs site, matplotlib, and the Plotly dashboard. Grounds
  are true-neutral silver in both modes, the signature is Prussian blue
  (`#23467E` light, `#8FB3E2` dark), and depth comes from hairline rule weight
  and a 24px drafting ground grid rather than shadow and noise. Type moves from
  DM Serif Display / DM Sans / JetBrains Mono to Instrument Serif / IBM Plex
  Sans / IBM Plex Mono.
- **`"cyanotype-dark"` plot theme** — the same ten hues on a graphite ground,
  lifted to hold against a dark background. Every plot function and `style()`
  accept it. `paper_mode=True` is rejected on it with an explanatory error,
  since a forced white background erases the theme's chrome.
- **`SERIES_COLORS_DARK`, `SEQUENTIAL_DARK`, `CHROME_DARK`, `EVAL_COLORS` and
  `EVAL_COLORS_DARK`** joined the public palette constants in `hotcoco.plot`,
  for styling a dark surface matplotlib does not own. The Plotly dashboard now
  derives its chrome and colorway from them instead of holding literal copies.
- **Confusion matrix cell annotations pick their ink from the cell's own
  luminance** rather than a hardcoded "white above the midpoint", which was
  wrong on any colormap whose high end is light — as `cyanotype-dark`'s is.
- **Bar-chart value labels get axis headroom.** `bar_label` places text past the
  bar end without widening the axes, so the longest bar's label clipped. The
  headroom is computed from the data, not tuned to a font.
- **The documentation has figures.** Every chart in `docs/assets/` is generated by
  `scripts/gen_docs_assets.py` in both schemes and swapped by the site's palette
  toggle, so the plotting guide shows its plots, the benchmarks page charts its
  own numbers, and the dataset browser guide shows the browser.
- **`hotcoco.plot.eval_colors(theme)`** — the single mapping from a theme name to
  its TP/FP/FN colors. `EVAL_COLORS` and `EVAL_COLORS_DARK` were two constants with
  nothing selecting between them, so every caller picked by hand; keyed off the
  theme's own `dark` flag rather than its name, a future dark theme is covered
  without a second string to keep in sync.
- **`scripts/test_theme.py` fails the suite on a color literal in a drawing call.**
  An AST walk over every module in `hotcoco/plot/` catches all four spellings
  matplotlib accepts — a `color=` keyword, a `set_*color()` setter, a `*_color`
  local, and a dict built inline in a call such as `bbox={"facecolor": ...}`.
  Palette *tables* are exempt on purpose: `_THEMES` and the PDF report's `_RC` are
  where the values live. The check is globbed rather than enumerated, so a new
  module in the package is covered by default.
- **IBM Plex Sans is vendored** at 400/500/600 in `python/hotcoco/_fonts/`,
  replacing the DM Sans the previous design system shipped. Shipped unmodified —
  "Plex" is an OFL Reserved Font Name, so a subset could not keep the family
  name. The browse UI, the PDF report, and matplotlib now all render Cyanotype's
  real body face with no CDN dependency.
- **`scripts/test_theme.py`** asserts the browse CSS tokens, the dashboard
  constants, and the vendored font files agree with `plot/theme.py`. The design
  system was previously a markdown convention with nothing enforcing it.

- **`COCO.load_warnings`** — everything the loader tolerated but flagged (duplicate
  annotation ids, non-finite JSON values normalized to `null`, orphaned result ids)
  is collected on the object as well as printed to stderr.
- **Custom JSON keys survive.** Keys outside the COCO schema on images, annotations,
  and categories now round-trip `load → filter/split/merge → save`, appear in the
  dicts `load_imgs`/`load_anns`/`load_cats` return, and are visible to `slice_by`
  callables — closing a pycocotools drop-in gap, at no measurable load cost.
- **`results()["params"]` records the full configuration** — `recall_thresholds`,
  `use_cats`, `kpt_oks_sigmas`, and `reference_deviations` join the existing keys,
  so a saved results file explains its own provenance.
- **Type stubs cover the LVIS / torchvision drop-in surface** — `LVIS`, `LVISEval`,
  `LVISeval`, `LVISResults`, `CocoDetection`, and `CocoEvaluator` are typed,
  `import hotcoco.mask` type-checks, and `metrics.is_computed` /
  `metrics.is_missing` are public.
- **`coco explore` gained `--iou-type`, `--iou-thr`, `--no-eval`, and `--slices`.**
  `--iou-thr` (and `COCO.browse(iou_thr=...)`) sets the UI slider's starting
  position, snapped to 0.50–0.95 in steps of 0.05.
- **`coco eval --diagnostics`** prints a worst-images-by-F1 table in human output,
  alongside the label-error candidates.
- **Docs: a ground-truth COCO JSON schema page** (The COCO Format) — which fields
  are required on `images`/`annotations`/`categories`, what is optional, and that
  unknown keys are preserved.

- **`Image`, `Annotation` and `Category` now implement `Default`** (Rust API), so a
  struct literal can name the fields it cares about and end with
  `..Default::default()`. `Dataset` already did, and every optional field on all
  three was already `#[serde(default)]` — "absent means default" was the schema's
  contract on load but could not be spelled at a construction site. Removes ~670
  lines whose only content was `None,` and `vec![],` from the converters and tests.
- **Open Images CSV conversion** — `COCO.from_oid(csv_path, class_descriptions=None,
  images_dir=None)`, `COCO.to_oid(output_csv)`, and `gt.load_res_oid(csv_path)`, plus
  `coco convert --from oid --to coco` with `--class-descriptions`. hotcoco has shipped
  Open Images evaluation since 0.3.0 but could not read the CSV format Open Images
  actually distributes, so the feature named after the dataset required the caller to
  write a parser first. Columns are resolved **by name**, which covers the full V6
  layout, the challenge subset, and detection CSVs with a `Score` column from one
  reader — and makes the format's `XMin,XMax,YMin,YMax` ordering (`XMax` before `YMin`)
  impossible to transpose. `IsGroupOf` maps to the `is_group_of` annotation field that
  drives IoA group-of matching. `load_res_oid` raises on a detection naming an unknown
  image or category rather than dropping it, since a silently discarded detection moves
  recall invisibly. Without `images_dir`, boxes stay normalized against a 1×1 image:
  Open Images AP is unaffected because IoU and IoA are ratios of areas scaled equally on
  both axes, but absolute areas and the small/medium/large ranges are not meaningful.
- **DOTA conversion is now available from Python** — `COCO.to_dota(output_dir)` and
  `COCO.from_dota(label_dir, images_dir=None, categories=None)`, plus
  `coco convert --from dota --to coco`. The Rust functions shipped in 0.4.0 and were
  never bound, so Python users could evaluate oriented boxes but could not load them
  from the standard oriented-box format.
- **`just docs-links` — a documentation link checker** (`scripts/check_docs_links.py`).
  Resolves every internal Markdown link, heading anchor, and `zensical.toml` nav entry,
  and fails on a link to a missing file, an anchor no page defines, a nav entry with no
  file, or a page missing from nav. Nothing validated documentation links before, which
  is how a docs-site button linking a notebook that 404s survived, along with a dozen
  anchors left dangling by page renames. Wired into the `/docs` and `/ship` workflows.
- **Two new user guide pages**, split out of `guide/evaluation.md`: **LVIS & Open
  Images** (federated AP, the 13 LVIS metrics, the Open Images Challenge protocol and
  group-of semantics) and **Model Diagnostics** (confusion matrix, TIDE, calibration,
  F-scores, model comparison, per-image diagnostics and label errors).
- **API reference entries for shipped surfaces that had none**: `COCOeval.slice_by`,
  `summary_lines`, `virtual_cat_names`, `COCO.healthcheck`, and an LVIS section
  covering `LVISeval`/`LVISEval`/`LVIS`/`LVISResults`.

- **`hotcoco.metrics` now reads numpy arrays natively.** The module docstring
  always said "lists or numpy arrays", but the bindings put numpy on the slow
  path: PyO3's list fast path doesn't fire for ndarrays, so every element of a
  `float64` array cost a boxed extraction — and every element of a bool array a
  Python-level `__bool__` call. `float64` and `bool` arrays (including strided
  views) are now read in a single copy across `average_precision`,
  `precision_recall_curve`, `calibration_curve`, and `calibration_error`; other
  dtypes and plain sequences still work through the fallback. The type stubs,
  which contradicted the docstring by requiring `Sequence`, now accept ndarrays
  too.

- **Provenance now reaches everything that draws the numbers.** `report()` has
  recorded whether a run is leaderboard-comparable since it landed, but every Python
  rendering surface ignored it: the PDF report, the browse dashboard, and
  `coco eval --json` presented Open Images, oriented-box, and custom-parameter
  results formatted identically to a pycocotools-parity run. The PDF is the artifact
  most likely to be circulated to someone who never ran the evaluation, which is
  exactly the case `Provenance` exists for.

  - The **PDF report** carries a provenance line under the run context, listing every
    reason when the run is an extension. Drawn in both states on purpose — a caveat
    that appears only sometimes cannot be distinguished from an older hotcoco.
  - The **browse dashboard** states provenance in the sidebar and shows a banner
    above the KPI tiles when the run is not parity-verified.
  - **`coco eval --json`** gained `reference_deviations` alongside the `provenance`
    it already carried. The `summarize()` warnings go to stderr and do not survive a
    pipe.

- **`COCOeval.provenance()` and `COCOeval.reference_deviations()`** in Python. Same
  values as `report()["provenance"]` and the `summarize()` warnings, but read from
  the configuration alone, so unlike `report()` and `results()` they work **before**
  `run()` — check comparability ahead of a long evaluation instead of after it. In
  Rust, `COCOeval::provenance()` is the single mapping from `reference_deviations()`
  to a `Provenance`, which `report()` now calls rather than inlining.

- **`EvalResults` carries `provenance`.** It reaches the CLI's `--json`, the PDF
  report, and `ev.results()` in Python - the artifacts users archive and come
  back to. Previously the marker existed only on `EvalReport`, which nothing that
  writes a file uses, so comparability died with the process.

- **`hotcoco.metrics` and `hotcoco.primitives` — the functional layer.** Metric
  functions you can call on plain arrays, with no evaluator, no dataset, and no COCO
  JSON:

  ```python
  from hotcoco import metrics, primitives

  ap = metrics.average_precision(scores, matched, num_gt=len(ground_truths))
  ece, mce = metrics.calibration_error(scores, matched)
  cm = metrics.confusion_matrix(gt_labels, dt_labels, num_classes=80)
  rows, cols = primitives.lsap(similarity, maximize=True)
  ```

  This is the shape `sklearn.metrics` and `torchmetrics.functional` use. Before 1.0
  every one of these lived only as a `COCOeval` method, so scoring anything that had
  not been through the COCO pipeline meant reimplementing it.

  The two namespaces split by what a function *produces*: `primitives` produces matches
  and similarities (`lsap`, `bbox_iou`, `mask_iou`), `metrics` produces numbers from
  matches (`average_precision`, `precision_recall_curve`, `calibration_curve`,
  `calibration_error`, `confusion_matrix`). `tests/architecture.rs` enforces the
  boundary: `metrics` may not import a family driver, and `primitives` may not import
  `metrics`.

  **Purely additive.** Every `COCOeval` method still exists and returns exactly what it
  did — the methods are now thin adapters that marshal `eval_imgs` into arrays and call
  these functions. Verified byte-identical on val2017 bbox/segm/keypoints plus LVIS and
  boundary fixtures, including bootstrap confidence intervals.

  Bootstrap CIs (`metrics::bootstrap::bootstrap_ci`) and greedy matching
  (`primitives::greedy::greedy_match_masked`) are Rust-only for now — see
  [the API reference](https://derekallman.github.io/hotcoco/api/metrics/) for why.

- `hotcoco.detection` — the first metric-family namespace, exposing `COCOeval`,
  `Params`, `Hierarchy`, and `compare`. Panoptic, tracking, and concepts follow the same
  shape, so code evaluating several families reads consistently:
  `from hotcoco import detection, panoptic`.

  **Purely additive.** `hotcoco.detection.COCOeval` *is* `hotcoco.COCOeval` — the same
  object under the family name. Top-level names are permanent compatibility guarantees,
  not deprecated aliases; if you are replacing `pycocotools`, keep importing from the
  top level. The LVIS helpers stay top-level too, since they exist to mirror
  `lvis-api`'s import paths.

- `hotcoco::report::EvalReport` — the shape every metric family reports in. Carries headline `metrics`, nested `per_class` and
  `per_group` breakdowns, renderable `curves`, and the producing `params`, so a renderer
  that can draw a detection result will be able to draw a panoptic or tracking one
  unchanged.

  `Provenance` (`ParityVerified` / `Extension` / `UserComposed`) records whether numbers
  may be compared against a leaderboard. Family drivers set it and it survives
  serialization, so a deserialized report renders as what it actually is. Oriented-box
  evaluation reports `Extension`: it is a real metric, but no reference implementation
  exists for it to be standard against.

- `COCOeval.report()` in Python and `COCOeval::report()` in Rust assemble one. `curves` holds the aggregate precision-recall
  curve per IoU threshold (`pr@0.50` …), averaged over categories at `area="all"` and the
  largest `max_dets`, plus the shared `rec_thrs` axis — the slice a chart actually draws.
  The full `T×R×K×A×M` tensor (~1M floats on COCO) stays reachable via `accumulated()`
  rather than being copied into the report.

- `EvalParams` is now exported. It was `pub` and appeared as a public field of
  `EvalResults`, but its module was private, so downstream code could receive the type
  and never name it.

- `just semver` — checks the public Rust API against the last published release via
  `cargo-semver-checks`. The 1.0 reorganization moves nearly every type between modules
  while promising the crate-root paths keep resolving, and this is the mechanical proof
  of that rather than re-reading `lib.rs` by hand.

- `primitives::greedy::coco_match_floor` — the canonical spelling of pycocotools'
  `min(t, 1 - 1e-10)` match floor, so the detection lineage has one definition of the
  clamp instead of a literal repeated at each call site.

- **`COCOeval.metric_defs()`** — the metric catalog as structured data (`name`, `ap`,
  `iou_thr`, `area`, `max_det`, `freq_group`) in `metric_keys()` order, so renderers
  read a metric's axes instead of regex-parsing its name. The PDF report and dashboard
  consume it; the regex they used misread `AR10` as "IoU 0.10" whenever the IoU grid
  contained 0.10.

- **`COCOeval.is_benchmark_standard()`** — the default-deny predicate behind
  `provenance()`, exposed so renderers stop re-deriving it with a string compare.
  `PlotData` and the browse dashboard now read it from Rust.

- **`COCO::cat_name(id)`** (Rust) — one owner of the category display-name fallback.
  An unnamed category previously rendered as `cat_7`, `7`, or `?` depending on the
  surface; every surface now says `cat_7`.

- **`Params::nearest_iou_thr_idx()`, `Params::max_det_idx()`, `Params::all_area_range()`**
  (Rust) — named owners for lookups that TIDE, per-image diagnostics, F-scores, and the
  report each hand-rolled (and, for the max-dets index, got wrong — see Fixed).

- **Healthcheck: `dt_nan_score` error.** NaN detection scores are now their own
  error-severity finding explaining why they poison score sorting; they previously
  disappeared into the `dt_score_out_of_range` *warning*, whose message did not
  describe them.

- **`scripts/bench.py --phases`** — splits each benchmark into load (constructor +
  `loadRes`) and eval (`evaluate` + `accumulate` + `summarize`) phases, plus a
  bbox-only-GT variant row; feeds the new "Where the time goes" table in
  `docs/benchmarks.md`.

- **Bounded `target/`: `just disk` / `just clean`.** Cargo never evicts build
  artifacts, and macOS's default `unpacked` split-debuginfo plus a test-running
  pre-commit hook grew `target/` to 18 GB of loose `.o` files. The dev/test profiles
  now use `split-debuginfo = "packed"` with `line-tables-only` debuginfo (rationale
  recorded in `Cargo.toml`), the pre-commit hook warns — never blocks — if stray `.o`
  files reappear, and `just disk` / `just clean` report and reclaim.

### Changed

- **Every documented fact now has one owning page.** A redundancy audit of the
  README and all 27 docs pages found the same content on two to four pages in
  about 40 places, some already drifting. API pages now hold signatures,
  parameters, and return shapes; guides hold worked examples and interpretation;
  the benchmarks page owns every parity figure, including new TIDE and Open
  Images subsections moved from the guides; every other surface links. The pass
  also fixed what the drift exposed: batch `mask.area` returns `uint32`, not
  `uint64`; plots save at 200 DPI, not 150; `report()`'s title derives from the
  eval mode rather than defaulting to a fixed string; `use_cats=False` does warn;
  LVIS "frequent" is more than 100 training images, not 100 or more; `coco
  explore` takes no `--json`; the CLI JSON shape lists `provenance` and
  `reference_deviations`; loader warnings print rather than staying silent;
  `CocoEvaluator` accepts a single `iou_type` string; and `hotcoco.metrics`'
  `is_computed`/`is_missing` are documented. Net effect: about 560 fewer lines of
  documentation saying the same things.
- **`COCOeval.eval` is the same dict object on every access**, as in
  pycocotools, rather than a fresh copy per read. In-place edits persist across
  reads; `summarize()`, `stats`, and `results()` read the evaluator's own
  arrays and do not see them. `eval['params']` is the `params` object
  `accumulate()` ran with, held by reference the way pycocotools holds
  `self.params`, so `ev.eval['params'] is ev.params`. The dict is rebuilt by
  `evaluate()`, `accumulate()`, and `run()`.
- **TIDE's classification and miss-count passes run in parallel**, and the
  per-category delta pass reuses one set of AP scratch buffers across the
  eight ranked-AP evaluations each category needs instead of allocating per
  call: 2.30 s → 1.44 s on Objects365, results bit-identical, with a test
  pinning that the rayon thread count does not change the numbers.
- **The top-level copy defines hotcoco as a perception evaluation toolkit, in plain
  words.** README, the docs home page, and `help(hotcoco)` now open on what the
  toolkit is and does — "hotcoco is a perception evaluation toolkit, written in Rust
  with Python bindings" — with the pycocotools drop-in demoted from *definition* to
  *stated fact* rather than deleted: it still appears inside the first 200 characters
  of every surface, which is where the search traffic lives. Slogan copy is gone
  ("Evaluation that tells you why", "fast enough for every epoch, lean enough for
  every dataset"); every sentence states a concrete fact. README's 20-feature list is
  grouped under four verbs — Evaluate, Diagnose, Explore your data, Compose and
  integrate — and the docs home page's four cards mirror them. The home page runs
  Quick start → Performance → Error analysis → The dataset browser → Use the metrics
  directly: a two-up figure gallery (confusion matrix beside per-category AP, via a
  new `.figure-gallery` grid in `extra.css`), the browse UI screenshot with three
  sentences of context, and `hotcoco.metrics` called on plain numpy arrays. Metric
  counts ("all 34 metrics") became "every COCO metric" on all surfaces — a total
  summed across three iou_types explained nothing. Copy only — no API, feature, or
  benchmark number changed.
- **`ROADMAP.md` states the 1.x ladder.** Detection ships today; 1.1 panoptic, 1.2
  tracking, 1.3 concepts (gated on a feasibility spike), each a sibling on the same
  `primitives` → `metrics` layering, with the verification reference named per family
  (panopticapi, TrackEval). A **Not planned** section records what stays out of
  scope — caption metrics and model-in-the-loop metrics — and why no parity claim is
  possible for them.
- **`STYLE.md` records the style-guide choice as settled.** Alternatives — Microsoft,
  Red Hat, GitLab, and the Diátaxis framework — were reviewed in 2026-08 and Google
  was reaffirmed; the intro now says so to keep the choice from being re-litigated.
  The voice summary adds "plain rather than promotional".
- **`CONTRIBUTING.md` explains why the crate is layered the way it is** — detection is
  the first metric family rather than the only one, so a kernel a second family would
  need belongs in a shared layer the day it is written.
- **Every documentation surface was brought onto the Google style guide.** Page
  titles, section headings, and `zensical.toml` nav labels moved to sentence case
  (`# Mask operations`, `# Working with results`); `&` became `and` in headings and
  bullet labels, which moves those anchors, so the two inbound links to
  `#per-image-diagnostics-label-error-detection` were repointed and every
  cross-reference's link text was realigned to the renamed titles. 21 instances of
  `e.g.`/`i.e.`/`etc.` and 16 "see below"/"the run above" spatial references were
  rewritten to `for example`/`such as` and named-section links. The same pass ran
  over `///` and `//!` doc comments, PyO3 `#[doc]` strings, Python docstrings,
  `.pyi` stubs, and CLI `help=` text — 58 files in all.
- **The `iou_thrs` deviation warning reads `lines might show -1.000`**, not `may
  show`. `may` denotes permission; possibility is `might`. The string is quoted
  verbatim in `docs/api/cocoeval.md`, so both moved together.
- **BREAKING: the plot theme `"cold-brew"` is renamed `"cyanotype"`.** The old
  name raises `ValueError`. The registered colormap is renamed to match:
  `hotcoco_coldbrew` → `hotcoco_cyanotype`. Callers passing the old string must
  update; callers relying on the default argument need no change.
- **BREAKING: the `"warm-slate"`, `"scientific-blue"`, and `"ember"` plot themes
  are removed.** All three carried the espresso chrome the reset exists to
  retire, and keeping them would have meant maintaining three palettes outside
  the design system. `"cyanotype"` and `"cyanotype-dark"` are the full set.
- **The caveat marker moved off red.** Cyanotype spends red exclusively on false
  positives, so the non-leaderboard-comparable flag is now Plum — chart series 5,
  `#7E5580` on light grounds and `#B189B3` on dark. `plot/report.py` and
  `dashboard.py` derive it from the palette instead of hardcoding a hex.
- **Eval semantics are palette-owned.** TP/FP/FN move off the Tailwind primaries
  to `#47714E`/`#B24A2E`/`#5A5FB0` (light) and `#7FBC98`/`#F0A050`/`#9296EE`
  (dark), and the browse canvas overlay now reads them from the CSS tokens
  rather than keeping a second copy — the boxes drawn on an image and the badges
  beside it can no longer disagree.

- **The five format converters now share one contract.** Malformed input raises an
  error naming the file and line/position; records the target format cannot express
  are skipped and **counted** — nothing vanishes uncounted. Missing image dimensions
  are an error wherever geometry must scale (`from_yolo` no longer produces
  degenerate 0×0 geometry; `to_oid` errors instead of skip-counting), with two
  documented exceptions (`from_oid` without `images_dir` keeps 1×1 normalized
  boxes; DOTA dimensions are metadata only). `file_name` is never invented:
  verbatim where the format records it (CVAT, VOC), bare stem otherwise
  (YOLO, DOTA, Open Images — no more fabricated `.png`/`.jpg`). Exports raise on
  file-stem collisions and on an annotation referencing an unknown `category_id`,
  in all five formats.
- **Converter stats dicts renamed to the `skipped_<reason>` scheme**:
  `missing_bbox` → `skipped_no_bbox` in `to_yolo`/`to_voc`/`to_oid`, `to_oid`
  drops `missing_dims` (now an error), and `to_cvat` adds `skipped_degenerate`.
- **`from_cvat` reads real CVAT exports**: shapes written as open/close pairs
  (the form CVAT uses when a shape has `<attribute>` children) now import;
  unsupported shapes and degenerate polygons are counted skips reported with a
  `UserWarning` instead of aborting the file.
- **YOLO `data.yaml` accepts the Ultralytics forms** of `names:` — index-keyed
  dict and block list — in addition to the flow list; exported category names
  containing commas are quoted.
- **VOC conversion applies the devkit's 1-based inclusive convention in both
  directions** (import `x = xmin − 1`, `w = xmax − xmin + 1`; export the
  inverse), accepts float coordinates on import, and imports `<difficult>` to
  `iscrowd` (the mapping was previously export-only).
- **`dota_to_coco`'s Python argument order is `(label_dir, categories,
  image_dims)`**, matching the Rust signature.
- **Converter errors surface as `ValueError` / `IOError`** (was `RuntimeError`),
  and the Rust CLI prints errors via `Display` instead of `Debug`.
- **`coco healthcheck` exits 1 on ERROR-level findings** in both human and
  `--json` modes, so the advertised CI-gate use works; `--help` documents the
  exit status.
- **`split(test_frac=0.0)` / `coco split --test-frac 0.0`** is honored as a
  three-way split with an empty test set instead of falling back to two-way.
- **Browse and the dashboard are fully offline** — DM Sans and plotly.js are
  vendored with the package; DM Serif Display and JetBrains Mono fall back to
  system font stacks.
- **The `-1.0` "not computed" sentinel is handled consistently**: the PDF report
  and dashboard render it as `n/a` (the CLI already did), printed metrics show
  `-1.000` only for the true sentinel, and an unknown area label or max-dets
  value in `summarize()` degrades to `-1.0` instead of silently reporting the
  `"all"` slice under a per-size name.
- **`compare()` raises `ValueError` on mismatched `iou_thrs`, `rec_thrs`,
  `max_dets`, or area ranges** instead of summarizing one run under the other's
  metric catalog.
- **`f_scores()` key style is `F1` / `F2` / `F0.5`** — beta formatted with no
  trailing zeros (was `F2.0`).
- **`ev.stats` matches pycocotools in both states**: an empty list before
  `summarize()`, a numpy `float64` array after (was `None` / plain list).
- **`annToRLE` / `ann_to_rle` returns `{'size': [h, w], 'counts': bytes}`** —
  the pycocotools and `mask.encode` format (was `{'h', 'w', 'counts': list}`).
- **`mask.area` on a list returns a `uint32` array** (pycocotools parity).
- **Pre-`evaluate()`/`accumulate()` guards are catchable `UserWarning`s** —
  previously raw stderr prints, invisible in Jupyter.
- **`params`, `dataset`, `coco_gt`, and `coco_dt` getters return copies** —
  assign back to apply. Now documented, with the pull-edit-assign idiom, in the
  migration guide and API reference.
- **Annotations missing `area` are excluded from explicit area-range queries**
  (`get_ann_ids(area_rng=...)`, `filter(area_rng=...)`) — a documented divergence
  from pycocotools, which raises `KeyError` there.
- **`plot.pr_curve(iou_thr=0.0)` is honored** — an explicit `0.0` was previously
  swallowed by the dispatcher's `iou_thr or 0.5` default.
- **The full benchmark tables live only on the Benchmarks page**; the README
  keeps a headline claim and a link, so the numbers cannot drift apart.

- **ROADMAP.md is forward-looking only** (333 → 49 lines). Shipped items are now
  deleted from it rather than marked `**Shipped.**` and struck through — that
  convention had turned the roadmap into a second, worse changelog in which roughly
  three quarters of the file described work already released. The architecture
  overview it carried (diagram, crate layout, registry table) moved to
  `CONTRIBUTING.md`, where contributor-facing reference belongs.
- **`guide/evaluation.md` reduced from 1,003 to 258 lines**, now covering just the
  evaluation pipeline, the four `iou_type` values, the standard metrics, params, and
  sliced evaluation. Its JSON-export and experiment-tracker sections moved to
  **Working with Results**, which is now the canonical home for both, and for the
  `provenance` explanation that had been written five different ways across five pages.
- **Guide and API pages divided by role**: guides hold worked examples and
  interpretation, API pages hold signatures, parameters, and return shapes. Return-shape
  and parameter tables duplicated into the guides were removed in favor of links.
- **26 repeated camelCase-alias admonitions** across `api/coco.md`, `api/params.md`,
  and `api/mask.md` collapsed into one note per page pointing at the alias table in the
  migration guide.
- **~300 lines of comments removed across 26 source files** with no loss of
  information. The recurring pattern was a change's rationale pasted into the file and
  left permanently — measurements of code that no longer exists, bug post-mortems, and
  in two cases a comment arguing with its own earlier revision. Parity references,
  invariants, and compatibility traps were kept. `hotcoco`'s crate-level docs dropped
  from 83 to 59 lines, with the 1.0 module-rename table moved to the migration guide.
- **Internal planning documents moved from `docs/plans/` to `plans/`** at the repo
  root. Zensical builds every Markdown file under `docs/` whether or not it appears in
  the nav, so planning notes were being rendered into the site output and its search
  index; only `.gitignore` kept them out of CI. They are now outside the docs tree
  entirely.
- **Benchmarks re-measured on 1.0** (they had been published from 0.5.0 and 0.3.0
  builds). On COCO val2017 bbox evaluation is now 0.14s against 5.11s for pycocotools
  — 36.2×, up from a published 33.4× — with segm at 20.8× and keypoints at 18.8×;
  the evaluation engine alone is 35–74× rather than 30–52×. Every figure is the
  per-cell median of three runs. The claim that hotcoco "scales better at higher
  detection counts" was removed: at 10× detections its advantage narrows to 8–33×,
  which the previous numbers already showed for segm and keypoints. The benchmark
  pages now state the **core count** of each machine and note that speedups scale
  with it — hotcoco evaluates in parallel and pycocotools does not, so a ratio
  measured on 8 cores does not describe a 4-core laptop.
- Docs-site description updated from "11-26x faster" (a figure matching no current
  measurement) to "up to 36× faster", and Benchmarks promoted to a top-level nav item.

- **TIDE's `FP` and `FN` ΔAP now follow tidecv's special-oracle definitions,
  and both are parity-gated.** Previously `FP` was the union fix of the five
  error types — which flips Cls/Loc errors into true positives and therefore
  raises recall — and `FN` was a literal alias of `Miss`, carrying no
  independent information. Now `FP` measures perfect precision (every false
  positive suppressed, recall untouched) and `FN` measures perfect recall
  (every unmatched ground truth leaves the denominator, precision untouched),
  exactly as tidecv computes them. **Both values change.** `parity_tide.py`
  gates `FP`, `FN`, and `ap_base` against tidecv at ±0.005 (measured val2017
  diffs: 0.0003, 0.0007, 0.0003); previously all three were printed but never
  compared.

- **Open Images group-of boxes now follow the Challenge protocol.** A group-of box
  is worth exactly one ground truth: the best-scoring detection inside it is a true
  positive, surplus detections inside it are ignored, and an undetected group-of box
  is a single false negative. Previously they were ignored entirely, contributing to
  neither the numerator nor the denominator.

  Both behaviors are real published protocols — the old one is the Open Images
  **V2** detection metric, the new one is the **Challenge** metric (2018/2019), i.e.
  TensorFlow's `group_of_weight = 1.0` and what FiftyOne implements. **OID AP values
  will change.** If you need V2 semantics, open an issue; the two differ by a single
  parameter and an enum is easy to add.

- **Open Images AP now uses VOC 2010 all-points integration.** The protocol says
  detections are "evaluated as in the PASCAL VOC 2010 protocol", and both reference
  implementations follow it — TensorFlow's `compute_average_precision` and
  FiftyOne's `_compute_AP`, the latter alongside a separate 101-point path for its
  COCO evaluation. hotcoco was reusing COCO's 101-point recall grid, which quantizes
  the result by up to ~`1/101` per class. **OID AP values change**; COCO and LVIS are
  untouched and keep the 101-point grid.

- **Open Images is now parity-checked against the TensorFlow Object Detection API.**
  `scripts/parity_oid.py` compares 70 cases — mAP *and* per-class AP — against
  frozen output from `OpenImagesDetectionEvaluator(group_of_weight=1.0)`. Worst
  difference **1.11e-16**, one ulp. Runs in CI; needs neither network nor
  TensorFlow. Regenerate fixtures with `just gen-oid-fixtures`.

  Open Images stays `Provenance::Extension`, but the warning now names the real gap:
  the challenge's non-exhaustive image-level-label rule is not implemented (it needs
  per-image label data COCO JSON cannot carry). It previously claimed no reference
  implementation existed, which was untrue.

- **`COCOeval::results()` and the Python result dicts are now byte-stable.** They
  used `HashMap`, so three identical runs produced three different key orders and
  anything archived, hashed, or diffed in CI churned for no reason. `EvalResults`
  is `BTreeMap` throughout and the PyO3 dict builders sort before inserting, since
  Python dicts preserve insertion order. `report::EvalReport` already did this.

- **`reference_deviations()` covers four more ways a run stops being comparable.**
  It now flags non-default area-range **bounds** (not just labels), `rec_thrs`,
  `use_cats=False`, and custom `kpt_oks_sigmas`. Each changes what the metric
  means while leaving its name intact, and each previously reported
  `parity_verified`.

  The area-range case was the sharpest: the check compared labels only, and the
  Python `params.areaRng` setter deliberately *preserves* labels — so the one path
  a caller takes to redefine "small" was the one the guard could not see.

  The early return for modes without a checked reference is now an exhaustive
  `match` rather than a default-allow `!=`, so a future `EvalMode` cannot inherit
  `parity_verified` by omission.

- **`summarize()` selects the nearest IoU threshold rather than every threshold
  within 1e-9.** A params list holding two thresholds that close silently reported
  their average under the name of one of them.

- **Open Images: detections absorbed by a group-of box are now ignored, not counted
  as true positives.** This changes Open Images AP values.

  A group-of GT carries no false-negative penalty, so it never enters the recall
  denominator. Crediting every detection that overlapped one as a true positive grew
  the numerator against a denominator that could not grow, and `recall` was unbounded
  above 1.0 — measured at 4.0 on a one-image case with three detections on a single
  group-of box. Absorbed detections now score as neither true positives nor false
  positives, which is what the reference `OpenImagesChallengeEvaluator` does and what
  keeps the metric coherent.

  Open Images reports only AP@0.5, and AP was unaffected by the old behavior, so most
  users will see no change in the summary line. `AccumulatedEval.recall` is public,
  however, and was wrong for any Open Images run with a detection on a group-of box.

  Found by an auditability sweep, not by a parity test — Open Images has no reference
  implementation checked against, which is why `report()` marks it
  `Provenance::Extension`. The one test covering this path used detections at IoU 0.16
  against the group-of box and never reached it; it now asserts the absorption itself
  alongside `recall <= 1.0`.

- **`primitives` narrowed to the matching kernels; the metric functions moved to
  `metrics`.** `primitives` now holds `sim`, `greedy`, and `assign` — the three kernels
  that produce matches. `primitives::counts` moved to `metrics::counts`, because
  computing AP from match flags is scoring, not matching.

  `EvalReport` and `Provenance` live in their own top-level `hotcoco::report` — they
  are the cross-family output *contract*, a peer of `metrics` and `primitives`
  rather than a member of either, and `metrics` is free functions over arrays. The
  crate-root `hotcoco::EvalReport` and `hotcoco::Provenance` paths are unchanged.

- The detection analysis methods are now adapters over `metrics`. `COCOeval::calibration`,
  `confusion_matrix`, and `compare` used to each own their own math; the math now lives
  in `metrics::calibration`, `metrics::confusion`, and `metrics::bootstrap`, and the
  methods marshal `eval_imgs` into arrays and call it. Return types and values are
  unchanged. `bootstrap_ci` takes the statistic as a closure, so it computes intervals
  for any resampled quantity rather than only detection metric deltas.

  Detection keeps what is genuinely detection-shaped: TIDE's Cls/Loc/Both/Dupe/Bkg
  taxonomy is about box localization versus classification, and `image_diagnostics`
  reports per-image fields. Both stay in `detection`.

- `detection::metrics` (the internal `MetricDef` catalog) is now `detection::catalog`, so
  it no longer collides with the crate-level `metrics`. It was already private.

- **The Rust `eval` module is now `detection`.** `eval` was its name while detection was
  the only metric family in the crate; it is now one family beside the panoptic,
  tracking, and concepts families that follow.

  **The Python API is completely unaffected.** `hotcoco.COCOeval`, `init_as_pycocotools()`,
  and the whole `pycocotools`/LVIS drop-in surface are permanent compatibility
  guarantees.

- Two relocations for the same reason:
  - `hotcoco::hierarchy` → `hotcoco::detection::hierarchy`. The Open Images label
    hierarchy is consumed only by the detection family's GT/DT expansion, so its
    top-level placement wrongly implied it was a cross-family primitive.
  - `hotcoco::healthcheck` → `hotcoco::quality::healthcheck`, joined by `COCO::stats`
    and the `SummaryStats`/`CategoryStats`/`DatasetStats` DTOs, which moved out of
    `hotcoco::types`. Dataset quality is its own tier, distinct from the schema
    (`types` says what a COCO file may contain) and from the metrics engine
    (`detection` scores predictions against one).

- `eval/types.rs` is dissolved. It held types for four unrelated concerns plus two
  sibling features' public types, so "where is `EvalImg` defined?" answered "in a junk
  drawer" rather than "in the matching stage". Each type now lives next to the stage
  that produces it: `EvalImg`/`EvalImgContext`/`IouMatrix` → `eval/matching.rs`,
  `AccumulatedEval`/`EvalShape` → `eval/accumulate.rs`, `ConfusionMatrix` →
  `eval/confusion.rs`, `TideErrors` → `eval/tide.rs`, and `EvalMode` plus the LVIS
  `FreqGroup`/`FreqGroups` buckets → a new `eval/mode.rs`. Every public path is
  unchanged — `hotcoco::EvalImg` and `hotcoco::eval::EvalImg` both still resolve.

- The analysis layer no longer reaches into `COCOeval`'s private state. TIDE read the
  whole-dataset `ious` similarity cache directly, and `compare`/`slice`/`report` read
  `freq_groups`; both now go through driver-private accessors. `cell_ious(img_id,
  cat_id)` deliberately returns **one cell rather than the map** — the 0.5 primitives
  contract review identified that cache as the likeliest route by which retention leaks
  into a shared contract, which would foreclose the recompute-instead-of-retain lever
  the tracking family needs. A new `similarity_cache_stays_driver_private` conformance
  test in `tests/architecture.rs` fails the build if anything outside the driver touches
  the field.

- Per-image matching moved out of `eval/evaluate.rs` into a new `eval/matching.rs`, and
  the 229-line `evaluate_img_static` is now four named steps over two explicit views:
  `gather_gt` (load + apply the mode-dependent ignore rules + partition
  non-ignored-first), `gather_dt` (load + score-descending order + `max_det` cap),
  `reordered_iou` (build the flat matrix the matcher's contract expects), and
  `match_cell` (invoke `greedy_match`, translate indices back to annotation ids, run
  the Open Images group-of pass). `evaluate.rs` is now purely the outer driver —
  parameter resolution, sparse-pair collection, and the parallel fan-out — with no
  matching math of its own. Behavior unchanged.

- `eval/summarize.rs` is split by responsibility. It was 861 lines doing five unrelated
  jobs — the metric catalog, the reduction, output formatting, pipeline orchestration,
  and `f_scores` — which made "where is AP75 defined?" and "where is it computed?" the
  same unhelpful answer. Now: `eval/metrics.rs` owns the catalog (pure `MetricDef`
  configuration, no evaluation data), `eval/summarize.rs` owns only the reduction
  (accumulated arrays + definitions → numbers, 127 lines), and `eval/report.rs` owns
  presentation (`summarize_lines`, `summarize`, `metric_keys`, `get_results`,
  `f_scores`, `print_results`, `results`). `run()` moved to the `COCOeval` facade in
  `eval/mod.rs`, where the other pipeline entry points live. No public path changed and
  no behavior changed.

- TIDE's false-positive classifier is now a free `classify_fp` function over an
  explicit `FpEvidence` struct, with `ErrType` promoted out of the function body.
  The tidecv priority order (`Loc > Cls > Dupe > Bkg > Both`) is a parity contract,
  and it previously lived as an inline `else` block inside a nested loop with a
  function-local enum — so it could not be read or tested on its own. It now carries
  the priority table in its docs and has unit tests covering every variant, each
  precedence pair, and the inclusive interval bounds. Behavior is unchanged.
  `ErrType::as_str` also replaces a second, separate enumeration of the variants in
  the aggregation step.

- **`iou_type` serializes lowercase.** `results()["params"]["iou_type"]` and saved
  results files now say `"bbox"` / `"segm"` / `"keypoints"` instead of `"Bbox"` / …,
  matching `Display` and `FromStr` — an archived results file round-trips into
  `COCOeval(...)` and the CLI without case-fixing.

- **`f_scores()` keys renamed** from `F150` / `F175` to `F1_50` / `F1_75` (general
  form `F{beta}_50` / `F{beta}_75`) — the names the documentation always advertised.

- **Byte-stable output ordering, now in the types.** `get_results()`, `f_scores()`,
  and `compare()`'s metric/delta/CI maps are ordered (`BTreeMap`) in Rust like
  `EvalResults` / `EvalReport` already were, instead of being sorted only at the
  Python boundary — the Rust API and serialized JSON no longer churn between runs.

- **`confusion_matrix()["matrix"]` dtype is `uint64`.** It was `int64` from
  `COCOeval.confusion_matrix()` but `uint64` from `hotcoco.metrics` — the two entry
  points now agree, via one shared converter.

- **`ev.params` mutations reach every method.** `confusion_matrix()`, `tide_errors()`,
  `calibration()`, `slice_by()`, `image_diagnostics()`, `get_results()`, `report()`,
  `f_scores()`, `results()`, and `save_results()` now observe params edits made before
  the call; previously `ev.params.catIds = [...]` was silently ignored by analysis
  methods until the next `evaluate()` — despite `confusion_matrix` being documented
  as standalone.

- **Presentation consistency across surfaces:** the CLI TIDE table lists error types
  in canonical order (Cls, Loc, Both, Dupe, Bkg, Miss — it led with Loc);
  `coco compare` prints `n/a` for not-computed metrics instead of `-1.000`; the
  dashboard and PDF recall KPI tile shows `AR100` / `AR@300` / `AR` per mode instead
  of `AR1`; `--title` defaults to a mode-derived report title ("LVIS Evaluation
  Report", …); `--json` runs of `eval` / `compare` / `convert` no longer animate
  spinners; and the PDF footer version comes from the extension itself rather than a
  best-effort `importlib.metadata` lookup that could silently vanish.

- **Evaluation allocates far less.** Flat per-threshold match matrices
  (`ThreshMatrix`), a per-`evaluate()` RLE cache mirroring pycocotools' `_prepare`
  (`SegmRles`), thread-local polygon-rasterization scratch, and per-(image, category)
  gathering shared across the four area ranges: evaluate + accumulate wall time on
  val2017 dropped 44% (bbox), 24% (segm), 32% (keypoints). Output verified
  bit-identical.

- **JSON loading is ~2× faster.** simd-json for parsing, a hand-written streaming
  `Segmentation` deserializer (replacing `#[serde(untagged)]`, which buffered every
  polygon twice regardless of parser), and a SIMD substring prefilter for the
  NaN/Infinity sanitizer: `COCO::new` on the val2017 instances file went 105 ms →
  58 ms, and the end-to-end bbox benchmark headline moved from 24× to 33× vs
  pycocotools.

- **Analysis surfaces scale.** The confusion matrix accumulates sparse label pairs
  instead of a dense (K+1)² grid per rayon split (528 ms → single-digit ms with
  1,000 mostly-empty categories) and collects per-image pairs in O(annotations)
  rather than O(images × categories); `accumulate` visits each (category, area)
  bucket once with the maxDets loop inside (~40% faster — bootstrap `compare` is
  accumulate-bound); TIDE sorts each category's detections twice instead of eight
  times and parallelizes across categories. All verified bit-identical on val2017
  and Objects365.

- **In-code documentation is contract, not changelog.** A pass over every comment
  and doc comment in the crate removed the refactor-history narration the 1.0 cycle
  accumulated ("spelled out at five sites", "free to drift from", 64 sites across
  `src/`), keeping the mechanism and contract explanations and cutting facts that
  were restated at three to five call sites down to one owner plus links. Net −100
  comment lines with no code change. `detection`'s module docs now describe LVIS,
  Open Images and TIDE rather than calling the module a port of `cocoeval.py`; the
  two `COCOeval` examples changed from ```` ```rust,ignore ```` to `no_run`, so
  rustdoc compiles them (8 → 10 doctests); and a paragraph stranded under a
  `# Panics` heading in `metrics::counts`, a broken intra-doc link, and a reference
  to a file under gitignored `plans/` are gone. `cargo doc` is warning-free.

- **American English throughout, and `typos` is a usable gate.** 39 spellings
  corrected in comments, docstrings, scripts, the 101 notebook, and the CHANGELOG.
  `_typos.toml` excluded `python/hotcoco/static/` wholesale to silence the vendored
  Plotly bundle, which also hid 3,492 lines of first-party browse-UI JS and CSS —
  now `*.min.js` / `*.ttf` by extension. Four abbreviated identifiers in tests and
  scripts that the checker read as misspellings were renamed rather than
  allowlisted — they are now `iou_fwd`/`iou_rev`, `hc_gt_ignore`, `dog_ancestors`,
  and `miscopied` — leaving one entry, under `extend-identifiers` rather than
  `extend-words` so it cannot suppress a word in prose. `typos .` exits clean
  repo-wide.

- **`COCOeval::evaluate` and the confusion matrix call
  `primitives::greedy::greedy_match_masked`**, whose `GtMasks` names the two
  per-GT policy masks. Both sites previously used the positional `greedy_match`,
  where the adjacent `Option<&[bool]>` arguments transpose silently — so the type
  that exists to prevent that mix-up had no callers. The switch is allocation-
  and result-identical, verified by val2017 parity; the positional wrapper is
  gone (see Removed).

### Fixed

- **Polygon rasterization matched pycocotools on arm64 only.** `mask::fr_poly`
  fused the edge interpolation `s*t+ys` into one multiply-add on every platform,
  on the premise that every shipped pycocotools wheel does. Only the arm64 wheels
  do; the x86-64 wheels on PyPI target baseline x86-64, which has no FMA
  instruction, so there the reference rounds twice and about 2 in 400 random
  polygons differ by a boundary pixel. The rasterizer now picks the arithmetic by
  target architecture, so `hotcoco.mask` agrees with the pycocotools installed on
  the same machine on both. Found by the 1.0.0 verify gate on Linux after the
  mask parity had passed on an arm64 Mac, where the fused form is the right one.
- **`scripts/test_parity.py` did not import on Python 3.9.** A `float | None`
  default in a helper signature is evaluated at definition time without
  `from __future__ import annotations`; the file now has it, like every other
  script CI runs.
- **CI tested a package nobody installs.** The Python job built its wheel from
  inside `crates/hotcoco-pyo3`, which yields a bare extension module with no
  `hotcoco.plot`, `cli`, or `integrations`, so every test of the Python layer
  either never ran or failed — the first `v1.0.0` tag failed its own verify gate
  on `ModuleNotFoundError: hotcoco.plot`, and CI on `main` had been green for
  months against the wrong artifact. The job now builds from the repo root, where
  `pyproject.toml`'s `python-source` pulls in the package, installs matplotlib, and
  runs the theme tests too. The GitHub Release job also waited on nothing but the
  wheel builds under `always()`, so it created a release page with zero assets for
  a tag that published nothing; it now depends on both publish jobs.

- **Result files carrying both `segmentation` and `keypoints` are typed as
  segmentation**, matching pycocotools' `elif` precedence — previously keypoints
  won, a silent parity divergence.
- **`evaluate()` no longer panics on empty `max_dets`** — it degrades to the
  default detection cap and `-1.0` summary stats, like every sibling entry point.
- **Malformed RLE strings, polygon coordinates, and mask dimensions now raise
  errors** instead of panicking or overflowing — the mask kernels no longer trust
  values from untrusted annotation files.
- **Extension import failures explain themselves**: when the package is present
  but its compiled extension fails to import, the CLI says so and suggests
  reinstalling, rather than claiming hotcoco is not installed.

- **`reliability_diagram` drew three colors the theme never chose.** Its ECE/MCE
  annotation box was `facecolor="white"`, so on `cyanotype-dark` it painted a white
  box under `#EAEAEA` text and both numbers were invisible; the calibration gap was
  matplotlib's `firebrick` and the reference diagonal `gray`. Firebrick was also a
  second red, which the rule reserving red for false positives exists to prevent.
  The gap now takes the FP semantic for the theme's ground, and the box and diagonal
  read the resolved chrome off `rcParams` inside the rc context — so they follow
  `paper_mode`'s swapped grounds too, which the theme dict cannot. The gap bars are
  also opaque in both directions: the solid gap sits on the plot background and the
  hatched one on top of an accuracy bar, so the old `alpha=0.35`/`0.5` composited one
  semantic into two colors — and resolved the amber to `#664A30`, a brown.
- **`pr_curve`'s ten-threshold default was unreadable, and `iou_thrs=[0.90]` drew
  nothing.** Ten curves sat under a ten-entry legend the IoU=0.65 line ran through,
  and the fill under the topmost curve tinted every curve below it. The legend now
  moves outside the axes past four curves, and the fill applies only at two or fewer.
  Separately, the threshold filter compared floats with `==` while the default grid
  stores 0.90 as `0.8999999999999999`, so the requested line was silently dropped and
  an empty axes drawn. `PlotData.iou_indices` now matches within a tolerance and
  raises when a threshold genuinely is not in the run's grid.
- **The dataset browser screenshot in the docs still showed the retired palette.**
  `browse-ui.webp` is the one asset `scripts/gen_docs_assets.py` cannot produce — it
  is captured by hand from a running server — so it kept shipping Cold Brew's warm
  browns while every generated figure followed the Cyanotype swap. The browse
  stylesheet itself was already correct; only the image was stale.

- **VOC conversion reported a `<part>`'s bounding box as its object's.** A VOC2012
  `<object>` may contain `<part>` sub-elements describing regions of itself — a person's
  head, hand, or foot — each with its own `<name>` and `<bndbox>`. The parser skipped a
  part's `<name>` but not its `<bndbox>`, and because parts always follow the object's
  own box in VOC2012, the *last part's* coordinates overwrote the object's. Every
  annotation with parts converted to the wrong box, silently, on the most common class
  in the dataset. Nothing caught it: no test used a `<part>` element, and the guard was
  spelled out on one branch of a five-flag state machine and forgotten on the next.
  The parser now tracks a single position value in which "inside a part" and "inside the
  object's box" are mutually exclusive, so there is no second branch to forget.
- **`mask.fr_py_objects` was unreachable, and `mask.fr_py_objects_snake` was public
  instead.** The snake_case alias carried no `#[pyo3(name)]`, so PyO3 exported the Rust
  identifier verbatim. Every other mask function ships both spellings, and both the API
  reference and the migration guide's alias table promised this one, so the documented
  call raised `AttributeError` while a name no one would type was the only way in.
- **`COCO.toYolo`, `COCO.fromYolo` and `Params.expandDt` were declared in the type stubs
  but never defined**, so an IDE would autocomplete all three into an `AttributeError`.
  The stub conformance test only asserted that every runtime member appears in the
  stub — never the reverse — so a stub could invent members freely.
  `test_members_covered` now checks both directions; it fails on all four defects
  above when run against the previous build.
- **Two documented Python APIs did not exist.** `COCO.to_dota()` / `from_dota()` were
  described with worked examples on five surfaces including the README, and
  `Hierarchy.from_categories` in the evaluation guide and API reference; both were
  Rust-only, so the Python examples raised `AttributeError`. DOTA is resolved by
  binding it (see Added). `Hierarchy.from_categories` stays Rust-only, and the
  hierarchy guidance now shows the automatic `supercategory` derivation that
  `oid_style=True` performs when no hierarchy is passed.
- **`COCO.browse()` was documented with 4 of its 8 parameters**, omitting `iou_type`,
  `iou_thr`, `slices`, and `eval` — the last being the one the README points at for the
  evaluation dashboard.
- **`iou_type` documentation omitted `"obb"`** in `api/cocoeval.md` and `api/params.md`,
  contradicting the guide and README. The CLI's `--iou-type` genuinely does not accept
  it, which is now stated rather than left as a surprise.
- **Stale version references**: the Rust install snippet pinned `hotcoco = "0.4"`, the
  CLI's sample JSON output showed `"hotcoco_version": "0.3.0"`, and the val2017
  benchmark table was labeled 0.5.0 despite reporting 1.0 timings.
- **A citation of hotcoco 0.6.0**, a release that never existed, in the Open Images
  protocol note.
- **The docs-site "Open Notebook" button and a quickstart link** pointed at a relative
  path outside `docs/`, which 404s on the published site; both now use the repository
  URL.
- **`api/integrations.md` showed a `CocoEvaluator` import path that does not exist**
  in torchvision (`torchvision.models.detection.coco_utils`).
- **A doc comment placed between `#[derive]` and `#[non_exhaustive]`** on
  `AccumulatedEval`, separating the struct from its documentation.
- **Two comments in `metrics/calibration.rs` contradicted each other** about whether
  out-of-range score handling was settled policy or an open question; the test now
  pins the documented behavior.

- **Unsorted `max_dets` no longer silently empties `image_diagnostics()`.** Five
  sites each derived the per-image detection cap independently — four spelled
  `max_dets.last()`, one spelled the maximum. Identical on the sorted default
  `[1, 10, 100]`, divergent on unsorted input: `evaluate()` stamped eval images
  with one value while diagnostics filtered on the other and matched nothing.
  `Params::max_det()` now owns the reduction (the maximum, as the name says — the
  same value pycocotools reaches by sorting `maxDets` in place), an architecture
  test bans any other derivation, and evaluation results are now independent of
  `max_dets` order.

- **Ground-truth annotations now feed the matcher in JSON array order,** exactly
  as pycocotools builds `_gts`, instead of being re-sorted by annotation id. The
  order is observable: on an exact IoU tie the greedy matcher takes the later
  ground truth, so files whose annotation ids are not in array order — converted
  or merged datasets, typically — could match a different GT than pycocotools.
  Official COCO files are id-ordered, so their numbers cannot move.

- **A keypoints PDF report was titled "COCO Evaluation Report".** `PlotData.iou_type`
  carried the Rust enum's spelling (`"Bbox"`, `"Keypoints"`) while `eval_mode` arrived
  lowercase, so `iou_type == "keypoints"` was never true. Lowercased at the boundary,
  where the field's documented contract already said it was.

- **Open Images group-of matching used IoU instead of IoA.** The protocol says a
  detection is inside a group-of box when intersection divided by the *detection's*
  area exceeds 0.5. hotcoco forced the plain-IoU formula for every OID ground truth,
  so a detection smaller than the group box — an individual object inside a cluster,
  the normal case — was never absorbed and leaked out as a false positive. An 80×80
  detection wholly inside a 200×200 group-of box scores IoA 1.00 but IoU 0.16, and
  was counted as an error.

  The tests that should have caught this used geometry below the threshold, so the
  group-of code path never ran; they passed on interpolation instead. They now assert
  the measure, and place the false positive before full recall where AP can see it.

- **Group-of tie-breaking disagreed with the reference.** When a detection sat
  inside two overlapping group-of boxes at equal IoA — which containment makes
  common, since it saturates at 1.0 — hotcoco credited the later box and left the
  earlier one permanently unmatched, converting a true positive into a miss. The
  reference's `np.argmax` takes the first maximum. The rule now lives in
  `primitives::greedy::best_above_floor` so a second caller cannot re-derive it.
  Found by the new Open Images parity check.

- **`AccumulatedEval` gained `ap_all_points`**, the exact area under the precision
  envelope, shaped and indexed like `recall`. `precision` holds that same envelope
  sampled at the 101 recall thresholds. `AccumulatedEval` is now `#[non_exhaustive]`
  so later fields are additive — done while still pre-1.0, when it is free.

- **`EvalImg` gained `gt_in_denominator`** (`gtInDenominator` in Python) and is now
  `#[non_exhaustive]`. `gt_ignore` means "excluded from matching", which for group-of
  boxes is no longer the same as "excluded from the recall denominator". Code
  computing `num_gt` should read the new field.

- **`ev.params.imgIds = [...]` was a silent no-op.** The `params` getter cloned
  into a fresh object on every access, so pycocotools' canonical configuration
  idiom - the one in its own demo - mutated a temporary and evaluation proceeded
  over the whole dataset anyway. `params` is now a persistent object, reconciled
  with the evaluator by a single `with_params` helper that `evaluate()`, `run()`
  and `summarize()` all route through: pulled in before, pushed back after, so a
  caller reading `ev.params.imgIds` sees the resolved list as pycocotools leaves
  it.

  One helper rather than one call site, because the first attempt patched
  `evaluate()` alone and left two holes: `run()` - the path the docstring points
  LVIS/Detectron2/MMDetection users at - still ignored `params` entirely, and a
  mutation *after* `evaluate()` stayed invisible to `summarize()`, which then
  reported `parity_verified` for an off-reference configuration. That is the
  silent downgrade `reference_deviations()` exists to prevent, arriving from the
  other direction.

- **Comparability warnings are now real Python warnings.** `summarize()` writes
  them with `eprintln!`, which goes to file descriptor 2 and bypasses
  `sys.stderr` - invisible in a Jupyter cell, invisible to `capsys`, uncatchable
  by `warnings.catch_warnings`. Notebook users are the primary audience and never
  saw them. `COCOeval::reference_deviations()` is public for the same reason:
  `Provenance` is one bit, and the reason is what a caller can act on.

- **`ImageSummary.ap` had no assertion anywhere in the repo**, despite surfacing
  in `coco eval --diagnostics`, the browse viewer, and the dashboard. Now pinned
  by closed-form cases, including the `n_gt == 0` convention that is deliberately
  the opposite of TIDE's.

- **`load_res()` rejects NaN detection scores.** Ranking sorts with
  `partial_cmp(..).unwrap_or(Equal)`, which is not transitive once NaN is present:
  the sort does not panic, it produces an arbitrary order, and AP becomes a
  function of the sort implementation. Loading from a *file* was already safe —
  `sanitize_non_finite` rewrites bare `NaN` to `null` — so this closes the
  programmatic path.

- **`compare()` paired per-category results by position instead of category id.**
  When two evaluators carried different category lists — different GT files, or the
  same file filtered differently — category *i* of model A was compared against
  whatever model B evaluated in slot *i*, and reported under A's name. `compare()`
  validates `eval_mode` and `iou_type` but never the category lists, so nothing
  caught it. Both sides are now keyed by category id, and the iteration covers the
  union so a category only one side evaluated is reported rather than silently
  dropped.

  Per-category deltas also treated a category missing from one side as a swing of up
  to 1.0 rather than as no evidence, so a category absent from B sorted to the top of
  the "worst regressions first" table. They now use the same `metric_delta` the
  summary metrics and bootstrap CIs use.

  Invisible to the previous tests, which all compared a model to itself — the case
  where positional and id-keyed lookup are indistinguishable.

- **`COCOeval::calibration()` now rejects detection scores outside `[0, 1]`.**
  Previously they were silently accepted and produced a meaningless result.

  Binning buckets by `score * n_bins` and clamps the *index*, not the score, so an
  out-of-range value saturates into an end bin and carries its raw magnitude into
  that bin's mean — a model exporting raw logits got a calibration error above 1.0
  with nothing to indicate why. `[0, 1]` was already a documented precondition;
  it is now enforced, with an error naming the offending score, how many
  detections are affected, and what to do about it.

- **Polygon rasterization now reproduces the reference's floating-point
  arithmetic, making segmentation parity exact.** All 12 segmentation metrics moved
  from ~1e-5 to ~1e-14 on COCO val2017.

  `maskApi.c` writes `(int)(ys+s*t+.5)`, and both clang and gcc default to
  `-ffp-contract=fast` — so every shipped pycocotools wheel fuses `s*t+ys` into a
  single FMA, one rounding where the unfused form has two. Rust never contracts
  implicitly, so the plain expression was a *more accurate* computation that
  disagreed with the reference. `mask::fr_poly` now uses `f64::mul_add`.

  It bites only at a boundary — with `s = -5/6`, `t = 57`, `ys = 75` the product
  lands a hair either side of `-47.5`, so the two forms round to 28 and 27 and the
  polygon differs by one pixel. Two of 400 random polygons hit it. But COCO
  ground-truth segmentations *are* polygons, so this path builds every segm GT
  mask, and one pixel was the whole residual.

- **`mask.frPyObjects` produced empty masks for bounding-box input.** It routed
  every coordinate list to the polygon rasterizer, which returns an empty RLE below
  three points — so a 4-element box was read as a 2-point polygon and rasterized to
  nothing, silently. It now dispatches on entry length the way pycocotools does:
  exactly 4 values is a box, more than 4 is a flattened polygon. Entries shorter
  than 4 raise instead of returning a blank mask.

  hotcoco accepts boxes as either a list of lists or a numpy array; pycocotools
  requires an array and raises `TypeError` on the list form. Being more permissive
  is safe — nothing that worked against pycocotools changes.

- **`mask.iou` and `mask.bbox_iou` rejected integer `iscrowd`.** COCO JSON stores
  `iscrowd` as `0`/`1`, so the idiomatic
  `maskUtils.iou(dt, gt, [a["iscrowd"] for a in anns])` — and any numpy array —
  raised `TypeError` against a `Vec<bool>` parameter. Both now accept bools, ints,
  and numpy arrays of either, matching what the crate already does when
  deserializing annotations. Non-numeric entries still raise, naming the type.

- **The adversarial corpus is tracked and actually runs.** 18 hand-curated edge
  cases (all-crowd, area boundaries, 1x1 images, boxes outside the frame) were
  gitignored, so they verified nothing on any machine but the author's, and
  nothing iterated them. `scripts/test_adversarial.py` runs them under pytest and
  in CI, comparing per-(image, category) matching *decisions* against
  pycocotools annotation by annotation — a check metrics cannot replace, since
  two detections swapped between images can leave AP identical to fifteen decimal
  places. Level 2 also runs unconditionally now; it used to be gated on level 1
  having already failed, which made it unreachable in exactly the case it is good
  at.

- **A pinned val2017 baseline** (`scripts/fixtures/val2017_expected.json`),
  produced by pycocotools rather than by hotcoco. `parity.py` checks it alongside
  the live comparison, which catches what the live run structurally cannot:
  hotcoco and the reference drifting *together*, as a pycocotools upgrade that
  silently changed a metric would.

- **The fuzzer checks invariants, not only parity.** It was purely differential,
  so its ~10,000 generated datasets only ever exercised surfaces pycocotools also
  computes — leaving Open Images, oriented boxes, LVIS frequency groups, TIDE,
  calibration, and the confusion matrix with no fuzz coverage at all, which are
  precisely the surfaces marked `Provenance::Extension` because no reference
  exists.

- **New: `scripts/parity_mask.py`,** a differential test of every `hotcoco.mask`
  operation against `pycocotools.mask` — encode, decode, round-trip, area, toBbox,
  iou across crowd modes, merge, the string codec, and `frPyObjects` for both
  polygons and boxes. Compared bit-for-bit, since RLE is integer run lengths and
  masks are `uint8`. The RLE codec was previously covered only transitively through
  segmentation AP, which never reached `merge`, `frPyObjects`, or the string codec —
  all three of the bugs above were in that gap.

- **The default IoU and recall grids now match `numpy.linspace` bit-for-bit.**
  This shifts metric values in the last few decimal places, closer to
  pycocotools.

  pycocotools builds both grids with `np.linspace`; hotcoco built them as
  `0.5 + 0.05 * i` and `i / 100.0`. Those disagree with numpy by one ulp at 2 of
  the 10 IoU thresholds and 10 of the 101 recall thresholds, because numpy
  computes a single `step` once and multiplies where the other forms round twice
  or divide exactly.

  One ulp is not harmless on the recall grid. `recall = tp / num_gt` is a ratio of
  small integers, so it lands exactly on a grid point routinely — at
  `num_gt = 20, tp = 7` it equaled the old `rec_thrs[35]` bit-for-bit while
  sitting strictly below numpy's, so the two-pointer scan stopped one detection
  early and reported a different precision there. Neither grid is more *correct*,
  which is exactly why matching the reference is free. `params::linspace` is now
  the single owner of both, pinned against captured numpy bit patterns.

- **Property tests over `primitives` and `metrics`.** Randomized coverage of the
  matcher contract (injectivity, output agreement, threshold clearance, phase-2
  eligibility), bbox IoU algebra, PR-curve well-formedness, `f_beta` bounds,
  confusion marginals, and calibration binning — roughly 80k generated cases,
  each verified to fail against an injected violation. These check hotcoco
  against its own stated contracts rather than against a reference, which is the
  only kind of check available on Open Images and oriented boxes, where no
  reference exists. The Open Images recall bug above was found this way.

- **Corrected the `primitives::greedy` note on exact-duplicate matching.** It
  claimed identical geometry yields exactly `1.0`, and concluded that unclamped
  callers therefore still match duplicates at `t == 1.0`. The intersection extent
  is computed as `(x + w) - x`, which does not round-trip to `w` in binary
  floating point, so a box against itself gives `0.9999999999999993`. pycocotools
  computes it the same way — the kernel is right and unchanged; the note was
  wrong, and backwards: the clamp is what makes duplicate matching work at
  `t == 1.0`, not a redundancy. For sub-pixel geometry even the clamp is not
  enough. Both regimes are now pinned by property tests.

- **`from lvis import LVISEval` now works after `init_as_lvis()`.** lvis-api spells the
  class `LVISEval` with a capital E, and that is what Detectron2 and MMDetection import.
  hotcoco exported only `LVISeval`, following pycocotools' `COCOeval` — so `init_as_lvis()`
  registered a `lvis` module that the canonical import could not use. Both spellings now
  exist and are the same object. Found by the 1.0 third-party-consumer smoke test, which
  is now a regression test.

- `scripts/parity_tide.py` can now fail. It exited 0 in both failure modes: when a ΔAP
  exceeded tolerance it printed "Some values exceed tolerance" and fell through, and
  when `tidecv` was absent it printed a skip notice and exited 0 after comparing
  hotcoco's numbers against nothing. Both now exit non-zero, and `tidecv` joins the dev
  extras beside the other reference implementations so it is installed rather than
  silently missing.

- `import hotcoco.mask` now works. It raised `ModuleNotFoundError` while
  `from hotcoco import mask` succeeded, because PyO3's `add_submodule` makes a submodule
  reachable as an attribute without registering it in `sys.modules`. Anyone migrating
  from `import pycocotools.mask` writes the failing form — and `init_as_pycocotools()`
  masked the problem, since it registers `pycocotools.mask` explicitly and so always
  worked.

- Detection matching now applies pycocotools' match floor. pycocotools starts each
  detection's search at `min(t, 1 - 1e-10)` rather than at `t`; hotcoco compared
  against the raw threshold, so a detection whose IoU fell in `[1 - 1e-10, 1.0)`
  matched in pycocotools but not in hotcoco. The clamp is inert for every threshold
  below 1.0, so the default `0.50:0.05:0.95` sweep and every published metric are
  unchanged — it is observable only at `iou_thr == 1.0`. On COCO val2017 at
  `iou_thr = 1.0` this closed a real divergence: 7,688 match decisions before,
  13,724 after, which is exactly pycocotools' count. AP is unaffected on that data
  because the affected pairs sit in ignored partitions, but on non-ignored geometry
  the difference is material (AP 0.25 → 1.0 on a two-image regression fixture).

  The policy is now settled and documented as a table in `primitives::greedy`:
  the detection `evaluate()` path (including the Open Images group-of pass) clamps;
  TIDE does not, because its parity contract is *tidecv* rather than pycocotools;
  and the confusion matrix, per-image diagnostics, and calibration do not, because
  they are hotcoco-native analysis over a user-chosen threshold.
  `greedy_match_masked` itself adds no epsilon of its own — the clamp remains
  caller-applied.

- **Per-class AP and F-scores could read the wrong maxDets slice.** With unsorted
  `max_dets` (e.g. `[100, 10, 1]`), `report()["per_class"]`, `compare()`'s
  per-category table, and `f_scores()` silently used the *last* slot (maxDets = 1)
  while the headline AP used the largest — 0.63 vs 0.91 within one `report()`. All
  three now resolve through `Params::max_det_idx()`; a regression test pins it.

- **`coco.browse(eval=ev)` rendered no dashboard.** The documented invocation only
  honored `eval=` when `dt=` was also passed; it now uses the evaluator directly and
  derives the detection overlay from it.

- **`plot.confusion_matrix(group_by=…, top_n=N)` ignored `top_n`**, and grouped
  matrices skipped the >30-category auto-subset.

- **PR plots fabricated the recall axis** as `linspace(0, 1, R)` — mislabeling every
  x coordinate on a run with custom `rec_thrs`. All curves now use the evaluator's
  own grid, and the standard aggregate curve is read from `report()["curves"]`
  instead of re-derived (the two disagreed about empty slices: NaN vs the `-1.0`
  sentinel).

- **`image_diagnostics()` per-image AP ignored custom `rec_thrs`**, always evaluating
  on the default 101-point grid.

- Open Images' `AP` row was captioned "IoU 0.50:0.95" in the PDF; it is AP@0.5.

- `just report type=kpt` failed from any directory but the repo root (relative
  paths) and rejected the `kpt` spelling its own recipe suggests.

- `scripts/parity.py` silently skipped its pinned val2017 baseline when the fixture
  was missing or a key renamed, printing "ALL METRICS PASS" having checked nothing;
  the baseline is now mandatory and a missing per-type key is a scored failure.

- **Five drop-in gaps found by the 1.0 third-party-consumer smoke test** —
  running torchvision's `CocoDetection` and torchmetrics'
  `MeanAveragePrecision(backend="pycocotools")` through `init_as_pycocotools()`:

  - The `import pycocotools.coco as pc` binding form failed — it binds via
    `getattr` on the parent, not `sys.modules`, so the module patch alone only
    covered `from pycocotools.coco import COCO`. The patch functions now also
    set the submodule names as attributes.
  - Scalar ids were rejected: `coco.getAnnIds(img_id)` with a bare int is how
    torchvision calls it. Every query/load method (and its camelCase twin) now
    accepts a scalar or a sequence, matching pycocotools' `_isArrayLike` —
    including `get_cat_ids("person")`, which pycocotools itself gets wrong by
    iterating the string.
  - `coco.dataset` was read-only, breaking pycocotools' in-memory construction
    flow `COCO(); coco.dataset = d; coco.createIndex()` — torchmetrics uses it
    verbatim, with image entries carrying only an `id`. The property is now
    writable (indexing eagerly on assignment), `createIndex()` exists as a
    re-index, and images without `width`/`height` are accepted at this boundary.
  - `COCOeval(cocoGt=…, cocoDt=…, iouType=…)` — pycocotools' constructor
    keyword spellings, which torchmetrics passes — were rejected. Both
    spellings are now accepted; mixing the two spellings of one argument is an
    error.
  - The camelCase aliases exposed snake_case *keyword* names, so Detectron2's
    canonical `getAnnIds(imgIds=…)` failed — while the type stubs had always
    advertised the camel spellings. The aliases now carry pycocotools'
    parameter names (`imgIds`, `catIds`, `areaRng`, `catNms`, `supNms`).

  With all five fixed, torchmetrics computes bit-identical mAP through real
  pycocotools and through hotcoco (worst metric diff 0.0), and torchvision
  loads val2017 through the patched `COCO` unchanged. Regression tests cover
  each gap without the torch dependency.

### Removed

- **`primitives::greedy::greedy_match`** (Rust), the positional wrapper around
  `greedy_match_masked`. Its two adjacent `Option<&[bool]>` arguments transpose
  silently, which is why `GtMasks` exists; once both callers moved to the masked
  form the wrapper had no callers and no reason to stay. Nothing in the Python
  API changes.
- **`hotcoco.eval_index`** — a deprecated shim with zero callers, removed before
  1.0 froze it.

- **The `toVoc`, `fromVoc`, `toCvat` and `fromCvat` camelCase aliases.** A camelCase
  alias exists for exactly one reason — pycocotools has a method of that name and
  hotcoco is a drop-in replacement — and pycocotools has no converters at all. (LVIS
  never motivated the convention either; its API is snake_case.) The four were
  undocumented on every surface, absent from the migration guide's alias table, and
  reachable only via `dir()`. `to_voc`, `from_voc`, `to_cvat` and `from_cvat` are
  unchanged, and the nine genuine pycocotools aliases — `getAnnIds`, `loadRes`,
  `annToMask` and friends — are untouched. No camelCase aliases were added for the new
  DOTA and Open Images converters, for the same reason.

- **Five pre-1.0 Rust *module* paths, with no compatibility aliases.** `cargo-semver-checks`
  reports 47 removed paths, and every one is under these five:

  | Gone | Use |
  |---|---|
  | `hotcoco::eval::*` | `hotcoco::detection::*` |
  | `hotcoco::hierarchy` | `hotcoco::detection::hierarchy` |
  | `hotcoco::healthcheck` | `hotcoco::quality::healthcheck` |
  | `hotcoco::types::{SummaryStats, CategoryStats, DatasetStats}` | `hotcoco::quality::*` |
  | `hotcoco::primitives::counts` | `hotcoco::metrics::counts` |

  **No crate-root path was removed.** `hotcoco::COCOeval`, `hotcoco::EvalImg`,
  `hotcoco::Hierarchy`, `hotcoco::HealthReport`, `hotcoco::SummaryStats` and the rest
  resolve exactly as they did in 0.5.0, so code using them needs no edit — which is
  most code, since the module paths are the verbose form.

  0.x is pre-release under SemVer ("anything MAY change at any time"), so these paths
  carried no stability promise. An earlier draft of 1.0 kept them as deprecated
  aliases; that was dropped because it committed the crate to carrying a compatibility
  tier through all of 1.x and removing it at 2.0 — a multi-year obligation to a Rust
  module surface with no known consumer, when the crate-root paths already absorb the
  moves. A compile error naming the new path is a better migration experience than a
  silent alias.

  **Python is entirely unaffected** and always was: these are Rust module paths.
  `hotcoco.COCOeval`, `hotcoco.mask`, `init_as_pycocotools()`, and the
  `pycocotools`/LVIS drop-in surface are permanent.

- **`Provenance::label()`** (Rust) — never called, and carried a second, hyphenated
  spelling of the provenance strings; removed before 1.0 froze it.

## [0.5.0] - 2026-07-26

### Added

- `COCOeval.eval` now includes the `params` and `date` keys, matching pycocotools'
  dict exactly (`params`, `counts`, `date`, `precision`, `recall`, `scores` in that
  order). `params` is the `Params` object used for evaluation; `date` uses
  pycocotools' `'%Y-%m-%d %H:%M:%S'` format. Closes a Tier-1 drop-in gap vs
  pycocotools. The numeric arrays are unchanged, so all metrics still match.
- `primitives::sim` now owns every similarity kernel: the `bbox_iou`, `mask_iou`, and
  `obb_iou` matrix kernels moved here from `mask`/`geometry`, joining `oks_matrix`.
  A new scalar `bbox_iou_pair` is the single definition of single-pair bbox IoU.
  `hotcoco::mask::iou`, `hotcoco::mask::bbox_iou`, and `hotcoco::geometry::obb_iou`
  still resolve — they are now re-exports, so the `pycocotools.mask` drop-in surface
  is unchanged.
- `primitives::counts::average_precision` — the mechanical AP core (sort → classify →
  cumsum → interpolate → mean), now shared by TIDE and per-image diagnostics.
- `SimKind` gained `From<IouType>`, so detection dispatches on the geometry axis.
- `coco-eval` gained subcommands: `coco-eval eval …` and `coco-eval completions <shell>`.
  The bare form (`coco-eval --gt … --dt …`) is unchanged and still supported.
- Architecture conformance tests (`crates/hotcoco/tests/architecture.rs`) that fail the
  build if the IoU formula, the parallelism threshold, or greedy matching is duplicated
  outside `primitives/`.
- `[tool.pyright]` config for the Python package, scoped to `python/` (excluding the
  untyped `scripts/` and vendored `external/`) at `basic` strictness with
  `pythonVersion = "3.9"` to match ruff's `target-version`. It resolves the compiled
  `hotcoco` extension through `.venv`, so it type-checks call sites against the
  hand-written `__init__.pyi` — catching the signature drift that
  `scripts/test_stubs.py` cannot see, since that test checks name coverage only.
- Python lint is now enforced instead of merely available. The pre-commit hook runs
  `ruff format --check` and `ruff check` whenever Python files are staged (and fails
  loudly if `uv` is missing rather than skipping), and CI gained a `python-lint` job.
  `just py-lint` and `just py-fmt-check` had existed for a while but gated nothing,
  which is how the tree accumulated 48 ruff errors.
- `just setup` now installs the `rust-analyzer` rustup component. Because the toolchain
  is pinned and the component was never installed for the pinned version, editors and
  LSP clients had no Rust code intelligence in this repo — silently, because
  `~/.cargo/bin/rust-analyzer` is a rustup proxy: `which` resolved it while every spawn
  failed with "Unknown binary". It is deliberately *not* listed in `rust-toolchain.toml`,
  since CI installs that file's components and the setup action's `components:` input
  only adds and cannot subtract — listing it there would make all five CI jobs download
  an editor backend they never use. Re-run `just setup` after a channel bump; switching
  channels drops any component not in the toolchain file.

### Changed

- Upgraded `pyo3` and `numpy` from 0.28 to 0.29, which clears the RUSTSEC-2026-0176 (OOB read in `PyList`/`PyTuple` iterators) and RUSTSEC-2026-0177 (missing `Sync` bound on `PyCFunction::new_closure`) security advisories. The corresponding `deny.toml` ignores have been removed. No public Python API changes; the binding sources compiled unchanged against the 0.29 API.
- The confusion matrix now uses the shared `primitives::greedy::greedy_match` instead of
  its own greedy loop, so every matcher in the crate shares one tie-breaking rule.
  Verified byte-identical on val2017 across 160 threshold/max-det/min-score
  configurations, including at the match boundary.
- `primitives::greedy` documents the pycocotools threshold-epsilon clamp
  (`min(t, 1 - 1e-10)`) as **caller-owned**. It is inert below `t = 1.0` and no caller
  applies it today, so behavior is unchanged; the difference at `t = 1.0` is now
  written down rather than implicit.
- The release workflow only triggers on true version tags (`v[0-9]+.[0-9]+.[0-9]+*`),
  and `cargo publish` failures now fail the release instead of being downgraded to a
  warning — only an already-published version is tolerated.
- The Python tree is now clean under `just py-lint` and `just py-fmt-check`, which had
  drifted to 48 ruff errors across 13 unformatted files because neither the pre-commit
  hook nor CI runs them. Beyond formatting, this removed five unused imports, three dead
  local variables, and one unresolvable annotation (`plot/core.py` annotated a return as
  `"np.ndarray"` while importing numpy only inside the function body — now declared
  under `TYPE_CHECKING`, so the annotation resolves without making numpy a hard import).
- The dashboard confusion matrix shows the raw count alongside the normalized rate in
  its hover, completing what the code already intended — the raw matrix was being read
  and discarded under a comment reading "Hover text with counts". A rate alone cannot
  distinguish one stray detection from a systematic confusion.
- `ruff` is pinned to `>=0.15,<0.16` instead of `>=0.4`. Lint results are now a gate
  (pre-commit and CI), so an unconstrained ruff could fail a PR that changed no code.

### Fixed

- `shapely` was never declared as a dependency, so `scripts/fuzz_obb_parity.py` — the
  OBB IoU fuzz harness the `/parity` skill offers on request — failed at collection with
  `ModuleNotFoundError` for anyone whose venv did not happen to have it. It is now in the
  `dev` extra, and all 9 OBB parity tests pass.
- `import hotcoco` raised `TypeError: unsupported operand type(s) for |` on Python 3.9,
  the oldest version declared by `requires-python` and served by the single `abi3-py39`
  wheel. `python/hotcoco/__init__.py` and `python/hotcoco/_style.py` used PEP 604
  `X | None` unions in annotations that Python evaluates at runtime (function
  signatures, and an annotated attribute assignment), which requires 3.10. Both files
  now carry `from __future__ import annotations`, so the annotations are never
  evaluated. Reproduced and verified fixed on CPython 3.9.6.
- CI now runs the Python smoke test on a `["3.9", "3.12"]` matrix instead of 3.12 only.
  The declared support floor had never been exercised, which is why the import failure
  above shipped.
- Bumped `crossbeam-epoch` (→0.9.20), `rand` (→0.9.5), and `quick-xml` (→0.41) to clear RUSTSEC-2026-0204, -0097, -0194, and -0195 security advisories.
- `coco-eval --completions <shell>` works standalone. It previously required `--gt` and
  `--dt`, which clap validated before the completions branch ran — so the flag could
  never generate a completion script on its own.
- `MIN_PARALLEL_WORK`, the threshold at which IoU kernels switch to rayon, was defined
  twice with different values (1024 in `mask`, 1000 in `geometry`) under a comment
  claiming they matched. There is now one constant.
- The internal path-dependency pins in `hotcoco-cli` and `hotcoco-pyo3` had lagged at
  `0.4.0` since the 0.4.1 release. crates.io resolves these for published builds, so a
  stale pin publishes against an older API than was tested. They now live in
  `[workspace.dependencies]` next to `[workspace.package] version`, and the members
  inherit them — so the mismatch is visible in one file instead of hidden in two.
- `primitives::assign::lsap` reported an infeasible cost matrix (a row with no finite
  entry) as a bare index-out-of-bounds panic in release builds, because the check was a
  `debug_assert!`. It now asserts with a message naming the cause, as scipy does.
- Six README links pointed at documentation pages that do not exist and returned 404:
  TIDE errors, confusion matrix, F-scores, and logging metrics are sections of
  `guide/evaluation/`; format conversion is a section of `guide/datasets/`; the PyTorch
  integrations page is at `api/integrations/`.
- The Rust install snippet in `docs/getting-started/installation.md` still suggested
  `hotcoco = "0.3"`.
- The Objects365 sentence in `README.md` gave `39×` and `14×` in parentheses two lines
  after stating that parenthesized speedups are versus pycocotools, though the `14×` is
  versus faster-coco-eval. Both figures are unchanged; the baselines are now named.

## [0.4.1] - 2026-07-23

### Fixed

- COCO JSON loading now tolerates non-standard `NaN`, `Infinity`, and `-Infinity` float tokens. Python's `json` module emits and accepts these by default, so files produced by pycocotools/numpy pipelines frequently contain them even though they are invalid JSON; serde_json previously rejected such files with an `expected value` error. `COCO::new` and `load_res` now normalize non-finite tokens to `null` (matching serde's own float serialization), which becomes `None` on `Option<f64>` fields such as `area` and `score`. String values that merely contain the substrings `NaN`/`Infinity` are left untouched. As part of this, `COCO::new` reads the file via `serde_json::from_slice` instead of `from_reader`.

## [0.4.0] - 2026-04-06

### Added

- CVAT for Images 1.1 format conversion — `COCO.to_cvat(output_path)` exports to a single CVAT XML file with `<box>` and `<polygon>` elements; `COCO.from_cvat(cvat_path)` imports CVAT XML back to COCO format with polygon segmentation support (shoelace area, bbox from vertex extents); `coco convert --from coco --to cvat` and `--from cvat --to coco` CLI support; `<polyline>`, `<points>`, `<cuboid>` elements skipped
- Pascal VOC format conversion — `COCO.to_voc(output_dir)` exports to VOC XML annotations (`Annotations/*.xml` + `labels.txt`); `COCO.from_voc(voc_dir)` imports VOC XML back to COCO format; `coco convert --from coco --to voc` and `--from voc --to coco` CLI support; COCO `iscrowd` maps to VOC `<difficult>`, VOC `<difficult>` dropped on import; integer-pixel round-trip within ≤1px
- Confidence calibration analysis — `COCOeval.calibration(n_bins=10, iou_threshold=0.5)` computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), per-bin accuracy vs confidence breakdown, and per-category ECE; measures how well predicted confidence scores align with actual detection accuracy
- `coco eval --calibration` CLI flag with `--cal-bins` and `--cal-iou-thr` options; formatted table output with per-bin breakdown and top-10 worst-calibrated categories; included in `--json` output
- `hotcoco.plot.reliability_diagram()` — matplotlib reliability diagram showing predicted confidence vs actual accuracy per bin, with perfect calibration diagonal, gap overlay, and ECE/MCE annotation; accepts either a calibration dict or COCOeval instance
- `CalibrationResult` and `CalibrationBin` Rust types exported from crate root
- `coco.browse()` dataset browser rewritten: replaced Gradio with FastAPI + HTMX + Jinja2 + vanilla JS Canvas; sidebar with multi-select category filter and shuffle; infinite-scroll thumbnail grid with server-side annotated thumbnails; lightbox with full-resolution canvas overlay for bbox/segmentation/keypoint annotations; hover-to-highlight syncs canvas and annotation sidebar; scroll-to-zoom and drag-to-pan; keyboard navigation (arrow keys, Escape); responsive layout adapts from 400px to 1400px+ viewports; works inline in Jupyter IFrames
- `coco.browse(dt=...)` — detection overlay: GT solid bboxes, DT dashed bboxes, confidence scores on labels; Sources toggle (GT/DT) and Min Score slider for filtering detections
- `coco explore --dt <results.json>` CLI flag — enables detection overlay from the command line
- Eval-aware browse: `coco.browse(dt=..., iou_type="bbox", iou_thr=0.5)` auto-runs evaluation and colors detections as TP (green), FP (red), FN (blue); "Category | Eval" toggle switches between standard and eval coloring; hover highlights matched DT↔GT pairs with connecting line; eval badges on annotation sidebar items
- `coco.browse(eval=coco_eval)` — accept pre-computed `COCOeval` for advanced users who want custom evaluation settings
- `coco.browse(slices={"daytime": [1,2,3], ...})` — sliced browsing with per-slice AP display; accepts dict or path to JSON file
- `coco explore --iou-type`, `--iou-thr`, `--no-eval`, `--slices` CLI flags for eval-aware browsing
- Gallery eval badges: TP/FP/FN count chips on thumbnail cards when eval data is available
- Gallery eval sorting: "Worst first", "Most FP", "Most FN" sort options; eval filter: "Has FP", "Has FN", "Has errors", "Perfect only"
- Interactive IoU threshold slider in sidebar — re-indexes cached eval data without re-evaluation
- Category hierarchy tree view: supercategory groupings with expand/collapse, group-level check/uncheck, keyboard navigation; flat/tree toggle persisted in localStorage
- `python/hotcoco/eval_index.py` — `build_eval_index(coco_eval, iou_thr)` extracts per-annotation TP/FP/FN status from `eval_imgs` at any IoU threshold
- `coco.browse(port=7860)` — new `port` parameter for custom server port
- `python/hotcoco/server.py` — new FastAPI server module with `create_app()`, `run_server()`, and `start_server_background()` for Jupyter
- `python/hotcoco/static/` — new static assets: `style.css` (responsive dark theme), `overlay.js` (Canvas annotation renderer), `htmx.min.js` (vendored HTMX 2.0.4)
- `python/hotcoco/templates/` — new Jinja2 templates: `base.html`, `index.html`, `partials/gallery.html`, `partials/detail.html`
- `COCO(annotation_file, image_dir=...)` — new `image_dir` constructor arg and settable attribute; propagated through `filter`, `split`, `sample`, and `load_res`
- `--json` flag on every `coco` subcommand (`eval`, `stats`, `healthcheck`, `filter`, `merge`, `split`, `sample`, `convert`) — writes a single JSON object to stdout; intended for CI/CD pipelines, dashboards, and shell scripts; stderr and exit codes are unchanged; errors also emit JSON when the flag is active
- `coco eval --json` suppresses the Rust-side metrics table (via fd-level stdout redirect) and returns `{metrics, params, tide?, slices?, healthcheck?}` — optional keys only present when their flags are passed
- `docs/cli.md` — new "JSON output mode" section with CI gating example and JSON error format; `--json` row added to every subcommand flags table; JSON output shape documented for `eval`
- Model comparison — `hotcoco.compare(eval_a, eval_b)` computes per-metric deltas, per-category AP differences, and optional bootstrap confidence intervals for statistical significance; `n_bootstrap` resamples images with replacement and re-accumulates in parallel (rayon); result includes `metric_keys`, `metrics_a`, `metrics_b`, `deltas`, `ci`, `per_category`
- `COCOeval.metric_keys()` — returns metric names in canonical display order for the current evaluation mode; single source of truth for metric ordering across all Python consumers
- `coco compare` CLI subcommand — `--gt`, `--dt-a`, `--dt-b`, `--bootstrap`, `--seed`, `--confidence`, `--name-a`, `--name-b`, `--json` flags; formatted table with deltas, CIs, and per-category breakdown
- `hotcoco.plot.comparison_bar()` — grouped bar chart comparing two models, with CI error bars from bootstrap
- `hotcoco.plot.category_deltas()` — horizontal bar chart of per-category AP deltas sorted by magnitude (green=improvement, red=regression)
- `ComparisonResult`, `CompareOpts`, `BootstrapCI`, `CategoryDelta` Rust types exported from crate root
- Per-image diagnostics & label error detection — `COCOeval.image_diagnostics(iou_thr=0.5, score_thr=0.5)` computes per-annotation TP/FP/FN classification, per-image F1 and AP scores, error profiles, and automatically flags suspected label errors (wrong_label: cross-category GT mislabels; missing_annotation: high-confidence FPs with no nearby GT)
- `coco eval --diagnostics` CLI flag with `--diag-iou-thr` and `--diag-score-thr` options; compact summary output with F1 distribution and top label error categories
- `ImageDiagnostics`, `AnnotationIndex`, `ImageSummary`, `LabelError`, `DtStatus`, `GtStatus`, `ErrorProfile`, `LabelErrorType` Rust types exported from crate root
- Eval dashboard — `/dashboard` route in browse server with KPI tiles (AP, AP50, AP75, AR100), IoU-sweep PR curves, per-category AP leaderboard with expand/collapse, confusion matrix heatmap (click cell → gallery), TIDE error breakdown, calibration reliability diagram (ECE/MCE), per-image F1 histogram colored by error profile, and suspected label errors table (click row → image); all charts Plotly.js with dark theme matching browse UI; Gallery↔Dashboard nav pills in sidebar; dashboard data cached after first compute
- `python/hotcoco/dashboard.py` — Plotly chart generation module: `build_dashboard()`, `kpi_tiles()`, `chart_pr_curves()`, `chart_per_category_ap()`, `chart_confusion_matrix()`, `chart_tide_errors()`, `chart_calibration()`, `chart_f1_distribution()`, `label_errors_table()`
- `python/hotcoco/templates/dashboard.html` — dashboard template with sidebar metadata, KPI row, chart grid, and label errors table
- Dashboard responsive layout — 5 breakpoints (1400px max-width, 1000px single-column charts, 768px toolbar mode with inline metadata, 480px compact with hidden chart hints and 3-column TIDE, 350px+ ultra-narrow); matches gallery responsive behavior at all widths
- Browse UI: HTMX loading spinner on gallery container (`hx-indicator`); error toast for failed HTMX requests (`htmx:responseError`, `htmx:sendError`) with 4-second auto-dismiss
- Browse UI: `@media (prefers-reduced-motion: reduce)` — disables all CSS animations and transitions for users with motion sensitivity preferences
- Browse UI: `:focus-visible` ring on custom range slider thumbs for keyboard accessibility
- Browse UI: `title` attributes on eval badges in annotation sidebar ("True positive", "False positive", "False negative") for screen reader and tooltip accessibility
- Oriented bounding box (OBB) evaluation — `IouType::Obb` / `iou_type="obb"` for rotated detection tasks (aerial imagery, document analysis, scene text); `Annotation.obb` field as `[cx, cy, w, h, angle]` (radians); rotated IoU via Sutherland-Hodgman polygon clipping in new `geometry` module; same 12 AP/AR metrics as bbox; `load_res` auto-computes `area` and axis-aligned `bbox` from OBB
- DOTA format conversion — `coco_to_dota()` exports OBB annotations to DOTA text format (one `.txt` per image, 8-point polygon corners); `dota_to_coco()` imports DOTA text files with auto-category discovery and corner-to-OBB reconstruction; `DotaStats` type exported from crate root
- `crates/hotcoco/src/geometry.rs` — new computational geometry module with `obb_to_corners()`, `obb_iou()` (rayon-parallelized D×G matrix), and Sutherland-Hodgman polygon clipping internals
- OBB visualization in browse — rotated rectangle overlays on canvas (lightbox) and PIL thumbnails (gallery); OBB-aware hover hit-testing via point-in-convex-polygon; eval coloring (TP/FP/FN) and match connector lines work with OBB annotations; dashed outlines for DT, solid for GT
- `scripts/fuzz_obb_parity.py` — hypothesis-based OBB IoU parity fuzzer using Shapely (GEOS) as reference implementation; 200 random cases + 8 deterministic known-value tests
- `python/hotcoco/_style.py` — new zero-dependency terminal styling module with `green()`, `red()`, `yellow()`, `dim()` color helpers, `status()` / `error()` / `warning()` output helpers, `Timer` context manager, and `Spinner` (delayed-start braille spinner, 100ms threshold, no-op in non-TTY/Jupyter)
- `COCOeval.summarize_lines()` (Rust) / `ev.summary_lines()` (Python) — returns metric summary as `Vec<String>` / `list[str]` without printing; `summarize()` now delegates to it
- Styled CLI output — all `coco` subcommands show colored status lines on stderr (green action verbs, dimmed file paths and timing); `NO_COLOR` env var respected; color works in Jupyter, spinners disabled in non-TTY
- Rust CLI (`coco-eval`) — `anstyle`/`anstream` colored output, `indicatif` braille spinners, elapsed timing on all operations; uses `summarize_lines()` for metrics output
- `_style.section(title, params)` — prints a section header with green title and dim `(params)`; used by all analysis outputs (TIDE, calibration, diagnostics, slices, compare)
- `_table(columns, rows, footer)` helper in `cli.py` — auto-aligned table with `─` separators; replaces 5 hand-built table implementations
- Type stubs — `python/hotcoco/__init__.pyi` and `py.typed` marker ship with the package; full coverage of COCO, COCOeval, Params, Hierarchy, mask, and compare APIs; enables autocomplete and type checking in VS Code, PyCharm, etc.
- `scripts/test_stubs.py` — stub coverage test verifying every public name in the runtime module has a corresponding stub entry; wired into `just test`
- `text_signature` on all mask functions and `compare()` — `help()` and IPython `?` now show real parameter names instead of `(*args, **kwargs)`
- `deny.toml` — cargo-deny configuration for dependency auditing (security advisories, license compliance, duplicate crate detection)
- Cold Brew design system — unified visual theme across all surfaces: browse UI (`style.css`), docs site (`extra.css`, `zensical.toml`), matplotlib plots (`theme.py`), Plotly dashboard (`dashboard.py`); canonical spec at `.claude/skills/theme-factory/themes/cold-brew.md`
- `"cold-brew"` matplotlib theme — 10-color infographic-optimized chart palette (warm/cool alternation), espresso-cream chrome, DM Sans bundled font family; added as default for all `hotcoco.plot` functions
- `python/hotcoco/_fonts/` — bundled DM Sans static font instances (Regular 400, Medium 500, Bold 700) extracted from variable font; auto-registered by matplotlib on import
- `hotcoco.plot.reliability_diagram()` — gap bars now render in both directions: solid fill for overconfident bins (accuracy < confidence), diagonal hatching for underconfident bins (accuracy > confidence)
- Title/subtitle positioning — `_place_title_and_subtitle()` helper computes dynamic figure-fraction spacing from actual figure height; reserves layout rect so titles never overlap axes on any figure size
- `Rle::new(h, w, counts)` constructor — validates that counts sum to `h * w` via `debug_assert` (zero overhead in release builds); provides a safe construction path for external callers

### Changed

- Default matplotlib theme changed from `"warm-slate"` to `"cold-brew"` in all `hotcoco.plot` functions and `style()` context manager
- Browse UI accent shifted from warm caramel (`#d4a574`) to Dusty Steel (`#8694A8`); background surfaces updated to espresso tones; GT badge uses Dusty Clay (`#A8806E`), DT badge uses Dusty Steel
- Dashboard Plotly colorway updated to 10-color infographic palette; confusion matrix midpoint shifted from gold to steel
- Docs site fonts changed to DM Sans (body) and JetBrains Mono (code); accent colors updated to Dusty Steel / Deep Steel
- `zensical.toml` font stack updated: `text = "DM Sans"`, `code = "JetBrains Mono"`
- Pre-commit hook clippy step changed from `-p hotcoco -p hotcoco-cli` to `--workspace`, matching CI; removed redundant `cargo check -p hotcoco-pyo3` step (now covered by workspace clippy); hook reduced from 4 steps to 3
- Rust edition 2021 → 2024, MSRV 1.74 → 1.85, workspace resolver 2 → 3
- PyO3 0.23 → 0.28, numpy crate 0.23 → 0.28 — `PyObject` → `Py<PyAny>`, `allow_threads` → `detach`, `downcast` → `cast`, `#[pyclass(from_py_object)]` for Clone types
- `[workspace.lints]` — centralized clippy/rust lint configuration across all 3 crates; `unsafe_code = "forbid"`, `unwrap_used = "warn"`, `dbg_macro = "deny"`, `clippy::pedantic` with targeted allows
- Cargo profiles — `dist-release` (LTO + codegen-units=1 + strip) for PyPI wheels, `profiling` (release + debug symbols) for flamegraphs, `opt-level = 1` for dev/test profiles
- GIL released during heavy computation — `py.detach()` wraps `evaluate()`, `accumulate()`, `run()`, `confusion_matrix()`, `tide_errors()`, `calibration()`, `slice_by()`, `image_diagnostics()`, `compare()`; enables concurrent Python workloads
- Library `.unwrap()` calls replaced with `.expect()` or safe alternatives; `#[allow(clippy::unwrap_used)]` in test modules only

- `crates/hotcoco/src/convert.rs` split into `convert/mod.rs` (shared types) + `convert/yolo.rs` + `convert/voc.rs` + `convert/cvat.rs` submodules; public API unchanged
- `quick-xml` 0.37 added as dependency for XML read/write in VOC conversion
- `ConvertError` gains `XmlError(String)` variant for XML parsing/writing failures
- `coco convert` CLI `--from`/`--to` choices extended from `{coco, yolo}` to `{coco, yolo, voc}`
- Metric display ordering is now derived from Rust `MetricDef` arrays everywhere; removed all hardcoded Python-side metric name lists from `cli.py`, `report.py`, `plots.py`, and parity scripts; `COCOeval.metric_keys()` and `ComparisonResult.metric_keys` are the canonical sources; `report.py` splits AP/AR by key prefix instead of maintaining parallel lists
- `python/hotcoco/__init__.py` — replaced `from .hotcoco import *` with explicit imports; added `__all__` listing all 13 public names
- `scripts/helpers.py` — extracted `suppress_stdout()`, keypoint constants (`COCO_KEYPOINT_NAMES`, `COCO_SKELETON`, `COCO_KPT_OKS_SIGMAS`), and path constants (`WORKSPACE`, `DATA_DIR`) shared across `parity.py`, `bench.py`, `test_parity.py`, `fuzz_parity.py`
- `crates/hotcoco/src/error.rs` — new unified `Error` enum with typed `#[from]` variants for `io::Error`, `serde_json::Error`, `ConvertError`, and a catch-all `Other(String)`; replaces `Box<dyn Error>`, bare `String`, and `io::Error` returns across 13 functions in `coco.rs`, `mask.rs`, and `eval/` submodules
- `crates/hotcoco-pyo3/src/lib.rs` — added `to_pyerr()` helper that maps `hotcoco::Error` variants to appropriate Python exceptions (`PyIOError`, `PyValueError`, `PyRuntimeError`); replaces 14 inline `.map_err(...)` calls
- `crates/hotcoco-pyo3/src/convert.rs` — added `opt!` and `req!` macros for extracting optional/required fields from Python dicts; replaces ~20 repetitions of the `dict.get_item("key")?.map(|v| v.extract()).transpose()?` pattern
- `COCOeval` fields `eval_imgs`, `eval`, `stats` changed from `pub` to `pub(crate)` with public getter methods: `eval_imgs()`, `accumulated()`, `stats()`; Python API unchanged (still `@property` getters)
- Browse UI: overlay toggles (Boxes/Segments/Keypoints/GT/DT/Eval) moved from bottom of image panel into the lightbox header bar for better visibility and space efficiency
- Browse UI: unified IoU threshold and Min Score sliders to use the same stacked layout (label + value on top, slider below) via shared `.range-slider` CSS class, eliminating ~70 lines of duplicate vendor-prefixed slider styling
- Browse UI: fixed segmentation mask misalignment on first lightbox open — replaced single `requestAnimationFrame` with double-rAF to guarantee layout is settled after `display:none → flex` transition
- Browse UI: fixed "ANNOTATIONS" header bounce during arrow-key navigation by adding `contain: size layout` on `.lightbox-card` and stable dimensions on `.info-panel-header`
- Browse UI: removed `scale(0.98)` from lightbox slide-up animation — was distorting `getBoundingClientRect()` measurements during the animation
- Browse UI: keyboard hints condensed from `← → navigate Esc close` to `← → Esc`
- Browse UI: canvas now fills the entire container (not just the image rect), preventing zoom clipping at edges
- Browse UI: gallery infinite scroll sentinel uses `hx-swap="outerHTML"` instead of `afterend` to prevent blank grid cells
- `coco explore` CLI — added `--iou-type`, `--iou-thr`, `--no-eval`, `--slices` flags; prints TP/FP/FN summary on startup when eval is active
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
- `examples/coco_evaluation_101.ipynb` — restructured and expanded: 5-act narrative (Getting Started → Understand Your Model → Compare & Slice → Dataset Tools → Integration); added sections for confusion matrix, calibration, per-image diagnostics, model comparison, sliced evaluation, dataset operations, interactive browse, and publication plots with inline visualizations; richer 4-image synthetic dataset for diagnostic demos
- `docs/getting-started/quickstart.md` — updated notebook description to reflect new content
- `coco --help` and subcommand help: new description and epilog examples on top-level parser and `eval`/`healthcheck` subparsers; `--gt`/`--dt`/`--slices` help text improved; `stats` one-liner updated
- `scripts/test_parity.py` renamed to `scripts/fuzz_parity.py` — clarifies that this is the slow hypothesis-based fuzzer (`just fuzz`), distinct from `scripts/test_parity.py` (the fast CI regression suite, `just test`)
- `scripts/fixtures/adversarial/` added to `.gitignore` and removed from tracking — hypothesis-generated fixtures are ephemeral outputs, not source files; the directory is recreated locally by running `just fuzz`
- `docs/stylesheets/extra.css` — full docs theme redesign: custom CSS variable palettes for light (stone-cream) and dark (cool charcoal) modes; all 12 `--md-code-hl-*` syntax token colors set to a warm editorial palette (dusty steel blue keywords, sage strings, clay numbers, plum functions); admonition type overrides (note/info/warning/tip) with flat tinted backgrounds and no title-bar box artifact; hero pill buttons, feature card lift-on-hover, warm-tinted shadows throughout
- `zensical.toml` — docs theme: `primary`/`accent` palette entries switched to `"custom"`; `navigation.tabs` added to features (top-level sections move to tab bar, freeing sidebar width); `[project.theme.font]` added with `text = "Nunito"` and `code = "IBM Plex Mono"`; color palette shifted from saturated warm-brown to desaturated gray-brown (`#4A4540`) with dusty slate blue accent (`#6B7E9A`) for a cooler, less heavy feel; logo icon updated to `lucide/coffee`
- Internal: extracted named constants `AREA_SMALL` (32²), `AREA_LARGE` (96²), `KPT_OKS_SIGMAS` and `default_iou_thrs()` helper in `params.rs`; replaced inline literals in `Params::new()` and deduplicated the IoU range formula between `params.rs` and `summarize.rs`
- Browse server (`server.py`) and CLI (`cli.py`) now use `COCOeval.image_diagnostics()` instead of the Python-side `build_eval_index()`; `eval_index.py` reduced to a backward-compatible thin wrapper
- Per-image AP in `image_diagnostics()` uses the shared `precision_recall_curve` from `accumulate.rs` (monotone precision correction, O(N+R) two-pointer scan) instead of a separate O(N×R) implementation
- Browse UI: extracted 367 lines of inline JavaScript from `index.html` into `static/gallery.js` (IIFE module, browser-cacheable) and 20 lines from `dashboard.html` into `static/dashboard.js`
- `python/hotcoco/cli.py` — all raw ANSI escape codes (`\033[91m`, etc.) replaced with `_style` module helpers; all `print("error: ...", file=sys.stderr)` calls replaced with `error()` helper; `cmd_eval --json` uses `ev.summary_lines()` instead of `dup2`/`devnull` fd-level stdout suppression; `cmd_stats` and `cmd_merge` use shared `_load_coco()` helper (removes redundant `ImportError` guards)
- `crates/hotcoco-cli` — added `anstyle`, `anstream`, `indicatif` dependencies; `clap` feature `color` enabled
- Unified CLI analysis output formatting — TIDE, calibration, diagnostics, slices, and compare tables all use `_table()` helper with consistent `─` separators, 2-space indent, and `section()` headers; sub-section labels use `dim()` for parenthetical qualifiers; diagnostics tip line dimmed
- Removed `Eval type: bbox` line from `summarize()` output — redundant with status line and `--iou-type` flag
- Browse UI: `overlay.js` restructured — scattered module-level variables consolidated into `_cache`/`_ui` namespaces; all `var` replaced with `const`/`let`; magic numbers extracted into named constants (`DASH_SEGMENT`, `MIN_FONT_SIZE`, `BASE_KPT_RADIUS`, etc.); JSDoc added to `drawOverlays()` documenting the 4-pass rendering pipeline
- Browse UI: checkbox styling DRYed — shared base selector for all 3 variants (category filter, overlay toggles, tree group) with per-variant size/position overrides; reduces ~90 lines of duplicated CSS
- Browse UI: `metric_fmt` Jinja2 filter added for consistent metric formatting across templates; replaces scattered `"%.3f" | format()` patterns in dashboard and gallery
- `ConvertError` — replaced manual `Display`/`Error`/`From<io::Error>` impls with `thiserror` derive macros, matching the crate's `Error` enum pattern
- `mask.rs` — `transpose_mask` doc comment clarified: now specifies `(h, w)` for row→column and `(w, h)` for the reverse, instead of claiming the operation is its own inverse
- CI — added `cargo-deny` dependency audit step (ubuntu-only); test step scoped to `-p hotcoco -p hotcoco-cli` (excludes cdylib)
- Pre-commit hook step numbering corrected from `[1/4]` to `[1/3]` (matches actual 3-step hook)
- `_typos.toml` — new typos-cli configuration: `en-us` locale, domain abbreviations allow-listed (`nd`, `obb`, `oks`, etc.), data/fixture dirs excluded
- `python/hotcoco/cli.py` — reformatted with ruff (line length compliance); `cmd_merge` import moved to function scope
- Docs — American English spelling throughout (`behaviour` → `behavior`, `normalised` → `normalized`, `maximises` → `maximizes`, `summarisation` → `summarization`, `Randomise` → `Randomize`); benchmark versions updated to 0.3.0
- Browse UI: `nav_query | safe` in `detail.html` replaced with JSON-encoded `<script type="application/json">` data element parsed in `overlay.js`, eliminating a potential XSS surface from string interpolation
- Browse UI: canvas `getBoundingClientRect()` result cached in `_cache.canvasRect` (updated on resize), avoiding forced layout reflow on every mousemove during hover hit-testing
- Browse UI: redundant `syncCatViews()` calls removed from per-checkbox handlers (`onCatChange`, `onTreeChildChange`, `toggleGroupCheck`); cross-view sync now runs only on view switch via `setCatView()`
- Browse server: inline HTML error strings in `server.py` replaced with `partials/error.html` template; `image_id` no longer duplicated inside `annotation_json` (dead `nav` key removed)
- `rand` 0.8 → 0.9 — `gen_range` → `random_range`, `small_rng` feature dropped (default in 0.9); `SmallRng::seed_from_u64` produces different sequences for the same seed (affects `split()`, `sample()`, bootstrap — dataset utilities only, no parity impact)
- `quick-xml` 0.37 → 0.39 — `BytesText::unescape()` → `decode()` in CVAT and VOC converters; no functional change for ASCII label names
- Internal: `img_cat_to_anns` HashMap pre-allocated with `reserve(n_anns)` in `create_index()`, eliminating rehashes during annotation indexing

### Fixed

- `accumulate.rs` — recall initialization moved into early-return path when `nd == 0` (GT exists but no detections); previously recall was written unconditionally then overwritten, now it's set once at the correct point
- `tide.rs` — replaced hardcoded `/ 101.0` with `/ rec_thrs.len() as f64` so TIDE ΔAP computation adapts if recall threshold count ever changes
- `mask.rs` — `iou()` and `bbox_iou()` now validate that `iscrowd` length matches `gt` length, raising `ValueError` with a clear message instead of panicking on out-of-bounds access
- `lib.rs` — browse `slice_by` callable that returns `None` now skips the image instead of raising a type error on extract
- `hotcoco-pyo3/src/lib.rs` — 13 clippy lints fixed: redundant closures replaced with method references (`String::as_str`, `str::to_lowercase`, `<[f64]>::to_vec`), `.map().unwrap_or()` → `.map_or()`, missing semicolons on unit-returning setter delegates
- Browse UI: `mouseup` and `touchstart` event listeners in `overlay.js` accumulated on every lightbox open (memory leak); now stored in module-level refs and cleaned up before re-attaching in `initOverlay()`
- Browse server: unhandled exceptions returned FastAPI's default JSON error instead of themed HTML; added `@app.exception_handler(Exception)` with styled error page and `logging.exception()` for 500s; `/detail/{id}` 404 now returns themed HTML instead of plain text
- Browse server: broad `except Exception: pass` on slice metric computation replaced with `except (KeyError, ValueError, TypeError)` and `logger.warning()` to avoid silently hiding bugs

## [0.3.0] - 2026-03-16

### Changed

- Internal: replaced `is_lvis: bool` with `eval_mode: EvalMode` enum (`Coco | Lvis | OpenImages`) across all evaluation branch points; `EvalParams.is_lvis` serialized field renamed to `eval_mode` (string: `"coco"`, `"lvis"`, `"openimages"`); no behavior change — prepares for Open Images evaluation support
- `hotcoco.plot` internal refactor: new `PlotData` dataclass (`python/hotcoco/plot/data.py`) centralizes eval extraction from `COCOeval`, exposes `area_idx`, `max_det_idx`, `nearest_iou_idx` helpers and cat-name lookup; all plot functions now consume `PlotData` instead of reaching into `COCOeval` internals directly
- `hotcoco.plot` figure saving: increased output DPI from 150 to 200; added `bbox_inches="tight"` to prevent label clipping on save
- `hotcoco.plot.confusion_matrix`: colorbar now uses `make_axes_locatable` for proportional sizing; normalized matrix clamped to `[0, 1]` with `vmax=1.0`; PR-curve plots switched to `layout="compressed"` for tighter axis packing
- `_annotate_bars` internal helper removed in favor of native `ax.bar_label`

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
- `COCOeval.f_scores(beta=1.0)` — compute F-beta scores after `accumulate()`; for each (IoU threshold, category) finds the confidence operating point that maximizes F-beta, then averages across categories; returns `{"F1": ..., "F150": ..., "F175": ...}` (key prefix reflects beta value); supports arbitrary beta for precision/recall trade-off weighting
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
- `COCOeval.tide_errors(pos_thr=0.5, bg_thr=0.1)` — TIDE error decomposition (Bolya et al., ECCV 2020); classifies every FP into six mutually exclusive types (Loc, Cls, Dupe, Bkg, Both, Miss) and reports ΔAP — the AP gain from eliminating each type; requires `evaluate()` first; priority order matches tidecv (Loc > Cls > Dupe > Bkg > Both); Bkg/Both/Dupe ΔAP uses suppression (not flip-to-TP) for correct curve behavior
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
