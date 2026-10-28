# panoptic

Panoptic quality — PQ, SQ, RQ — against panopticapi's protocol.

=== "Python"

    ```python
    from hotcoco import panoptic
    ```

=== "Rust"

    ```rust
    use hotcoco::panoptic::{PanopticDataset, PanopticEval};
    ```

The [panoptic guide](../guide/panoptic.md) covers what the metric measures,
the two input forms, and how the numbers were checked. This page is the
signatures and return shapes.

---

## `PanopticEval`

=== "Python"

    ```python
    PanopticEval(
        gt: str | os.PathLike | COCO | dict,
        pred: str | os.PathLike | COCO | dict,
        *,
        gt_folder: str | os.PathLike | None = None,
        pred_folder: str | os.PathLike | None = None,
    )
    ```

    | Parameter | Description |
    |-----------|-------------|
    | `gt` | Ground truth: a COCO panoptic JSON path, a `COCO` dataset whose annotations carry masks (one per segment), or a dict of either shape |
    | `pred` | Predictions, in any of the same forms. Its `categories` are ignored |
    | `gt_folder`, `pred_folder` | PNG folders for the JSON form. Default: the JSON path without `.json` |

=== "Rust"

    ```rust
    PanopticEval::new(gt: PanopticDataset, pred: PanopticDataset) -> PanopticEval

    PanopticDataset::from_file(path: &Path) -> Result<PanopticDataset>   // folder = path minus .json
    PanopticDataset::from_dataset(dataset: &Dataset) -> PanopticDataset  // masks, no PNG files
    PanopticDataset::with_folder(self, folder) -> PanopticDataset
    ```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate()` | `None` | Score every ground-truth image against its prediction, in parallel. Raises `ValueError` on the inputs panopticapi rejects |
| `summarize()` | `None` | Print the table — PQ, SQ, RQ, N for All, Things, Stuff — in percent |
| `summary_lines()` | `list[str]` | The same table as strings, without printing. Empty, with a `CallOrderWarning`, before `evaluate()` |
| `run()` | `None` | `evaluate()` then `summarize()` |
| `stats` | `list[float]` | The nine headline values, in `METRIC_NAMES` order; empty before `evaluate()` |
| `eval_imgs` | `list[dict]` | Per-image match records, below; empty before `evaluate()` |
| `results()` | `dict` | panopticapi's result shape, below |
| `report()` | `dict` | The [`EvalReport`](cocoeval.md#report) every family produces |
| `reference_deviations()` | `list[str]` | Why this run is not comparable to panopticapi; empty when it is |
| `provenance()` | `str` | `"parity_verified"` or `"extension"` |

In Rust the same names return `Result` where Python raises: `results()`
returns a `PanopticResults` with `to_json()` and `save()`, `result()` exposes
the raw `PanopticResult` counts behind it, and `summary_lines()` is spelled
`summarize_lines()`. `PanopticResult.pixels` is the category confusion matrix
the mIoU comes from, with the unlabeled-prediction column a predicted void
lands in, and `PanopticResult::semantic()` reads the `SemanticScores` off it.

### `eval_imgs`

What the matcher decided in each image — the panoptic counterpart of
[`COCOeval.eval_imgs`](cocoeval.md#eval_imgs), and the records the per-class
counts are summed from. This attribute is provisional, like the
[matchers](primitives.md#match_segments) whose output it holds.

=== "Python"

    ```python
    eval_imgs: list[dict]
    ```

=== "Rust"

    ```rust
    fn eval_imgs(&self) -> &[ImageMatches]   // { image_id, matches: MatchIds }
    ```

One record per ground-truth image, in the order `evaluate()` visits them:

```python
{
    "image_id": 139,
    "matched": [(3226956, 1001, 0.8817), (6979964, 1002, 0.9512)],   # (gt_id, pred_id, iou)
    "missed": [8345037],
    "false_positives": [1003],
    "ignored": [1004],
}
```

The four lists are what [`primitives.match_segments`](primitives.md#match_segments)
returns for the image; that page owns which segments land in which list, double
matches included. What the evaluator adds: ids are the input's, the PNG colors
of the COCO panoptic format or the annotation ids of a `COCO` dataset, repeated
in the records where the annotations repeat them, as converters that write
`id: 0` everywhere do. `matched` is in the order its IoUs were summed: by id for
the COCO panoptic format, and by annotation order for a `COCO` dataset. The
other three lists follow `segments_info`, or annotation, order.

Each read of `eval_imgs` converts every record to Python, so bind it once
(`records = ev.eval_imgs`) rather than indexing the attribute in a loop, as with
`COCOeval.eval_imgs`.

### `results()`

```python
{
    "All":    {"pq": 0.6249, "sq": 0.9346, "rq": 0.6684, "n": 133},
    "Things": {"pq": 0.6040, "sq": 0.9320, "rq": 0.6481, "n": 80},
    "Stuff":  {"pq": 0.6564, "sq": 0.9387, "rq": 0.6991, "n": 53},
    "per_class": {
        1: {"pq": 0.71, "sq": 0.86, "rq": 0.83, "tp": 2693, "fp": 509, "fn": 598, "iou": 2316.4},
        ...
    },
    "provenance": "parity_verified",
    "reference_deviations": [],
    "hotcoco_version": "1.3.0",
}
```

Scores are fractions; `n` is the number of categories averaged, which excludes
any with no segment on either side. `per_class` is keyed by category id and
covers every ground-truth category; one with no segment on either side reports `-1.0`
for the three scores and zeros for the counts. The last three keys and the
four counts are hotcoco's additions to panopticapi's dict.

### `report()`

| Key | Contents |
|-----|----------|
| `task` | `"panoptic"` |
| `provenance` | `"parity_verified"`, or `"extension"` when a category lacks `isthing` |
| `metrics` | `PQ`, `SQ`, `RQ`, `PQ_th`, `SQ_th`, `RQ_th`, `PQ_st`, `SQ_st`, `RQ_st` — `-1.0` for a split with nothing to average — plus `mIoU`, `fwIoU`, `mAcc`, `aAcc`, `mDice`, the same pixels scored as [semantic segmentation](../guide/panoptic.md#miou-from-the-same-pixels) |
| `per_class` | category name → `PQ`, `SQ`, `RQ` for evaluable categories, and `IoU`, `Acc`, `Dice` where computed |
| `per_group` | `all`, `things`, `stuff` → `{"PQ", "SQ", "RQ", "n"}`, splits with `n > 0` only |
| `curves` | empty; PQ has no curve |
| `params` | `n_images`, `n_categories`, `gt_folder`, `pred_folder` |

---

## `pq_compute`

=== "Python"

    ```python
    pq_compute(
        gt_json_file: str | os.PathLike,
        pred_json_file: str | os.PathLike,
        gt_folder: str | os.PathLike | None = None,
        pred_folder: str | os.PathLike | None = None,
    ) -> dict
    ```

panopticapi's function: evaluate two COCO panoptic JSON files, print the
table, return [`results()`](#results). Same positional arguments, same
defaults.

---

## `METRIC_NAMES`

```python
["PQ", "SQ", "RQ", "PQ_th", "SQ_th", "RQ_th", "PQ_st", "SQ_st", "RQ_st"]
```

The order of `stats` and the keys of `report()["metrics"]`.

---

## The layers underneath

The driver composes the two shared layers, as detection does:

| | Python | Rust |
|---|---|---|
| Segment overlaps and matching | [`primitives.segment_overlaps`](primitives.md#segment_overlaps), [`primitives.match_segments`](primitives.md#match_segments) | `primitives::panoptic::{Overlaps, match_segments, pq_iou}` |
| Counts and formulas | [`metrics.panoptic_quality`](metrics.md#panoptic_quality) | `metrics::panoptic::{PqCounts, PqScores, pq_average}` |

`segment_overlaps` is the per-image histogram of `(gt id, pred id)` pixel
co-occurrences; `match_segments` applies the protocol's rules to it;
`panoptic_quality` turns counts into PQ, SQ, RQ. Summed per category and
averaged, the three reproduce `PanopticEval`'s numbers exactly — the
[`match_segments`](primitives.md#match_segments) section has the counting
rule. Add the IoUs in a running sum, image by image: Python's `sum()`
compensates for rounding since 3.12, and the last bit then differs.

## Data model

`Category` gained `isthing: bool | None`, read from `1`/`0` or `true`/`false`
and written back as a bool. In Rust, `PanopticDataset` holds the COCO
panoptic schema — `images`, `annotations` of `{image_id, file_name,
segments_info}`, `categories` — plus the PNG `folder`; `Segmentation::to_rle(h, w)`
rasterizes any segmentation onto a canvas.
