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
| `evaluate()` | `None` | Score every ground-truth image against its prediction, in parallel. Raises `RuntimeError` on the inputs panopticapi rejects |
| `summarize()` | `None` | Print the table — PQ, SQ, RQ, N for All, Things, Stuff — in percent |
| `summary_lines()` | `list[str]` | The same table as strings, without printing |
| `run()` | `None` | `evaluate()` then `summarize()` |
| `stats` | `list[float]` | The nine headline values, in `METRIC_NAMES` order; empty before `evaluate()` |
| `results()` | `dict` | panopticapi's result shape, below |
| `report()` | `dict` | The [`EvalReport`](cocoeval.md#report) every family produces |
| `reference_deviations()` | `list[str]` | Why this run is not comparable to panopticapi; empty when it is |
| `provenance()` | `str` | `"parity_verified"` or `"extension"` |

In Rust the same names return `Result` where Python raises, `result()`
exposes the `PanopticResult` behind `results()`, and `summary_lines()` is
spelled `summarize_lines()`.

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
| `metrics` | `PQ`, `SQ`, `RQ`, `PQ_th`, `SQ_th`, `RQ_th`, `PQ_st`, `SQ_st`, `RQ_st` — `-1.0` for a split with nothing to average |
| `per_class` | category name → `{"PQ", "SQ", "RQ"}`, evaluable categories only |
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
| Segment overlaps and matching | — | `primitives::panoptic::{Overlaps, match_segments, pq_iou}` |
| Counts and formulas | [`metrics.panoptic_quality`](metrics.md#panoptic_quality) | `metrics::panoptic::{PqCounts, PqScores, pq_average}` |

`Overlaps::compute(gt, pred)` is the per-image histogram of `(gt id, pred id)`
pixel co-occurrences; `match_segments` applies the protocol's rules to it;
`PqCounts::scores()` and `pq_average` turn counts into PQ, SQ, RQ. The
formulas are callable from Python as `metrics.panoptic_quality`; the matcher
is not bound yet, and lands with the rest of the composability work.

## Data model

`Category` gained `isthing: bool | None`, read from `1`/`0` or `true`/`false`
and written back as a bool. In Rust, `PanopticDataset` holds the COCO
panoptic schema — `images`, `annotations` of `{image_id, file_name,
segments_info}`, `categories` — plus the PNG `folder`; `Segmentation::to_rle(h, w)`
rasterizes any segmentation onto a canvas.
