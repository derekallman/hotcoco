# semantic

Semantic segmentation — mIoU, fwIoU, mAcc, aAcc — against mmsegmentation's
protocol.

=== "Python"

    ```python
    from hotcoco import semantic
    ```

=== "Rust"

    ```rust
    use hotcoco::semantic::SemanticEval;
    ```

The [semantic guide](../guide/semantic.md) covers what the metric measures,
the two input forms, and how the numbers were checked. This page is the
signatures and return shapes.

---

## `SemanticEval`

=== "Python"

    ```python
    SemanticEval(
        classes: int | Sequence[str],
        *,
        ignore_index: int | None = 255,
        reduce_zero_label: bool = False,
    )
    ```

    | Parameter | Description |
    |-----------|-------------|
    | `classes` | The number of classes, labeled `0..classes - 1`, or their names in class-id order |
    | `ignore_index` | Ground-truth label of an unlabeled pixel, left out on both sides. `None` counts every pixel |
    | `reduce_zero_label` | mmseg's rule for files with "other" at `0`: ground-truth `0` becomes `255` and every other ground-truth label shifts down by one. Needs `ignore_index=255`; the constructor raises `ValueError` otherwise |

=== "Rust"

    ```rust
    SemanticEval::new(num_classes: usize) -> SemanticEval              // names class_<id>, ignore 255
    SemanticEval::with_class_names(names: Vec<String>) -> SemanticEval
    SemanticEval::ignore_index(self, Option<u64>) -> SemanticEval
    SemanticEval::reduce_zero_label(self, bool) -> SemanticEval
    SemanticEval::check_options(&self) -> Result<()>                   // the builder cannot fail; this is the error the first update would raise
    ```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `update(gt, pred)` | `None` | Count one ground-truth map against one prediction: integer arrays of the same shape, `(H, W)` or `(N, H, W)`, any integer dtype. `ValueError` on a shape mismatch or a label out of range; nothing from the pair is counted then |
| `update_png(gt, pred)` | `None` | Count one pair of PNG label maps — 8- or 16-bit grayscale, or a palette image whose index is the class |
| `update_folders(gt_folder, pred_folder)` | `int` | Count every PNG in `gt_folder` against the file of the same name in `pred_folder`, in parallel; returns the pairs counted. `ValueError` on a missing prediction or a bad pair, with nothing counted |
| `merge(other)` | `None` | Add another evaluator's counts — the shards of a distributed run. `ValueError` if it was built with different classes or options |
| `confusion()` | `ndarray[uint64]` | The `(num_classes, num_classes)` matrix behind every number: row the ground-truth class, column the predicted class |
| `summarize()` | `None` | Print one row per class with IoU and Acc in percent, then the four means |
| `summary_lines()` | `list[str]` | The same table as strings, without printing |
| `stats` | `list[float]` | The four means as fractions, in `METRIC_NAMES` order; `-1.0` where nothing was counted |
| `class_names`, `num_classes`, `n_images` | | The configuration and how many updates were counted: one per `update` call, so an `(N, H, W)` batch is one, and one per PNG pair |
| `results()` | `dict` | The results shape, below |
| `report()` | `dict` | The [`EvalReport`](cocoeval.md#report) every family produces |
| `reference_deviations()` | `list[str]` | Why this run is not comparable to mmseg; empty today, since every option is a reference option |
| `provenance()` | `str` | `"parity_verified"` or `"extension"` |

An evaluator pickles; `pickle.loads(pickle.dumps(ev))` carries the
configuration and the matrix, which is what `torch.distributed`'s
`all_gather_object` needs. In Rust the same names return `Result` where
Python raises, `update` is generic over any integer slice type on either
side with `update_maps` for two `LabelMap`s of run-time width,
`results()` returns a `SemanticResults` with `to_json()` and `save()`,
`confusion()` returns the `metrics::semantic::PixelConfusion`, and
`summary_lines()` is spelled `summarize_lines()`.

### `results()`

```python
{
    "mIoU": 0.7642, "fwIoU": 0.9210, "mAcc": 0.8407, "aAcc": 0.9588,
    "per_class": [
        {"name": "road", "IoU": 0.9781, "Acc": 0.9892, "tp": 3_204_118, "gt_area": 3_239_101, "pred_area": 3_250_777},
        ...
    ],
    "n_images": 500,
    "n_pixels": 1_046_249_374,
    "provenance": "parity_verified",
    "reference_deviations": [],
    "hotcoco_version": "1.3.0",
}
```

Scores are fractions. `per_class` is a list in class-id order covering every
class; a class on neither side reports `-1.0` for `IoU`, and one with no
ground truth `-1.0` for `Acc`, beside the counts that say so. `n_pixels`
counts after ignoring.

### `report()`

| Key | Contents |
|-----|----------|
| `task` | `"semantic"` |
| `provenance` | `"parity_verified"` |
| `metrics` | `mIoU`, `fwIoU`, `mAcc`, `aAcc` — `-1.0` when nothing was counted |
| `per_class` | class name → `{"IoU", "Acc"}`, computed values only |
| `per_group` | empty; there are no splits |
| `curves` | empty; mIoU has no curve |
| `params` | `n_images`, `n_pixels`, `num_classes`, `ignore_index`, `reduce_zero_label` |

---

## `METRIC_NAMES`

```python
["mIoU", "fwIoU", "mAcc", "aAcc"]
```

The order of `stats` and the keys of `report()["metrics"]`.

---

## The layers underneath

The driver composes the shared layers, as detection and panoptic do. There
is no matcher: pixels pair by position.

| | Python | Rust |
|---|---|---|
| The confusion matrix | [`metrics.pixel_confusion`](metrics.md#pixel_confusion) | `metrics::semantic::PixelConfusion` — `new`, `with_unlabeled_pred`, `update`, `update_with`, `update_maps`, `add_count`, `add`, `from_counts`, `class_counts` |
| A map of run-time width | — | `metrics::semantic::{LabelMap, LabelPairOp}` — a map as `U8`, `U16`, or `I64`, and `LabelMap::with_pair` to run an op generic over both widths |
| The numbers | [`metrics.mean_iou`](metrics.md#mean_iou) | `metrics::semantic::{semantic_scores, SemanticScores, ClassCounts, ClassScores}` |
| The IoU kernel | — | `primitives::sim::iou_from_counts(intersection, a, b)` |

`PixelConfusion::update` takes any integer slice type on either side (every
dtype numpy stores a label map in) and the ignore index; `update_with` takes
a closure over the ground-truth label instead, which is how the driver
applies `reduce_zero_label`; `update_maps`, on `PixelConfusion` and on
`SemanticEval`, takes two `LabelMap`s, the form a decoded file or a numpy
array arrives in. `with_unlabeled_pred` adds a column for a
prediction that names no class — a panoptic void — which counts as a miss and
never as a class; `PanopticEval` fills it through `add_count`. `ClassCounts::iou()` and `acc()` return `None`
where the reference has NaN; `semantic_scores` turns a class list into the
four means with the `-1.0` sentinel filled in.
