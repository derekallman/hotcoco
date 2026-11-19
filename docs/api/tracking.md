# tracking

Multi-object tracking — HOTA, CLEAR (MOTA, MOTP), and Identity (IDF1) —
against TrackEval, on MOTChallenge 2D boxes.

=== "Python"

    ```python
    from hotcoco.tracking import TrackingEval
    ```

=== "Rust"

    ```rust
    use hotcoco::tracking::{Benchmark, TrackingEval};
    ```

The [tracking guide](../guide/tracking.md) covers what the metrics measure,
the two input forms, what each benchmark cleans up, and what is not covered
yet. This page is the signatures and return shapes.

---

## `TrackingEval`

=== "Python"

    ```python
    TrackingEval(
        benchmark: str | None = None,
        *,
        clear_threshold: float = 0.5,
        identity_threshold: float = 0.5,
    )
    ```

    | Parameter | Description |
    |-----------|-------------|
    | `benchmark` | `"MOT15"`, `"MOT16"`, `"MOT17"`, or `"MOT20"`, in any case: the MOTChallenge clean-up to apply. `None` evaluates every row as given. An unknown name raises `ValueError` |
    | `clear_threshold` | The IoU at which CLEAR counts a match, TrackEval's `THRESHOLD` option |
    | `identity_threshold` | The same for Identity. Either threshold outside 0 to 1 raises `ValueError` |

=== "Rust"

    ```rust
    TrackingEval::new(benchmark: Option<Benchmark>, clear_threshold: f64, identity_threshold: f64) -> Result<TrackingEval>
    "MOT17".parse::<Benchmark>() -> Result<Benchmark>      // any case; Benchmark::Mot15, Mot16, Mot17, Mot20
    ```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `update(name, gt, pred, *, num_frames)` | `None` | Evaluate one sequence and keep its result. `gt` and `pred` are 2-D arrays of MOTChallenge rows, `frame, id, x, y, w, h, conf, class, visibility`, as a numpy array of any integer or float dtype or a list of rows. Ground truth needs 8 columns with a benchmark and 6 without; predictions need 6. Frames run from 1 to `num_frames`. The rows are copied and evaluated without the GIL. `ValueError`, naming the sequence, on a name already evaluated, too few columns, a frame outside `1..num_frames`, a fractional or negative frame or id, a box or confidence that is not finite, a track id twice in one frame, or a benchmark's class rule; nothing is kept then. `TypeError` for an array that is not 2-D |
| `update_folders(gt_folder, pred_folder, *, seqmap=None)` | `int` | Evaluate every sequence of a benchmark in TrackEval's layout, in parallel; returns how many. Sequence `s` is `{gt_folder}/{s}/gt/gt.txt` with the `seqLength` of `{gt_folder}/{s}/seqinfo.ini`, against `{pred_folder}/{s}.txt`. The sequences are those `seqmap` names after its header line, or every folder in `gt_folder` not starting with a dot. `FileNotFoundError` naming a missing folder or file; `ValueError` for a line that is not numbers, a `seqinfo.ini` with no `seqLength`, a sequence already evaluated, or anything `update` refuses. Nothing is kept then |
| `merge(other)` | `None` | Add another evaluator's sequences — the shards of a distributed run. `ValueError` if the two differ in benchmark or thresholds, share a sequence, or are the same evaluator |
| `results()` | `dict` | The combined run; see [`results()`](#results) |
| `sequence_results()` | `dict` | `{name: dict}` in name order, each the 44 values of `results()` plus `Frames` |
| `curves(sequence=None)` | `dict` | The per-threshold values of the combined run, or of the sequence named; see [`curves()`](#curves). `KeyError` for a sequence not evaluated |
| `stats` | `list[float]` | The 44 values of `results()` as floats, in `METRIC_NAMES` order |
| `summarize()` | `None` | Print TrackEval's HOTA, CLEAR, Identity, and Count tables: a row per sequence and a `COMBINED` row, scores in percent |
| `summary_lines()` | `list[str]` | The same tables as strings, without printing |
| `report()` | `dict` | The [`EvalReport`](cocoeval.md#report) every family produces; see [`report()`](#report) |
| `benchmark`, `clear_threshold`, `identity_threshold` | | The configuration; `benchmark` is the upper-case name or `None` |
| `reference_deviations()` | `list[str]` | Why this run is not comparable to TrackEval. Empty unless `benchmark` is `None` and a ground-truth row has `conf` 0, which TrackEval never evaluates |
| `provenance()` | `str` | `"parity_verified"` or `"extension"` |

`results()`, `sequence_results()`, `curves()`, `stats`, `summarize()`,
`summary_lines()`, and `report()` raise `RuntimeError` before any sequence is added. An evaluator
pickles; `pickle.loads(pickle.dumps(ev))` carries the configuration and every
sequence's result.

In Rust the same names return `Result` where Python raises. `update` takes
each side as a row-major `&[f64]` with its column count,
`update(name, gt, gt_ncols, pred, pred_ncols, num_frames)`; `update_folders`
takes `&Path`s and an `Option<&Path>` seqmap; `results()` returns a
`TrackingResult` with `hota`, `clear`, `identity`, and `count` fields (the
`metrics::tracking` types `HotaResult`, `ClearResult`, `IdentityResult`, and
`CountResult`), which serializes flat under TrackEval's names;
`sequence_results()` returns a `&BTreeMap<String, TrackingResult>`; `stats()`
returns a `[f64; 44]`; `curves()` is the combined run's, and
`TrackingResult::curves()` a sequence's; and `summary_lines()` is spelled
`summarize_lines()`.

### `results()`

```python
{
    "hotcoco_version": "1.5.0",
    "provenance": "parity_verified",
    "reference_deviations": [],
    "HOTA": 0.5257, "DetA": 0.8042, "AssA": 0.3463, "DetRe": 0.8120, "DetPr": 0.9474,
    "AssRe": 0.4683, "AssPr": 0.5392, "LocA": 0.9083, "OWTA": 0.5292,
    "HOTA(0)": 0.5563, "LocA(0)": 0.9055, "HOTALocA(0)": 0.5038,
    "MOTA": 0.5714, "MOTP": 0.9055, "MODA": 0.8571, "CLR_Re": 0.8571, "CLR_Pr": 1.0,
    "MTR": 0.5, "PTR": 0.5, "MLR": 0.0, "sMOTA": 0.4904, "CLR_F1": 0.9231,
    "FP_per_frame": 0.0, "MOTAL": 0.8141, "MOTP_sum": 5.4329,
    "CLR_TP": 6, "CLR_FN": 1, "CLR_FP": 0, "IDSW": 2, "MT": 1, "PT": 1, "ML": 0, "Frag": 0, "CLR_Frames": 4,
    "IDF1": 0.6154, "IDR": 0.5714, "IDP": 0.6667, "IDTP": 4, "IDFN": 3, "IDFP": 2,
    "Dets": 6, "GT_Dets": 7, "IDs": 2, "GT_IDs": 2,
}
```

The keys are `METRIC_NAMES`, in order, under TrackEval's field names. Scores
are fractions and counts are ints. `HOTA` through `OWTA` are the mean over
the 19 localization thresholds, as TrackEval's summary takes them; `HOTA(0)`,
`LocA(0)`, and `HOTALocA(0)` are read at the first threshold, 0.05.
`MOTP_sum` is the summed IoU of CLEAR's matches, the numerator of `MOTP`. A
sequence's entry in `sequence_results()` adds `Frames`, its frame count; the
combined run has none, as in TrackEval.

### `curves()`

```python
{
    "alphas": [0.05, 0.1, 0.15000000000000002, ..., 0.9500000000000001],
    "HOTA": [...], "DetA": [...], "AssA": [...], "DetRe": [...], "DetPr": [...],
    "AssRe": [...], "AssPr": [...], "LocA": [...], "OWTA": [...],
    "HOTA_TP": [...], "HOTA_FN": [...], "HOTA_FP": [...],
}
```

19 values each, one per threshold. `alphas` are the thresholds exactly as
TrackEval builds them, a numpy range in steps of 0.05, so nine of them sit one
ulp above the decimal. The three counts are floats here.

### `report()`

| Key | Contents |
|-----|----------|
| `task` | `"tracking"` |
| `provenance` | `"parity_verified"`, or `"extension"` when `reference_deviations()` has an entry |
| `metrics` | the 44 values of `results()`, every one a float |
| `per_class` | empty; only pedestrians are scored |
| `per_group` | sequence name → its `METRIC_NAMES` values |
| `curves` | `curves()` of the combined run |
| `params` | `benchmark`, `clear_threshold`, `identity_threshold`, `n_sequences`, `num_alphas` |

---

## `METRIC_NAMES`

```python
["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA",
 "HOTA(0)", "LocA(0)", "HOTALocA(0)",
 "MOTA", "MOTP", "MODA", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR", "sMOTA", "CLR_F1",
 "FP_per_frame", "MOTAL", "MOTP_sum",
 "CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag", "CLR_Frames",
 "IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP",
 "Dets", "GT_Dets", "IDs", "GT_IDs"]
```

The order of `stats` and the keys of `results()` and `report()["metrics"]`.
In Rust, `tracking::METRIC_NAMES`.

---

## The layers underneath

The driver composes the shared layers, as the other families do.

| | Python | Rust |
|---|---|---|
| The assignment | [`primitives.lsap`](primitives.md#lsap) | `primitives::assign::lsap`, scipy's `linear_sum_assignment`, tie-breaking included |
| The numbers | — | `metrics::tracking::{HotaResult, ClearResult, IdentityResult, CountResult}`, each with `combine`; the counts `ClearCounts` and `IdentityCounts`; `association_scores`, `track_coverage`, `fragmentations`, `hota_alphas`, `mean_over_alphas` |

`HotaResult::from_counts` builds one sequence's HOTA from its per-threshold
counts, association scores, and summed similarity; `ClearCounts::scores` and
`IdentityCounts::scores` turn counts into CLEAR's and Identity's scores, and
`ClearResult::unmatched` is TrackEval's early-return row for a sequence with
no ground truth or no predictions.
