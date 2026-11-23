# Report

The result of an evaluation, the same shape for every metric family. An
evaluator's `compute()` returns one:

```python
from hotcoco import semantic

ev = semantic.SemanticEval(19)
for gt, pred in loader:
    ev.update(gt, pred)
report = ev.compute()

print(report)                      # the family's table
report.metrics["mIoU"]             # a float, or None where nothing was computed
```

`compute()` exists on [`SemanticEval`](semantic.md), [`TrackingEval`](tracking.md),
and [`PanopticEval`](panoptic.md). A `Report` has no public constructor: an
evaluator or [`Report.load`](#load) makes it, so its `provenance` is always the
producer's.

---

## Attributes

| Attribute | Type | Contents |
|---|---|---|
| `task` | `str` | `"detection"`, `"panoptic"`, `"semantic"`, or `"tracking"` |
| `metrics` | `dict[str, float \| None]` | Headline metric to value. `None` means nothing was computed for that configuration, such as panoptic's stuff split with no stuff categories. It is never a low score, and a negative value, such as a tracking MOTA below 0, stays a number |
| `per_class` | `dict[str, dict[str, float]]` | Class name to metric to value, with only the values that were computed |
| `per_group` | `dict[str, dict[str, float]]` | Group name to metric to value: tracking's sequences, panoptic's splits, LVIS's frequency groups |
| `curves` | `dict[str, numpy.ndarray]` | Curve name to a `float64` array. A point not computed is `NaN`, which plotting libraries skip |
| `params` | `dict` | The configuration that produced the report |
| `provenance` | `str` | `"parity_verified"`, `"extension"`, or `"user_composed"`; see [provenance](../guide/results.md#check-provenance-before-you-publish-a-number) |
| `deviations` | `list[str]` | Why the run is not comparable to the reference; empty when it is |
| `hotcoco_version` | `str` | The version that produced the report |

The containers are plain dicts, lists, and arrays, built once: `report.metrics`
is the same dict on every read, so it goes straight to `json.dumps`, a logger,
or `pandas.DataFrame(report.per_class).T`. An edit to one is what
[`to_dict`](#to_dict) and [`save`](#save) write.

`compute()` emits each entry of `deviations` as a
[`ReferenceDeviationWarning`](../guide/results.md#filtering-the-warnings), as
`summarize()` does.

---

## Methods

### `print(report)`

The family's table, the lines `summarize()` prints: panopticapi's table for
panoptic, one row per class for semantic, TrackEval's four tables for tracking.

### `to_dict`

```python
to_dict() -> dict
```

The report as plain, JSON-ready containers: a copy, with `None` for a metric or
curve point that was not computed, and the printed table under `summary`.

### `save`

```python
save(path: str | os.PathLike) -> None
```

Write the report as JSON. Raises `ValueError` if an edited container no longer
makes a report, and `OSError` naming the path if it cannot be written.

### `load`

```python
Report.load(path: str | os.PathLike) -> Report
```

Read a report `save()` wrote. Its `provenance` is the one recorded. Raises
`FileNotFoundError` or another `OSError` naming the path, and `ValueError` for
a file that is not a report.

A `Report` also pickles and copies, so it crosses processes as it is.

---

## In Rust

`compute()` returns the core `hotcoco::EvalReport`, the same fields:
`metrics: BTreeMap<String, Option<f64>>`, `per_class`, `per_group`, `curves`,
`params`, `provenance`, `deviations`, `summary` (the table's lines), and
`hotcoco_version`. Each family's `report()` returns it; `EvalReport::save` and
`EvalReport::load` read and write the same JSON as Python.
