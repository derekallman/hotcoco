# TrackingEval

Multi-object tracking evaluator — HOTA, CLEAR/MOTA, Identity/IDF1.

=== "Python"

    ```python
    from hotcoco import COCO, TrackingEval

    coco_gt = COCO("tracking_gt.json")
    coco_dt = coco_gt.load_res("tracking_dt.json")

    ev = TrackingEval(coco_gt, coco_dt)
    ev.run()
    ```

=== "Rust"

    ```rust
    use hotcoco::{COCO, IouType};
    use hotcoco::eval::tracking::TrackingEval;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("tracking_gt.json"))?;
    let coco_dt = coco_gt.load_res(Path::new("tracking_dt.json"))?;

    let mut ev = TrackingEval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.run();
    ```

---

## Constructor

=== "Python"

    ```python
    TrackingEval(
        coco_gt: COCO,
        coco_dt: COCO,
        iou_type: str = "bbox",
    )
    ```

=== "Rust"

    ```rust
    TrackingEval::new(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self
    ```

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_gt` | `COCO` | Ground truth dataset with tracking annotations (`video_id`, `track_id`, `frame_index`). |
| `coco_dt` | `COCO` | Detection/tracking results. |
| `iou_type` | `str` | `"bbox"` (default) or `"segm"`. |

---

## Methods

### `evaluate()`

Run per-sequence matching and metric computation. Groups images by `video_id`, sorts by `frame_index`, then evaluates each sequence in parallel.

### `accumulate()`

Aggregate per-sequence results into dataset-level metrics. Integer counts are summed across sequences; derived ratios (HOTA, MOTA, IDF1) are recomputed from the totals.

### `summarize()`

Print a formatted table of all tracking metrics to stdout.

### `run()`

Convenience: calls `evaluate()`, `accumulate()`, and `summarize()` in sequence.

### `get_results(prefix=None)`

Returns a flat `dict[str, float]` of all metrics.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | `str \| None` | `None` | Key prefix (e.g. `"val/track"` → `"val/track/HOTA"`). |

**Returns:** `dict[str, float]` with keys: `HOTA`, `DetA`, `AssA`, `LocA`, `DetRe`, `DetPr`, `AssRe`, `AssPr`, `MOTA`, `MOTP`, `IDSW`, `CLR_TP`, `CLR_FN`, `CLR_FP`, `MT`, `PT`, `ML`, `Frag`, `IDF1`, `IDP`, `IDR`, `IDTP`, `IDFN`, `IDFP`.

---

## Module-level functions

### `mot_to_coco(path)`

Convert a MOTChallenge `gt.txt` file to a COCO tracking dataset dict.

```python
from hotcoco import mot_to_coco

dataset = mot_to_coco("MOT17-02/gt/gt.txt")
coco = COCO(dataset)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to a MOTChallenge-format `gt.txt` file. |

**Returns:** `dict` — a COCO-format dataset dict with `images`, `annotations`, `categories`, `videos`, and `tracks`.

### `init_as_trackeval()`

Patch `sys.modules` so that `from trackeval import ...` resolves to hotcoco.

```python
from hotcoco import init_as_trackeval
init_as_trackeval()
```
