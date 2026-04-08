# Tracking Evaluation

hotcoco evaluates multi-object tracking (MOT) results using three metric families: **HOTA**, **CLEAR/MOTA**, and **Identity/IDF1**. All three are computed in a single pass.

## Data format

Tracking evaluation uses TAO-style COCO JSON with three extra fields:

| Field | Where | Description |
|-------|-------|-------------|
| `video_id` | images, annotations | Which video sequence this belongs to |
| `frame_index` | images | Frame position within its video (0-indexed) |
| `track_id` | annotations | Object track ID (consistent across frames) |

```json
{
  "videos": [{"id": 1, "name": "MOT17-02"}],
  "images": [
    {"id": 1, "video_id": 1, "frame_index": 0, "width": 1920, "height": 1080, "file_name": "000001.jpg"},
    {"id": 2, "video_id": 1, "frame_index": 1, "width": 1920, "height": 1080, "file_name": "000002.jpg"}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "track_id": 1, "video_id": 1, "bbox": [100, 100, 50, 80], "area": 4000},
    {"id": 2, "image_id": 2, "category_id": 1, "track_id": 1, "video_id": 1, "bbox": [110, 100, 50, 80], "area": 4000}
  ],
  "categories": [{"id": 1, "name": "pedestrian"}],
  "tracks": [{"id": 1, "category_id": 1, "video_id": 1}]
}
```

Standard COCO JSON (without these fields) still loads normally — the tracking fields are optional everywhere.

## Quick start

=== "Python"

    ```python
    from hotcoco import COCO, TrackingEval

    coco_gt = COCO("tracking_gt.json")
    coco_dt = coco_gt.load_res("tracking_dt.json")

    ev = TrackingEval(coco_gt, coco_dt)
    ev.run()  # evaluate + accumulate + summarize
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

The three-step pipeline is the same as `COCOeval`: `evaluate()` → `accumulate()` → `summarize()`. The `run()` method calls all three.

## Output

`summarize()` prints a table for each metric family:

```
 HOTA  | DetA   | AssA   | LocA   | DetRe  | DetPr  | AssRe  | AssPr
 0.5423 | 0.6012 | 0.4891 | 0.7834 | 0.7523 | 0.7234 | 0.6123 | 0.6789

 MOTA  | MOTP   | IDSW  | CLR_TP | CLR_FN | CLR_FP | MT  | PT  | ML  | Frag
 0.6234 | 0.7834 |    12 |   4523 |   1234 |    567 |  34 |  12 |   4 |   23

 IDF1  | IDP    | IDR    | IDTP  | IDFN  | IDFP
 0.5678 | 0.6012 | 0.5378 |  3456 |  1234 |   567
```

## Extracting results

`get_results()` returns a flat `dict[str, float]` — ready for experiment trackers:

```python
results = ev.get_results()
# {"HOTA": 0.5423, "MOTA": 0.6234, "IDF1": 0.5678, "IDSW": 12.0, ...}

# With prefix for logging:
results = ev.get_results(prefix="val/track")
# {"val/track/HOTA": 0.5423, "val/track/MOTA": 0.6234, ...}
```

## Metrics reference

### HOTA (Higher Order Tracking Accuracy)

HOTA balances detection and association quality at multiple IoU thresholds (0.05–0.95). It decomposes into:

| Metric | Description |
|--------|-------------|
| **HOTA** | Geometric mean of DetA and AssA, averaged over IoU thresholds |
| **DetA** | Detection accuracy: TP / (TP + FN + FP) |
| **AssA** | Association accuracy: how well matched pairs share track identity |
| **LocA** | Localization accuracy: mean IoU of matched pairs |
| **DetRe/DetPr** | Detection recall/precision |
| **AssRe/AssPr** | Association recall/precision |

### CLEAR (MOTA/MOTP)

The classic MOT metrics:

| Metric | Description |
|--------|-------------|
| **MOTA** | 1 - (FN + FP + IDSW) / num_GT |
| **MOTP** | Mean IoU of matched pairs |
| **IDSW** | Number of identity switches |
| **MT/PT/ML** | Mostly tracked / partially tracked / mostly lost |
| **Frag** | Number of track fragmentations |

### Identity (IDF1)

Global identity-based assignment:

| Metric | Description |
|--------|-------------|
| **IDF1** | F1 score of global identity assignment |
| **IDP/IDR** | Identity precision/recall |
| **IDTP/IDFN/IDFP** | Identity true positives/false negatives/false positives |

## MOTChallenge import

hotcoco can convert MOTChallenge `gt.txt` files to COCO tracking format:

```python
from hotcoco import mot_to_coco

dataset = mot_to_coco("MOT17-02/gt/gt.txt")
# Returns a dict with images, annotations, videos, tracks
```

The standard MOT format is: `frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility`.

## Drop-in for TrackEval

If your codebase imports from `trackeval`, you can switch to hotcoco with one line:

```python
from hotcoco import init_as_trackeval
init_as_trackeval()

# Existing trackeval imports now resolve to hotcoco
from trackeval import TrackingEval
```
