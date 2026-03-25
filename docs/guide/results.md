# Working with Results

Beyond the 12 summary metrics, hotcoco gives you access to per-image evaluation results and the full precision/recall arrays.

## Loading results

`load_res` returns a new `COCO` object containing your detections, with images
and categories copied from the ground truth. It accepts three input formats:

=== "JSON file"

    ```python
    coco_gt = COCO("instances_val2017.json")
    coco_dt = coco_gt.load_res("detections.json")
    ```

=== "List of dicts"

    ```python
    detections = [
        {"image_id": 42, "category_id": 1, "bbox": [10, 20, 100, 80], "score": 0.95},
        {"image_id": 42, "category_id": 3, "bbox": [200, 150, 60, 40], "score": 0.72},
    ]
    coco_dt = coco_gt.load_res(detections)
    ```

=== "NumPy array"

    ```python
    import numpy as np

    # Shape (N, 7): [image_id, x, y, w, h, score, category_id]
    arr = np.array([
        [42, 10, 20, 100, 80, 0.95, 1],
        [42, 200, 150, 60, 40, 0.72, 3],
    ], dtype=np.float64)
    coco_dt = coco_gt.load_res(arr)

    # Shape (N, 6): category_id defaults to 1
    coco_dt = coco_gt.load_res(arr[:, :6])
    ```

=== "Rust"

    ```rust
    // From a file
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

    // From in-memory annotations
    let coco_dt = coco_gt.load_res_anns(my_annotations)?;
    ```

`load_res` automatically computes missing fields based on the detection format:

| Detection type | Auto-computed fields |
|---------------|---------------------|
| bbox | `area` from bbox, polygon `segmentation` from bbox |
| segm | `area` from RLE mask |
| keypoints | `area` from keypoint extent bbox |

## Per-image evaluation results

After calling `evaluate()`, the `eval_imgs` field contains per-image, per-category, per-area-range results:

=== "Python"

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()

    # eval_imgs is a list — some entries may be None
    for e in ev.eval_imgs:
        if e is not None:
            print(f"Image {e['image_id']}, Cat {e['category_id']}")
            print(f"  DT matches: {e['dtMatches']}")
            print(f"  GT matches: {e['gtMatches']}")
            print(f"  DT scores:  {e['dtScores']}")
    ```

=== "Rust"

    ```rust
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();

    for e in ev.eval_imgs().iter().flatten() {
        println!("Image {}, Cat {}", e.image_id, e.category_id);
        println!("  DT matches: {:?}", e.dt_matches);
        println!("  GT matches: {:?}", e.gt_matches);
        println!("  DT scores:  {:?}", e.dt_scores);
    }
    ```

Each `eval_img` entry contains:

| Field | Description |
|-------|-------------|
| `image_id` | Image ID |
| `category_id` | Category ID |
| `dt_matches` / `dtMatches` | Detection-to-GT matches per IoU threshold |
| `gt_matches` / `gtMatches` | GT-to-detection matches per IoU threshold |
| `dt_scores` / `dtScores` | Detection confidence scores |
| `gt_ignore` / `gtIgnore` | Whether each GT was ignored (crowd or out of area range) |
| `dt_ignore` / `dtIgnore` | Whether each detection was ignored per IoU threshold |

## Precision and recall arrays

After calling `accumulate()`, the full precision/recall curves are available:

=== "Python"

    ```python
    ev.accumulate()

    # Access the accumulated evaluation
    acc = ev.eval

    # Precision array: shape [T x R x K x A x M]
    # T = IoU thresholds, R = recall thresholds (101),
    # K = categories, A = area ranges, M = max detections
    precision = acc["precision"]
    recall = acc["recall"]
    scores = acc["scores"]

    print(f"Precision shape: {len(precision)}")
    ```

=== "Rust"

    ```rust
    ev.accumulate();

    if let Some(acc) = ev.accumulated() {
        // Index into the 5D precision array [T x R x K x A x M]
        let idx = acc.precision_idx(
            0,  // IoU threshold index
            0,  // recall threshold index
            0,  // category index
            0,  // area range index
            2,  // max detections index
        );
        println!("Precision: {}", acc.precision[idx]);

        // Recall array [T x K x A x M]
        let idx = acc.recall_idx(0, 0, 0, 2);
        println!("Recall: {}", acc.recall[idx]);
    }
    ```

### Array dimensions

| Dimension | Name | Default size | Description |
|-----------|------|-------------|-------------|
| T | IoU thresholds | 10 | `[0.50, 0.55, ..., 0.95]` |
| R | Recall thresholds | 101 | `[0.00, 0.01, ..., 1.00]` |
| K | Categories | varies | Number of evaluated categories |
| A | Area ranges | 4 | `[all, small, medium, large]` |
| M | Max detections | 3 | `[1, 10, 100]` |

Precision has shape `[T x R x K x A x M]`. Recall has shape `[T x K x A x M]`. A value of `-1` means no data (e.g. no GT annotations for that category/area combination).

## Extracting per-category AP

The simplest way is `get_results(per_class=True)`, which returns a flat dict with one entry per category:

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

per_class = ev.get_results(per_class=True)
# {"AP": 0.382, ..., "AP/person": 0.72, "AP/car": 0.41, ...}

for key, val in per_class.items():
    if key.startswith("AP/"):
        print(f"{key[3:]}: {val:.3f}")
```

For direct access to the raw precision arrays (e.g. to compute AP at a non-standard IoU or area range):

=== "Python"

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()

    acc = ev.eval
    precision = acc["precision"]  # shape [T x R x K x A x M]

    # Get category IDs and names
    cat_ids = ev.params.cat_ids
    cats = coco_gt.load_cats(cat_ids)

    # AP per category (IoU=0.50:0.95, area=all, maxDets=100)
    # A=0 (all areas), M=2 (maxDets=100)
    import numpy as np
    for i, cat in enumerate(cats):
        prec = precision[:, :, i, 0, 2]   # shape [T x R]
        prec = prec[prec >= 0]             # exclude -1 (no data)
        ap = float(np.mean(prec)) if prec.size else -1.0
        print(f"{cat['name']}: AP = {ap:.3f}")
    ```

=== "Rust"

    ```rust
    ev.evaluate();
    ev.accumulate();

    if let Some(acc) = ev.accumulated() {
        for (k, &cat_id) in ev.params.cat_ids.iter().enumerate() {
            if let Some(cat) = ev.coco_gt.get_cat(cat_id) {
                // Mean precision across IoU thresholds and recall points
                // for category k, area=all (0), maxDets=100 (2)
                let mut sum = 0.0;
                let mut count = 0;
                for t in 0..acc.t {
                    for r in 0..acc.r {
                        let val = acc.precision[acc.precision_idx(t, r, k, 0, 2)];
                        if val >= 0.0 {
                            sum += val;
                            count += 1;
                        }
                    }
                }
                let ap = if count > 0 { sum / count as f64 } else { -1.0 };
                println!("{}: AP = {:.3}", cat.name, ap);
            }
        }
    }
    ```
