# Quick start

A complete COCO evaluation in under a minute.

## 1. Install

```bash
pip install hotcoco
```

Rust and CLI installs: [Installation](installation.md).

## 2. Load ground truth

The ground truth is a COCO-format JSON file containing your dataset's annotations (bounding boxes, segmentation masks, or keypoints). See [The COCO format](coco-format.md) for the expected file layout and which fields are required.

=== "Python"

    ```python
    from hotcoco import COCO

    coco_gt = COCO("instances_val2017.json")

    print(f"Images: {len(coco_gt.get_img_ids())}")
    print(f"Categories: {len(coco_gt.get_cat_ids())}")
    print(f"Annotations: {len(coco_gt.get_ann_ids())}")
    ```

=== "Rust"

    ```rust
    use hotcoco::COCO;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("instances_val2017.json"))?;

    println!("Images: {}", coco_gt.get_img_ids(&[], &[]).len());
    println!("Categories: {}", coco_gt.get_cat_ids(&[], &[], &[]).len());
    println!("Annotations: {}", coco_gt.get_ann_ids(&[], &[], None, None).len());
    ```

=== "CLI"

    The CLI handles loading automatically — skip to step 4.

## 3. Load detection results

=== "Python"

    ```python
    coco_dt = coco_gt.load_res("detections.json")
    ```

=== "Rust"

    ```rust
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;
    ```

Your results file should be a JSON array of detection dicts:

```json
[
  {"image_id": 42, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.95},
  ...
]
```

!!! warning "Bbox format: `[x, y, width, height]` — not `[x1, y1, x2, y2]`"
    The wrong box format produces plausible-looking but incorrect metrics with no error or warning — see [Bbox format](troubleshooting.md#bbox-format-x1-y1-x2-y2-vs-x-y-w-h) for the conversion.

## 4. Run evaluation

=== "Python"

    ```python
    from hotcoco import COCOeval

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.run()  # shorthand for evaluate() + accumulate() + summarize()
    ```

=== "Rust"

    ```rust
    use hotcoco::COCOeval;
    use hotcoco::params::IouType;

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.run()?;  // shorthand for evaluate() + accumulate() + summarize()
    ```

=== "CLI"

    ```bash
    coco eval --gt instances_val2017.json --dt detections.json --iou-type bbox
    ```

    The Rust binary takes the same flags: `coco-eval --gt ... --dt ... --iou-type bbox`.

Output:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.584
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.209
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.323
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680
```

## 5. Access metrics programmatically

=== "Python"

    ```python
    results = ev.get_results()
    # {"AP": 0.382, "AP50": 0.584, "AP75": 0.412, "APs": 0.209, "APm": 0.420, "APl": 0.529, ...}

    ap = results["AP"]
    ap_50 = results["AP50"]
    ```

    For per-class breakdowns or experiment trackers, see [Logging metrics](../guide/results.md#logging-metrics).

=== "Rust"

    ```rust
    if let Some(stats) = ev.stats() {
        let ap = stats[0];       // AP @ IoU=0.50:0.95, area=all
        let ap_50 = stats[1];    // AP @ IoU=0.50
        let ap_75 = stats[2];    // AP @ IoU=0.75
        println!("AP: {ap:.3}, AP50: {ap_50:.3}, AP75: {ap_75:.3}");
    }
    ```

## 6. Customize evaluation

Set `ev.params` — categories, images, IoU thresholds, max detections — before calling `evaluate()`; see [Customizing parameters](../guide/evaluation.md#customizing-parameters).

## Next steps

- [Evaluation](../guide/evaluation.md) — bbox, segm, keypoint, and OBB workflows explained
- [LVIS and Open Images](../guide/lvis-open-images.md) — federated annotation, 13-metric LVIS output, hierarchy-aware Open Images eval
- [Model diagnostics](../guide/diagnostics.md) — TIDE errors, confusion matrix, calibration, F-scores, model comparison
- [PyTorch integration](../guide/pytorch.md) — `CocoDetection` dataset and `CocoEvaluator` for training loops
- [Working with results](../guide/results.md) — load_res, eval_imgs, precision/recall arrays
- [API Reference](../api/coco.md) — full class and method reference
- [Notebook: COCO Evaluation 101](https://github.com/derekallman/hotcoco/blob/main/examples/coco_evaluation_101.ipynb) — end-to-end walkthrough: diagnostics (TIDE, confusion matrix, calibration, label errors), model comparison, dataset ops, plots, and more
