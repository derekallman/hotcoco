# Quick Start

A complete COCO evaluation in under a minute.

## 1. Install

=== "Python"

    ```bash
    pip install hotcoco
    ```

=== "Rust"

    ```bash
    cargo add hotcoco
    ```

=== "CLI"

    ```bash
    cargo install hotcoco-cli
    ```

## 2. Load ground truth

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

!!! tip
    `load_res` automatically computes missing `area` fields from bounding boxes or segmentation masks.

## 4. Run evaluation

=== "Python"

    ```python
    from hotcoco import COCOeval

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ```

=== "Rust"

    ```rust
    use hotcoco::COCOeval;
    use hotcoco::params::IouType;

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

=== "CLI"

    ```bash
    coco-eval --gt instances_val2017.json --dt detections.json --iou-type bbox
    ```

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

    For per-class breakdowns or experiment tracker integration:

    ```python
    import wandb
    wandb.log(ev.get_results(prefix="val/bbox", per_class=True), step=epoch)
    # {"val/bbox/AP": 0.382, ..., "val/bbox/AP/person": 0.72, ...}
    ```

=== "Rust"

    ```rust
    if let Some(stats) = &ev.stats {
        let ap = stats[0];       // AP @ IoU=0.50:0.95, area=all
        let ap_50 = stats[1];    // AP @ IoU=0.50
        let ap_75 = stats[2];    // AP @ IoU=0.75
        println!("AP: {ap:.3}, AP50: {ap_50:.3}, AP75: {ap_75:.3}");
    }
    ```

## 6. Customize evaluation

=== "Python"

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")

    # Evaluate only specific categories
    ev.params.cat_ids = [1, 2, 3]

    # Evaluate only specific images
    ev.params.img_ids = [42, 139, 285]

    # Change max detections
    ev.params.max_dets = [1, 10, 50]

    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ```

=== "Rust"

    ```rust
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);

    // Evaluate only specific categories
    ev.params.cat_ids = vec![1, 2, 3];

    // Evaluate only specific images
    ev.params.img_ids = vec![42, 139, 285];

    // Change max detections
    ev.params.max_dets = vec![1, 10, 50];

    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

## Next steps

- [Evaluation](../guide/evaluation.md) — bbox, segm, and keypoint workflows explained
- [Working with Results](../guide/results.md) — load_res, eval_imgs, precision/recall arrays
- [API Reference](../api/coco.md) — full class and method reference
