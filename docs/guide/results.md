# Working with results

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

`load_res` fills in fields the detection format leaves out, such as `area` — the per-format list is in the [`load_res` API reference](../api/coco.md#load_res).

## Per-image evaluation results

After calling `evaluate()`, the `eval_imgs` field contains per-image, per-category, per-area-range results:

=== "Python"

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()

    # eval_imgs is a list — some entries can be None
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

The keys match pycocotools' `evalImgs`; the full list is in the [`eval_imgs` API reference](../api/cocoeval.md#eval_imgs).

`evaluate()` does not build these records up front. It keeps only what
`accumulate()` reads, about 20 bytes per detection, and materializes the full
records the first time something asks for them: this attribute, `tide_errors()`,
`calibration()`, or `image_diagnostics()`. That first access repeats the matching
pass, roughly the cost of `evaluate()` itself, and is cached afterwards. The
records describe the `evaluate()` run, whatever `params` has been changed to since.

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

The five dimensions and their default sizes are in the [`eval` API reference](../api/cocoeval.md#eval).

## The evaluation report

`report()` returns everything about a finished evaluation in one dict — metrics,
per-class and per-group breakdowns, plottable curves, and the parameters that
produced them.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

report = ev.report()
report["metrics"]["AP"]              # 0.377
report["per_class"]["person"]["AP"]  # 0.521
```

Every hotcoco metric family reports in this shape, so a function that renders one
renders them all.

### Check provenance before you publish a number

`provenance` says whether a result is comparable to a published leaderboard:

```python
report["provenance"]   # 'parity_verified'
```

`"parity_verified"` means the numbers were checked against the reference
implementation — bbox, segm, and keypoints match pycocotools **at reference
parameters**. `"extension"` means a real metric or configuration with no reference implementation checked against
it. That covers more than geometry:

- oriented bounding boxes — no reference protocol exists to check against
- Open Images — the group-of protocol is checked, but the challenge's image-level-label
  rule is not implemented; see [Verification](lvis-open-images.md#verification)
- any run with non-default `iou_thrs`, `rec_thrs`, `max_dets`, area-range labels
  or bounds, `use_cats=False`, or custom `kpt_oks_sigmas` — the metric is real,
  but nobody checked *that* configuration against a reference

Extension numbers are fine for comparing your own models against each other; they
are not leaderboard numbers.

The distinction goes missing the moment results reach a chart, so it travels with the
data and survives saving and reloading:

```python
if report["provenance"] != "parity_verified":
    print(f"note: {report['provenance']} — not benchmark-standard")
```

It also travels into everything hotcoco draws. The PDF report carries a provenance
line, the browse dashboard carries a banner, and `coco eval --json` carries both the
marker and the reasons — so a chart handed to someone who never ran the evaluation
still says what it is.

#### Checking before you evaluate

`provenance()` reads only the configuration, so unlike `report()` it works before
`run()`. Worth checking ahead of a long evaluation rather than discovering afterwards
that the numbers cannot be published:

```python
ev = hotcoco.COCOeval(gt, dt, "bbox")
ev.params.iouThrs = [0.5]

ev.provenance()             # 'extension' — already, before evaluating
ev.reference_deviations()   # ['iou_thrs differ from default (0.50:0.05:0.95). ...']
```

`reference_deviations()` returns one sentence per reason and is empty exactly when
the run is parity-verified. It is the same predicate behind the warnings `summarize()`
prints, so a report cannot claim parity while the warnings disagree.

Read these rather than inferring comparability from `iou_type` or the eval mode:
parity is a property of the whole configuration, so the preceding run is an extension
even though it is ordinary COCO bbox evaluation.

### Plotting precision-recall curves

`curves` holds one aggregate PR curve per IoU threshold, all sharing the
`"rec_thrs"` x-axis:

```python
import matplotlib.pyplot as plt

curves = report["curves"]
for iou in ("pr@0.50", "pr@0.75", "pr@0.95"):
    plt.plot(curves["rec_thrs"], curves[iou], label=iou)

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
```

These are averaged over categories at `area="all"` and the largest `max_dets` — the
slice a chart draws. For per-category curves, read the `eval["precision"]` array
directly, as [Extracting per-category AP](#extracting-per-category-ap) shows; on COCO
that array is roughly a million floats, which is why the report carries only the
aggregate.

## Extracting per-category AP

The simplest way is `get_results(per_class=True)`, which adds one `"AP/{category}"` entry per category to the flat dict shown in [Logging metrics](#logging-metrics):

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

per_class = ev.get_results(per_class=True)
for key, val in per_class.items():
    if key.startswith("AP/"):
        print(f"{key[3:]}: {val:.3f}")
```

For direct access to the raw precision arrays, for example to compute AP at a non-standard IoU or area range:

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
            if let Some(cat) = ev.coco_gt().get_cat(cat_id) {
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

## Saving results to JSON

`results()` and `save_results()` serialize the full evaluation output — parameters, summary metrics, and optionally per-category AP — to a JSON dict or file. Both require `summarize()` (or `run()`) first.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

# Get results as a dict
r = ev.results()
print(r["metrics"]["AP"])      # 0.378
print(r["params"]["iou_type"]) # "bbox"

# Save to a file
ev.save_results("results.json")

# Include per-category AP
ev.save_results("results.json", per_class=True)
```

The JSON structure is in the [`results()` API reference](../api/cocoeval.md#results).
`reference_deviations` travels with the file, so a saved results file explains its
own provenance — see [Checking before you evaluate](#checking-before-you-evaluate).

The `--output` flag of the [`coco-eval` Rust CLI](../cli.md#coco-eval-rust-cli) writes the same JSON.

## Logging metrics

`get_results()` accepts an optional `prefix` and `per_class` flag, returning a flat `dict[str, float]` that plugs directly into any experiment tracker.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

metrics = ev.get_results(prefix="val/bbox", per_class=True)
# {"val/bbox/AP": 0.578, ..., "val/bbox/AP/person": 0.82, "val/bbox/AP/car": 0.71, ...}
```

### Weights & Biases

```python
import wandb
wandb.log(ev.get_results(prefix="val/bbox", per_class=True), step=epoch)
```

### MLflow

```python
import mlflow
mlflow.log_metrics(ev.get_results(prefix="val/bbox"), step=epoch)
```

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
for k, v in ev.get_results(prefix="val/bbox").items():
    writer.add_scalar(k, v, global_step=epoch)
```

See [`get_results`](../api/cocoeval.md#get_results) in the API reference for full parameter details.
