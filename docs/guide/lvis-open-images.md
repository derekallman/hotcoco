# LVIS and Open Images

COCO's protocol is not the only one hotcoco speaks. LVIS needs federated evaluation over its 1,200-category long tail, and Open Images has its own single-threshold AP with category hierarchies and group-of boxes. Both are built in — this page covers when each applies and how to run them.

## LVIS evaluation

[LVIS](https://www.lvisdataset.org/) ([Gupta et al., ECCV 2019](https://arxiv.org/abs/1908.03195)) is a large-vocabulary instance segmentation dataset with ~1,200 categories. It uses **federated annotation** — each image is only exhaustively labeled for a subset of categories. Running standard COCO eval on LVIS over-penalizes detectors by treating every unannotated category as a missed detection. hotcoco handles this correctly out of the box.

### Drop-in replacement for lvis-api

If your pipeline already uses lvis-api, see [LVIS-based pipelines](frameworks.md#lvis-based-pipelines).

### Direct usage

If you're not using lvis-api, use `LVISeval` or pass `lvis_style=True` to `COCOeval`:

```python
from hotcoco import COCO, LVISeval

lvis_gt = COCO("lvis_v1_val.json")
lvis_dt = lvis_gt.load_res("detections.json")

ev = LVISeval(lvis_gt, lvis_dt, "segm")  # lvis_style=True is set automatically
ev.run()
results = ev.get_results()
# {"AP": 0.42, "APr": 0.38, "APc": 0.44, "APf": 0.45, "AR@300": ..., ...}
```

Or equivalently:

```python
from hotcoco import COCO, COCOeval

ev = COCOeval(lvis_gt, lvis_dt, "segm", lvis_style=True)
ev.evaluate()
ev.accumulate()
ev.summarize()
results = ev.get_results()
```

### The 13 LVIS metrics

| Metric | Description |
|--------|-------------|
| AP | mAP @ IoU[0.5:0.05:0.95] |
| AP50 | mAP @ IoU=0.5 |
| AP75 | mAP @ IoU=0.75 |
| APs | AP for small objects |
| APm | AP for medium objects |
| APl | AP for large objects |
| APr | AP for rare categories (1–10 training images) |
| APc | AP for common categories (11–100 training images) |
| APf | AP for frequent categories (more than 100 training images) |
| AR@300 | Mean recall @ max 300 detections per image |
| ARs@300 | AR for small objects |
| ARm@300 | AR for medium objects |
| ARl@300 | AR for large objects |

Small, medium, and large are the COCO area ranges — see [`area_rng`](../api/params.md#area_rng). The frequency split (rare / common / frequent) is determined by the `frequency` field on each category in the LVIS annotation file (`"r"`, `"c"`, `"f"`). These correspond to the number of training images in which the category appears, as defined in the LVIS paper.

`get_results()` returns all 13 metrics as a dict for programmatic access.

## Open Images evaluation

[Open Images](https://storage.googleapis.com/openimages/web/index.html) uses a different evaluation protocol from COCO: a single AP@IoU=0.5, a category hierarchy for annotation expansion, and `is_group_of` annotations in place of `iscrowd`. hotcoco supports all three.

### Quick start

```python
from hotcoco import COCO, COCOeval

coco_gt = COCO("oid_annotations.json")
coco_dt = coco_gt.load_res("detections.json")

ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True)
ev.run()

result = ev.get_results()
# {"AP": 0.573}
```

`oid_style=True` evaluates at a single IoU threshold of 0.50, over the `all` area range only, with 100 max detections — so `summarize()` prints one line and `get_results()` returns `{"AP": <float>}`.

Starting from the CSV files Open Images ships rather than from COCO JSON, load them with `COCO.from_oid` and `load_res_oid` — see [Open Images](datasets.md#open-images) for the readers, their options, and what happens without image dimensions. `IsGroupOf` carries through to the matching rules in [Group-of annotations](#group-of-annotations).

### Category hierarchy

Open Images categories form a hierarchy — a "Dog" detection also counts as an "Animal" detection if Animal is an ancestor of Dog. Pass a `Hierarchy` to expand GT annotations automatically at evaluation time.
Each box expands on its own, as the TF Object Detection API expands it: a dog box and a cat box with the same coordinates become two "Animal" ground truths. A ground truth that is already expanded — the input holds the identical box at the ancestor — is not copied again, so pre-expanded Open Images files evaluate unchanged.

```python
from hotcoco import COCO, COCOeval, Hierarchy

# From the OID hierarchy JSON (bbox_labels_600_hierarchy.json)
label_to_id = {cat["name"]: cat["id"] for cat in coco_gt.dataset["categories"]}
h = Hierarchy.from_file("bbox_labels_600_hierarchy.json", label_to_id=label_to_id)

ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True, hierarchy=h)
ev.run()
```

If you don't have a hierarchy JSON, omit `hierarchy=` entirely and `oid_style=True` derives one from each category's `supercategory` field. A hierarchy can also be built from an explicit parent map or an in-memory dict — see [Hierarchy](../api/hierarchy.md) for the constructors and the derivation rules.

### Detection expansion

By default only GT annotations are expanded up the hierarchy; set [`expand_dt`](../api/params.md#expand_dt) to also expand detections, so a "Dog" detection counts as an "Animal" detection too.

### Group-of annotations

OID uses `is_group_of: true` on annotations covering a *cluster* of objects — five or more instances of the same class, occluding each other, where no individual box can be drawn. A cluster is one thing you either found or didn't:

- **A group-of box is worth exactly one ground truth.** The best-scoring detection inside it is a true positive. Every other detection inside it is ignored — neither true positive nor false positive. Detecting the pile twice earns nothing extra. This is where it differs from COCO's `iscrowd`: a crowd region is dropped from the denominator, while a group-of box is counted once and can be found.
- **Missing it costs one false negative.** An undetected group-of box counts once against recall.
- **"Inside" is IoA, not IoU** — intersection divided by the *detection's* area, the same measure COCO uses for `iscrowd`. A detection wholly inside the box qualifies however small it is, which is the point: individual objects are much smaller than the cluster that contains them.

This is the [Open Images Challenge protocol](https://storage.googleapis.com/openimages/web/evaluation.html), equivalently TensorFlow's `group_of_weight = 1.0`, and it is what FiftyOne implements. It is checked against the TensorFlow Object Detection API — see [Verification](#verification).

Open Images AP also uses **VOC 2010 all-points integration** — the exact area under the precision-recall curve — rather than COCO's 101-point recall grid. The protocol specifies it and both reference implementations do it, so an OID number here is not directly comparable to a COCO number computed on the same data.

!!! note "Not the same as the Open Images V2 metric"

    The older V2 detection metric ignored group-of boxes entirely — they contributed to neither the numerator nor the denominator (`group_of_weight = 0.0`). Both are real published protocols. hotcoco implements the Challenge metric, so **numbers here don't match a V2-era leaderboard**.

Your annotations need `"is_group_of": true` in the JSON for this to take effect. Standard annotations without this field default to `false`.

### Verification

Open Images evaluation is compared against the TensorFlow Object Detection API, and every case agrees to within one ulp — the cases, the figures, and the two things the comparison does not cover (which is why `provenance` still reports `"extension"`) are in [Open Images parity](../benchmarks.md#open-images).

See [Hierarchy](../api/hierarchy.md) in the API reference for full construction and query methods.
