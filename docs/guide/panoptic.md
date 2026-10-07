# Panoptic segmentation

Panoptic segmentation labels every pixel with a category and, for countable
"things", an instance — one map per image that covers people and cars as
instances and sky and road as regions. Its metric is **panoptic quality (PQ)**
([Kirillov et al., CVPR 2019](https://arxiv.org/abs/1801.00868)), and the
reference implementation is
[panopticapi](https://github.com/cocodataset/panopticapi). hotcoco computes PQ
with the same rules and the same numbers, from the same files or from
annotations that carry masks, and reports it in the same
[`EvalReport`](results.md#the-evaluation-report) shape as detection.

## What PQ measures

Ground-truth and predicted segments of the same category are matched when
their IoU is above 0.5. The threshold makes the match unique in both
directions, so no assignment solver is involved. From the matched pairs
(TP), the unmatched ground truth (FN), and the unmatched predictions (FP):

```
PQ = Σ IoU / (TP + ½ FP + ½ FN)      segmentation and recognition together
SQ = Σ IoU / TP                      mean IoU of the matched pairs
RQ = TP / (TP + ½ FP + ½ FN)         an F1 over segments
```

so `PQ = SQ × RQ`. Each is computed per category, then averaged over the
categories that have at least one segment on either side — over all of them,
over things, and over stuff. Two rules keep the protocol fair to the model:

- a prediction more than half covered by **void** (unlabeled pixels) or by a
  **crowd** region of its own category is ignored rather than counted as a
  false positive, and void pixels are left out of the IoU denominator;
- a crowd segment is never matched and never a miss.

## Evaluate the COCO panoptic format

The COCO panoptic release is a JSON file plus a folder of PNG files, one per image,
where each pixel's color encodes its segment id. Predictions take the same
shape. Point `PanopticEval` at the two JSON files; the PNG folders default to
each path without its `.json`, which is where the COCO release and panopticapi
put them:

```python
from hotcoco import panoptic

ev = panoptic.PanopticEval("panoptic_val2017.json", "predictions.json")
ev.run()
```

```
          |    PQ     SQ     RQ     N
--------------------------------------
All       |  62.5   93.5   66.8   133
Things    |  60.4   93.2   64.8    80
Stuff     |  65.6   93.9   69.9    53
```

Pass `gt_folder=` and `pred_folder=` when the PNG files live elsewhere. The
predictions' `categories` are ignored; the ground truth's are scored, as in
panopticapi. `evaluate()` and `summarize()` are the two halves of `run()`
when you want the numbers without the table.

### Drop-in for `pq_compute`

Code that calls panopticapi's one function changes one import:

```python
from hotcoco.panoptic import pq_compute   # was: from panopticapi.evaluation import pq_compute

res = pq_compute("panoptic_val2017.json", "predictions.json")
res["All"]["pq"], res["Things"]["pq"], res["Stuff"]["pq"]
```

Same arguments, same return shape, scores as fractions. The one difference is
in `per_class`: a category with no segment on either side reports `-1.0`,
hotcoco's "not computed" marker, where panopticapi prints `0.0` and then
leaves it out of the average. The `tp`, `fp`, `fn`, and `iou` counts beside
each class are additions.

## Evaluate without PNG files

The PNG round trip is a packaging choice, not part of the metric. A
detection-style dataset whose annotations carry masks — RLE or polygon, one
annotation per segment, `iscrowd` where it applies — evaluates directly:

```python
from hotcoco import COCO, panoptic

gt = COCO("instances_panoptic.json")          # one annotation per segment, with masks
pred = gt.load_res("segments.json")           # the same shape; scores are ignored

ev = panoptic.PanopticEval(gt, pred)
ev.run()
```

Masks are rasterized onto the image's `height × width`, later annotations
over earlier ones where they overlap, and the pixels found are the segment —
a stale `area` field cannot move an IoU on this path. The two inputs can be
mixed: ground truth from PNG files, predictions as masks, or the reverse. On the
same pixels, both paths give identical numbers.

Categories need `isthing` (`1` or `true` for things, `0` or `false` for
stuff) for the things and stuff splits. Without it a category is scored in
`All` and in neither split, and `provenance` reports `"extension"` because
panopticapi would not have evaluated the file at all — see
[provenance](results.md#check-provenance-before-you-publish-a-number).

## Results and the report

`results()` is panopticapi's dict with the counts added; `report()` is the
family-neutral form:

```python
r = ev.report()
r["metrics"]["PQ"], r["metrics"]["PQ_th"], r["metrics"]["PQ_st"]
r["per_class"]["person"]           # {"PQ": ..., "SQ": ..., "RQ": ...}
r["per_group"]["stuff"]["n"]       # categories averaged into the stuff split
r["provenance"]                    # "parity_verified"
```

`ev.stats` holds the nine headline values in `panoptic.METRIC_NAMES` order
— `PQ, SQ, RQ, PQ_th, SQ_th, RQ_th, PQ_st, SQ_st, RQ_st`, fractions in
`[0, 1]` — which is what detection frameworks log, usually multiplied by 100.

## From the CLI

```bash
coco panoptic eval --gt panoptic_val2017.json --pred predictions.json
coco panoptic eval --gt gt.json --pred pred.json --gt-folder gt_png/ --pred-folder pred_png/ --json
```

The Rust binary has the same subcommand: `coco-eval panoptic --gt ... --pred ...`.
Both are in the [CLI reference](../cli.md#coco-panoptic-eval).

## What panopticapi rejects, hotcoco rejects

A predicted segment listed in the JSON but absent from its PNG, a PNG id
missing from the JSON, a prediction with a category the ground truth does not
define, and a ground-truth image with no prediction are all errors, raised
from `evaluate()` with the image id. A ground-truth PNG id with no JSON entry
is accepted and treated as neither a segment nor void, as the reference does.

Where panopticapi would divide by zero — a things or stuff split with no
evaluable category — hotcoco reports `-1.0` with `n = 0` instead of failing.

## Verification

PQ, SQ, and RQ match panopticapi on COCO panoptic val2017 to the last
printed digit, with every per-category count identical — the figures are in
[Panoptic parity](../benchmarks.md#panoptic). The same comparison runs in CI
on synthetic label maps that exercise every rule above.
