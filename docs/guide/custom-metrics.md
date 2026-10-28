# Build your own metric

hotcoco's evaluators report the standard numbers. When you need one they don't
report — F1 at the threshold you deploy at, a confusion matrix with your own
matching rule, a variant of PQ — build it from `hotcoco.primitives`, which decides
what pairs with what, and `hotcoco.metrics`, which turns counts into numbers. The
matchers are the ones `COCOeval` and `PanopticEval` run, so whatever your metric
shares with theirs comes out the same.

The matchers this page uses are provisional, as the
[primitives](../api/primitives.md) reference explains; it and the
[metrics](../api/metrics.md) reference hold every argument and return shape this
page relies on.

## The shape of a metric

Every metric here takes four steps: score each prediction against each ground
truth in one image, match them, count the outcomes, and apply a formula. hotcoco
supplies the first two.

| Step | Detection | Panoptic |
|---|---|---|
| Similarity | `primitives.bbox_iou` or `primitives.mask_iou` | `primitives.segment_overlaps` |
| Matching | `primitives.greedy_match` | `primitives.match_segments` |
| Counting | yours, per image and category | yours, per image and category |
| Formula | yours: precision, recall, F1 | `metrics.panoptic_quality` |

Count per image, sum the counts over the dataset, and apply the formula once, at
the end. Averaging a ratio per image gives a different number: an image with one
object would weigh as much as an image with fifty. torchmetrics splits a metric
the same way, into `update` and `compute` with states that reduce by summing, and
so does TrackEval, into `eval_sequence` and `combine_sequences`.

## Precision, recall, and F1 at one IoU

AP summarizes every confidence threshold; a deployed detector runs at one. This
metric counts true positives, false positives, and misses per category at IoU
0.50 and a minimum score. [`f_scores`](../api/cocoeval.md#f_scores) reports the
best F1 over all thresholds instead.

Group the annotations by image and category first:

```python
from collections import defaultdict
from operator import itemgetter

import numpy as np
from hotcoco import COCO, metrics, primitives


def group(coco, key):
    """A dataset's annotations grouped by ``key(ann)``, each group in dataset order."""
    groups = defaultdict(list)
    for ann in coco.load_anns(coco.get_ann_ids()):
        groups[key(ann)].append(ann)
    return groups
```

Then match each group and count:

```python
def detection_counts(
    gt, dt, iou_thr=0.5, score_thr=float("-inf"), area_range=(0, 1e10), max_det=100
):
    """``[tp, fp, fn]`` per category id, at one IoU threshold."""
    lo, hi = area_range
    by_cell = itemgetter("image_id", "category_id")
    gt_cells, dt_cells = group(gt, by_cell), group(dt, by_cell)
    category_ids = set(gt.get_cat_ids())
    counts = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for image_id, category_id in gt_cells.keys() | dt_cells.keys():
        if category_id not in category_ids:
            continue  # COCOeval scores only the ground truth's categories
        gts = gt_cells[image_id, category_id]
        dts = [d for d in dt_cells[image_id, category_id] if d["score"] >= score_thr]
        dts = sorted(dts, key=lambda d: -d["score"])[:max_det]

        iscrowd = np.array([g["iscrowd"] for g in gts], dtype=bool)
        gt_in_range = np.array([lo <= g["area"] <= hi for g in gts], dtype=bool)
        dt_in_range = np.array([lo <= d["area"] <= hi for d in dts], dtype=bool)
        dt_boxes, gt_boxes = [d["bbox"] for d in dts], [g["bbox"] for g in gts]
        sim = primitives.bbox_iou(dt_boxes, gt_boxes, iscrowd)
        m = primitives.greedy_match(
            sim,
            [d["score"] for d in dts],
            iou_thr,
            gt_ignore=~gt_in_range,
            iscrowd=iscrowd,
            dt_ignore=~dt_in_range,
        )

        matched, ignored = m["dt_matches"] >= 0, m["dt_ignore"]
        tp = np.sum(matched & ~ignored)
        fp = np.sum(~matched & ~ignored)
        fn = np.sum(gt_in_range & ~iscrowd) - tp
        counts[category_id] += (tp, fp, fn)
    return counts
```

`greedy_match` takes the detections best score first and gives each the unmatched
ground truth it overlaps most, at or above the threshold. `dt_matches` holds the
column each detection matched, or `-1`. A crowd region absorbs any number of
detections without counting them: those come back with `dt_ignore` set, neither
true nor false positives. Pass `iscrowd` to both calls, since `bbox_iou` scores a
crowd region by the detection's own area.

The rest is COCO's protocol. `max_det` is COCO's cap of 100 detections per image
and category, and a detection whose category the ground truth does not list is
never scored. Dropping detections below `score_thr` before matching gives the
same matches as dropping them after, because a detection's match depends only on
the detections that score higher.

Sum the counts, then divide:

```python
def precision_recall_f1(tp, fp, fn):
    """Each ratio is the mean of an indicator over what it divides by, so
    `mean_or_missing` gives `-1.0`, hotcoco's "not computed", where there is nothing."""
    ratio = metrics.mean_or_missing
    return ratio(tp, tp + fp), ratio(tp, tp + fn), ratio(2 * tp, 2 * tp + fp + fn)


gt = COCO("instances_val2017.json")
dt = gt.load_res("detections.json")

counts = detection_counts(gt, dt, score_thr=0.5)
precision, recall, f1 = precision_recall_f1(*sum(counts.values()))
names = {c["id"]: c["name"] for c in gt.load_cats(gt.get_cat_ids())}
per_category = {names[c]: precision_recall_f1(*n) for c, n in counts.items()}
```

`sum(counts.values())` pools every category before dividing, the micro average;
`per_category` has one row per class. A ratio with nothing to divide — recall for
a category no ground truth has — is `-1.0`, so `metrics.is_computed` can leave it
out of a mean over categories. With no `score_thr`, the true and false positives
are the ones `COCOeval` records in
[`eval_imgs`](results.md#per-image-evaluation-results) at the same IoU threshold.

### One area range

COCO's small, medium, and large scores keep every ground truth in the matching and
flag those outside the range with `gt_ignore`. A flagged ground truth can still
absorb a detection, so a box on a medium object is not a false positive for
small. Detections outside the range are flagged with `dt_ignore`, and count only
when they match a ground truth in the range. Both rules are in
`detection_counts` already:

```python
small = detection_counts(gt, dt, area_range=(0, 32**2))
```

## Cross-class confusion matrix

`COCOeval` matches each category on its own. Drop that split — match every
detection in an image against every ground truth, whatever its class — and each
match records which class a ground truth was taken for.
[`metrics.confusion_matrix`](../api/metrics.md#confusion_matrix) counts those
records, with index `K` for background: a ground truth nothing matched, or a
detection that matched nothing. This uses `group` from the preceding section.

```python
def confusion(gt, dt, iou_thr=0.5, max_det=100):
    """``(K + 1, K + 1)`` counts, rows the ground truth's class; ``K`` is background."""
    label = {c: i for i, c in enumerate(gt.get_cat_ids())}
    by_image, by_category = itemgetter("image_id"), itemgetter("category_id")
    gt_images, dt_images = group(gt, by_image), group(dt, by_image)
    gt_labels, dt_labels = [], []
    for image_id in gt_images.keys() | dt_images.keys():
        # Category-major within a score, as COCOeval orders them, so ties break alike.
        gts = sorted((g for g in gt_images[image_id] if not g["iscrowd"]), key=by_category)
        dts = [d for d in dt_images[image_id] if d["category_id"] in label]
        dts = sorted(dts, key=lambda d: (-d["score"], d["category_id"]))[:max_det]

        dt_boxes, gt_boxes = [d["bbox"] for d in dts], [g["bbox"] for g in gts]
        sim = primitives.bbox_iou(dt_boxes, gt_boxes, [False] * len(gts))
        m = primitives.greedy_match(sim, [d["score"] for d in dts], iou_thr)
        for d, j in zip(dts, m["dt_matches"]):
            gt_labels.append(label[gts[j]["category_id"]] if j >= 0 else None)
            dt_labels.append(label[d["category_id"]])
        for g, i in zip(gts, m["gt_matches"]):
            if i < 0:
                gt_labels.append(label[g["category_id"]])
                dt_labels.append(None)
    return metrics.confusion_matrix(gt_labels, dt_labels, len(label))


cm = confusion(gt, dt)
```

Crowd regions are left out: a region that absorbs any number of detections would
count one object many times. A detection on a crowd region therefore lands in the
background row, where `detection_counts` ignores it, so that row is not COCO's
false-positive count.

`cm` is the `"matrix"` that [`ev.confusion_matrix()`](diagnostics.md#confusion-matrix)
returns, so [Finding class confusions](diagnostics.md#finding-class-confusions)
reads it the same way. Build it yourself to change the rule: let same-class
matches claim their ground truths first, or keep the label pairs per image to find
the images the confusions come from.

## Panoptic quality from the segment matcher

Panoptic quality starts from a different similarity: the pixel count of every
pair of ground-truth and predicted segments, which
[`segment_overlaps`](../api/primitives.md#segment_overlaps) reads off two label
maps. [`match_segments`](../api/primitives.md#match_segments) applies PQ's
rules — same category, IoU above 0.5, void pixels left out of the union — and
returns the matched pairs with their IoU, the missed ground truths, the false
positives, and the predictions that count for neither. Per category, the counts
are what [`metrics.panoptic_quality`](../api/metrics.md#panoptic_quality) takes.

`image_counts` is one image, from label maps in memory — a model's output, say.
`panoptic_counts` reads the COCO panoptic files and sums the images, which takes
Pillow for the PNG files.

```python
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from hotcoco import metrics, primitives
from PIL import Image


def image_counts(gt_ids, gt_segments, pred_ids, pred_segments):
    """One image's ``[iou_sum, tp, fp, fn]`` per category id."""
    overlaps = primitives.segment_overlaps(gt_ids, pred_ids)
    m = primitives.match_segments(gt_segments, pred_segments, overlaps)
    gt_category = {s["id"]: s["category_id"] for s in gt_segments}
    pred_category = {s["id"]: s["category_id"] for s in pred_segments}
    counts = defaultdict(lambda: [0.0, 0, 0, 0])
    for gt_id, _, iou in m["matched"]:
        counts[gt_category[gt_id]][0] += iou
        counts[gt_category[gt_id]][1] += 1
    for pred_id in m["false_positives"]:
        counts[pred_category[pred_id]][2] += 1
    for gt_id in m["missed"]:
        counts[gt_category[gt_id]][3] += 1
    return counts


def read_ids(path):
    """A COCO panoptic PNG as segment ids: R + 256 G + 256² B."""
    rgb = np.asarray(Image.open(path), dtype=np.uint32)
    return rgb[..., 0] + 256 * rgb[..., 1] + 256**2 * rgb[..., 2]


def panoptic_counts(gt_json, pred_json, gt_folder, pred_folder):
    """The categories, and ``[iou_sum, tp, fp, fn]`` per category id over every image."""
    with open(gt_json) as f:
        gt = json.load(f)
    with open(pred_json) as f:
        pred = {a["image_id"]: a for a in json.load(f)["annotations"]}
    totals = {c["id"]: [0.0, 0, 0, 0] for c in gt["categories"]}
    for g in gt["annotations"]:
        p = pred[g["image_id"]]
        gt_ids = read_ids(Path(gt_folder) / g["file_name"])
        pred_ids = read_ids(Path(pred_folder) / p["file_name"])
        counts = image_counts(gt_ids, g["segments_info"], pred_ids, p["segments_info"])
        for category_id, n in counts.items():
            totals[category_id] = [t + x for t, x in zip(totals[category_id], n)]
    return gt["categories"], totals


def pq_average(categories, counts):
    """PQ, SQ, and RQ averaged over the categories with something to score."""
    scores = [metrics.panoptic_quality(*counts[c["id"]]) for c in categories]
    scores = [s for s in scores if metrics.is_computed(s[0])]
    totals = np.reshape(scores, (-1, 3)).sum(axis=0)  # three zeros when nothing scored
    return tuple(metrics.mean_or_missing(total, len(scores)) for total in totals)
```

```python
categories, pq_counts = panoptic_counts(
    "panoptic_val2017.json",
    "panoptic_predictions.json",
    "panoptic_val2017",
    "panoptic_predictions",
)
pq, sq, rq = pq_average(categories, pq_counts)
pq_things = pq_average([c for c in categories if c["isthing"]], pq_counts)
pq_stuff = pq_average([c for c in categories if not c["isthing"]], pq_counts)
```

A category with nothing on either side is left out of the average, as
panopticapi leaves it out. Every count, `iou_sum` included, equals
`PanopticEval`'s: each image's IoUs are added before the image totals, the order
`PanopticEval` adds them in. The averages agree to within 1e-12, because
`numpy.mean` adds pairwise. For the last bit too, average with a running `+=` as
`PanopticEval` does; since Python 3.12, `sum()` compensates for rounding and
lands one bit away as well.

`PanopticEval` keeps the same match records per image in
[`eval_imgs`](panoptic.md#per-image-matches), for when the standard matching is
all you need.

## Next steps

- [primitives](../api/primitives.md) — `mask_iou` scores RLE masks the way
  `bbox_iou` scores boxes, and `lsap` finds an optimal one-to-one assignment
  where greedy matching is not the protocol.
- [metrics](../api/metrics.md) — `average_precision` turns scores and match flags
  into AP, and `calibration_error` checks whether the scores mean what they say.
