# Semantic segmentation

Semantic segmentation labels every pixel with a class and nothing else — no
instances, no scores, one class-id map per image. Its metric is **mean
intersection over union (mIoU)**, reported beside per-class IoU, pixel
accuracy, Dice, and their frequency-weighted and class-averaged forms, and the
reference implementation most published numbers come from is
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)'s `IoUMetric`.
**Boundary IoU** restricts IoU to a band along each class's contour, which
mIoU is blind to on large regions. hotcoco computes all of them from the
same maps — in memory during a validation loop, or from saved PNG files — and
reports them in the same [`EvalReport`](results.md#the-evaluation-report)
shape as detection and panoptic.

## What the metrics measure

All but boundary IoU come from one `num_classes × num_classes` confusion
matrix: how many pixels of ground-truth class `g` were predicted as class
`p`, summed over every image in the split. From its diagonal (TP), rows
(TP + FN), and columns (TP + FP):

```
IoU_c  = TP_c / (TP_c + FP_c + FN_c)       per class
Acc_c  = TP_c / (TP_c + FN_c)              per class: recall over the class's pixels
Dice_c = 2 TP_c / (2 TP_c + FP_c + FN_c)   per class: the F1 of the class's pixels
mIoU   = mean of IoU_c over the classes that have one
mAcc   = mean of Acc_c over the classes that have ground truth
mDice  = mean of Dice_c over the classes that have one
fwIoU  = Σ IoU_c · (TP_c + FN_c) / Σ (TP_c + FN_c)    weighted by ground-truth share
aAcc   = Σ TP_c / Σ (TP_c + FN_c)          overall pixel accuracy
```

Two rules decide which pixels and classes count:

- a ground-truth pixel equal to the **ignore index** — `255` by default, as in
  mmseg, Cityscapes, and VOC — is left out on both sides, so a prediction there
  is neither right nor wrong;
- a class with no pixel on either side has no IoU or Dice and is left out of
  `mIoU` and `mDice`; one with no ground truth has no accuracy and is left
  out of `mAcc`. Each reads as `-1.0`, hotcoco's
  ["not computed"](results.md#the-evaluation-report) marker, never as a low
  score.

### Boundary IoU

A segmentation can score a high IoU on a large region while getting every
contour wrong, because the interior pixels dominate. Boundary IoU
([Cheng et al., CVPR 2021](https://arxiv.org/abs/2103.16562)) keeps only a
band along each mask's contour: the pixels of the mask within `d` of a pixel
outside it or of the image edge, with `d` 2% of the image diagonal (46
pixels on a 2048 × 1024 Cityscapes frame). Per class, the ground-truth band
and the predicted band are intersected and unioned, summed over the split,
and divided:

```
BIoU_c = Σ |B(G_c) ∩ B(P_c)| / Σ |B(G_c) ∪ B(P_c)|    per class, over the split
mBIoU  = mean of BIoU_c over the classes whose union is not empty
```

Ignored ground-truth pixels drop out of both bands, as they do from the
matrix. The band is a function of the maps, not of the confusion matrix, so
this is the one metric here that needs each map's shape; `update` reads it
off the array, a PNG file carries it, and the pass costs a fraction of the
counting. It is on by default and `boundary_dilation=None` leaves it out,
which reads as `-1.0` in `mBIoU`; `boundary_dilation=0.01` narrows the band.

## Evaluate from a validation loop

The evaluator is fed incrementally. Each `update` takes a ground-truth map and
a prediction of the same shape — `(H, W)`, or `(N, H, W)` for a batch — in
any integer dtype:

```python
from hotcoco import semantic

ev = semantic.SemanticEval(19)                      # Cityscapes train ids
for images, gt in val_loader:
    pred = model(images).argmax(1)
    ev.update(gt.numpy(), pred.numpy())
ev.summarize()
```

```
Class    |    IoU    Acc   Dice   BIoU
---------------------------------------
class_0  |  97.81  98.92  98.89  81.20
class_1  |  83.12  90.44  90.78  62.35
...
---------------------------------------
mIoU 76.42  fwIoU 92.10  mAcc 84.07  aAcc 95.88  mDice 85.31  mBIoU 58.70   (500 images, 1046249374 pixels)
```

Pass the class names instead of a count for a readable table and a
`per_class` keyed by name: `SemanticEval(["road", "sidewalk", ...])`. Every
label must be below the number of classes, on either side, except a
ground-truth pixel equal to `ignore_index`; one that is not raises
`ValueError` naming the pixel, and nothing from that pair is counted.

### Distributed runs

The whole state is one confusion matrix and the boundary counts beside it,
so an evaluator pickles and two evaluators `merge()`:

```python
import torch.distributed as dist

gathered = [None] * dist.get_world_size()
dist.all_gather_object(gathered, ev)                # each rank's SemanticEval
if dist.get_rank() == 0:
    total = gathered[0]
    for shard in gathered[1:]:
        total.merge(shard)
    total.report()
```

The merged result equals the single-process one exactly: the counts are
integers, and the order they are added in cannot change the sum.

## Evaluate saved predictions

A folder of predicted PNG files, named like the ground truth's, evaluates in
one call, reading pairs in parallel. By default the PNG files at the top of
each folder are paired by file name:

```python
ev = semantic.SemanticEval(150, reduce_zero_label=True)       # ADE20K
ev.update_folders("ADEChallengeData2016/annotations/validation", "predictions/")
ev.report()["metrics"]["mIoU"]
```

Cityscapes keeps a folder per city and several files per frame, told apart by
suffix, and predictions are usually saved flat under the image's name. Three
options pair that layout: `recursive` walks the subfolders, and the suffixes
say what to strip from each side's name to get the key the two sides share.
A prediction is found wherever it sits in its tree, so nested ground truth
pairs with flat predictions; two predictions with one key raise `ValueError`.

```python
ev = semantic.SemanticEval(19)
ev.update_folders(
    "gtFine/val", "predictions/",
    recursive=True, gt_suffix="_gtFine_labelTrainIds.png", pred_suffix="_leftImg8bit.png",
)
```

This is mmsegmentation's `seg_map_suffix` and `img_suffix` rule. A pairing
no suffix expresses goes through `update_png` on the pairs you choose.

A label-map PNG is single-channel: 8-bit grayscale (Cityscapes
`labelTrainIds`, ADE20K), 16-bit grayscale for more than 256 classes, or a
palette image whose *index* is the class (Pascal VOC, whose border color sits
at index 255). The palette is never expanded — the color is only how a viewer
shows the id. An RGB file is refused, since it either carries no ids or is a
COCO panoptic file, whose colors encode segment ids and which
[`hotcoco.panoptic`](panoptic.md) reads.

`reduce_zero_label` is mmseg's `LoadAnnotations` rule for datasets whose files
put "other" at `0`: ground-truth `0` becomes `255` and every other ground-truth
label shifts down by one, so a file with 151 labels scores 150 classes. It
applies to the ground truth only; predictions are already in the reduced label
space. mmseg hardcodes that `255`, so the rule needs `ignore_index=255`, the
default; any other value is refused rather than given a meaning the reference
does not have.

## Results, the report, and the confusion matrix

```python
r = ev.report()
r["metrics"]                        # {"mIoU", "fwIoU", "mAcc", "aAcc", "mDice", "mBIoU"}, fractions in [0, 1]
r["per_class"]["road"]              # {"IoU", "Acc", "Dice", "BIoU"}, computed values only
r["provenance"]                     # "parity_verified"

ev.results()["per_class"][0]        # {"name", "IoU", "Acc", "Dice", "BIoU", "tp", "gt_area", "pred_area"}
ev.stats                            # the six means in semantic.METRIC_NAMES order
```

`confusion()` returns the matrix itself as a `uint64` array, row the
ground-truth class, column the predicted class. It is the diagnostic this
family has: a class with a low IoU is either being missed (its row leaks into
other columns) or over-predicted (its column collects other rows), and the
matrix says which, and into what. A class whose `BIoU` sits well below its
`IoU` is the other diagnosis: the region is found and its outline is not.

The formulas are callable without an evaluator, the way `sklearn.metrics` and
`torchmetrics.functional` are, for a loop that keeps its own counts:
[`metrics.pixel_confusion`](../api/metrics.md#pixel_confusion) builds one
pair's matrix and [`metrics.mean_iou`](../api/metrics.md#mean_iou) reads the
numbers off a summed one, boundary IoU excepted.

## mIoU of a panoptic prediction

A panoptic model is also scored as a semantic one in its papers. Nothing
needs re-reading for that: `PanopticEval.report()` already carries `mIoU`,
`fwIoU`, `mAcc`, `aAcc`, and `mDice` from the pixels it matched, under the
same definitions as this page — see
[mIoU from the same pixels](panoptic.md#miou-from-the-same-pixels). Boundary
IoU is not among them, since that pass keeps the matrix and not the maps.

## From the CLI

```bash
coco semantic eval --gt annotations/validation --pred predictions/ --classes 150 --reduce-zero-label
coco semantic eval --gt gt/ --pred predictions/ --classes names.txt --json
coco semantic eval --gt gt/ --pred predictions/ --classes 19 --no-boundary-iou
```

The Rust binary has the same subcommand: `coco-eval semantic --gt ... --pred ... --classes ...`.
Both are in the [CLI reference](../cli.md#coco-semantic-eval).

## Verification

Three implementations of this metric are quoted in papers, and they agree on
everything but one corner. mmseg's `IoUMetric`, torchmetrics'
`multiclass_jaccard_index`, and detectron2's `SemSegEvaluator` all mask the
ignore index, build one dataset-wide confusion matrix, and take per-class IoU
as TP over the union. They differ on a class that is predicted somewhere but
absent from every ground-truth map: its union is non-empty and its IoU is
`0`. mmseg and torchmetrics count it toward `mIoU`; detectron2 drops it.
hotcoco follows mmseg and torchmetrics — the convention two of three
references share, and the one SegFormer-lineage numbers are reported under —
and the difference is visible only on a split where some class is never in
the ground truth. `fwIoU` is detectron2's definition; mmseg has none. `mDice`
is mmseg's, under the same rules as `mIoU`.

Boundary IoU has a paper and a kernel but no semantic evaluator to compare
against: the authors' `boundary-iou-api` covers instance and panoptic
segmentation, and the per-class accumulation above is the paper's description.
hotcoco holds its band to the authors' `mask_to_boundary`, run on every class
of every map, and the sums to the definition. detectron2's `SemSegEvaluator`
also reports a `BoundaryIoU`, but it erodes the whole label map and keeps
`label − window minimum`, so a class has a boundary only beside a
lower-numbered class and the number depends on how the classes are numbered.
hotcoco does not follow it, and the fixture records its numbers so the
difference is observed on most cases rather than assumed.

The comparison is frozen rather than run live, because every reference needs
torch or OpenCV: `scripts/gen_semantic_fixtures.py` downloads the sources at
pinned commits (mmseg v1.2.2, detectron2 `1e3e13b`, `boundary-iou-api`
`37d2558`, torchmetrics 1.9.0), runs them on 67 synthetic cases that exercise
ignored pixels, `reduce_zero_label`, classes on one side only, all-ignored
splits, and maps large enough for a band wider than one pixel, and writes the
outputs to `tests/fixtures/semantic_expected.json`. The Rust and Python
suites then check per-class IoU, Acc, and Dice, `mIoU`, `mAcc`, `aAcc`, and
`mDice` against mmseg to 1e-9, per-class IoU and `mIoU` against torchmetrics
to 1e-6 (it reduces in `float32`), `fwIoU`, `mAcc`, and `aAcc` against
detectron2 to 1e-9, per-class and mean boundary IoU against the authors'
kernel to 1e-9, that detectron2's `mIoU` *differs* on every case with a
predicted-but-absent class, and that its boundary IoU differs on at least half
the cases, so both choices are observed, not assumed. The confusion-matrix
metrics are pure functions of the matrix and the boundary band is a pure
function of the maps, so synthetic maps cover them; there is no real-dataset
parity run for this family. See [Semantic parity](../benchmarks.md#semantic).
