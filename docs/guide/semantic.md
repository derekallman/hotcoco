# Semantic segmentation

Semantic segmentation labels every pixel with a class and nothing else — no
instances, no scores, one class-id map per image. Its metric is **mean
intersection over union (mIoU)**, reported beside per-class IoU, pixel
accuracy, and their frequency-weighted and class-averaged forms, and the
reference implementation most published numbers come from is
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)'s `IoUMetric`.
hotcoco computes the same numbers from the same maps — in memory during a
validation loop, or from saved PNG files — and reports them in the same
[`EvalReport`](results.md#the-evaluation-report) shape as detection and
panoptic.

## What mIoU measures

Everything comes from one `num_classes × num_classes` confusion matrix:
how many pixels of ground-truth class `g` were predicted as class `p`, summed
over every image in the split. From its diagonal (TP), rows (TP + FN), and
columns (TP + FP):

```
IoU_c  = TP_c / (TP_c + FP_c + FN_c)       per class
Acc_c  = TP_c / (TP_c + FN_c)              per class: recall over the class's pixels
mIoU   = mean of IoU_c over the classes that have one
mAcc   = mean of Acc_c over the classes that have ground truth
fwIoU  = Σ IoU_c · (TP_c + FN_c) / Σ (TP_c + FN_c)    weighted by ground-truth share
aAcc   = Σ TP_c / Σ (TP_c + FN_c)          overall pixel accuracy
```

Two rules decide which pixels and classes count:

- a ground-truth pixel equal to the **ignore index** — `255` by default, as in
  mmseg, Cityscapes, and VOC — is left out on both sides, so a prediction there
  is neither right nor wrong;
- a class with no pixel on either side has no IoU and is left out of `mIoU`;
  one with no ground truth has no accuracy and is left out of `mAcc`. Both
  read as `-1.0`, hotcoco's ["not computed"](results.md#the-evaluation-report)
  marker, never as a low score.

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
Class    |    IoU    Acc
-------------------------
class_0  |  97.81  98.92
class_1  |  83.12  90.44
...
-------------------------
mIoU 76.42  fwIoU 92.10  mAcc 84.07  aAcc 95.88   (500 images, 1046249374 pixels)
```

Pass the class names instead of a count for a readable table and a
`per_class` keyed by name: `SemanticEval(["road", "sidewalk", ...])`. Every
label must be below the number of classes, on either side, except a
ground-truth pixel equal to `ignore_index`; one that is not raises
`ValueError` naming the pixel, and nothing from that pair is counted.

### Distributed runs

The whole state is one confusion matrix, so an evaluator pickles and two
evaluators `merge()`:

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
one call, reading pairs in parallel:

```python
ev = semantic.SemanticEval(150, reduce_zero_label=True)       # ADE20K
ev.update_folders("ADEChallengeData2016/annotations/validation", "predictions/")
ev.report()["metrics"]["mIoU"]
```

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
r["metrics"]                        # {"mIoU", "fwIoU", "mAcc", "aAcc"}, fractions in [0, 1]
r["per_class"]["road"]              # {"IoU": ..., "Acc": ...}, computed values only
r["provenance"]                     # "parity_verified"

ev.results()["per_class"][0]        # {"name", "IoU", "Acc", "tp", "gt_area", "pred_area"}
ev.stats                            # the four means in semantic.METRIC_NAMES order
```

`confusion()` returns the matrix itself as a `uint64` array, row the
ground-truth class, column the predicted class. It is the diagnostic this
family has: a class with a low IoU is either being missed (its row leaks into
other columns) or over-predicted (its column collects other rows), and the
matrix says which, and into what.

The formulas are callable without an evaluator, the way `sklearn.metrics` and
`torchmetrics.functional` are, for a loop that keeps its own counts:
[`metrics.pixel_confusion`](../api/metrics.md#pixel_confusion) builds one
pair's matrix and [`metrics.mean_iou`](../api/metrics.md#mean_iou) reads the
numbers off a summed one.

## mIoU of a panoptic prediction

A panoptic model is also scored as a semantic one in its papers. Nothing
needs re-reading for that: `PanopticEval.report()` already carries `mIoU`,
`fwIoU`, `mAcc`, and `aAcc` from the pixels it matched, under the same
definitions as this page — see
[mIoU from the same pixels](panoptic.md#miou-from-the-same-pixels).

## From the CLI

```bash
coco semantic eval --gt gtFine/val --pred predictions/ --classes 19
coco semantic eval --gt annotations/validation --pred predictions/ --classes 150 --reduce-zero-label --json
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
the ground truth. `fwIoU` is detectron2's definition; mmseg has none.

The comparison is frozen rather than run live, because every reference needs
torch: `scripts/gen_semantic_fixtures.py` downloads the three sources at
pinned commits (mmseg v1.2.2, detectron2 `1e3e13b`, torchmetrics 1.9.0), runs
them on 64 synthetic cases that exercise ignored pixels, `reduce_zero_label`,
classes on one side only, and all-ignored splits, and writes the outputs to
`tests/fixtures/semantic_expected.json`. The Rust and Python suites then
check per-class IoU and Acc, `mIoU`, `mAcc`, and `aAcc` against mmseg to
1e-9, per-class IoU and `mIoU` against torchmetrics to 1e-6 (it reduces in
`float32`), `fwIoU`, `mAcc`, and `aAcc` against detectron2 to 1e-9, and that
detectron2's `mIoU` *differs* on every case with a predicted-but-absent class,
so the convention choice is observed, not assumed. The metric is a pure
function of the confusion matrix, so synthetic maps cover it completely; there
is no real-dataset parity run for this family. See
[Semantic parity](../benchmarks.md#semantic).

## Not included

**Dice** is the same matrix read differently — per class, `2 · IoU / (1 + IoU)`
— and the convention of medical segmentation rather than of the ADE20K and
Cityscapes literature. **Boundary IoU** restricts IoU to a band along each
mask's contour and needs an erosion primitive and its own reference. Neither
ships; the [roadmap](https://github.com/derekallman/hotcoco/blob/main/ROADMAP.md)
records both as open if asked for.
