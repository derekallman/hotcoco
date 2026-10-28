"""Semantic segmentation metrics — mIoU, fwIoU, mAcc, aAcc, mDice, and boundary IoU.

The third metric family, beside :mod:`hotcoco.detection` and
:mod:`hotcoco.panoptic`, on the same engine: the formulas live in ``metrics``
(``pixel_confusion`` and ``mean_iou``), and this module drives them over a
dataset and reports an ``EvalReport``. Boundary IoU is the one metric here
that is not a function of the confusion matrix: it reads a band along each
class's contour off the maps themselves, so ``update`` takes maps with their
shape, and ``boundary_dilation=None`` leaves it out.

The evaluator is fed incrementally, the way a validation loop produces label
maps. Each call takes a ground-truth map and a prediction of the same shape,
``(H, W)`` or a ``(N, H, W)`` batch, in any integer dtype::

    from hotcoco import semantic

    ev = semantic.SemanticEval(19)                  # Cityscapes train ids; 255 is ignored
    for gt, logits in loader:
        ev.update(gt.numpy(), logits.argmax(1).numpy())
    ev.summarize()                                  # per-class IoU, Acc, Dice, BIoU, then the means
    ev.report()["metrics"]["mIoU"]

Or from saved predictions — PNG files named like the ground truth's, with the
class id as the pixel value or the palette index::

    ev = semantic.SemanticEval(150, reduce_zero_label=True)   # ADE20K's "other" at 0
    ev.update_folders("ADEChallengeData2016/annotations/validation", "preds/")

Its whole state is one ``num_classes × num_classes`` confusion matrix, so an
evaluator pickles and two evaluators over disjoint shards ``merge()`` into the
whole-split result. ``confusion()`` returns the matrix, which is the
diagnostic: which classes are being mistaken for which.

The numbers match mmsegmentation's ``IoUMetric`` and torchmetrics'
``multiclass_jaccard_index`` on frozen label maps, with one documented choice
where detectron2 differs, and boundary IoU matches the kernel its authors
published; see the semantic guide.
"""

from __future__ import annotations

from .hotcoco import semantic as _semantic

SemanticEval = _semantic.SemanticEval
METRIC_NAMES = _semantic.METRIC_NAMES

__all__ = ["METRIC_NAMES", "SemanticEval"]
