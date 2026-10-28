"""Type stubs for the metric-function namespace.

Unlike ``detection.pyi``, these names are *not* re-exported from the top level —
they exist only under ``hotcoco.metrics`` — so the real signatures live here.

Float and bool inputs accept lists or numpy arrays. A ``bool`` ndarray, or a
float input as an ndarray of any integer or float dtype, is read in one pass;
anything else falls back to per-element extraction but still works.
"""

from typing import Any, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

_Floats: TypeAlias = Sequence[float] | npt.NDArray[Any]
_Bools: TypeAlias = Sequence[bool] | npt.NDArray[Any]

def average_precision(
    scores: _Floats, matched: _Bools, num_gt: int, ignored: _Bools | None = None, rec_thrs: _Floats | None = None
) -> float: ...
def precision_recall_curve(
    tp_cum: _Floats, fp_cum: _Floats, num_gt: int, rec_thrs: _Floats | None = None
) -> tuple[float, list[tuple[int, float, int]]]: ...
def calibration_curve(scores: _Floats, matched: _Bools, n_bins: int = 10) -> list[dict[str, Any]]: ...
def calibration_error(scores: _Floats, matched: _Bools, n_bins: int = 10) -> tuple[float, float]: ...
def confusion_matrix(
    gt: Sequence[int | None], dt: Sequence[int | None], num_classes: int
) -> npt.NDArray[np.uint64]: ...
def panoptic_quality(iou_sum: float, tp: int, fp: int, fn_: int) -> tuple[float, float, float]:
    """``(PQ, SQ, RQ)`` from match counts; ``(-1.0, -1.0, -1.0)`` when there is nothing to score."""
    ...

def pixel_confusion(
    gt: npt.ArrayLike, pred: npt.ArrayLike, num_classes: int, *, ignore_index: int | None = None
) -> npt.NDArray[np.uint64]:
    """The ``(num_classes, num_classes)`` pixel confusion matrix of one label-map pair; row the ground truth."""
    ...

def mean_iou(confusion: npt.ArrayLike) -> dict[str, Any]:
    """``mIoU``, ``fwIoU``, ``mAcc``, ``aAcc``, ``mDice`` and per-class ``IoU``/``Acc``/``Dice`` lists from a square
    count matrix."""
    ...

def mean_or_missing(total: float, count: int) -> float:
    """The mean of ``count`` values summing to ``total``, or ``-1.0`` when ``count`` is 0."""
    ...

def is_computed(v: float) -> bool:
    """Whether a metric value was actually computed (``-1.0`` means "not computed")."""
    ...

def is_missing(v: float) -> bool:
    """Whether a metric value is the ``-1.0`` "not computed" sentinel."""
    ...

__all__ = [
    "average_precision",
    "calibration_curve",
    "calibration_error",
    "confusion_matrix",
    "is_computed",
    "is_missing",
    "mean_iou",
    "mean_or_missing",
    "panoptic_quality",
    "pixel_confusion",
    "precision_recall_curve",
]
