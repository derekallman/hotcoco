"""Type stubs for the matching-kernel namespace.

The strict layer: ``bbox_iou`` takes boxes and ``mask_iou`` takes RLEs, with no
type dispatch. ``hotcoco.mask.iou`` is the lenient, pycocotools-shaped wrapper
over both. The RLE alias is imported from ``mask.pyi`` so the two stubs share one
definition.
"""

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

from .mask import _Rle

_Flags = Sequence[bool] | Sequence[int] | npt.NDArray[np.bool_] | npt.NDArray[np.integer[Any]]

class _GreedyMatches(TypedDict):
    """What ``greedy_match`` returns: ``(T, D)``, ``(T, G)``, ``(T, D)``, or one axis fewer
    for a single threshold."""

    dt_matches: npt.NDArray[np.int64]
    gt_matches: npt.NDArray[np.int64]
    dt_ignore: npt.NDArray[np.bool_]

class _SegmentMatches(TypedDict):
    """What ``match_segments`` returns, by segment id."""

    matched: list[tuple[int, int, float]]
    missed: list[int]
    false_positives: list[int]
    ignored: list[int]

def lsap(
    cost: npt.NDArray[np.number[Any]] | Sequence[Sequence[float]], maximize: bool = False
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
def bbox_iou(
    dt: npt.NDArray[np.number[Any]] | Sequence[Sequence[float]],
    gt: npt.NDArray[np.number[Any]] | Sequence[Sequence[float]],
    iscrowd: Sequence[bool] | npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.float64]:
    """Pairwise IoU of ``[x, y, w, h]`` boxes, shape ``(D, G)``; ``dt``/``gt`` are ``(N, 4)``.

    ``iscrowd`` is one flag per ``gt`` row; ``None`` means no crowd region."""
    ...

def mask_iou(
    dt: Sequence[_Rle], gt: Sequence[_Rle], iscrowd: Sequence[bool] | npt.NDArray[np.bool_] | None = None
) -> npt.NDArray[np.float64]:
    """Pairwise IoU of RLE masks, shape ``(D, G)``.

    ``iscrowd`` is one flag per ``gt`` mask; ``None`` means no crowd region."""
    ...

def greedy_match(
    sim: npt.NDArray[np.number[Any]] | Sequence[Sequence[float]],
    scores: npt.NDArray[np.number[Any]] | Sequence[float],
    iou_thrs: float | Sequence[float] | npt.NDArray[np.floating[Any]] | None = None,
    *,
    gt_ignore: _Flags | None = None,
    iscrowd: _Flags | None = None,
    dt_ignore: _Flags | None = None,
) -> _GreedyMatches:
    """COCO's greedy matching of ``(D, G)`` similarity, detections in descending score order.

    ``dt_ignore`` flags the detections ignored when nothing matches them, as
    ``COCOeval`` flags those outside the area range; the returned ``dt_ignore``
    is ``evalImgs``' ``dtIgnore``.
    """
    ...

def segment_overlaps(
    gt: npt.NDArray[np.integer[Any]] | npt.NDArray[np.bool_] | Sequence[Any],
    pred: npt.NDArray[np.integer[Any]] | npt.NDArray[np.bool_] | Sequence[Any],
) -> dict[tuple[int, int], int]:
    """``{(gt_id, pred_id): pixels}`` for two label maps of segment ids, 0 for void."""
    ...

_Segments = list[dict[str, Any]] | tuple[dict[str, Any], ...]

def match_segments(
    gt_segments: _Segments, pred_segments: _Segments, overlaps: dict[tuple[int, int], int]
) -> _SegmentMatches:
    """Panoptic quality's matching of one image's ``segments_info`` lists, by id."""
    ...

__all__ = ["bbox_iou", "greedy_match", "lsap", "mask_iou", "match_segments", "segment_overlaps"]
