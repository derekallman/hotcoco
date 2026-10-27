"""Matching kernels — deciding what pairs with what.

The layer underneath :mod:`hotcoco.metrics`: similarity between two sets, and the
matchers that turn similarity into pairs::

    from hotcoco import primitives

    sim = primitives.bbox_iou(detections, ground_truths, iscrowd)
    m = primitives.greedy_match(sim, scores, 0.5, iscrowd=iscrowd)   # COCO's matcher
    rows, cols = primitives.lsap(sim, maximize=True)                   # optimal assignment

Panoptic segments are matched from the pixel overlaps of two label maps::

    overlaps = primitives.segment_overlaps(gt_map, pred_map)
    m = primitives.match_segments(gt_segments_info, pred_segments_info, overlaps)

Nothing here scores. Feed the matches to :mod:`hotcoco.metrics` for that.

The IoU kernels are strict: ``bbox_iou`` takes boxes, ``mask_iou`` takes RLEs.
``hotcoco.mask`` mirrors ``pycocotools.mask`` and is a permanent compatibility
surface — its ``iou`` accepts either and dispatches on type the way pycocotools
does, then forwards here. ``mask.bbox_iou`` is this module's ``bbox_iou``. One
implementation underneath, so the two surfaces cannot disagree.

``greedy_match`` and ``match_segments`` are the matchers ``COCOeval`` and
``PanopticEval`` run, not copies of them.

Stability
---------

Provisional through the 1.x series, like :mod:`hotcoco.metrics`. The IoU kernels
are the exception: those are frozen, since ``pycocotools`` parity depends on them.
"""

from __future__ import annotations

from .hotcoco import primitives as _primitives

lsap = _primitives.lsap
bbox_iou = _primitives.bbox_iou
mask_iou = _primitives.mask_iou
greedy_match = _primitives.greedy_match
segment_overlaps = _primitives.segment_overlaps
match_segments = _primitives.match_segments

__all__ = ["bbox_iou", "greedy_match", "lsap", "mask_iou", "match_segments", "segment_overlaps"]
