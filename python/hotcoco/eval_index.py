"""Backward-compatible wrapper — delegates to COCOeval.image_diagnostics()."""
from __future__ import annotations


def build_eval_index(coco_eval, iou_thr: float = 0.5) -> dict:
    """Extract per-annotation TP/FP/FN status from eval results.

    .. deprecated::
        Use ``coco_eval.image_diagnostics(iou_thr=iou_thr)`` instead.
    """
    return coco_eval.image_diagnostics(iou_thr=iou_thr)
