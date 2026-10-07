"""Metric functions — numbers computed from matches.

Every function here takes plain arrays and returns plain values. No evaluator, no
dataset, no COCO JSON::

    from hotcoco import metrics

    ap = metrics.average_precision(scores, matched, num_gt=len(ground_truths))
    ece, mce = metrics.calibration_error(scores, matched)

This is the shape ``sklearn.metrics`` and ``torchmetrics.functional`` use, and for
the same reason: a metric is a pure function of its inputs, so tying it to an
evaluator object only makes it harder to reach. ``COCOeval`` is still there when
you want the whole COCO pipeline — its analysis methods call straight into these.

``metrics`` produces numbers from matches; :mod:`hotcoco.primitives` produces the
matches themselves. ``primitives.lsap`` decides *which* prediction pairs with which
ground truth; ``metrics.average_precision`` turns that decision into a number.

Stability
---------

Provisional through the 1.x series — public so drivers and users can share them,
but not frozen until 1.4. Expect additive change rather than reshaping of what is
here. ``COCOeval`` and the ``pycocotools`` drop-in surface are the permanent,
frozen part of the API and are unaffected.
"""

from __future__ import annotations

from .hotcoco import metrics as _metrics

average_precision = _metrics.average_precision
precision_recall_curve = _metrics.precision_recall_curve
calibration_curve = _metrics.calibration_curve
calibration_error = _metrics.calibration_error
confusion_matrix = _metrics.confusion_matrix
panoptic_quality = _metrics.panoptic_quality
is_computed = _metrics.is_computed
is_missing = _metrics.is_missing

__all__ = [
    "average_precision",
    "calibration_curve",
    "calibration_error",
    "confusion_matrix",
    "is_computed",
    "is_missing",
    "panoptic_quality",
    "precision_recall_curve",
]
