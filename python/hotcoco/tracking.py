"""Multi-object tracking metrics — HOTA, CLEAR (MOTA, MOTP), and Identity (IDF1).

The fourth metric family, beside :mod:`hotcoco.detection`,
:mod:`hotcoco.panoptic`, and :mod:`hotcoco.semantic`, on the same engine: the
assignment is ``primitives.lsap``, the formulas live in ``metrics``, and this
module drives them over MOTChallenge sequences and reports an ``EvalReport``.
The numbers and the field names are TrackEval's.

The evaluator takes one sequence per ``update``, as rows in MOTChallenge
column order — ``frame, id, x, y, w, h, conf, class, visibility`` — with the
sequence's frame count::

    import numpy as np
    from hotcoco.tracking import TrackingEval

    ev = TrackingEval("MOT17")                      # MOT17's clean-up; None for your own data
    gt = np.loadtxt("MOT17-02-FRCNN/gt/gt.txt", delimiter=",", ndmin=2)
    pred = np.loadtxt("MyTracker/data/MOT17-02-FRCNN.txt", delimiter=",", ndmin=2)
    ev.update("MOT17-02-FRCNN", gt, pred, num_frames=600)
    ev.summarize()                                  # TrackEval's HOTA, CLEAR, Identity, Count tables
    ev.results()["HOTA"]

Or a whole benchmark in the folder layout TrackEval reads::

    ev.update_folders("gt/mot_challenge/MOT17-train", "trackers/mot_challenge/MOT17-train/MyTracker/data")

Each sequence is evaluated when it is added and only its result is kept, so an
evaluator pickles and two evaluators over different sequences ``merge()``.
Only MOTChallenge 2D boxes are covered so far; see the tracking guide for what
is not.
"""

from __future__ import annotations

from .hotcoco import tracking as _tracking

TrackingEval = _tracking.TrackingEval
METRIC_NAMES = _tracking.METRIC_NAMES

__all__ = ["METRIC_NAMES", "TrackingEval"]
