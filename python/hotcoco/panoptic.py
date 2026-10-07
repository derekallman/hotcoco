"""Panoptic segmentation metrics — PQ, SQ, and RQ.

The second metric family, beside :mod:`hotcoco.detection`, on the same engine:
the segment matcher lives in ``primitives``, the formulas in ``metrics``, and
this module drives them over a dataset and reports an ``EvalReport``.

Two ways in. The COCO panoptic format — a JSON file plus a folder of PNG files
whose pixel colors encode segment ids — reads from paths, with the folder
defaulting to the JSON path without ``.json`` as panopticapi assumes::

    from hotcoco import panoptic

    ev = panoptic.PanopticEval("panoptic_val2017.json", "predictions.json")
    ev.run()                        # prints the panopticapi table
    ev.results()["All"]["pq"]       # 0.0 to 1.0
    ev.report()                     # metrics, per_class, per_group, provenance

Or, with no PNG files at all, from detection-style datasets whose annotations
carry RLE or polygon masks, one annotation per segment::

    gt = COCO("instances.json")
    ev = panoptic.PanopticEval(gt, gt.load_res("segments.json"))

``pq_compute`` has panopticapi's signature and return shape, for code that
already calls it::

    from hotcoco.panoptic import pq_compute      # was: from panopticapi.evaluation import pq_compute

    res = pq_compute("panoptic_val2017.json", "predictions.json")

The numbers match panopticapi on COCO val2017 (see the benchmarks page for the
measured figures). Categories need ``isthing`` for the things/stuff split;
without it ``provenance`` is ``"extension"`` and ``reference_deviations()``
says why.
"""

from __future__ import annotations

from .hotcoco import panoptic as _panoptic

PanopticEval = _panoptic.PanopticEval
pq_compute = _panoptic.pq_compute
METRIC_NAMES = _panoptic.METRIC_NAMES

__all__ = ["METRIC_NAMES", "PanopticEval", "pq_compute"]
