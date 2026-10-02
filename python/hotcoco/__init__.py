"""Perception evaluation in pure Rust.

Covers detection today — bbox, segmentation, keypoints, and oriented boxes on
the COCO, LVIS, and Open Images protocols. Works as a drop-in pycocotools
replacement::

    from hotcoco import init_as_pycocotools
    init_as_pycocotools()

Also included: TIDE error analysis, confusion matrices, calibration, per-image
diagnostics, dataset tools, and metric functions that work on plain arrays.
Docs: https://derekallman.github.io/hotcoco/
"""

from __future__ import annotations

from .hotcoco import (  # noqa: F401
    COCO,
    COCOeval,
    Hierarchy,
    Params,
    StreamingEval,
    __version__,
    compare,
    init_as_lvis,
    init_as_pycocotools,
    mask,
)

# `COCO` is the Rust class itself, `browse()` included (a Rust method that
# forwards to `hotcoco.browse.browse_coco`). Do not wrap it in a Python
# subclass: every dataset the Rust core returns — `split()`, `filter()`,
# `load_res()`, `sample()`, `merge()` — is constructed as the base class, so a
# subclass's methods vanish from derived datasets.


class LVISeval:
    """Drop-in replacement for lvis-api LVISEval.

    Returns a ``COCOeval`` instance configured for LVIS federated evaluation
    (``lvis_style=True``). Supports ``run()`` / ``print_results()`` /
    ``get_results()`` as required by Detectron2 and MMDetection.

    Parameters
    ----------
    gt : COCO
        Ground-truth COCO object loaded from an LVIS annotation file.
    dt : COCO
        Detection results COCO object, for example from ``gt.load_res(...)``.
    iou_type : str
        One of ``"bbox"``, ``"segm"``, or ``"keypoints"``.
    """

    def __new__(cls, gt, dt, iou_type="segm"):
        return COCOeval(gt, dt, iou_type, lvis_style=True)


# lvis-api spells it `LVISEval`, with a capital E — `from lvis import LVISEval`
# is the canonical import, and it is what Detectron2 and MMDetection write. The
# `LVISeval` spelling above follows pycocotools' `COCOeval`, so both exist: one
# for consistency with the rest of hotcoco, one so `init_as_lvis()` actually
# satisfies the import it promises to.
LVISEval = LVISeval

# lvis-api uses LVIS as the dataset class name, not COCO.
LVIS = COCO


class LVISResults:
    """Drop-in replacement for lvis-api LVISResults.

    ``LVISResults(lvis_gt, predictions, max_dets=300)`` returns a ``COCO``
    object. ``max_dets`` is accepted for API compatibility; detection
    truncation is handled by ``LVISeval`` params (``max_dets=300``).
    """

    def __new__(cls, lvis_gt, results, max_dets=300):  # noqa: ARG003
        return lvis_gt.load_res(results)


import sys as _sys  # noqa: E402

from . import detection, metrics, primitives  # noqa: E402, F401

# `mask` is a PyO3 submodule object, which `from hotcoco import mask` finds as an
# attribute but `import hotcoco.mask` does not — the import system looks in
# sys.modules, and PyO3's add_submodule does not register there. Anyone migrating
# from `import pycocotools.mask` writes the second form, so register it.
_sys.modules.setdefault("hotcoco.mask", mask)
from .integrations import CocoDetection, CocoEvaluator  # noqa: E402, F401

__all__ = [
    "COCO",
    "COCOeval",
    "CocoDetection",
    "CocoEvaluator",
    "Hierarchy",
    "LVIS",
    "LVISEval",
    "LVISResults",
    "LVISeval",
    "Params",
    "StreamingEval",
    "__version__",
    "compare",
    "detection",
    "init_as_lvis",
    "init_as_pycocotools",
    "mask",
    "metrics",
    "primitives",
]
