"""Detection metrics — AP, AR, and the detection diagnostic suite.

hotcoco is growing from a COCO evaluation library into a perception evaluation
toolkit, with one namespace per metric family. This is the first of them;
:mod:`hotcoco.panoptic` is the second, and tracking follows the same shape.

Nothing here is new, and nothing is going away. ``hotcoco.detection.COCOeval``
*is* ``hotcoco.COCOeval`` — the same object, re-exported under the family name so
that code evaluating several families reads consistently::

    from hotcoco import detection, panoptic

    det = detection.COCOeval(gt, dt, "bbox")
    det.run()

The top-level names are permanent. ``hotcoco.COCOeval``, ``hotcoco.mask``, and
the ``pycocotools``/LVIS drop-in surface are compatibility guarantees, not
deprecated aliases — if you are replacing ``pycocotools``, keep importing from
the top level and ignore this module entirely.

The LVIS helpers (``LVISeval``, ``LVISResults``, ``LVIS``) deliberately stay at
the top level: they exist to mirror ``lvis-api``'s import paths, so moving them
under a family namespace would defeat their purpose.
"""

from __future__ import annotations

from .hotcoco import COCOeval, Hierarchy, Params, StreamingEval, compare

__all__ = ["COCOeval", "Hierarchy", "Params", "StreamingEval", "compare"]
