from .hotcoco import *  # noqa: F401, F403


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
        Detection results COCO object (e.g. from ``gt.load_res(...)``).
    iou_type : str
        One of ``"bbox"``, ``"segm"``, or ``"keypoints"``.
    """

    def __new__(cls, gt, dt, iou_type="segm"):  # noqa: F405
        return COCOeval(gt, dt, iou_type, lvis_style=True)  # noqa: F405


# lvis-api uses LVIS as the dataset class name, not COCO.
LVIS = COCO  # noqa: F405


class LVISResults:
    """Drop-in replacement for lvis-api LVISResults.

    ``LVISResults(lvis_gt, predictions, max_dets=300)`` returns a ``COCO``
    object. ``max_dets`` is accepted for API compatibility; detection
    truncation is handled by ``LVISeval`` params (``max_dets=300``).
    """

    def __new__(cls, lvis_gt, results, max_dets=300):  # noqa: ARG003
        return lvis_gt.load_res(results)
