from .hotcoco import COCO as _RustCOCO
from .hotcoco import *  # noqa: F401, F403


class COCO(_RustCOCO):
    """COCO dataset — extends the Rust core with Python-only methods."""

    def __init__(self, annotation_file=None, *, image_dir=None):  # noqa: ARG002
        # Rust __new__ handles construction and stores image_dir.
        # This __init__ exists only to accept the same kwargs without complaint.
        pass

    def browse(self, image_dir: str | None = None, dt=None, batch_size: int = 12):
        """Launch an interactive Gradio dataset browser.

        Parameters
        ----------
        image_dir : str, optional
            Root directory for image files. Overrides ``self.image_dir``.
        dt : COCO or str, optional
            Detection results to overlay. Pass a COCO object (from
            ``self.load_res()``) or a path string (auto-loaded).
        batch_size : int
            Number of images loaded per batch (default 12).

        Returns
        -------
        gr.Blocks
            The Gradio app object (already launched).

        Raises
        ------
        ValueError
            If ``image_dir`` is ``None`` and ``self.image_dir`` is also ``None``.
        ImportError
            If ``gradio`` is not installed (``pip install hotcoco[browse]``).
        """
        from . import browse as _browse

        dt_coco = None
        if dt is not None:
            dt_coco = self.load_res(dt) if isinstance(dt, str) else dt

        app = _browse.build_app(self, image_dir=image_dir, batch_size=batch_size, dt_coco=dt_coco)
        _browse.launch_app(app)
        return app


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
LVIS = COCO


class LVISResults:
    """Drop-in replacement for lvis-api LVISResults.

    ``LVISResults(lvis_gt, predictions, max_dets=300)`` returns a ``COCO``
    object. ``max_dets`` is accepted for API compatibility; detection
    truncation is handled by ``LVISeval`` params (``max_dets=300``).
    """

    def __new__(cls, lvis_gt, results, max_dets=300):  # noqa: ARG003
        return lvis_gt.load_res(results)


from .integrations import CocoDetection, CocoEvaluator  # noqa: E402, F401
