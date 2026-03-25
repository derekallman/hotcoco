"""PlotData: validated data extraction from a COCOeval object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlotData:
    """Extracted, validated data from a COCOeval object.

    Built via :meth:`from_coco_eval`. All plot functions consume this
    instead of reaching into COCOeval internals directly.
    """

    eval_mode: str  # "coco" | "lvis" | "openimages"
    iou_type: str  # "bbox" | "segm" | "keypoints"
    iou_thresholds: list[float]  # T values — matches precision axis 0
    area_labels: list[str]  # ordered area range labels — matches precision axis 3
    area_ranges: dict[str, tuple[float, float]]
    max_dets: list[int]  # matches precision axis 4
    metrics: dict[str, float]
    per_class: dict[str, float] | None
    precision: np.ndarray  # shape (T, R, K, A, M)
    recall_pts: np.ndarray  # linspace(0, 1, R)
    cat_ids: list[int]  # ordered — matches precision axis 2
    cat_names: dict[int, str]
    metric_key_order: list[str]  # canonical display order from Rust
    version: str

    # ------------------------------------------------------------------
    # Index helpers — used by plot functions to resolve axis positions
    # ------------------------------------------------------------------

    def area_idx(self, label: str) -> int:
        """Return the area-range axis index for *label*, defaulting to 0."""
        return self.area_labels.index(label) if label in self.area_labels else 0

    def max_det_idx(self, max_det: int | None) -> int:
        """Return the max-det axis index, defaulting to the last entry."""
        if max_det is not None and max_det in self.max_dets:
            return self.max_dets.index(max_det)
        return len(self.max_dets) - 1

    def nearest_iou_idx(self, target: float) -> int:
        """Return the IoU-threshold index closest to *target*."""
        return min(range(len(self.iou_thresholds)), key=lambda i: abs(self.iou_thresholds[i] - target))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_coco_eval(cls, coco_eval, *, per_class: bool = False) -> "PlotData":
        """Extract and validate plot data from a COCOeval object.

        Parameters
        ----------
        coco_eval : COCOeval
            Must have ``run()`` called first.
        per_class : bool
            Whether to include per-category AP values. Default False.

        Raises
        ------
        ValueError
            If ``run()`` has not been called, the eval mode is unrecognised,
            or the precision array does not have the expected 5D shape.
        """
        if coco_eval.eval is None:
            raise ValueError("Call coco_eval.run() before plotting.")

        r = coco_eval.results(per_class=per_class)
        params_dict = r["params"]

        valid_modes = {"coco", "lvis", "openimages"}
        if params_dict["eval_mode"] not in valid_modes:
            raise ValueError(f"Unknown eval_mode: {params_dict['eval_mode']!r}")

        precision = np.asarray(coco_eval.eval["precision"])
        if precision.ndim != 5:
            raise ValueError(f"Expected 5-dimensional precision array, got shape {precision.shape}")

        area_labels = list(coco_eval.params.area_rng_lbl)
        cat_ids = list(coco_eval.params.cat_ids)

        try:
            cats = coco_eval.coco_gt.load_cats(cat_ids)
            cat_names = {c["id"]: c["name"] for c in cats}
        except Exception:
            cat_names = {cid: str(cid) for cid in cat_ids}

        return cls(
            eval_mode=params_dict["eval_mode"],
            iou_type=params_dict["iou_type"],
            iou_thresholds=params_dict["iou_thresholds"],
            area_labels=area_labels,
            area_ranges={k: tuple(v) for k, v in params_dict["area_ranges"].items()},
            max_dets=params_dict["max_dets"],
            metrics=r["metrics"],
            per_class=r.get("per_class"),
            precision=precision,
            recall_pts=np.linspace(0.0, 1.0, precision.shape[1]),
            cat_ids=cat_ids,
            cat_names=cat_names,
            metric_key_order=coco_eval.metric_keys(),
            version=r["hotcoco_version"],
        )
