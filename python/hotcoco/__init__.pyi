"""Type stubs for hotcoco — a pure Rust port of pycocotools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

# Submodule namespaces. Re-exported explicitly (`as` form) so `from hotcoco
# import metrics` type-checks, not only `import hotcoco.metrics` — a stub
# package shadows the runtime `__init__.py`, so a checker sees only what this
# file declares. `mask` gets the same treatment: its signatures live in
# mask.pyi, which also makes `import hotcoco.mask` (the pycocotools migration
# form) resolve for the type checker.
from . import detection as detection
from . import mask as mask
from . import metrics as metrics
from . import primitives as primitives

# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------

class COCO:
    """COCO dataset API — load, query, and convert annotations."""

    image_dir: str | None

    def __init__(
        self, annotation_file: str | dict[str, Any] | None = None, *, image_dir: str | None = None
    ) -> None: ...

    # --- Query ---
    def get_ann_ids(
        self,
        img_ids: int | list[int] = ...,
        cat_ids: int | list[int] = ...,
        area_rng: list[float] | None = None,
        iscrowd: bool | None = None,
    ) -> list[int]: ...
    def get_cat_ids(
        self, cat_nms: str | list[str] = ..., sup_nms: str | list[str] = ..., cat_ids: int | list[int] = ...
    ) -> list[int]: ...
    def get_img_ids(self, img_ids: int | list[int] = ..., cat_ids: int | list[int] = ...) -> list[int]: ...

    # --- Load ---
    def load_anns(self, ids: int | list[int]) -> list[dict[str, Any]]: ...
    def load_cats(self, ids: int | list[int]) -> list[dict[str, Any]]: ...
    def load_imgs(self, ids: int | list[int]) -> list[dict[str, Any]]: ...
    def load_res(self, res: str | list[dict[str, Any]] | npt.NDArray[Any]) -> COCO: ...

    # --- Masks ---
    def ann_to_rle(self, ann: dict[str, Any]) -> dict[str, Any]:
        """RLE in pycocotools format: ``{"size": [h, w], "counts": bytes}``."""
        ...
    def ann_to_mask(self, ann: dict[str, Any]) -> npt.NDArray[np.uint8]: ...

    # --- Properties ---
    @property
    def dataset(self) -> dict[str, Any]:
        """A fresh copy on every access — in-place mutation is a no-op; assign back to apply."""
        ...
    @dataset.setter
    def dataset(self, value: dict[str, Any]) -> None: ...
    @property
    def load_warnings(self) -> list[str]:
        """Warnings collected while loading/indexing; empty for a clean load."""
        ...
    @property
    def imgs(self) -> dict[int, dict[str, Any]]: ...
    @property
    def anns(self) -> dict[int, dict[str, Any]]: ...
    @property
    def cats(self) -> dict[int, dict[str, Any]]: ...

    # --- Utilities ---
    def create_index(self) -> None: ...
    def update_anns(self, anns: list[dict[str, Any]]) -> None:
        """Replace whole annotations by ``id`` and re-index; unknown id raises ``KeyError``."""
        ...
    def set_ann_field(self, field: str, values: dict[int, Any], *, create: bool = False) -> None:
        """Set one field on the named annotations and re-index, keeping their other fields."""
        ...
    def stats(self) -> dict[str, Any]: ...
    def healthcheck(self, dt: COCO | None = None) -> dict[str, Any]: ...
    def filter(
        self,
        cat_ids: list[int] | None = None,
        img_ids: list[int] | None = None,
        area_rng: list[float] | None = None,
        drop_empty_images: bool = True,
    ) -> COCO: ...
    def save(self, path: str) -> None: ...
    def split(
        self, val_frac: float = 0.2, test_frac: float | None = None, seed: int = 42
    ) -> tuple[COCO, COCO] | tuple[COCO, COCO, COCO]: ...
    def sample(self, n: int | None = None, frac: float | None = None, seed: int = 42) -> COCO: ...

    # --- Conversion ---
    def to_yolo(self, output_dir: str) -> dict[str, int]: ...
    def to_voc(self, output_dir: str) -> dict[str, int]: ...
    def to_cvat(self, output_path: str) -> dict[str, int]: ...
    def to_dota(self, output_dir: str) -> dict[str, int]: ...
    def to_oid(self, output_csv: str) -> dict[str, int]: ...
    def load_res_oid(self, csv_path: str, class_descriptions: str | None = None) -> COCO: ...
    @classmethod
    def merge(cls, datasets: list[COCO]) -> COCO: ...
    @classmethod
    def from_yolo(cls, yolo_dir: str, images_dir: str | None = None) -> COCO: ...
    @classmethod
    def from_voc(cls, voc_dir: str) -> COCO: ...
    @classmethod
    def from_cvat(cls, cvat_path: str) -> COCO: ...
    @classmethod
    def from_dota(cls, label_dir: str, images_dir: str | None = None, categories: list[str] | None = None) -> COCO: ...
    @classmethod
    def from_oid(cls, csv_path: str, class_descriptions: str | None = None, images_dir: str | None = None) -> COCO: ...

    # --- Browse (Python-only) ---
    def browse(
        self,
        image_dir: str | None = None,
        dt: COCO | str | None = None,
        iou_type: str = "bbox",
        iou_thr: float = 0.5,
        eval: COCOeval | None = None,
        slices: dict[str, list[int]] | str | None = None,
        batch_size: int = 12,
        port: int = 7860,
    ) -> None: ...

    # --- pycocotools camelCase aliases ---
    def createIndex(self) -> None: ...
    def getAnnIds(
        self,
        imgIds: int | list[int] = ...,
        catIds: int | list[int] = ...,
        areaRng: list[float] | None = None,
        iscrowd: bool | None = None,
    ) -> list[int]: ...
    def getCatIds(
        self, catNms: str | list[str] = ..., supNms: str | list[str] = ..., catIds: int | list[int] = ...
    ) -> list[int]: ...
    def getImgIds(self, imgIds: int | list[int] = ..., catIds: int | list[int] = ...) -> list[int]: ...
    def loadAnns(self, ids: int | list[int]) -> list[dict[str, Any]]: ...
    def loadCats(self, ids: int | list[int]) -> list[dict[str, Any]]: ...
    def loadImgs(self, ids: int | list[int]) -> list[dict[str, Any]]: ...
    def loadRes(self, res: str | list[dict[str, Any]] | npt.NDArray[Any]) -> COCO: ...
    def annToRLE(self, ann: dict[str, Any]) -> dict[str, Any]: ...
    def annToMask(self, ann: dict[str, Any]) -> npt.NDArray[np.uint8]: ...
    # No camelCase converter aliases: an alias exists only where pycocotools has
    # that exact method name, and pycocotools has no converters.

# ---------------------------------------------------------------------------
# COCOeval
# ---------------------------------------------------------------------------

class COCOeval:
    """COCO evaluation — compute AP, AR, and diagnostic metrics."""

    def __init__(
        self,
        coco_gt: COCO | None = None,
        coco_dt: COCO | None = None,
        iou_type: str | None = None,
        lvis_style: bool = False,
        oid_style: bool = False,
        hierarchy: Hierarchy | None = None,
        *,
        cocoGt: COCO | None = None,
        cocoDt: COCO | None = None,
        iouType: str | None = None,
    ) -> None: ...

    # --- Core pipeline ---
    def evaluate(self) -> None: ...
    def accumulate(self) -> None: ...
    def summarize(self) -> None: ...
    def run(self) -> None: ...

    # --- Results ---
    def summary_lines(self) -> list[str]: ...
    def metric_keys(self) -> list[str]: ...
    def metric_defs(self) -> list[dict[str, Any]]: ...
    def get_results(self, prefix: str | None = None, per_class: bool = False) -> dict[str, float]: ...
    def print_results(self) -> None: ...
    def report(self) -> dict[str, Any]: ...
    def provenance(self) -> str: ...
    def is_benchmark_standard(self) -> bool: ...
    def reference_deviations(self) -> list[str]: ...
    def results(self, per_class: bool = False) -> dict[str, Any]: ...
    def save_results(self, path: str, per_class: bool = False) -> None: ...
    def f_scores(self, beta: float = 1.0) -> dict[str, float]: ...

    # --- Diagnostics ---
    def confusion_matrix(
        self, iou_thr: float = 0.5, max_det: int | None = None, min_score: float | None = None
    ) -> dict[str, Any]: ...
    def tide_errors(self, pos_thr: float = 0.5, bg_thr: float = 0.1) -> dict[str, Any]: ...
    def calibration(self, n_bins: int = 10, iou_threshold: float = 0.5) -> dict[str, Any]: ...
    def slice_by(self, slices: dict[str, list[int]] | Callable[[dict[str, Any]], str]) -> dict[str, Any]: ...
    def image_diagnostics(self, iou_thr: float = 0.5, score_thr: float = 0.5) -> dict[str, Any]: ...

    # --- Properties ---
    @property
    def params(self) -> Params:
        """The same live Params object every access; attribute *reads* on it return copies."""
        ...
    @params.setter
    def params(self, value: Params) -> None:
        """Stores a copy of ``value`` — mutate before assigning, or mutate ``ev.params`` after."""
        ...
    @property
    def stats(self) -> npt.NDArray[np.float64] | list[float]:
        """``[]`` before ``summarize()``; numpy float64 array after (pycocotools semantics)."""
        ...
    @property
    def coco_gt(self) -> COCO:
        """A fresh copy on every access — mutations never reach the evaluator."""
        ...
    @property
    def coco_dt(self) -> COCO:
        """A fresh copy on every access — mutations never reach the evaluator."""
        ...
    @property
    def eval_imgs(self) -> list[dict[str, Any] | None]: ...
    @property
    def eval(self) -> dict[str, Any] | None: ...
    @property
    def virtual_cat_names(self) -> list[str]: ...

    # --- pycocotools camelCase aliases ---
    @property
    def cocoGt(self) -> COCO: ...
    @property
    def cocoDt(self) -> COCO: ...
    @property
    def evalImgs(self) -> list[dict[str, Any] | None]: ...

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

class Params:
    """Evaluation parameters controlling IoU thresholds, area ranges, and max detections.

    Attribute reads return **copies**: ``p.max_dets.append(200)`` mutates a
    temporary and is a silent no-op. Assign whole values instead:
    ``p.max_dets = [1, 10, 100, 200]``.
    """

    def __init__(self, iou_type: str = "bbox") -> None: ...

    iou_type: str
    img_ids: list[int]
    cat_ids: list[int]
    iou_thrs: list[float]
    rec_thrs: list[float]
    max_dets: list[int]
    area_rng: list[list[float]]
    area_rng_lbl: list[str]
    use_cats: bool
    expand_dt: bool
    kpt_oks_sigmas: list[float]

    # --- pycocotools camelCase aliases ---
    iouType: str
    imgIds: list[int]
    catIds: list[int]
    iouThrs: list[float]
    recThrs: list[float]
    maxDets: list[int]
    areaRng: list[list[float]]
    areaRngLbl: list[str]
    useCats: bool
    # `expand_dt` has no alias here: it is hotcoco's Open Images extension, and
    # pycocotools has no field of that name to be compatible with.

# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

class Hierarchy:
    """Category hierarchy for OID-style evaluation."""

    @staticmethod
    def from_parent_map(parent_map: dict[int, int]) -> Hierarchy: ...
    @staticmethod
    def from_file(path: str, label_to_id: dict[str, int] | None = None) -> Hierarchy: ...
    @staticmethod
    def from_dict(tree_dict: dict[str, Any], label_to_id: dict[str, int] | None = None) -> Hierarchy: ...
    def ancestors(self, cat_id: int) -> list[int]: ...
    def children(self, cat_id: int) -> list[int]: ...
    def parent(self, cat_id: int) -> int | None: ...

# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def compare(
    eval_a: COCOeval, eval_b: COCOeval, n_bootstrap: int = 0, seed: int = 42, confidence: float = 0.95
) -> dict[str, Any]: ...
def init_as_pycocotools() -> None: ...
def init_as_lvis() -> None: ...

# ---------------------------------------------------------------------------
# LVIS drop-in surface (lvis-api spellings)
# ---------------------------------------------------------------------------

class LVISeval:
    """Drop-in for lvis-api ``LVISEval`` — constructs a ``COCOeval`` with ``lvis_style=True``."""

    def __new__(cls, gt: COCO, dt: COCO, iou_type: str = "segm") -> COCOeval: ...

# lvis-api spells it `LVISEval` (capital E); Detectron2 and MMDetection import
# that spelling. `LVIS` is lvis-api's dataset class name.
LVISEval = LVISeval
LVIS = COCO

class LVISResults:
    """Drop-in for lvis-api ``LVISResults`` — returns ``lvis_gt.load_res(results)``."""

    def __new__(
        cls, lvis_gt: COCO, results: str | list[dict[str, Any]] | npt.NDArray[Any], max_dets: int = 300
    ) -> COCO: ...

# ---------------------------------------------------------------------------
# torchvision drop-in surface
# ---------------------------------------------------------------------------

class CocoDetection:
    """Drop-in for ``torchvision.datasets.CocoDetection`` backed by hotcoco."""

    root: str
    coco: COCO
    ids: list[int]
    transform: Callable[[Any], Any] | None
    target_transform: Callable[[Any], Any] | None
    transforms: Callable[[Any, Any], tuple[Any, Any]] | None

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
        transforms: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    ) -> None: ...
    def __getitem__(self, index: int) -> tuple[Any, Any]: ...
    def __len__(self) -> int: ...

class CocoEvaluator:
    """Drop-in for the ``CocoEvaluator`` in torchvision's detection reference scripts."""

    coco_gt: COCO
    iou_types: list[str]
    coco_eval: dict[str, COCOeval | None]
    results: dict[str, list[dict[str, Any]]]

    def __init__(self, coco_gt: COCO, iou_types: str | list[str]) -> None: ...
    def update(self, predictions: dict[int, dict[str, Any]]) -> None: ...
    def synchronize_between_processes(self) -> None: ...
    def accumulate(self) -> None: ...
    def summarize(self) -> None: ...
    def get_results(self) -> dict[str, dict[str, float]]: ...
