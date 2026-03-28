"""Type stubs for hotcoco — a pure Rust port of pycocotools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, overload

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------

class COCO:
    """COCO dataset API — load, query, and convert annotations."""

    image_dir: str | None

    def __init__(
        self,
        annotation_file: str | dict[str, Any] | None = None,
        *,
        image_dir: str | None = None,
    ) -> None: ...

    # --- Query ---
    def get_ann_ids(
        self,
        img_ids: list[int] = ...,
        cat_ids: list[int] = ...,
        area_rng: list[float] | None = None,
        iscrowd: bool | None = None,
    ) -> list[int]: ...
    def get_cat_ids(
        self,
        cat_nms: list[str] = ...,
        sup_nms: list[str] = ...,
        cat_ids: list[int] = ...,
    ) -> list[int]: ...
    def get_img_ids(
        self,
        img_ids: list[int] = ...,
        cat_ids: list[int] = ...,
    ) -> list[int]: ...

    # --- Load ---
    def load_anns(self, ids: list[int]) -> list[dict[str, Any]]: ...
    def load_cats(self, ids: list[int]) -> list[dict[str, Any]]: ...
    def load_imgs(self, ids: list[int]) -> list[dict[str, Any]]: ...
    def load_res(self, res: str | list[dict[str, Any]] | npt.NDArray[Any]) -> COCO: ...

    # --- Masks ---
    def ann_to_rle(self, ann: dict[str, Any]) -> dict[str, Any]: ...
    def ann_to_mask(self, ann: dict[str, Any]) -> npt.NDArray[np.uint8]: ...

    # --- Properties ---
    @property
    def dataset(self) -> dict[str, Any]: ...
    @property
    def imgs(self) -> dict[int, dict[str, Any]]: ...
    @property
    def anns(self) -> dict[int, dict[str, Any]]: ...
    @property
    def cats(self) -> dict[int, dict[str, Any]]: ...

    # --- Utilities ---
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
        self,
        val_frac: float = 0.2,
        test_frac: float | None = None,
        seed: int = 42,
    ) -> tuple[COCO, COCO] | tuple[COCO, COCO, COCO]: ...
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        seed: int = 42,
    ) -> COCO: ...

    # --- Conversion ---
    def to_yolo(self, output_dir: str) -> dict[str, int]: ...
    def to_voc(self, output_dir: str) -> dict[str, int]: ...
    def to_cvat(self, output_path: str) -> dict[str, int]: ...
    @staticmethod
    def merge(datasets: list[COCO]) -> COCO: ...
    @staticmethod
    def from_yolo(yolo_dir: str, images_dir: str | None = None) -> COCO: ...
    @staticmethod
    def from_voc(voc_dir: str) -> COCO: ...
    @staticmethod
    def from_cvat(cvat_path: str) -> COCO: ...

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
    def getAnnIds(
        self,
        imgIds: list[int] = ...,
        catIds: list[int] = ...,
        areaRng: list[float] | None = None,
        iscrowd: bool | None = None,
    ) -> list[int]: ...
    def getCatIds(
        self,
        catNms: list[str] = ...,
        supNms: list[str] = ...,
        catIds: list[int] = ...,
    ) -> list[int]: ...
    def getImgIds(
        self,
        imgIds: list[int] = ...,
        catIds: list[int] = ...,
    ) -> list[int]: ...
    def loadAnns(self, ids: list[int]) -> list[dict[str, Any]]: ...
    def loadCats(self, ids: list[int]) -> list[dict[str, Any]]: ...
    def loadImgs(self, ids: list[int]) -> list[dict[str, Any]]: ...
    def loadRes(self, res: str | list[dict[str, Any]] | npt.NDArray[Any]) -> COCO: ...
    def annToRLE(self, ann: dict[str, Any]) -> dict[str, Any]: ...
    def annToMask(self, ann: dict[str, Any]) -> npt.NDArray[np.uint8]: ...
    def toYolo(self, output_dir: str) -> dict[str, int]: ...
    def toVoc(self, output_dir: str) -> dict[str, int]: ...
    def toCvat(self, output_path: str) -> dict[str, int]: ...
    @staticmethod
    def fromYolo(yolo_dir: str, images_dir: str | None = None) -> COCO: ...
    @staticmethod
    def fromVoc(voc_dir: str) -> COCO: ...
    @staticmethod
    def fromCvat(cvat_path: str) -> COCO: ...

# ---------------------------------------------------------------------------
# COCOeval
# ---------------------------------------------------------------------------

class COCOeval:
    """COCO evaluation — compute AP, AR, and diagnostic metrics."""

    def __init__(
        self,
        coco_gt: COCO,
        coco_dt: COCO,
        iou_type: str,
        lvis_style: bool = False,
        oid_style: bool = False,
        hierarchy: Hierarchy | None = None,
    ) -> None: ...

    # --- Core pipeline ---
    def evaluate(self) -> None: ...
    def accumulate(self) -> None: ...
    def summarize(self) -> None: ...
    def run(self) -> None: ...

    # --- Results ---
    def summary_lines(self) -> list[str]: ...
    def metric_keys(self) -> list[str]: ...
    def get_results(
        self, prefix: str | None = None, per_class: bool = False
    ) -> dict[str, float]: ...
    def print_results(self) -> None: ...
    def results(self, per_class: bool = False) -> dict[str, Any]: ...
    def save_results(self, path: str, per_class: bool = False) -> None: ...
    def f_scores(self, beta: float = 1.0) -> dict[str, float]: ...

    # --- Diagnostics ---
    def confusion_matrix(
        self,
        iou_thr: float = 0.5,
        max_det: int | None = None,
        min_score: float | None = None,
    ) -> dict[str, Any]: ...
    def tide_errors(
        self, pos_thr: float = 0.5, bg_thr: float = 0.1
    ) -> dict[str, Any]: ...
    def calibration(
        self, n_bins: int = 10, iou_threshold: float = 0.5
    ) -> dict[str, Any]: ...
    def slice_by(
        self, slices: dict[str, list[int]] | Callable[[dict[str, Any]], str]
    ) -> dict[str, Any]: ...
    def image_diagnostics(
        self, iou_thr: float = 0.5, score_thr: float = 0.5
    ) -> dict[str, Any]: ...

    # --- Properties ---
    @property
    def params(self) -> Params: ...
    @params.setter
    def params(self, value: Params) -> None: ...
    @property
    def stats(self) -> list[float] | None: ...
    @property
    def coco_gt(self) -> COCO: ...
    @property
    def coco_dt(self) -> COCO: ...
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
    """Evaluation parameters controlling IoU thresholds, area ranges, etc."""

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
    expandDt: bool

# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

class Hierarchy:
    """Category hierarchy for OID-style evaluation."""

    @staticmethod
    def from_parent_map(parent_map: dict[int, int]) -> Hierarchy: ...
    @staticmethod
    def from_file(
        path: str, label_to_id: dict[str, int] | None = None
    ) -> Hierarchy: ...
    @staticmethod
    def from_dict(
        tree_dict: dict[str, Any], label_to_id: dict[str, int] | None = None
    ) -> Hierarchy: ...
    def ancestors(self, cat_id: int) -> list[int]: ...
    def children(self, cat_id: int) -> list[int]: ...
    def parent(self, cat_id: int) -> int | None: ...

# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def compare(
    eval_a: COCOeval,
    eval_b: COCOeval,
    n_bootstrap: int = 0,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]: ...
def init_as_pycocotools() -> None: ...
def init_as_lvis() -> None: ...

# ---------------------------------------------------------------------------
# mask submodule
# ---------------------------------------------------------------------------

class mask:
    """RLE mask operations matching the pycocotools.mask API."""

    @staticmethod
    @overload
    def encode(mask: npt.NDArray[np.uint8]) -> dict[str, Any]: ...
    @staticmethod
    @overload
    def encode(mask: npt.NDArray[np.uint8]) -> list[dict[str, Any]]: ...
    @staticmethod
    def encode(mask: npt.NDArray[np.uint8]) -> dict[str, Any] | list[dict[str, Any]]:
        """Encode a binary mask to RLE. 2D → single dict, 3D → list."""
        ...
    @staticmethod
    @overload
    def decode(rle: dict[str, Any]) -> npt.NDArray[np.uint8]: ...
    @staticmethod
    @overload
    def decode(rle: list[dict[str, Any]]) -> npt.NDArray[np.uint8]: ...
    @staticmethod
    def decode(
        rle: dict[str, Any] | list[dict[str, Any]],
    ) -> npt.NDArray[np.uint8]:
        """Decode RLE to binary mask. Single dict → 2D, list → 3D."""
        ...
    @staticmethod
    @overload
    def area(rle: dict[str, Any]) -> int: ...
    @staticmethod
    @overload
    def area(rle: list[dict[str, Any]]) -> npt.NDArray[np.uint64]: ...
    @staticmethod
    def area(
        rle: dict[str, Any] | list[dict[str, Any]],
    ) -> int | npt.NDArray[np.uint64]:
        """Compute mask area. Single dict → int, list → array."""
        ...
    @staticmethod
    def to_bbox(
        rle: dict[str, Any] | list[dict[str, Any]],
    ) -> npt.NDArray[np.float64]:
        """Convert RLE to bounding box [x, y, w, h]."""
        ...
    @staticmethod
    def toBbox(
        rle: dict[str, Any] | list[dict[str, Any]],
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def merge(
        rles: dict[str, Any] | list[dict[str, Any]], intersect: bool = False
    ) -> dict[str, Any]:
        """Merge RLE masks via union (default) or intersection."""
        ...
    @staticmethod
    def iou(
        dt: dict[str, Any] | list[dict[str, Any]],
        gt: dict[str, Any] | list[dict[str, Any]],
        iscrowd: list[bool],
    ) -> npt.NDArray[np.float64]:
        """Compute IoU between dt and gt RLE masks. Shape: (D, G)."""
        ...
    @staticmethod
    def bbox_iou(
        dt: list[list[float]],
        gt: list[list[float]],
        iscrowd: list[bool],
    ) -> npt.NDArray[np.float64]:
        """Compute IoU between dt and gt bounding boxes. Shape: (D, G)."""
        ...
    @staticmethod
    def fr_poly(xy: list[float], h: int, w: int) -> dict[str, Any]:
        """Convert polygon to RLE."""
        ...
    @staticmethod
    def frPoly(xy: list[float], h: int, w: int) -> dict[str, Any]: ...
    @staticmethod
    def fr_bbox(bb: list[float], h: int, w: int) -> dict[str, Any]:
        """Convert bounding box to RLE."""
        ...
    @staticmethod
    def frBbox(bb: list[float], h: int, w: int) -> dict[str, Any]: ...
    @staticmethod
    def frPyObjects(
        seg: list[list[float]] | dict[str, Any] | list[dict[str, Any]],
        h: int,
        w: int,
    ) -> list[dict[str, Any]]:
        """Convert segmentation objects to RLE (pycocotools compat)."""
        ...
    @staticmethod
    def fr_py_objects(
        seg: list[list[float]] | dict[str, Any] | list[dict[str, Any]],
        h: int,
        w: int,
    ) -> list[dict[str, Any]]: ...
    @staticmethod
    def fr_py_objects_snake(
        seg: list[list[float]] | dict[str, Any] | list[dict[str, Any]],
        h: int,
        w: int,
    ) -> list[dict[str, Any]]: ...
    @staticmethod
    def rle_to_string(rle: dict[str, Any]) -> str:
        """Encode RLE to compact string format."""
        ...
    @staticmethod
    def rle_from_string(s: str, h: int, w: int) -> dict[str, Any]:
        """Decode compact string to RLE."""
        ...
