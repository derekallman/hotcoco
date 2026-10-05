"""Type stubs for ``hotcoco.mask`` — the ``pycocotools.mask``-compatible RLE surface.

An RLE dict is ``{"size": [h, w], "counts": bytes}``, exactly what
``pycocotools.mask.encode`` returns. On input the functions also accept
``{"size": [h, w], "counts": str}`` (compressed string) and
``{"size": [h, w], "counts": [int, ...]}`` (uncompressed counts); output is
always the pycocotools ``bytes`` format.
"""

from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import numpy.typing as npt

_Rle = dict[str, Any]

#: ``[x, y, w, h]`` boxes: an ``(N, 4)`` array or a sequence of 4-element rows.
_Boxes = npt.NDArray[Any] | Sequence[Sequence[float]] | Sequence[npt.NDArray[Any]]

_MaskDtype = np.dtype[np.uint8] | np.dtype[np.bool_]

@overload
def encode(mask: np.ndarray[tuple[int, int], _MaskDtype]) -> _Rle:
    """Encode a 2-D ``(H, W)`` binary mask to a single RLE dict."""
    ...

@overload
def encode(mask: np.ndarray[tuple[int, int, int], _MaskDtype]) -> list[_Rle]:
    """Encode a 3-D ``(H, W, N)`` mask stack to a list of N RLE dicts."""
    ...

@overload
def decode(rle: _Rle) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    """Decode a single RLE dict to a 2-D ``(H, W)`` Fortran-order mask."""
    ...

@overload
def decode(rle: list[_Rle]) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
    """Decode a list of N RLE dicts to a 3-D ``(H, W, N)`` Fortran-order stack."""
    ...

@overload
def area(rle: _Rle) -> int: ...
@overload
def area(rle: list[_Rle]) -> npt.NDArray[np.uint32]: ...
def to_bbox(rle: _Rle | list[_Rle]) -> npt.NDArray[np.float64]:
    """Bounding box(es) ``[x, y, w, h]``: shape ``(4,)`` for a dict, ``(N, 4)`` for a list."""
    ...

def toBbox(rle: _Rle | list[_Rle]) -> npt.NDArray[np.float64]: ...
def merge(rles: _Rle | list[_Rle], intersect: int | bool = False) -> _Rle:
    """Merge RLE masks via union (default) or intersection."""
    ...

@overload
def iou(
    dt: _Rle | Sequence[_Rle], gt: _Rle | Sequence[_Rle], iscrowd: Sequence[bool | int] | npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    """IoU between dt and gt RLE masks, shape ``(D, G)``. Crowd GTs use IoA."""
    ...

@overload
def iou(dt: _Boxes, gt: _Boxes, iscrowd: Sequence[bool | int] | npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """IoU between dt and gt ``[x, y, w, h]`` boxes, shape ``(D, G)`` — as ``pycocotools.mask.iou``
    dispatches. An array of any numeric dtype is boxes; dt and gt must be the same kind."""
    ...

def bbox_iou(dt: _Boxes, gt: _Boxes, iscrowd: Sequence[bool | int] | npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """IoU between dt and gt ``[x, y, w, h]`` boxes, shape ``(D, G)``. The same object as
    ``hotcoco.primitives.bbox_iou``."""
    ...

def fr_poly(xy: Sequence[float], h: int, w: int) -> _Rle:
    """Rasterize a flattened polygon ``[x1, y1, x2, y2, ...]`` to RLE."""
    ...

def frPoly(xy: Sequence[float], h: int, w: int) -> _Rle: ...
def fr_bbox(bb: Sequence[float], h: int, w: int) -> _Rle:
    """Rasterize a ``[x, y, w, h]`` box to RLE."""
    ...

def frBbox(bb: Sequence[float], h: int, w: int) -> _Rle: ...
@overload
def frPyObjects(seg: _Rle, h: int, w: int) -> _Rle:
    """A single RLE dict in gives a single RLE dict out — as pycocotools does."""
    ...

@overload
def frPyObjects(seg: Sequence[Sequence[float]] | list[_Rle] | npt.NDArray[Any], h: int, w: int) -> list[_Rle]:
    """Boxes/polygons/RLE dicts in a list give a list of RLE dicts."""
    ...

@overload
def fr_py_objects(seg: _Rle, h: int, w: int) -> _Rle: ...
@overload
def fr_py_objects(seg: Sequence[Sequence[float]] | list[_Rle] | npt.NDArray[Any], h: int, w: int) -> list[_Rle]: ...
def rle_to_string(rle: _Rle) -> str:
    """Encode RLE to the compact LEB128 string format."""
    ...

def rle_from_string(s: str, h: int, w: int) -> _Rle:
    """Decode a compact string to RLE."""
    ...
