"""Type stubs for the matching-kernel namespace.

The strict layer: ``bbox_iou`` takes boxes and ``mask_iou`` takes RLEs, with no
type dispatch. ``hotcoco.mask.iou`` is the lenient, pycocotools-shaped wrapper
over both. The RLE alias is imported from ``mask.pyi`` so the two stubs share one
definition.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .mask import _Rle

def lsap(
    cost: npt.NDArray[np.float64] | Sequence[Sequence[float]], maximize: bool = False
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
def bbox_iou(
    dt: npt.NDArray[np.float64] | Sequence[Sequence[float]],
    gt: npt.NDArray[np.float64] | Sequence[Sequence[float]],
    iscrowd: Sequence[bool] | npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Pairwise IoU of ``[x, y, w, h]`` boxes, shape ``(D, G)``; ``dt``/``gt`` are ``(N, 4)``."""
    ...

def mask_iou(
    dt: Sequence[_Rle], gt: Sequence[_Rle], iscrowd: Sequence[bool] | npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Pairwise IoU of RLE masks, shape ``(D, G)``."""
    ...

__all__ = ["bbox_iou", "lsap", "mask_iou"]
