"""Type stubs for hotcoco.panoptic — PQ, SQ, RQ."""

from __future__ import annotations

import os
from typing import Any

from . import COCO

METRIC_NAMES: list[str]

class PanopticEval:
    """Panoptic quality evaluation against panopticapi's protocol."""

    def __init__(
        self,
        gt: str | os.PathLike[str] | COCO | dict[str, Any],
        pred: str | os.PathLike[str] | COCO | dict[str, Any],
        *,
        gt_folder: str | os.PathLike[str] | None = None,
        pred_folder: str | os.PathLike[str] | None = None,
    ) -> None: ...
    def evaluate(self) -> None: ...
    def summarize(self) -> None: ...
    def summary_lines(self) -> list[str]: ...
    def run(self) -> None: ...
    @property
    def stats(self) -> list[float]: ...
    def results(self) -> dict[str, Any]: ...
    def report(self) -> dict[str, Any]: ...
    def reference_deviations(self) -> list[str]: ...
    def provenance(self) -> str: ...

def pq_compute(
    gt_json_file: str | os.PathLike[str],
    pred_json_file: str | os.PathLike[str],
    gt_folder: str | os.PathLike[str] | None = None,
    pred_folder: str | os.PathLike[str] | None = None,
) -> dict[str, Any]: ...
