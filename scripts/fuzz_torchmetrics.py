"""Differential fuzzer: torchmetrics' MeanAveragePrecision on hotcoco vs pycocotools.

RF-DETR evaluates detection and segmentation through torchmetrics this way: it
builds the metric under a backend name torchmetrics knows, then replaces the
backend's COCO, COCOeval and mask modules with hotcoco's. Everything torchmetrics
does on top runs unchanged: `mask.encode` of every predicted mask in `update()`,
`float32` threshold grids from `torch.linspace`, `mask.area` per annotation,
`COCO()` + `dataset =` + `createIndex()`, `params` field writes, and one
`COCOeval` per class with `class_metrics`.

Each case is a random batch of states shaped like a detector's output: `xyxy`
boxes, `bool` masks, crowd regions, an optional `area`, scores with ties, sparse
class ids, empty images, boxes on the 32**2 and 96**2 area edges, whole-pixel
boxes whose IoU lands exactly on a threshold, and duplicate detections. Every
output tensor must equal the pycocotools backend's exactly. faster-coco-eval
runs alongside for reference and does not fail the run.

One adjustment to the reference: pycocotools reports the headline AP
(`stats[0]`, torchmetrics' `map`) at `maxDets=100` whatever the configured caps
are, and -1 when 100 is not one of them. hotcoco and faster-coco-eval read it at
the largest cap, and RF-DETR, which evaluates at `[1, 10, 500]`, makes its
pycocotools-exact backend do the same. The reference here does too.

bbox+segm in one metric is not fuzzed: torchmetrics switches each prediction's
`area` between the two IoU types by editing `coco_preds.dataset` in place, and
hotcoco's `dataset` is a copy, so the edit never reaches it. RF-DETR evaluates
each IoU type on its own prediction dataset for that reason.

Usage:
    uv run --group torchmetrics python scripts/fuzz_torchmetrics.py [--cases N] [--seed S]
    just fuzz-torchmetrics
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import sys
import time
import warnings

import hotcoco
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval as PyCOCOeval
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import CocoBackend


class PyCOCOevalAtLargestCap(PyCOCOeval):
    """pycocotools, with the headline AP read at the largest cap; see the module docstring."""

    def summarize(self) -> None:
        super().summarize()
        precision = self.eval["precision"][:, :, :, self.params.areaRngLbl.index("all"), -1]
        valid = precision[precision > -1]
        self.stats[0] = np.mean(valid) if valid.size else -1


class PycocotoolsBackend(CocoBackend):
    def __init__(self) -> None:
        super().__init__("pycocotools")

    @property
    def cocoeval(self) -> object:
        return PyCOCOevalAtLargestCap


class HotcocoBackend(CocoBackend):
    """torchmetrics' COCO backend with hotcoco's modules, as RF-DETR swaps them in."""

    # How many evaluators torchmetrics asked this backend for: zero at the end
    # means the swap never reached the evaluation, and the run compared nothing.
    evaluators_built = 0

    def __init__(self) -> None:
        super().__init__("faster_coco_eval")

    @property
    def coco(self) -> object:
        return hotcoco.COCO

    @property
    def cocoeval(self) -> object:
        HotcocoBackend.evaluators_built += 1
        return hotcoco.COCOeval

    @property
    def mask_utils(self) -> object:
        return hotcoco.mask


# Backends built under a name torchmetrics knows, then replaced.
SWAPPED = {"hotcoco": HotcocoBackend, "pycocotools": PycocotoolsBackend}


def rect_mask(h: int, w: int, box: list[float]) -> torch.Tensor:
    m = torch.zeros(h, w, dtype=torch.bool)
    x0, y0, x1, y1 = (round(v) for v in box)
    m[max(0, y0) : max(0, min(h, y1)), max(0, x0) : max(0, min(w, x1))] = True
    return m


def blob_mask(rng: np.random.Generator, h: int, w: int) -> torch.Tensor:
    if rng.random() < 0.3:
        return torch.zeros(h, w, dtype=torch.bool)
    return torch.from_numpy(rng.random((h, w)) < rng.uniform(0.0, 0.6))


def gen_case(rng: np.random.Generator) -> tuple[dict, list[tuple[dict, dict]]]:
    """One metric configuration and the per-image (prediction, target) pairs."""
    n_images = int(rng.integers(1, 7))
    if rng.random() < 0.5:
        label_pool = list(range(int(rng.integers(1, 5))))
    else:
        label_pool = sorted(rng.choice(np.arange(120), size=int(rng.integers(1, 6)), replace=False).tolist())
    with_masks = rng.random() < 0.5
    give_area = rng.random() < 0.25
    bool_crowd = rng.random() < 0.2
    score_mode = rng.choice(["uniform", "ties", "allsame"])
    geom = rng.choice(["float", "integer", "edge"])
    many = rng.random() < 0.15
    big = rng.random() < 0.3
    images = []
    for _ in range(n_images):
        if geom == "edge":
            h, w = int(rng.integers(100, 140)), int(rng.integers(100, 140))
        elif big:
            h, w = int(rng.integers(60, 180)), int(rng.integers(60, 180))
        else:
            h, w = int(rng.integers(3, 40)), int(rng.integers(3, 40))

        def box() -> list[float]:
            if geom == "edge":  # area 32**2 or 96**2, give or take one pixel
                side = int(rng.choice([32, 96]))
                bw = side + int(rng.integers(-1, 2))
                bh = (side * side + int(rng.integers(-1, 2))) / bw
                x0, y0 = float(rng.integers(0, max(1, w - bw))), float(rng.integers(0, max(1, h - int(bh))))
                return [x0, y0, x0 + bw, y0 + bh]
            if geom == "integer":
                x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
                return [x0, y0, x0 + int(rng.integers(0, w + 1)), y0 + int(rng.integers(0, h + 1))]
            x0, y0 = rng.uniform(-2, w), rng.uniform(-2, h)
            if rng.random() < 0.08:
                return [x0, y0, x0, y0 + rng.uniform(0, 5)]  # zero width
            return [x0, y0, x0 + rng.uniform(0.5, w), y0 + rng.uniform(0.5, h)]

        n_gt = int(rng.integers(0, 6))
        gt_boxes = [box() for _ in range(n_gt)]
        gt_labels = [int(rng.choice(label_pool)) for _ in range(n_gt)]
        gt_masks = [rect_mask(h, w, b) if rng.random() < 0.7 else blob_mask(rng, h, w) for b in gt_boxes]
        dt_boxes, dt_labels, dt_masks = [], [], []
        for _ in range(int(rng.integers(0, 40 if many else 14))):
            if dt_boxes and rng.random() < 0.08:  # exact duplicate
                k = int(rng.integers(len(dt_boxes)))
                dt_boxes.append(list(dt_boxes[k]))
                dt_labels.append(dt_labels[k])
                dt_masks.append(dt_masks[k].clone())
                continue
            if gt_boxes and rng.random() < 0.65:  # near a ground truth
                j = int(rng.integers(len(gt_boxes)))
                jit = rng.normal(0, rng.choice([0.5, 2.0, 6.0]), 4)
                if geom != "float":
                    jit = np.round(jit)
                b = [gt_boxes[j][k] + jit[k] for k in range(4)]
                dt_boxes.append([b[0], b[1], max(b[0], b[2]), max(b[1], b[3])])
                dt_labels.append(gt_labels[j] if rng.random() < 0.8 else int(rng.choice(label_pool)))
                shift = (int(rng.integers(-3, 4)), int(rng.integers(-3, 4)))
                dt_masks.append(torch.roll(gt_masks[j], shift, (0, 1)))
            else:
                b = box()
                dt_boxes.append(b)
                dt_labels.append(int(rng.choice(label_pool)))
                dt_masks.append(rect_mask(h, w, b) if rng.random() < 0.6 else blob_mask(rng, h, w))
        n_dt = len(dt_boxes)
        if score_mode == "uniform":
            scores = rng.random(n_dt)
        elif score_mode == "ties":
            scores = rng.integers(1, 4, n_dt) / 4
        else:
            scores = np.full(n_dt, 0.5)
        pred = {
            "boxes": torch.tensor(dt_boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(dt_labels, dtype=torch.int64),
        }
        target = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(gt_labels, dtype=torch.int64),
            "iscrowd": torch.tensor(
                [int(rng.random() < 0.15) for _ in range(n_gt)], dtype=torch.bool if bool_crowd else torch.int64
            ),
        }
        if give_area:
            target["area"] = torch.tensor(rng.uniform(0, 4000, n_gt) * (rng.random(n_gt) < 0.8), dtype=torch.float32)
        if with_masks:
            pred["masks"] = torch.stack(dt_masks) if dt_masks else torch.zeros(0, h, w, dtype=torch.bool)
            target["masks"] = torch.stack(gt_masks) if gt_masks else torch.zeros(0, h, w, dtype=torch.bool)
        images.append((pred, target))
    config = {
        "iou_type": "segm" if with_masks else "bbox",
        "max_detection_thresholds": [[1, 10, 100], [1, 10, 500], [1, 2, 3], [1, 5, 10]][int(rng.integers(4))],
        "class_metrics": bool(rng.random() < 0.5),
    }
    return config, images


def compute(backend: str, config: dict, images: list[tuple[dict, dict]]) -> dict[str, torch.Tensor] | Exception:
    try:
        metric = MeanAveragePrecision(backend="faster_coco_eval", **config)
        if backend in SWAPPED:
            metric._coco_backend = SWAPPED[backend]()
        metric.warn_on_many_detections = False
        # One update() per image, as a validation loop with batch size 1 makes.
        for pred, target in images:
            metric.update([copy.deepcopy(pred)], [copy.deepcopy(target)])
        with contextlib.redirect_stdout(io.StringIO()):
            return {k: v.clone() for k, v in metric.compute().items()}
    except Exception as error:  # an error on one backend only is a finding too
        return error


def differences(ref: dict, got: dict) -> list[tuple[str, object, object]]:
    out = []
    for key in sorted(set(ref) | set(got)):
        a, b = ref.get(key), got.get(key)
        if a is None or b is None or a.shape != b.shape or not torch.equal(a, b):
            out.append((key, None if a is None else a.tolist(), None if b is None else b.tolist()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuzz torchmetrics' MeanAveragePrecision on hotcoco against pycocotools."
    )
    parser.add_argument("--cases", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0, help="first seed; case i uses seed + i")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    # The swap writes this attribute; under another name it would add an unused
    # one, and every case would compare faster-coco-eval with itself.
    if not hasattr(MeanAveragePrecision(), "_coco_backend"):
        print("torchmetrics no longer keeps its backend in `_coco_backend`; the swap would do nothing")
        return 1

    failures = {"hotcoco": [], "faster_coco_eval": []}
    compared = 0
    started = time.perf_counter()
    for i in range(args.cases):
        seed = args.seed + i
        config, images = gen_case(np.random.default_rng(seed))
        ref = compute("pycocotools", config, images)
        compared += not isinstance(ref, Exception)
        for backend in failures:
            got = compute(backend, config, images)
            if isinstance(ref, Exception) or isinstance(got, Exception):
                if type(ref) is not type(got):
                    failures[backend].append((seed, config, [("error", repr(ref), repr(got))]))
                continue
            diff = differences(ref, got)
            if diff:
                failures[backend].append((seed, config, diff))

    print(
        f"{args.cases} cases, seeds {args.seed}..{args.seed + args.cases - 1}, "
        f"{compared} with metrics to compare, {time.perf_counter() - started:.0f} s"
    )
    for backend, found in failures.items():
        role = "must match" if backend == "hotcoco" else "reference only"
        print(f"{backend} vs pycocotools ({role}): {len(found)} differing cases")
        for seed, config, diff in found[:5]:
            print(f"  seed {seed} {config}")
            for key, ref, got in diff[:4]:
                print(f"    {key}: pycocotools={ref} {backend}={got}")
    # A run where nearly every case raised on both sides, or where hotcoco never
    # evaluated anything, agrees with the reference without checking it.
    if compared < args.cases / 2 or HotcocoBackend.evaluators_built == 0:
        print(
            f"only {compared} of {args.cases} cases produced metrics, and hotcoco built "
            f"{HotcocoBackend.evaluators_built} evaluators: nothing was compared"
        )
        return 1
    return 1 if failures["hotcoco"] else 0


if __name__ == "__main__":
    sys.exit(main())
