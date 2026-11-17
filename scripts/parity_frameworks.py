#!/usr/bin/env python3
"""Parity check: Detectron2's and MMDetection's own evaluators, on hotcoco
behind `init_as_pycocotools()` / `init_as_lvis()` versus on pycocotools and
lvis-api.

Usage:
    just parity-frameworks            # builds .venv-frameworks on first run
    just parity-frameworks d2-bbox    # one case

Each case runs twice in a fresh process, once per backend, on the first 500
val2017 images, and the two result dicts must be equal: Detectron2's to six
decimals, MMDetection's as it rounds them (three). Exits non-zero if any case
differs, fails to run, or ran on the wrong backend. Needs `data/`
(`just download-coco`).

`tests/test_framework_shims.py` is the CI gate for the same surface: it repeats
the calls these evaluators make, read from their source. This script is what
tells you that list is complete. Its first run found three calls the list had
missed.

What it does not cover
----------------------
- Detectron2 `COCOEvaluator(use_fast_impl=True)`, the default: its C++ kernel
  reads pycocotools' internals. Every case here passes `use_fast_impl=False`.
- MMDetection `CocoMetric(sort_categories=True)` and `use_mp_eval=True`, and
  `LVISFixedAPMetric`.
- MMDetection `LVISMetric(metric="proposal")`, a known difference: MMDetection
  sets `params.useCats` and `params.maxDets`, pycocotools' names, which
  lvis-api ignores and hotcoco applies. See docs/guide/frameworks.md.

Two shims the environment needs, neither of which touches the code under test:
MMDetection imports `mmcv`'s compiled ops at import time and its metrics never
call them, so `mmcv-lite` is installed and `mmcv._ext` is a stub that raises if
called; and lvis-api 0.5.3 uses `np.float`, removed in numpy 1.24, so the
reference run restores the alias.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
N_IMAGES = 500
BACKENDS = ("reference", "hotcoco")
# Each case takes seconds; one that takes minutes has gone quadratic.
CASE_TIMEOUT_S = 300

# case -> (framework, dataset kind, evaluator options)
CASES: dict[str, tuple[str, str, dict]] = {
    "d2-bbox": ("d2", "bbox", {}),
    "d2-bbox-max-dets": ("d2", "bbox", {"max_dets_per_image": 250}),
    "d2-segm": ("d2", "segm", {}),
    "d2-keypoints": ("d2", "keypoints", {}),
    "d2-lvis": ("d2", "lvis", {}),
    # The dataset loaders, on all 5,000 images: they index `imgToAnns` once per
    # image, which took minutes when each read rebuilt the map.
    "d2-load-coco": ("d2", "load-coco", {}),
    "d2-load-lvis": ("d2", "load-lvis", {}),
    "mmdet-bbox": ("mmdet", "bbox", {}),
    "mmdet-bbox-classwise": ("mmdet", "bbox", {"classwise": True}),
    "mmdet-segm": ("mmdet", "segm", {}),
    "mmdet-proposal": ("mmdet", "proposal", {}),
    "mmdet-lvis": ("mmdet", "lvis", {}),
    "mmdet-lvis-classwise": ("mmdet", "lvis", {"classwise": True}),
}


# ---------------------------------------------------------------------------
# Inputs, shared by both frameworks
# ---------------------------------------------------------------------------


def load_inputs(kind: str, workdir: str) -> dict:
    """An annotation file of the first `N_IMAGES` val2017 images, and their
    detections grouped by image. `lvis` makes the file LVIS-shaped."""
    keypoints = kind == "keypoints"
    ann_file = DATA / "annotations" / ("person_keypoints_val2017.json" if keypoints else "instances_val2017.json")
    lvis = kind in ("lvis", "load-lvis")
    results = DATA / {"bbox": "bbox", "proposal": "bbox", "keypoints": "kpt"}.get(kind, "segm")
    results = results.with_name(results.name + "_val2017_results.json")
    gt = json.loads(ann_file.read_text())
    images = sorted(gt["images"], key=lambda im: im["id"])[:N_IMAGES]
    keep = {im["id"] for im in images}
    cats = sorted(gt["categories"], key=lambda c: c["id"])
    by_image: dict[int, list[dict]] = {}
    for det in json.loads(results.read_text()):
        if det["image_id"] in keep:
            by_image.setdefault(det["image_id"], []).append(det)
    # A file holding only those images: MMDetection scores every image in the
    # annotation file, and 4,500 with no detections would flatten the numbers.
    if lvis:
        subset = lvis_shaped(gt, images, cats, keep)
    else:
        subset = {**gt, "images": images, "annotations": [a for a in gt["annotations"] if a["image_id"] in keep]}
    ann_file = Path(workdir) / "annotations.json"
    ann_file.write_text(json.dumps(subset))
    return {
        "ann_file": str(ann_file),
        "images": images,
        "cats": cats,
        "label": {c["id"]: k for k, c in enumerate(cats)},
        "by_image": by_image,
    }


def lvis_shaped(gt: dict, images: list[dict], cats: list[dict], keep: set[int]) -> dict:
    """val2017 as an LVIS file: no crowds, a frequency per category, and some
    negative and non-exhaustive categories per image, so the federated rules
    have something to do."""
    anns = [a for a in gt["annotations"] if a["image_id"] in keep and not a["iscrowd"]]
    present: dict[int, set[int]] = {}
    images_of: dict[int, set[int]] = {}
    for a in anns:
        a.pop("iscrowd")
        present.setdefault(a["image_id"], set()).add(a["category_id"])
        images_of.setdefault(a["category_id"], set()).add(a["image_id"])
    for c in cats:
        n = len(images_of.get(c["id"], ()))
        c.update(
            image_count=n,
            instance_count=sum(a["category_id"] == c["id"] for a in anns),
            frequency="r" if n <= 3 else "c" if n <= 20 else "f",
            synonyms=[c["name"]],
            synset=c["name"],
        )
        c["def"] = ""
    cat_ids = [c["id"] for c in cats]
    for k, im in enumerate(images):
        here = present.get(im["id"], set())
        im["neg_category_ids"] = [c for c in cat_ids if c not in here][k % 7 :: 9]
        im["not_exhaustive_category_ids"] = sorted(here)[: k % 3]
    return {"info": {}, "licenses": [], "images": images, "annotations": anns, "categories": cats}


def predictions(inputs: dict, kind: str):
    """Per image: `(image, xyxy boxes, scores, labels, masks or keypoints)` as
    torch tensors, the shape both frameworks take from a model."""
    import numpy as np
    import pycocotools.mask as mask_utils
    import torch

    for im in inputs["images"]:
        dets = inputs["by_image"].get(im["id"], [])
        h, w = im["height"], im["width"]
        extra = None
        if kind in ("segm", "lvis"):
            rles = [
                d["segmentation"]
                if isinstance(d["segmentation"], dict)
                else mask_utils.merge(mask_utils.frPyObjects(d["segmentation"], h, w))
                for d in dets
            ]
            boxes = np.array([mask_utils.toBbox(r) for r in rles], dtype=np.float32).reshape(-1, 4)
            masks = np.stack([mask_utils.decode(r) for r in rles]) if rles else np.zeros((0, h, w))
            extra = torch.from_numpy(masks.astype(bool))
        elif kind == "keypoints":
            kps = np.array([d["keypoints"] for d in dets], dtype=np.float32).reshape(-1, 17, 3)
            lo, hi = kps[:, :, :2].min(1), kps[:, :, :2].max(1)
            boxes = np.concatenate([lo, hi - lo], 1) if len(dets) else np.zeros((0, 4), np.float32)
            extra = torch.from_numpy(kps)
        else:
            boxes = np.array([d["bbox"] for d in dets], dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        yield (
            im,
            torch.from_numpy(boxes),
            torch.tensor([d["score"] for d in dets], dtype=torch.float32),
            torch.tensor([inputs["label"][d["category_id"]] for d in dets], dtype=torch.long),
            extra,
        )


# ---------------------------------------------------------------------------
# The frameworks' evaluators
# ---------------------------------------------------------------------------


def load_with_detectron2(kind: str, inputs: dict) -> dict:
    """`load_coco_json` on val2017 as it ships, or `load_lvis_json` on the
    LVIS-shaped file: the record count and a digest of every record."""
    import hashlib

    from detectron2.data.datasets import load_coco_json, load_lvis_json

    if kind == "load-coco":
        records = load_coco_json(str(DATA / "annotations" / "instances_val2017.json"), "/images")
    else:
        records = load_lvis_json(inputs["ann_file"], "/images")

    def plain(value):
        """Records by value. hotcoco holds a box edge as a float, so one the
        file writes as `416` comes back `416.0`; Detectron2 makes a float
        tensor of either."""
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, bytes):  # compressed RLE counts
            return value.decode()
        return value if isinstance(value, (str, bool, type(None))) else float(value)

    text = json.dumps(plain(records), sort_keys=True)
    return {
        "records": len(records),
        "annotations": sum(len(r["annotations"]) for r in records),
        "digest": hashlib.sha256(text.encode()).hexdigest(),
    }


def run_detectron2(kind: str, options: dict, inputs: dict, workdir: str) -> dict:
    if kind.startswith("load-"):
        return load_with_detectron2(kind, inputs)
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.evaluation import COCOEvaluator, LVISEvaluator
    from detectron2.structures import Boxes, Instances

    DatasetCatalog.register("parity", list)
    meta = MetadataCatalog.get("parity")
    meta.set(
        json_file=inputs["ann_file"],
        image_root="/nonexistent",
        thing_classes=[c["name"] for c in inputs["cats"]],
        thing_dataset_id_to_contiguous_id=inputs["label"],
    )
    if kind == "lvis":
        evaluator = LVISEvaluator("parity", tasks=("bbox", "segm"), output_dir=workdir)
    else:
        if kind == "keypoints":
            meta.set(keypoint_names=inputs["cats"][0]["keypoints"])
        tasks = {"bbox": ("bbox",), "segm": ("bbox", "segm"), "keypoints": ("keypoints",)}[kind]
        evaluator = COCOEvaluator("parity", tasks=tasks, output_dir=workdir, use_fast_impl=False, **options)
    evaluator.reset()
    for im, boxes, scores, labels, extra in predictions(inputs, kind):
        inst = Instances((im["height"], im["width"]))
        inst.pred_boxes, inst.scores, inst.pred_classes = Boxes(boxes), scores, labels
        if kind == "keypoints":
            inst.pred_keypoints = extra
        elif extra is not None:
            inst.pred_masks = extra
        evaluator.process([{"image_id": im["id"]}], [{"instances": inst}])
    if kind == "lvis":
        out = evaluator.evaluate()
    else:
        out = evaluator.evaluate(img_ids=[im["id"] for im in inputs["images"]])
    return {
        f"{task}/{name}": None if value != value else round(float(value), 6)
        for task, metrics in out.items()
        for name, value in metrics.items()
    }


class _UncalledOps(types.ModuleType):
    """Stands in for `mmcv._ext`, the compiled ops `mmcv-lite` leaves out."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def op(*_args, **_kwargs):
            raise RuntimeError(f"mmcv op {name} was called; this harness stubs the ops")

        return op


def run_mmdet(kind: str, options: dict, inputs: dict, _workdir: str) -> dict:
    sys.modules["mmcv._ext"] = _UncalledOps("mmcv._ext")
    from mmdet.evaluation.metrics import CocoMetric, LVISMetric

    if kind == "lvis":
        metric = LVISMetric(ann_file=inputs["ann_file"], metric=["bbox", "segm"], **options)
    else:
        names = {"bbox": ["bbox"], "segm": ["bbox", "segm"], "proposal": ["proposal"]}[kind]
        metric = CocoMetric(ann_file=inputs["ann_file"], metric=names, **options)
    metric.dataset_meta = {"classes": tuple(c["name"] for c in inputs["cats"])}
    samples = []
    for im, boxes, scores, labels, masks in predictions(inputs, kind):
        pred = {"bboxes": boxes, "scores": scores, "labels": labels}
        if masks is not None:
            pred["masks"] = masks
        samples.append({"img_id": im["id"], "ori_shape": (im["height"], im["width"]), "pred_instances": pred})
    for start in range(0, len(samples), 50):
        metric.process({}, samples[start : start + 50])
    out = metric.evaluate(len(samples))
    # A category with no ground truth here has no precision; NaN != NaN.
    return {name: None if value != value else float(value) for name, value in out.items()}


def child(case: str, backend: str) -> None:
    """One case on one backend; the result dict is the last line of stdout."""
    framework, kind, options = CASES[case]
    if backend == "hotcoco":
        import hotcoco

        hotcoco.init_as_pycocotools()
        hotcoco.init_as_lvis()
    else:
        import numpy as np

        np.float = float  # lvis-api 0.5.3; see the module docstring

    import lvis
    import pycocotools.coco

    # A run on the wrong backend compares a tool with itself and passes.
    for cls in (pycocotools.coco.COCO, lvis.LVIS):
        on_hotcoco = cls.__module__.split(".")[0] == "hotcoco"
        if on_hotcoco != (backend == "hotcoco"):
            raise SystemExit(f"{backend} run got {cls.__module__}.{cls.__name__}")

    with tempfile.TemporaryDirectory() as workdir:
        inputs = load_inputs(kind, workdir)
        run = run_detectron2 if framework == "d2" else run_mmdet
        result = run(kind, options, inputs, workdir)
    if not result:
        raise SystemExit("the evaluator returned no metrics")
    print(json.dumps(result, sort_keys=True))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_case(case: str) -> str | None:
    """`None` if the two backends agree, else what went wrong."""
    results = {}
    for backend in BACKENDS:
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--child", case, backend],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONWARNINGS": "ignore"},
                check=False,
                timeout=CASE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"{backend} run did not finish in {CASE_TIMEOUT_S} s"
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
            return f"{backend} run failed:\n{tail}"
        results[backend] = json.loads(proc.stdout.strip().splitlines()[-1])
    ref, ours = results["reference"], results["hotcoco"]
    differ = sorted(k for k in ref.keys() | ours.keys() if ref.get(k) != ours.get(k))
    if differ:
        shown = ", ".join(f"{k}: {ref.get(k)} vs {ours.get(k)}" for k in differ[:4])
        return f"{len(differ)} of {len(ref)} metrics differ (reference vs hotcoco): {shown}"
    print(f"  PASS  {case}: {len(ref)} metrics equal")
    return None


def main() -> int:
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], sys.argv[3])
        return 0
    cases = sys.argv[1:] or list(CASES)
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        print(f"unknown case {unknown[0]!r}; choose from: {', '.join(CASES)}")
        return 2
    if not (DATA / "bbox_val2017_results.json").exists():
        print("data/ is missing; run `just download-coco` first")
        return 2
    failed = 0
    for case in cases:
        problem = run_case(case)
        if problem:
            failed += 1
            print(f"  FAIL  {case}: {problem}")
    print(f"\n{len(cases) - failed} of {len(cases)} framework cases match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
