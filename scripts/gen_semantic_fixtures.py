"""gen_semantic_fixtures.py — freeze semantic segmentation scores from three references.

Semantic mIoU is a pure function of one pixel confusion matrix, and the field has
three implementations of that function people quote numbers from: mmsegmentation's
`IoUMetric` (the SegFormer and ADE20K lineage), torchmetrics'
`multiclass_jaccard_index`, and detectron2's `SemSegEvaluator` (the Mask2Former
lineage). None is light to install — mmseg needs mmcv and torch, detectron2 needs
torch — so, as `gen_oid_fixtures.py` does for the TF Object Detection API, this
script downloads the reference *source* at a pinned commit, stubs the framework
imports the metric code never calls, runs it, and freezes the output.
`tests/test_parity_semantic.py` and `crates/hotcoco/tests/semantic.rs` then need no
network and no torch.

    uv run --with torchmetrics python scripts/gen_semantic_fixtures.py

torch itself comes from the `torchmetrics` dependency group (`uv sync --group
torchmetrics` once); mmseg's static methods run on torch tensors.

Where the three agree and where they do not
--------------------------------------------
All three mask out pixels whose ground truth is `ignore_index`, build one confusion
matrix over the whole split, and take per-class IoU as TP / (TP + FP + FN). They
differ in one corner: a class that is predicted somewhere but absent from every
ground-truth map has union > 0 and IoU 0. mmseg (`np.nanmean` over classes with a
finite IoU) and torchmetrics (`average="macro"` over classes present on either
side) count it toward the mean; detectron2 (`iou_valid = pos_gt > 0 & union > 0`)
drops it. hotcoco follows mmseg and torchmetrics. The fixture records all three so
the test can assert the choice is the one documented, on cases built to hit it.

Two other conventions are recorded, not chosen: fwIoU is detectron2's (mmseg has
none), and `reduce_zero_label` is mmseg's `LoadAnnotations` relabel — ground-truth
`0` becomes ignore and every other label shifts down by one — applied to the
ground truth only, which is where mmseg applies it.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import logging
import random
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from helpers import FIXTURES_DIR, fetch_source, load_module, stub_module

OUT = FIXTURES_DIR / "semantic_expected.json"

# Pinned, not `main`. An oracle tracking a moving branch is the same defect as a
# pinned semver baseline running zero checks: green, and you cannot say what it
# checked. mmseg is the v1.2.2 tag, the version vernier reports parity with;
# detectron2 has had no release since 2021, so it is the main commit of 2026-09-30.
MMSEG_COMMIT = "c685fe6767c4cadf6b051983ca6208f1b9d1ccb8"  # v1.2.2
MMSEG_URL = (
    f"https://raw.githubusercontent.com/open-mmlab/mmsegmentation/{MMSEG_COMMIT}/mmseg/evaluation/metrics/iou_metric.py"
)
D2_COMMIT = "1e3e13bbf607b54f62205c4c33922521822fb298"
D2_URL = f"https://raw.githubusercontent.com/facebookresearch/detectron2/{D2_COMMIT}/detectron2/evaluation/sem_seg_evaluation.py"

# Minimum share of cases that must exercise the convention corner (a class
# predicted but absent from the ground truth), and the share with ignored pixels.
# A corpus without them agrees with every implementation, including a wrong one.
MIN_PRED_ONLY_CASES = 0.25
MIN_IGNORED_CASES = 0.5

rng = random.Random(0x5E6)
nprng = np.random.default_rng(0x5E6)


def _install_mmseg(root: Path):
    """`IoUMetric`'s two static methods, with mmengine and the registry stubbed.

    `intersect_and_union` and `total_area_to_metrics` touch nothing but torch and
    numpy. The imports at module scope — mmengine's `BaseMetric`, logging,
    `mkdir_or_exist`, the `METRICS` registry, PrettyTable, PIL — serve
    `process()`/`compute_metrics()` plumbing this never calls.
    """
    dest = root / "iou_metric.py"
    fetch_source(MMSEG_URL, dest)

    class BaseMetric:  # noqa: D401 — a stand-in with the attributes the subclass reads
        def __init__(self, collect_device="cpu", prefix=None, **_):
            self.results = []

    stub_module("mmengine")
    stub_module("mmengine.dist", is_main_process=lambda: True)
    stub_module("mmengine.evaluator", BaseMetric=BaseMetric)
    stub_module(
        "mmengine.logging",
        MMLogger=type("MMLogger", (), {"get_current_instance": staticmethod(lambda: None)}),
        print_log=lambda *a, **k: None,
    )
    stub_module("mmengine.utils", mkdir_or_exist=lambda *a, **k: None)
    stub_module("mmseg")
    stub_module("mmseg.registry", METRICS=types.SimpleNamespace(register_module=lambda *a, **k: lambda cls: cls))
    stub_module("prettytable", PrettyTable=object)
    return load_module(dest, "mmseg_iou_metric").IoUMetric


def _install_detectron2(root: Path):
    """`SemSegEvaluator.evaluate()`, with detectron2's data and comm layers stubbed.

    `evaluate()` reads only `_conf_matrix`, `_num_classes`, `_class_names`,
    `_compute_boundary_iou`, `_distributed`, `_output_dir` and `_logger` off the instance;
    with `_distributed=False` and no output dir, the comm and file I/O paths are
    skipped. The constructor needs a `MetadataCatalog`, so the instance is made
    with `__new__` and those attributes set directly — the same state `process()`
    would have left, whose `np.bincount` fill is reproduced in `d2_confusion`.
    """
    dest = root / "sem_seg_evaluation.py"
    fetch_source(D2_URL, dest)

    stub_module("detectron2")
    stub_module("detectron2.data", DatasetCatalog=object, MetadataCatalog=object)
    stub_module("detectron2.utils")
    stub_module(
        "detectron2.utils.comm", is_main_process=lambda: True, synchronize=lambda: None, all_gather=lambda x: [x]
    )
    stub_module("detectron2.utils.file_io", PathManager=object)
    # Loaded under its real dotted name so `from .evaluator import ...` resolves.
    stub_module("detectron2.evaluation", __path__=[])
    stub_module("detectron2.evaluation.evaluator", DatasetEvaluator=object)
    stub_module("pycocotools")
    stub_module("pycocotools.mask")
    return load_module(dest, "detectron2.evaluation.sem_seg_evaluation").SemSegEvaluator


def d2_confusion(gt_maps, pred_maps, num_classes, ignore_label):
    """detectron2 `SemSegEvaluator.process()`'s confusion-matrix fill, verbatim:
    rows are predictions, columns ground truth, index `num_classes` is ignore."""
    conf = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    for gt, pred in zip(gt_maps, pred_maps):
        gt = gt.astype(np.int64).copy()
        pred = pred.astype(np.int64)
        gt[gt == ignore_label] = num_classes
        conf += np.bincount((num_classes + 1) * pred.reshape(-1) + gt.reshape(-1), minlength=conf.size).reshape(
            conf.shape
        )
    return conf


def reduce_zero_label(gt: np.ndarray) -> np.ndarray:
    """mmseg `LoadAnnotations` with `reduce_zero_label=True`, verbatim: `0` becomes
    255, every other label shifts down by one, and the shifted 254 becomes 255.
    The 255 is hardcoded there, which is why hotcoco requires `ignore_index=255`
    with this option."""
    out = gt.astype(np.int64).copy()
    out[out == 0] = 255
    out = out - 1
    out[out == 254] = 255
    return out


def _random_case(idx: int):
    num_classes = rng.randint(2, 9)
    reduce = rng.random() < 0.2
    ignore_index = rng.choice([255, 255, 255, num_classes + 3]) if not reduce else 255
    n_images = rng.randint(1, 4)
    # Which classes exist on each side, so some are absent from both, some are
    # ground-truth only (IoU 0, Acc 0) and some prediction only (the corner).
    gt_classes = sorted(rng.sample(range(num_classes), rng.randint(1, num_classes)))
    pred_only = [c for c in range(num_classes) if c not in gt_classes and rng.random() < 0.6]
    pred_classes = sorted(set(gt_classes) | set(pred_only)) if rng.random() < 0.8 else gt_classes
    ignore_frac = rng.choice([0.0, 0.0, 0.1, 0.3])

    gts, preds = [], []
    for _ in range(n_images):
        h, w = rng.randint(6, 20), rng.randint(6, 20)
        gt = nprng.choice(gt_classes, size=(h, w))
        # Predictions mostly agree, so IoUs land strictly between 0 and 1.
        agree = nprng.random((h, w)) < rng.choice([0.5, 0.7, 0.9])
        noise = nprng.choice(pred_classes, size=(h, w))
        pred = np.where(agree, gt, noise)
        if ignore_frac:
            gt = np.where(nprng.random((h, w)) < ignore_frac, ignore_index, gt)
        if reduce:
            # Stored labels are the raw file's: 1..=num_classes, with 0 for
            # "other"; the metric sees them through `reduce_zero_label`.
            gt = np.where(gt == ignore_index, 0, gt + 1)
        gts.append(gt.astype(np.int64))
        preds.append(pred.astype(np.int64))
    return {
        "id": idx,
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "reduce_zero_label": reduce,
        "gt": [g.tolist() for g in gts],
        "pred": [p.tolist() for p in preds],
    }


def _hand_cases():
    """Corners a random draw may miss: a perfect prediction, a class never
    predicted, every pixel ignored, and the convention corner in isolation."""
    return [
        {
            "num_classes": 3,
            "ignore_index": 255,
            "reduce_zero_label": False,
            "gt": [[[0, 1], [2, 2]]],
            "pred": [[[0, 1], [2, 2]]],
        },
        {
            "num_classes": 3,
            "ignore_index": 255,
            "reduce_zero_label": False,
            "gt": [[[0, 0], [1, 1]]],
            "pred": [[[0, 0], [0, 0]]],
        },
        {
            "num_classes": 2,
            "ignore_index": 255,
            "reduce_zero_label": False,
            "gt": [[[255, 255], [255, 255]]],
            "pred": [[[0, 1], [1, 0]]],
        },
        # Class 2 appears only in the prediction: mmseg and torchmetrics average
        # its IoU of 0 in (mIoU (1 + ½ + 0) / 3 = ½); detectron2 leaves it out
        # (mIoU (1 + ½) / 2 = ¾).
        {
            "num_classes": 3,
            "ignore_index": 255,
            "reduce_zero_label": False,
            "gt": [[[0, 0], [1, 1]]],
            "pred": [[[0, 0], [1, 2]]],
        },
    ]


def _num(v) -> float | None:
    """A reference scalar, with NaN as JSON `null` — JSON has no NaN, and the
    consumers read `null` as hotcoco's "not computed"."""
    v = float(v)
    return None if np.isnan(v) else v


def evaluate_case(case, IoUMetric, SemSegEvaluator, tm):
    num_classes = case["num_classes"]
    ignore = case["ignore_index"]
    gts = [np.array(g, dtype=np.int64) for g in case["gt"]]
    preds = [np.array(p, dtype=np.int64) for p in case["pred"]]
    if case["reduce_zero_label"]:
        gts = [reduce_zero_label(g) for g in gts]

    # mmseg: per-image areas summed, then the dataset-level formulas.
    totals = [torch.zeros(num_classes, dtype=torch.float64) for _ in range(4)]
    for gt, pred in zip(gts, preds):
        parts = IoUMetric.intersect_and_union(torch.from_numpy(pred), torch.from_numpy(gt), num_classes, ignore)
        for t, p in zip(totals, parts):
            t += p
    mm = IoUMetric.total_area_to_metrics(*totals, metrics=["mIoU"], nan_to_num=None)
    mm_iou = np.asarray(mm["IoU"], dtype=np.float64)
    mm_acc = np.asarray(mm["Acc"], dtype=np.float64)
    mmseg = {
        "IoU": [_num(v) for v in mm_iou],
        "Acc": [_num(v) for v in mm_acc],
        "aAcc": _num(mm["aAcc"]),
        "mIoU": _num(np.nanmean(mm_iou)) if np.isfinite(mm_iou).any() else None,
        "mAcc": _num(np.nanmean(mm_acc)) if np.isfinite(mm_acc).any() else None,
    }

    # torchmetrics: whole split as one tensor pair. It reduces in float32, so
    # its numbers sit within ~1e-7 of mmseg's, not on them.
    pred_t = torch.cat([torch.from_numpy(p).reshape(-1) for p in preds])
    gt_t = torch.cat([torch.from_numpy(g).reshape(-1) for g in gts])
    kw = dict(num_classes=num_classes, ignore_index=ignore)
    torchmetrics = {
        "IoU": tm.multiclass_jaccard_index(pred_t, gt_t, average=None, **kw).double().tolist(),
        "mIoU": _num(tm.multiclass_jaccard_index(pred_t, gt_t, average="macro", **kw)),
        "aAcc": _num(tm.multiclass_accuracy(pred_t, gt_t, average="micro", multidim_average="global", **kw)),
    }

    # detectron2: the evaluate() of a SemSegEvaluator whose process() saw these maps.
    ev = SemSegEvaluator.__new__(SemSegEvaluator)
    ev._num_classes = num_classes
    ev._class_names = [f"c{i}" for i in range(num_classes)]
    ev._compute_boundary_iou = False
    ev._distributed = False
    ev._output_dir = None
    ev._conf_matrix = d2_confusion(gts, preds, num_classes, ignore)
    ev._b_conf_matrix = np.zeros_like(ev._conf_matrix)
    ev._predictions = []
    ev._logger = logging.getLogger("detectron2.stub")
    res = ev.evaluate()["sem_seg"]
    detectron2 = {
        "mIoU": _num(res["mIoU"] / 100),
        "fwIoU": _num(res["fwIoU"] / 100),
        "mACC": _num(res["mACC"] / 100),
        "pACC": _num(res["pACC"] / 100),
        "IoU": [_num(res[f"IoU-c{i}"] / 100) for i in range(num_classes)],
        "ACC": [_num(res[f"ACC-c{i}"] / 100) for i in range(num_classes)],
    }
    return {"mmseg": mmseg, "torchmetrics": torchmetrics, "detectron2": detectron2}


def main():
    import torchmetrics.functional.classification as tm

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        IoUMetric = _install_mmseg(root)
        SemSegEvaluator = _install_detectron2(root)

        cases = [_random_case(i) for i in range(60)]
        for i, hand in enumerate(_hand_cases()):
            cases.append({"id": len(cases), **hand})
        # Pure-numpy evaluate() emits RuntimeWarnings on 0/0; they are the NaNs
        # the fixture records as null.
        with np.errstate(divide="ignore", invalid="ignore"):
            for case in cases:
                case["expected"] = evaluate_case(case, IoUMetric, SemSegEvaluator, tm)

    pred_only = sum(
        1
        for c in cases
        if any(
            iou == 0.0 and acc is None for iou, acc in zip(c["expected"]["mmseg"]["IoU"], c["expected"]["mmseg"]["Acc"])
        )
        and c["expected"]["detectron2"]["mIoU"] is not None
    )
    ignored = sum(1 for c in cases if any(c["ignore_index"] in np.array(g) for g in c["gt"]) or c["reduce_zero_label"])
    assert pred_only >= MIN_PRED_ONLY_CASES * len(cases), f"only {pred_only} cases hit the prediction-only corner"
    assert ignored >= MIN_IGNORED_CASES * len(cases), f"only {ignored} cases have ignored pixels"

    OUT.write_text(
        json.dumps(
            {
                "reference": {
                    "mmsegmentation": {
                        "commit": MMSEG_COMMIT,
                        "tag": "v1.2.2",
                        "file": "mmseg/evaluation/metrics/iou_metric.py",
                    },
                    "detectron2": {"commit": D2_COMMIT, "file": "detectron2/evaluation/sem_seg_evaluation.py"},
                    "torchmetrics": md.version("torchmetrics"),
                    "torch": md.version("torch"),
                },
                "convention": (
                    "mmseg/torchmetrics: a class present only in predictions counts (IoU 0); detectron2 drops it"
                ),
                "min_pred_only_cases": MIN_PRED_ONLY_CASES,
                "cases": cases,
            },
            indent=None,
            separators=(",", ":"),
        )
        + "\n"
    )
    print(
        f"wrote {len(cases)} cases to {OUT} ({OUT.stat().st_size / 1024:.0f} KB); {pred_only} hit the convention corner"
    )


if __name__ == "__main__":
    main()
