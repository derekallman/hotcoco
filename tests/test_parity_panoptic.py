"""Panoptic parity: hotcoco vs panopticapi on synthetic label maps.

Every case is written to disk as COCO panoptic PNG files plus JSON, run through
panopticapi's own worker (`pq_compute_single_core`, the function `pq_compute`
fans out to), and through `hotcoco.panoptic.PanopticEval` on the same files.
Compared below the averages, where a mistake cannot cancel:

- per-category TP, FP, FN exactly and summed IoU to 1e-9;
- PQ, SQ, RQ and N for All, Things, Stuff to 1e-9.

The same label maps also go through the path with no PNG files — detection-
style datasets carrying RLEs — which must reproduce the PNG path exactly.

What the random cases exercise, by construction: crowd segments absorbing
predictions of their category, predictions mostly on void being ignored,
category mismatches, IoU on both sides of 0.5, segments dropped, merged and
shifted, images with empty predictions, and ground-truth `area` fields that
disagree with the PNG (panopticapi believes the JSON). The reference here is
the pinned panopticapi commit the dev extra installs; `scripts/parity_panoptic.py`
runs the same comparison on COCO val2017.

    uv run pytest tests/test_parity_panoptic.py -v
"""

from __future__ import annotations

import copy
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
from helpers import panopticapi_reference, perturb_label_map, pq_disagreements, suppress_output
from hotcoco import COCO, mask, panoptic
from panopticapi.evaluation import pq_compute
from panopticapi.utils import id2rgb
from PIL import Image

TOL = 1e-9

# Share of random cases that must land strictly inside (0, 1): a corpus of
# perfect or empty matches agrees with any implementation that returns 0 or 1.
MIN_DISCRIMINATING = 0.5

CATEGORIES = [
    {"id": 1, "name": "person", "isthing": 1},
    {"id": 2, "name": "car", "isthing": 1},
    {"id": 7, "name": "dog", "isthing": 1},
    {"id": 100, "name": "sky", "isthing": 0},
    {"id": 101, "name": "road", "isthing": 0},
]
THING_IDS = [c["id"] for c in CATEGORIES if c["isthing"]]
STUFF_IDS = [c["id"] for c in CATEGORIES if not c["isthing"]]


# ── building cases ──────────────────────────────────────────────────────────


class Case:
    """Ground truth and prediction for a few images, as label maps + segments."""

    def __init__(self, name: str, *, png_only: bool = False):
        self.name = name
        # True when the case leans on something only the PNG format can say —
        # a PNG id with no segments_info entry, or a stated area that differs
        # from the pixels — so the RLE path is not expected to reproduce it.
        self.png_only = png_only
        self.images: list[dict] = []
        # image_id -> (label map, segments_info)
        self.gt: dict[int, tuple[np.ndarray, list[dict]]] = {}
        self.pred: dict[int, tuple[np.ndarray, list[dict]]] = {}
        self.categories = [dict(c) for c in CATEGORIES]

    def add(self, image_id: int, gt_map, gt_segs, pred_map, pred_segs):
        h, w = gt_map.shape
        self.images.append({"id": image_id, "height": h, "width": w, "file_name": f"{image_id}.jpg"})
        self.gt[image_id] = (gt_map, gt_segs)
        self.pred[image_id] = (pred_map, pred_segs)
        return self

    def write(self, root: Path) -> dict:
        """Write PNG files + JSON the way the COCO panoptic release lays them out."""
        paths = {}
        for side, data in (("gt", self.gt), ("pred", self.pred)):
            folder = root / f"panoptic_{side}"
            folder.mkdir(parents=True, exist_ok=True)
            annotations = []
            for image_id, (label_map, segs) in data.items():
                Image.fromarray(id2rgb(label_map)).save(folder / f"{image_id}.png")
                annotations.append({"image_id": image_id, "file_name": f"{image_id}.png", "segments_info": segs})
            payload = {"images": self.images, "annotations": annotations, "categories": self.categories}
            json_path = root / f"panoptic_{side}.json"
            json_path.write_text(json.dumps(payload))
            paths[side] = (json_path, folder)
        return paths

    def as_coco(self, side: str) -> COCO:
        """The same maps as a detection-style dataset with RLE masks."""
        data = self.gt if side == "gt" else self.pred
        annotations = []
        next_id = 1
        for image_id, (label_map, segs) in data.items():
            for seg in segs:
                binary = np.asfortranarray((label_map == seg["id"]).astype(np.uint8))
                annotations.append(
                    {
                        "id": next_id,
                        "image_id": image_id,
                        "category_id": seg["category_id"],
                        "iscrowd": seg.get("iscrowd", 0),
                        "segmentation": mask.encode(binary),
                        "area": int(binary.sum()),
                    }
                )
                next_id += 1
        return COCO({"images": self.images, "annotations": annotations, "categories": self.categories})


def segments_from_map(label_map: np.ndarray, category_of, rng: random.Random, *, crowd_rate=0.0, area_noise=0.0):
    """segments_info for every non-zero id in `label_map`."""
    segs = []
    for sid in sorted(int(v) for v in np.unique(label_map) if v != 0):
        area = int((label_map == sid).sum())
        if area_noise and rng.random() < area_noise:
            # panopticapi takes the JSON's word for a ground-truth area.
            area += rng.randint(-area // 4, area // 2)
        segs.append(
            {"id": sid, "category_id": category_of(sid), "area": area, "iscrowd": 1 if rng.random() < crowd_rate else 0}
        )
    return segs


def random_partition(rng: random.Random, h: int, w: int, n: int, first_id: int = 1) -> np.ndarray:
    """`n` rectangles painted in order, later over earlier; 0 stays void."""
    m = np.zeros((h, w), dtype=np.uint32)
    sid = first_id
    for _ in range(n):
        rh, rw = rng.randint(2, max(3, h // 2)), rng.randint(2, max(3, w // 2))
        y, x = rng.randint(0, h - rh), rng.randint(0, w - rw)
        m[y : y + rh, x : x + rw] = sid
        sid += 1
    return m


def random_case(seed: int, area_noise: float = 0.0) -> Case:
    rng = random.Random(seed)
    case = Case(f"random-{seed}" + ("-area-noise" if area_noise else ""), png_only=bool(area_noise))
    for image_id in (1, 2):
        h, w = rng.randint(12, 40), rng.randint(12, 40)
        gt_map = random_partition(rng, h, w, rng.randint(2, 7))
        cats = {}

        def category_of(sid, cats=cats, rng=rng):
            return cats.setdefault(sid, rng.choice(THING_IDS + STUFF_IDS))

        gt_segs = segments_from_map(gt_map, category_of, rng, crowd_rate=0.15, area_noise=area_noise)
        pred_map, pred_segs = perturb_label_map(
            gt_map, gt_segs, THING_IDS + STUFF_IDS, rng, shift=3, erode_px=(1,), first_id=1000, spurious=3
        )
        case.add(image_id, gt_map, gt_segs, pred_map, pred_segs)
    return case


def hand_cases() -> list[Case]:
    """One case per rule, with the geometry written out."""
    cases = []

    # Crowd absorption: a prediction mostly on a crowd region of its own
    # category is ignored; the same prediction of another category is a FP.
    gt = np.zeros((8, 8), dtype=np.uint32)
    gt[:, :4] = 1  # crowd person
    gt[0:2, 6:8] = 2  # person
    pred = np.zeros_like(gt)
    pred[0:3, :4] = 11  # on the crowd, cat person -> ignored
    pred[4:7, :4] = 12  # on the crowd, cat car -> FP
    pred[0:4, 6:8] = 13  # covers segment 2 plus void -> matched, IoU 1.0
    cases.append(
        Case("crowd").add(
            1,
            gt,
            [
                {"id": 1, "category_id": 1, "area": 32, "iscrowd": 1},
                {"id": 2, "category_id": 1, "area": 4, "iscrowd": 0},
            ],
            pred,
            [{"id": 11, "category_id": 1}, {"id": 12, "category_id": 2}, {"id": 13, "category_id": 1}],
        )
    )

    # Exactly 0.5 is not a match.
    gt = np.zeros((4, 4), dtype=np.uint32)
    gt[:, 0:3] = 1
    gt[:, 3] = 2
    pred = np.zeros_like(gt)
    pred[:, 0:2] = 11
    pred[:, 2:4] = 12
    # gt1 (12 px) vs pred 11 (8 px): inter 8, union 12 -> 0.667 match.
    # gt2 (4 px) vs pred 12 (8 px): inter 4, union 8 -> 0.5 no match.
    cases.append(
        Case("half").add(
            1,
            gt,
            [
                {"id": 1, "category_id": 100, "area": 12, "iscrowd": 0},
                {"id": 2, "category_id": 101, "area": 4, "iscrowd": 0},
            ],
            pred,
            [{"id": 11, "category_id": 100}, {"id": 12, "category_id": 101}],
        )
    )

    # A ground-truth PNG id with no segments_info entry: not void, so it
    # neither shrinks the union nor helps a prediction get ignored.
    gt = np.zeros((4, 8), dtype=np.uint32)
    gt[:, 0:4] = 9  # unknown to the JSON
    gt[:, 4:8] = 1
    pred = np.zeros_like(gt)
    pred[:, 2:8] = 11  # 8 on gt1, 8 on the unknown id: iou 16/(24+16-16) = 2/3
    cases.append(
        Case("unknown-gt-label", png_only=True).add(
            1, gt, [{"id": 1, "category_id": 2, "area": 16, "iscrowd": 0}], pred, [{"id": 11, "category_id": 2}]
        )
    )

    # The JSON's ground-truth area, not the PNG's, enters the union.
    gt = np.zeros((4, 4), dtype=np.uint32)
    gt[:, :] = 1
    pred = np.zeros_like(gt)
    pred[:, :] = 11
    cases.append(
        Case("json-area", png_only=True)
        .add(1, gt, [{"id": 1, "category_id": 7, "area": 40, "iscrowd": 0}], pred, [{"id": 11, "category_id": 7}])
        .add(2, gt, [{"id": 1, "category_id": 7, "area": 16, "iscrowd": 0}], pred, [{"id": 11, "category_id": 7}])
    )

    # An image where nothing was predicted, and one with nothing to find.
    gt = np.zeros((6, 6), dtype=np.uint32)
    gt[1:5, 1:5] = 1
    empty = np.zeros_like(gt)
    pred = np.zeros_like(gt)
    pred[2:4, 2:4] = 11
    cases.append(
        Case("empty-sides")
        .add(1, gt, [{"id": 1, "category_id": 1, "area": 16, "iscrowd": 0}], empty, [])
        .add(2, empty, [], pred, [{"id": 11, "category_id": 100}])
    )

    return cases


# ── running both ────────────────────────────────────────────────────────────


def run_reference(paths: dict) -> tuple[dict, dict]:
    """panopticapi on the written files: (averages, per-category counts)."""
    return panopticapi_reference(paths["gt"][0], paths["pred"][0], paths["gt"][1], paths["pred"][1], multi_core=False)


def run_hotcoco_png(paths: dict) -> dict:
    gt_json, gt_folder = paths["gt"]
    pred_json, pred_folder = paths["pred"]
    ev = panoptic.PanopticEval(gt_json, pred_json, gt_folder=gt_folder, pred_folder=pred_folder)
    ev.evaluate()
    return ev.results()


def run_hotcoco_rle(case: Case) -> dict:
    ev = panoptic.PanopticEval(case.as_coco("gt"), case.as_coco("pred"))
    ev.evaluate()
    return ev.results()


def disagreements(ref_avg: dict, ref_counts: dict, got: dict) -> list[str]:
    return pq_disagreements(ref_avg, ref_counts, got, [c["id"] for c in CATEGORIES], tol=TOL)


RANDOM_CASES = [random_case(seed) for seed in range(40)] + [random_case(seed, area_noise=0.5) for seed in range(10)]
CASES = hand_cases() + RANDOM_CASES


@pytest.fixture(params=CASES, ids=[c.name for c in CASES])
def written_case(request, tmp_path):
    case = request.param
    return case, case.write(tmp_path)


def test_random_cases_discriminate():
    """Most random cases must score strictly between 0 and 1."""
    inside = 0
    for case in RANDOM_CASES:
        with tempfile.TemporaryDirectory() as d:
            pq = run_hotcoco_png(case.write(Path(d)))["All"]["pq"]
        inside += 0.0 < pq < 1.0
    assert inside >= len(RANDOM_CASES) * MIN_DISCRIMINATING, f"{inside}/{len(RANDOM_CASES)} cases inside (0, 1)"


def test_png_path_matches_panopticapi(written_case):
    case, paths = written_case
    ref_avg, ref_counts = run_reference(paths)
    got = run_hotcoco_png(paths)
    why = disagreements(ref_avg, ref_counts, got)
    assert not why, f"{case.name} diverges from panopticapi:\n  " + "\n  ".join(why)


def test_rle_path_reproduces_png_path(written_case):
    """No PNG files, same pixels, same numbers — exactly."""
    case, paths = written_case
    if case.png_only:
        pytest.skip("case depends on PNG-only semantics")
    from_png = run_hotcoco_png(paths)
    from_rle = run_hotcoco_rle(case)
    for key in ("All", "Things", "Stuff", "per_class"):
        assert from_rle[key] == from_png[key], f"{case.name}: {key} differs between the RLE and PNG paths"


def test_pq_compute_is_a_drop_in(tmp_path):
    """Same signature, same keys, same values as panopticapi's `pq_compute`."""
    case = RANDOM_CASES[3]
    paths = case.write(tmp_path)
    with suppress_output(stderr=False):
        ref = pq_compute(str(paths["gt"][0]), str(paths["pred"][0]), str(paths["gt"][1]), str(paths["pred"][1]))
        got = panoptic.pq_compute(paths["gt"][0], paths["pred"][0], paths["gt"][1], paths["pred"][1])
    assert set(ref) <= set(got), f"panopticapi keys missing: {set(ref) - set(got)}"
    for split in ("All", "Things", "Stuff"):
        for key in ("pq", "sq", "rq", "n"):
            assert abs(ref[split][key] - got[split][key]) <= TOL, (split, key)
    for cid, ref_scores in ref["per_class"].items():
        g = got["per_class"][cid]
        if g["tp"] + g["fp"] + g["fn"] == 0:
            # panopticapi prints 0.0 for a category it then skips; hotcoco
            # says "not computed". The counts beside it are the tie-breaker.
            assert ref_scores == {"pq": 0.0, "sq": 0.0, "rq": 0.0}
            assert (g["pq"], g["sq"], g["rq"]) == (-1.0, -1.0, -1.0)
        else:
            for key in ("pq", "sq", "rq"):
                assert abs(ref_scores[key] - g[key]) <= TOL, (cid, key)

    # The default folders are the JSON paths without `.json` on both sides.
    ref_default = ref
    with suppress_output(stderr=False):
        got_default = panoptic.pq_compute(paths["gt"][0], paths["pred"][0])
    assert got_default["All"] == got["All"]
    assert abs(ref_default["All"]["pq"] - got_default["All"]["pq"]) <= TOL


def test_comparator_catches_a_wrong_number(tmp_path):
    """The check above must be able to fail: perturb one count and one score."""
    case = RANDOM_CASES[0]
    paths = case.write(tmp_path)
    ref_avg, ref_counts = run_reference(paths)
    got = run_hotcoco_png(paths)
    assert not disagreements(ref_avg, ref_counts, got)

    wrong = copy.deepcopy(got)
    wrong["All"]["pq"] += 1e-6
    assert any(p.startswith("All.pq") for p in disagreements(ref_avg, ref_counts, wrong))

    wrong = copy.deepcopy(got)
    cid = next(c for c, v in got["per_class"].items() if v["tp"] + v["fp"] + v["fn"] > 0)
    wrong["per_class"][cid]["fp"] += 1
    assert any(p.startswith(f"category {cid}.fp") for p in disagreements(ref_avg, ref_counts, wrong))


@pytest.mark.parametrize(
    "break_it, message",
    [
        (lambda pred: pred["annotations"][0]["segments_info"].pop(), "is in the PNG but not in segments_info"),
        (
            lambda pred: pred["annotations"][0]["segments_info"].append({"id": 424242, "category_id": 1}),
            "is in segments_info but not in the mask",
        ),
        (
            lambda pred: pred["annotations"][0]["segments_info"][0].__setitem__("category_id", 999),
            "unknown category_id 999",
        ),
        (lambda pred: pred["annotations"].pop(), "no prediction for the image with id"),
    ],
    ids=["png-id-not-in-json", "json-id-not-in-png", "unknown-category", "missing-image"],
)
def test_reference_errors_raise_here_too(tmp_path, break_it, message):
    """Where panopticapi raises, hotcoco raises, naming the same problem."""
    case = RANDOM_CASES[1]
    paths = case.write(tmp_path)
    pred_json = json.loads(paths["pred"][0].read_text())
    break_it(pred_json)
    paths["pred"][0].write_text(json.dumps(pred_json))

    with pytest.raises(Exception):
        with suppress_output(stderr=False):
            pq_compute(str(paths["gt"][0]), str(paths["pred"][0]), str(paths["gt"][1]), str(paths["pred"][1]))
    with pytest.raises(RuntimeError, match=message):
        run_hotcoco_png(paths)


def test_missing_isthing_is_an_extension(tmp_path):
    case = Case("no-isthing")
    gt = np.ones((4, 4), dtype=np.uint32)
    case.add(1, gt, [{"id": 1, "category_id": 1, "area": 16, "iscrowd": 0}], gt, [{"id": 1, "category_id": 1}])
    for c in case.categories:
        c.pop("isthing")
    paths = case.write(tmp_path)
    ev = panoptic.PanopticEval(paths["gt"][0], paths["pred"][0])
    assert ev.provenance() == "extension"
    assert "isthing" in ev.reference_deviations()[0]
    with pytest.warns(UserWarning, match="isthing"):
        ev.run()
    res = ev.results()
    assert res["All"]["pq"] == 1.0
    assert res["Things"]["n"] == 0 and res["Things"]["pq"] == -1.0
    report = ev.report()
    assert report["provenance"] == "extension"
    assert "things" not in report["per_group"]
    assert report["metrics"]["PQ"] == 1.0


def test_report_has_the_family_shape(tmp_path):
    case = RANDOM_CASES[2]
    paths = case.write(tmp_path)
    ev = panoptic.PanopticEval(paths["gt"][0], paths["pred"][0], gt_folder=paths["gt"][1], pred_folder=paths["pred"][1])
    ev.evaluate()
    report = ev.report()
    assert report["task"] == "panoptic"
    assert report["provenance"] == "parity_verified"
    assert list(report["metrics"]) == sorted(panoptic.METRIC_NAMES)
    assert ev.stats == [report["metrics"][k] for k in panoptic.METRIC_NAMES]
    assert set(report["per_group"]) <= {"all", "things", "stuff"}
    assert set(report["per_group"]["all"]) == {"PQ", "SQ", "RQ", "n"}
    for name, scores in report["per_class"].items():
        assert set(scores) == {"PQ", "SQ", "RQ"}, name
        assert all(0.0 <= v <= 1.0 for v in scores.values()), name
    assert report["curves"] == {}
    assert report["params"]["n_images"] == 2
