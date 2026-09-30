"""Run the whole adversarial corpus through the two-level harness.

`scripts/adversarial_harness.py` is the sharpest check in the repo: level 1
compares the 12 metrics, and level 2 compares the *per-match decisions* —
`dtIds`, `gtIds`, `dtMatches`, `gtIgnore` — annotation by annotation against
pycocotools. This runs both levels over every fixture in the corpus, in process;
`just adversarial <fixture>` is the same two calls with a printed report.

Both halves matter, and level 2 is the one metrics cannot replace: two detections
swapped between images, or a crowd flag on the wrong annotation, can leave AP
identical to fifteen decimal places while every decision underneath is wrong.

Each fixture's iou_type is read from its contents — see `_iou_type`.

    uv run pytest tests/test_adversarial.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from adversarial_harness import compare_eval_imgs, load_fixture, run_hotcoco, run_pycocotools
from helpers import FIXTURES_DIR, compare_metrics, suppress_output, written_json

CORPUS = FIXTURES_DIR / "adversarial"

# Level-1 metric tolerance per iou_type. These are the *harness's* documented
# thresholds for flagging a metric difference on small synthetic fixtures, and are
# deliberately not `parity.py`'s 1e-12 real-data gate — different question, different
# inputs. Kept in one place so the two do not drift into four spellings.
METRIC_TOLERANCE = {"bbox": 1e-4, "segm": 2e-4, "keypoints": 1e-4}


def _iou_type(path: Path) -> str:
    """Classify a fixture from its contents, not its filename.

    Filename suffixes looked tidy and matched nothing: `crowd_polygon_seg.json`
    ends in `_seg`, not `_segm`, so the segmentation branch was unreachable and
    every fixture silently ran as bbox. A dispatch that defaults to a mode rather
    than failing is the "green and verifying nothing" shape this repo keeps
    producing.

    Reading the data cannot drift. Note that `crowd_polygon_seg` is genuinely a
    *bbox* case despite the name — its GT carries a polygon but its detections are
    boxes, so it exercises "crowd + polygon GT must not break bbox eval". The
    corpus currently has no segmentation fixture; the branch stays because adding
    one should just work.
    """
    data = json.loads(path.read_text())
    anns = data.get("annotations", [])
    dets = data.get("detections", [])

    if any("keypoints" in a for a in anns):
        return "keypoints"
    # Segmentation eval needs masks on *both* sides; GT-only polygons are a bbox
    # case wearing a segmentation coat.
    seg = (a.get("segmentation") for a in anns)
    det_seg = (d.get("segmentation") for d in dets)
    if any(x is not None for x in seg) and any(x is not None for x in det_seg):
        return "segm"
    return "bbox"


def _fixtures() -> list[Path]:
    if not CORPUS.is_dir():
        return []
    return sorted(CORPUS.glob("*.json"))


FIXTURES = _fixtures()


def test_corpus_is_present():
    """A corpus that vanished would otherwise turn this file into zero tests."""
    assert FIXTURES, f"no fixtures found in {CORPUS} — the corpus is tracked and should not be empty"


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_matches_pycocotools_decision_for_decision(fixture: Path):
    iou_type = _iou_type(fixture)
    gt_data, detections = load_fixture(fixture)
    with written_json(gt_data) as (gt_path,), suppress_output():
        hc_ev = run_hotcoco(gt_path, detections, iou_type)
        py_ev = run_pycocotools(gt_path, detections, iou_type)

    mismatches = compare_metrics(py_ev.stats, hc_ev.stats, hc_ev.metric_keys(), tolerance=METRIC_TOLERANCE[iou_type])
    divergences = compare_eval_imgs(hc_ev, py_ev)
    problems = [m.line() for m in mismatches] + [
        f"image_id={d['image_id']} category_id={d['category_id']} aRng={d['aRng']}:\n" + "\n".join(d["issues"])
        for d in divergences
    ]
    assert not problems, f"{fixture.name} ({iou_type}) diverges from pycocotools:\n" + "\n".join(problems)
