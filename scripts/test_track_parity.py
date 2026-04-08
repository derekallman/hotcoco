"""Parity tests: hotcoco tracking metrics vs TrackEval.

Constructs tracking sequences, runs both implementations, and verifies
that HOTA, CLEAR MOT, and IDF1 metrics match within tolerance.

Usage:
    uv run pytest scripts/test_track_parity.py -v -x --tb=short
"""

import json
import os
import tempfile

import numpy as np
import pytest
from trackeval.metrics import CLEAR as TrackEvalCLEAR
from trackeval.metrics import HOTA as TrackEvalHOTA
from trackeval.metrics import Identity as TrackEvalIdentity

from hotcoco import COCO, TrackingEval

# Tolerance for floating-point comparison
TOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_frames(frames):
    """Round box coordinates to 2 decimal places (matching MOTChallenge CSV precision)."""
    return {
        fid: [(tid, round(x, 2), round(y, 2), round(w, 2), round(h, 2)) for tid, x, y, w, h in boxes]
        for fid, boxes in frames.items()
    }


def frames_to_coco_json(frames, is_gt=True):
    """Convert frame dict → TAO-style COCO JSON dict."""
    images = []
    annotations = []
    ann_id = 1
    for fid in sorted(frames.keys()):
        images.append(
            {"id": fid, "file_name": f"{fid:06d}.jpg", "height": 480, "width": 640, "video_id": 1, "frame_index": fid}
        )
        for tid, x, y, w, h in frames[fid]:
            ann = {
                "id": ann_id,
                "image_id": fid,
                "category_id": 1,
                "track_id": tid,
                "video_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            if not is_gt:
                ann["score"] = 1.0
            annotations.append(ann)
            ann_id += 1

    # Ensure DT images include all GT images (even empty ones)
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
        "videos": [{"id": 1, "name": "seq1"}],
    }


def build_trackeval_data(gt_frames, dt_frames):
    """Convert frame dicts to the data dict format TrackEval expects."""
    all_frame_ids = sorted(set(list(gt_frames.keys()) + list(dt_frames.keys())))
    num_timesteps = len(all_frame_ids)

    gt_id_set = set()
    dt_id_set = set()
    for boxes in gt_frames.values():
        for tid, *_ in boxes:
            gt_id_set.add(tid)
    for boxes in dt_frames.values():
        for tid, *_ in boxes:
            dt_id_set.add(tid)

    gt_ids_sorted = sorted(gt_id_set)
    dt_ids_sorted = sorted(dt_id_set)
    gt_id_map = {tid: i for i, tid in enumerate(gt_ids_sorted)}
    dt_id_map = {tid: i for i, tid in enumerate(dt_ids_sorted)}

    data = {
        "num_timesteps": num_timesteps,
        "num_gt_ids": len(gt_ids_sorted),
        "num_tracker_ids": len(dt_ids_sorted),
        "num_gt_dets": 0,
        "num_tracker_dets": 0,
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
    }

    for fid in all_frame_ids:
        gt_boxes = gt_frames.get(fid, [])
        dt_boxes = dt_frames.get(fid, [])

        gt_ids_t = np.array([gt_id_map[b[0]] for b in gt_boxes], dtype=int)
        dt_ids_t = np.array([dt_id_map[b[0]] for b in dt_boxes], dtype=int)

        data["num_gt_dets"] += len(gt_boxes)
        data["num_tracker_dets"] += len(dt_boxes)
        data["gt_ids"].append(gt_ids_t)
        data["tracker_ids"].append(dt_ids_t)

        num_gt = len(gt_boxes)
        num_dt = len(dt_boxes)
        if num_gt == 0 or num_dt == 0:
            data["similarity_scores"].append(np.zeros((num_gt, num_dt)))
            continue

        iou = np.zeros((num_gt, num_dt))
        for gi, (_, gx, gy, gw, gh) in enumerate(gt_boxes):
            for di, (_, dx, dy, dw, dh) in enumerate(dt_boxes):
                x1 = max(gx, dx)
                y1 = max(gy, dy)
                x2 = min(gx + gw, dx + dw)
                y2 = min(gy + gh, dy + dh)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                union = gw * gh + dw * dh - inter
                iou[gi, di] = inter / union if union > 0 else 0.0

        data["similarity_scores"].append(iou)

    return data


def run_both(gt_frames, dt_frames, clear_threshold=0.5):
    """Run hotcoco TrackingEval and TrackEval on the same data, return both results."""
    gt_frames = _round_frames(gt_frames)
    dt_frames = _round_frames(dt_frames)

    # --- hotcoco ---
    # Ensure DT images include all frames from GT (TrackingEval groups by GT images)
    all_fids = sorted(set(list(gt_frames.keys()) + list(dt_frames.keys())))

    gt_json = frames_to_coco_json({fid: gt_frames.get(fid, []) for fid in all_fids}, is_gt=True)
    dt_json = frames_to_coco_json({fid: dt_frames.get(fid, []) for fid in all_fids}, is_gt=False)

    # Write to temp files
    gt_f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(gt_json, gt_f)
    gt_f.close()

    dt_f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(dt_json, dt_f)
    dt_f.close()

    try:
        coco_gt = COCO(gt_f.name)
        coco_dt = coco_gt.load_res(dt_f.name)
        ev = TrackingEval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        hc = ev.get_results()
    finally:
        os.unlink(gt_f.name)
        os.unlink(dt_f.name)

    # Map our key names to the ones the tests expect
    hc_flat = {
        "HOTA": hc.get("HOTA", 0.0),
        "DetA": hc.get("DetA", 0.0),
        "AssA": hc.get("AssA", 0.0),
        "LocA": hc.get("LocA", 0.0),
        "DetRe": hc.get("DetRe", 0.0),
        "DetPr": hc.get("DetPr", 0.0),
        "AssRe": hc.get("AssRe", 0.0),
        "AssPr": hc.get("AssPr", 0.0),
        "MOTA": hc.get("MOTA", 0.0),
        "MOTP": hc.get("MOTP", 0.0),
        "IDSw": hc.get("IDSW", 0.0),
        "Frag": hc.get("Frag", 0.0),
        "MT": hc.get("MT", 0.0),
        "PT": hc.get("PT", 0.0),
        "ML": hc.get("ML", 0.0),
        "CLR_TP": hc.get("CLR_TP", 0.0),
        "CLR_FN": hc.get("CLR_FN", 0.0),
        "CLR_FP": hc.get("CLR_FP", 0.0),
        "IDF1": hc.get("IDF1", 0.0),
        "IDP": hc.get("IDP", 0.0),
        "IDR": hc.get("IDR", 0.0),
    }

    # --- TrackEval ---
    data = build_trackeval_data(gt_frames, dt_frames)

    te_hota = TrackEvalHOTA()
    te_hota_res = te_hota.eval_sequence(data)

    te_clear = TrackEvalCLEAR({"THRESHOLD": clear_threshold, "PRINT_CONFIG": False})
    te_clear_res = te_clear.eval_sequence(data)

    te_id = TrackEvalIdentity({"THRESHOLD": clear_threshold, "PRINT_CONFIG": False})
    te_id_res = te_id.eval_sequence(data)

    return hc_flat, None, te_hota_res, te_clear_res, te_id_res


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestPerfectTracking:
    """Both implementations should report perfect scores for identical GT and DT."""

    @pytest.fixture(autouse=True)
    def setup(self):
        frames = {}
        for f in range(1, 11):
            frames[f] = [(1, 100.0, 200.0, 50.0, 80.0), (2, 300.0, 200.0, 50.0, 80.0)]
        self.hc, self.hc_d, self.te_hota, self.te_clear, self.te_id = run_both(frames, frames)

    def test_hota(self):
        te_hota_avg = float(np.mean(self.te_hota["HOTA"]))
        assert abs(self.hc["HOTA"] - te_hota_avg) < TOL, (
            f"HOTA: hotcoco={self.hc['HOTA']:.6f} vs TrackEval={te_hota_avg:.6f}"
        )

    def test_det_a(self):
        te_det_a = float(np.mean(self.te_hota["DetA"]))
        assert abs(self.hc["DetA"] - te_det_a) < TOL

    def test_ass_a(self):
        te_ass_a = float(np.mean(self.te_hota["AssA"]))
        assert abs(self.hc["AssA"] - te_ass_a) < TOL

    def test_mota(self):
        assert abs(self.hc["MOTA"] - self.te_clear["MOTA"]) < TOL

    def test_motp(self):
        assert abs(self.hc["MOTP"] - self.te_clear["MOTP"]) < TOL

    def test_idsw(self):
        assert self.hc["IDSw"] == self.te_clear["IDSW"]

    def test_idf1(self):
        assert abs(self.hc["IDF1"] - self.te_id["IDF1"]) < TOL


class TestNoDetections:
    """No tracker output — everything should be FN."""

    @pytest.fixture(autouse=True)
    def setup(self):
        gt = {1: [(1, 0, 0, 10, 10)], 2: [(1, 5, 0, 10, 10)]}
        dt = {}
        self.hc, self.hc_d, self.te_hota, self.te_clear, self.te_id = run_both(gt, dt)

    def test_hota_zero(self):
        te_hota_avg = float(np.mean(self.te_hota["HOTA"]))
        assert abs(self.hc["HOTA"] - te_hota_avg) < TOL

    def test_mota_zero(self):
        assert abs(self.hc["MOTA"] - self.te_clear["MOTA"]) < TOL

    def test_idf1_zero(self):
        assert abs(self.hc["IDF1"] - self.te_id["IDF1"]) < TOL


class TestIdentitySwitch:
    """GT track 1 is matched to DT track 1 in frame 1, DT track 2 in frame 2."""

    @pytest.fixture(autouse=True)
    def setup(self):
        gt = {1: [(1, 0, 0, 10, 10)], 2: [(1, 0, 0, 10, 10)]}
        dt = {1: [(1, 0, 0, 10, 10)], 2: [(2, 0, 0, 10, 10)]}
        self.hc, self.hc_d, self.te_hota, self.te_clear, self.te_id = run_both(gt, dt)

    def test_hota(self):
        te_hota_avg = float(np.mean(self.te_hota["HOTA"]))
        assert abs(self.hc["HOTA"] - te_hota_avg) < TOL, (
            f"HOTA: hotcoco={self.hc['HOTA']:.6f} vs TrackEval={te_hota_avg:.6f}"
        )

    def test_ass_a(self):
        te_ass_a = float(np.mean(self.te_hota["AssA"]))
        assert abs(self.hc["AssA"] - te_ass_a) < TOL, f"AssA: hotcoco={self.hc['AssA']:.6f} vs TrackEval={te_ass_a:.6f}"

    def test_idsw(self):
        assert self.hc["IDSw"] == self.te_clear["IDSW"]

    def test_mota(self):
        assert abs(self.hc["MOTA"] - self.te_clear["MOTA"]) < TOL

    def test_idf1(self):
        assert abs(self.hc["IDF1"] - self.te_id["IDF1"]) < TOL


class TestPartialOverlap:
    """DT boxes overlap GT but aren't identical (IoU < 1)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        gt = {}
        dt = {}
        for f in range(1, 6):
            gt[f] = [(1, 0, 0, 10, 10), (2, 30, 0, 10, 10)]
            dt[f] = [(1, 3, 0, 10, 10), (2, 33, 0, 10, 10)]
        self.hc, self.hc_d, self.te_hota, self.te_clear, self.te_id = run_both(gt, dt)

    def test_hota(self):
        te_hota_avg = float(np.mean(self.te_hota["HOTA"]))
        assert abs(self.hc["HOTA"] - te_hota_avg) < TOL, (
            f"HOTA: hotcoco={self.hc['HOTA']:.6f} vs TrackEval={te_hota_avg:.6f}"
        )

    def test_loc_a(self):
        te_loc_a = float(np.mean(self.te_hota["LocA"]))
        assert abs(self.hc["LocA"] - te_loc_a) < TOL

    def test_mota(self):
        assert abs(self.hc["MOTA"] - self.te_clear["MOTA"]) < TOL

    def test_motp(self):
        assert abs(self.hc["MOTP"] - self.te_clear["MOTP"]) < TOL

    def test_idf1(self):
        assert abs(self.hc["IDF1"] - self.te_id["IDF1"]) < TOL


class TestMixedScenario:
    """Multiple tracks, some missed frames, one ID switch."""

    @pytest.fixture(autouse=True)
    def setup(self):
        gt = {
            1: [(1, 0, 0, 10, 10), (2, 50, 0, 10, 10)],
            2: [(1, 2, 0, 10, 10), (2, 52, 0, 10, 10)],
            3: [(1, 4, 0, 10, 10), (2, 54, 0, 10, 10)],
            4: [(1, 6, 0, 10, 10), (2, 56, 0, 10, 10)],
            5: [(1, 8, 0, 10, 10), (2, 58, 0, 10, 10)],
        }
        dt = {
            1: [(1, 0, 0, 10, 10), (2, 50, 0, 10, 10)],
            2: [(1, 2, 0, 10, 10)],
            3: [(1, 4, 0, 10, 10), (2, 54, 0, 10, 10)],
            4: [(1, 6, 0, 10, 10), (3, 56, 0, 10, 10)],
            5: [(1, 8, 0, 10, 10), (3, 58, 0, 10, 10)],
        }
        self.hc, self.hc_d, self.te_hota, self.te_clear, self.te_id = run_both(gt, dt)

    def test_hota(self):
        te_hota_avg = float(np.mean(self.te_hota["HOTA"]))
        assert abs(self.hc["HOTA"] - te_hota_avg) < TOL, (
            f"HOTA: hotcoco={self.hc['HOTA']:.6f} vs TrackEval={te_hota_avg:.6f}"
        )

    def test_det_a(self):
        te_det_a = float(np.mean(self.te_hota["DetA"]))
        assert abs(self.hc["DetA"] - te_det_a) < TOL

    def test_ass_a(self):
        te_ass_a = float(np.mean(self.te_hota["AssA"]))
        assert abs(self.hc["AssA"] - te_ass_a) < TOL

    def test_mota(self):
        assert abs(self.hc["MOTA"] - self.te_clear["MOTA"]) < TOL, (
            f"MOTA: hotcoco={self.hc['MOTA']:.6f} vs TrackEval={self.te_clear['MOTA']:.6f}"
        )

    def test_idsw(self):
        assert self.hc["IDSw"] == self.te_clear["IDSW"], (
            f"IDSw: hotcoco={self.hc['IDSw']} vs TrackEval={self.te_clear['IDSW']}"
        )

    def test_idf1(self):
        assert abs(self.hc["IDF1"] - self.te_id["IDF1"]) < TOL, (
            f"IDF1: hotcoco={self.hc['IDF1']:.6f} vs TrackEval={self.te_id['IDF1']:.6f}"
        )


class TestFalsePositivesOnly:
    """No GT, only DT — everything is FP."""

    @pytest.fixture(autouse=True)
    def setup(self):
        gt = {}
        dt = {1: [(1, 0, 0, 10, 10)], 2: [(1, 5, 0, 10, 10)]}
        self.hc, self.hc_d, self.te_hota, self.te_clear, self.te_id = run_both(gt, dt)

    def test_hota_zero(self):
        te_hota_avg = float(np.mean(self.te_hota["HOTA"]))
        assert abs(self.hc["HOTA"] - te_hota_avg) < TOL

    def test_idf1_zero(self):
        assert abs(self.hc["IDF1"] - self.te_id["IDF1"]) < TOL


class TestManyTracksSwapped:
    """3 GT tracks, DT has IDs swapped — Hungarian should resolve globally."""

    @pytest.fixture(autouse=True)
    def setup(self):
        gt = {}
        dt = {}
        for f in range(1, 8):
            gt[f] = [(1, 0, 0, 10, 10), (2, 30, 0, 10, 10), (3, 60, 0, 10, 10)]
            dt[f] = [(3, 0, 0, 10, 10), (1, 30, 0, 10, 10), (2, 60, 0, 10, 10)]
        self.hc, self.hc_d, self.te_hota, self.te_clear, self.te_id = run_both(gt, dt)

    def test_hota(self):
        te_hota_avg = float(np.mean(self.te_hota["HOTA"]))
        assert abs(self.hc["HOTA"] - te_hota_avg) < TOL

    def test_idf1(self):
        assert abs(self.hc["IDF1"] - self.te_id["IDF1"]) < TOL

    def test_mota(self):
        assert abs(self.hc["MOTA"] - self.te_clear["MOTA"]) < TOL
