"""
adversarial_harness.py — two-level parity checker for hotcoco vs pycocotools.

Level 1 (metric): compare final AP/AR stats. Fast, coarse.
Level 2 (eval_imgs): compare per-(image, category, area_rng) matching decisions.
  Fires automatically when level 1 finds a diff, to pinpoint the exact annotation.

Fixture format (JSON):
    {
      "images": [...],
      "categories": [...],
      "annotations": [...],   // ground truth
      "detections": [...]     // predictions — list of COCO result dicts
    }

Usage:
    python adversarial_harness.py fixture.json
    python adversarial_harness.py fixture.json --iou-type segm --metric-thr 2e-4
    python adversarial_harness.py fixture.json --eval-imgs-only
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# IoU thresholds used by COCO eval (np.linspace(0.5, 0.95, 10))
IOU_THRS = np.linspace(0.5, 0.95, 10).round(2).tolist()

# How close an IoU value has to be to a threshold to flag as "boundary jitter"
BOUNDARY_EPS = 1e-6


# ---------------------------------------------------------------------------
# Load fixture — split GT and DT, write GT to a temp file
# ---------------------------------------------------------------------------


def load_fixture(fixture_path):
    """
    Returns (gt_tmp_path, detections_list, tmpfile_handle).
    Caller must close tmpfile_handle when done.
    """
    with open(fixture_path) as f:
        data = json.load(f)

    detections = data.pop("detections", [])

    # Write GT-only COCO JSON to a temp file both tools can read
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.flush()
    return tmp.name, detections, tmp


# ---------------------------------------------------------------------------
# Run both tools
# ---------------------------------------------------------------------------


def _write_tmp_json(data):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.flush()
    return tmp


def run_hotcoco(gt_path, detections, iou_type):
    import hotcoco as hc

    gt = hc.COCO(gt_path)
    dt_tmp = _write_tmp_json(detections)
    try:
        dt = gt.load_res(dt_tmp.name)
    finally:
        dt_tmp.close()
        Path(dt_tmp.name).unlink(missing_ok=True)
    ev = hc.COCOeval(gt, dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


def run_pycocotools(gt_path, detections, iou_type):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(gt_path)
    dt = gt.loadRes(detections) if detections else gt.loadRes([])
    ev = COCOeval(gt, dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


# ---------------------------------------------------------------------------
# Level 1: metric comparison
# ---------------------------------------------------------------------------


def compare_metrics(hc_ev, py_ev, threshold):
    """Return dict of metric_name → (hc_val, py_val, diff) for diffs > threshold."""
    metric_names = hc_ev.metric_keys()
    hc_stats = dict(zip(metric_names, hc_ev.stats))
    py_stats = dict(zip(metric_names, py_ev.stats))
    failures = {}
    for k in py_stats:
        hc_v = hc_stats.get(k, float("nan"))
        py_v = py_stats.get(k, float("nan"))
        diff = abs(hc_v - py_v)
        if diff > threshold:
            failures[k] = (hc_v, py_v, diff)
    return failures


# ---------------------------------------------------------------------------
# Level 2: eval_imgs comparison
# ---------------------------------------------------------------------------


def _index_eval_imgs(eval_imgs):
    """Build (image_id, category_id, aRng_tuple) → entry dict."""
    idx = {}
    for ei in eval_imgs:
        if ei is None:
            continue
        key = (ei["image_id"], ei["category_id"], tuple(ei["aRng"]))
        idx[key] = ei
    return idx


def _to_array(v):
    """Coerce list-of-lists or ndarray to float64 ndarray."""
    return np.array(v, dtype=float)


def _check_pair(key, hc_ei, py_ei):
    """
    Compare one (image, category, aRng) pair between the two tools.
    Returns a list of issue strings, empty if clean.
    """
    image_id, category_id, aRng = key
    issues = []

    hc_dt_ids = list(hc_ei["dtIds"])
    py_dt_ids = list(py_ei["dtIds"])
    hc_gt_ids = list(hc_ei["gtIds"])
    py_gt_ids = list(py_ei["gtIds"])

    # Check the DT/GT sets are the same — if not, something is very wrong upstream
    if set(hc_dt_ids) != set(py_dt_ids):
        issues.append(
            f"    DT ID sets differ:\n      hotcoco={sorted(hc_dt_ids)}\n      pycocotools={sorted(py_dt_ids)}"
        )
        return issues  # can't align further

    if set(hc_gt_ids) != set(py_gt_ids):
        issues.append(
            f"    GT ID sets differ:\n      hotcoco={sorted(hc_gt_ids)}\n      pycocotools={sorted(py_gt_ids)}"
        )
        return issues

    # Build ID → index maps for alignment (indices within this eval_img)
    hc_dt_idx = {ann_id: i for i, ann_id in enumerate(hc_dt_ids)}
    py_dt_idx = {ann_id: i for i, ann_id in enumerate(py_dt_ids)}
    hc_gt_idx = {ann_id: i for i, ann_id in enumerate(hc_gt_ids)}
    py_gt_idx = {ann_id: i for i, ann_id in enumerate(py_gt_ids)}

    dt_ids = sorted(set(hc_dt_ids))
    gt_ids = sorted(set(hc_gt_ids))

    # dtMatches: (T, D) — matched GT ID or 0
    hc_dtm = _to_array(hc_ei["dtMatches"])  # hotcoco
    py_dtm = _to_array(py_ei["dtMatches"])  # pycocotools
    hc_dti = _to_array(hc_ei["dtIgnore"])  # (T, D) bool
    py_dti = _to_array(py_ei["dtIgnore"])  # (T, D) bool

    # gtIgnore: (G,) bool
    hc_gti = _to_array(hc_ei["gtIgnore"])
    py_gti = _to_array(py_ei["gtIgnore"])

    # --- gtIgnore divergence ---
    for gt_id in gt_ids:
        hi = hc_gt_idx[gt_id]
        pi = py_gt_idx[gt_id]
        hc_v = bool(hc_gti[hi])
        py_v = bool(py_gti[pi])
        if hc_v != py_v:
            issues.append(f"    GT ann_id={gt_id}: gtIgnore hotcoco={hc_v}, pycocotools={py_v}")

    # --- dtIgnore and dtMatches divergence, per IoU threshold ---
    for t_idx, thr in enumerate(IOU_THRS):
        if t_idx >= hc_dtm.shape[0]:
            break
        for dt_id in dt_ids:
            hi = hc_dt_idx[dt_id]
            pi = py_dt_idx[dt_id]

            hc_match = int(hc_dtm[t_idx, hi])
            py_match = int(py_dtm[t_idx, pi])
            hc_ign = bool(hc_dti[t_idx, hi])
            py_ign = bool(py_dti[t_idx, pi])

            # Match divergence: one matched, other didn't (or matched different GT)
            if hc_match != py_match:
                note = _boundary_note(hc_ei, py_ei, hc_dt_idx, py_dt_idx, hc_gt_idx, py_gt_idx, dt_id, gt_ids, thr)
                issues.append(
                    f"    IoU@{thr:.2f} DT ann_id={dt_id}: "
                    f"matched GT hotcoco={hc_match or 'none'}, "
                    f"pycocotools={py_match or 'none'}" + (f"  [{note}]" if note else "")
                )

            # dtIgnore divergence
            if hc_ign != py_ign:
                issues.append(f"    IoU@{thr:.2f} DT ann_id={dt_id}: dtIgnore hotcoco={hc_ign}, pycocotools={py_ign}")

    return issues


def _boundary_note(hc_ei, py_ei, hc_dt_idx, py_dt_idx, hc_gt_idx, py_gt_idx, dt_id, gt_ids, thr):
    """
    If the disagreement is on a DT whose best IoU is within BOUNDARY_EPS of the
    threshold, flag it as potential float jitter rather than a real bug.
    Requires 'ious' to be present in the eval_img (not always available).
    """
    hc_ious = hc_ei.get("ious")
    if hc_ious is None:
        return None
    hc_ious = np.array(hc_ious)  # (D, G)
    d = hc_dt_idx[dt_id]
    if d >= hc_ious.shape[0]:
        return None
    row = hc_ious[d]
    best = row.max() if row.size > 0 else 0.0
    if abs(best - thr) < BOUNDARY_EPS:
        return f"⚠ IoU={best:.8f} ≈ threshold — likely float jitter, not a real bug"
    return None


def compare_eval_imgs(hc_ev, py_ev):
    """
    Compare eval_imgs from both tools. Returns list of dicts describing each
    divergent (image, category, aRng) pair with specific annotation-level details.
    """
    hc_idx = _index_eval_imgs(hc_ev.eval_imgs)
    py_idx = _index_eval_imgs(py_ev.evalImgs)

    all_keys = set(hc_idx) | set(py_idx)
    divergences = []

    for key in sorted(all_keys):
        image_id, category_id, aRng = key

        in_hc = key in hc_idx
        in_py = key in py_idx

        if in_hc and not in_py:
            divergences.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "aRng": aRng,
                    "issues": ["    pair present in hotcoco but missing from pycocotools"],
                }
            )
            continue

        if in_py and not in_hc:
            py_ei = py_idx[key]
            # Only flag if it's non-trivial (has DTs or GTs)
            if py_ei and (py_ei.get("dtIds") or py_ei.get("gtIds")):
                divergences.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "aRng": aRng,
                        "issues": ["    pair present in pycocotools but missing from hotcoco"],
                    }
                )
            continue

        issues = _check_pair(key, hc_idx[key], py_idx[key])
        if issues:
            divergences.append({"image_id": image_id, "category_id": category_id, "aRng": list(aRng), "issues": issues})

    return divergences


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_metric_failures(failures):
    print("\n[LEVEL 1] METRIC DIVERGENCES:")
    for name, (hc_v, py_v, diff) in failures.items():
        print(f"  {name:6s}: hotcoco={hc_v:.6f}  pycocotools={py_v:.6f}  diff={diff:.2e}")


def print_eval_img_divergences(divergences, limit=20):
    print(f"\n[LEVEL 2] EVAL_IMG DIVERGENCES ({len(divergences)} pairs):")
    if not divergences:
        print("  none — matching decisions are identical")
        return
    shown = 0
    for d in divergences:
        if shown >= limit:
            print(f"  ... ({len(divergences) - limit} more pairs omitted)")
            break
        print(f"\n  image_id={d['image_id']}  category_id={d['category_id']}  aRng={d['aRng']}:")
        for issue in d["issues"]:
            print(issue)
        shown += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture", help="Path to fixture JSON")
    ap.add_argument("--iou-type", default="bbox", choices=["bbox", "segm", "keypoints"])
    ap.add_argument(
        "--metric-thr", type=float, default=1e-4, help="Metric diff threshold to flag (default 1e-4; use 2e-4 for segm)"
    )
    ap.add_argument("--eval-imgs-only", action="store_true", help="Skip metric check; always run eval_imgs comparison")
    ap.add_argument("--max-divergences", type=int, default=20, help="Max eval_img pairs to print")
    args = ap.parse_args()

    fixture_path = args.fixture

    print(f"Fixture: {fixture_path}")
    print(f"IoU type: {args.iou_type}  metric threshold: {args.metric_thr}")

    gt_path, detections, tmp = load_fixture(fixture_path)
    try:
        hc_ev = run_hotcoco(gt_path, detections, args.iou_type)
        py_ev = run_pycocotools(gt_path, detections, args.iou_type)
    finally:
        tmp.close()
        Path(gt_path).unlink(missing_ok=True)

    found_issue = False

    # Level 1
    if not args.eval_imgs_only:
        failures = compare_metrics(hc_ev, py_ev, args.metric_thr)
        if failures:
            print_metric_failures(failures)
            found_issue = True
        else:
            print("\n[LEVEL 1] Metrics OK — all within threshold")

    # Level 2: run if metric diff found, or if --eval-imgs-only
    if found_issue or args.eval_imgs_only:
        divergences = compare_eval_imgs(hc_ev, py_ev)
        print_eval_img_divergences(divergences, limit=args.max_divergences)
        if divergences:
            found_issue = True

    if found_issue:
        print("\nRESULT: DIVERGENCE FOUND")
        sys.exit(1)
    else:
        print("\nRESULT: OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
