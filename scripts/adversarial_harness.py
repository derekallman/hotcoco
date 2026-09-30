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

import numpy as np
from helpers import compare_metrics, written_json

# How close an IoU value has to be to a threshold to flag as "boundary jitter"
BOUNDARY_EPS = 1e-6


# ---------------------------------------------------------------------------
# Load fixture — split GT and DT
# ---------------------------------------------------------------------------


def load_fixture(fixture_path):
    """Returns (gt_dataset, detections_list). Both tools read GT from a file, so
    the caller writes it with `helpers.written_json`."""
    with open(fixture_path) as f:
        data = json.load(f)

    detections = data.pop("detections", [])
    return data, detections


# ---------------------------------------------------------------------------
# Run both tools
# ---------------------------------------------------------------------------


def run_hotcoco(gt_path, detections, iou_type):
    import hotcoco as hc

    gt = hc.COCO(gt_path)
    with written_json(detections) as (dt_path,):
        dt = gt.load_res(dt_path)
    ev = hc.COCOeval(gt, dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


def run_pycocotools(gt_path, detections, iou_type):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(gt_path)
    dt = gt.loadRes(detections)
    ev = COCOeval(gt, dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


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


def _check_pair(hc_ei, py_ei, iou_thrs):
    """
    Compare one (image, category, aRng) pair between the two tools.
    Returns a list of issue strings, empty if clean.
    """
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
    hc_dtm = np.array(hc_ei["dtMatches"], dtype=float)  # hotcoco
    py_dtm = np.array(py_ei["dtMatches"], dtype=float)  # pycocotools
    hc_dti = np.array(hc_ei["dtIgnore"], dtype=float)  # (T, D) bool
    py_dti = np.array(py_ei["dtIgnore"], dtype=float)  # (T, D) bool

    # gtIgnore: (G,) bool
    hc_gt_ignore = np.array(hc_ei["gtIgnore"], dtype=float)
    py_gt_ignore = np.array(py_ei["gtIgnore"], dtype=float)

    # --- gtIgnore divergence ---
    for gt_id in gt_ids:
        hi = hc_gt_idx[gt_id]
        pi = py_gt_idx[gt_id]
        hc_v = bool(hc_gt_ignore[hi])
        py_v = bool(py_gt_ignore[pi])
        if hc_v != py_v:
            issues.append(f"    GT ann_id={gt_id}: gtIgnore hotcoco={hc_v}, pycocotools={py_v}")

    # --- dtIgnore and dtMatches divergence, per IoU threshold ---
    for t_idx, thr in enumerate(iou_thrs[: hc_dtm.shape[0]]):
        for dt_id in dt_ids:
            hi = hc_dt_idx[dt_id]
            pi = py_dt_idx[dt_id]

            hc_match = int(hc_dtm[t_idx, hi])
            py_match = int(py_dtm[t_idx, pi])
            hc_ign = bool(hc_dti[t_idx, hi])
            py_ign = bool(py_dti[t_idx, pi])

            # Match divergence: one matched, other didn't (or matched different GT)
            if hc_match != py_match:
                note = _boundary_note(hc_ei, hc_dt_idx[dt_id], thr)
                issues.append(
                    f"    IoU@{thr:.2f} DT ann_id={dt_id}: "
                    f"matched GT hotcoco={hc_match or 'none'}, "
                    f"pycocotools={py_match or 'none'}" + (f"  [{note}]" if note else "")
                )

            # dtIgnore divergence
            if hc_ign != py_ign:
                issues.append(f"    IoU@{thr:.2f} DT ann_id={dt_id}: dtIgnore hotcoco={hc_ign}, pycocotools={py_ign}")

    return issues


def _boundary_note(hc_ei, d, thr):
    """
    If the disagreement is on DT row `d` whose best IoU is within BOUNDARY_EPS of
    the threshold, flag it as potential float jitter rather than a real bug.
    Requires 'ious' to be present in the eval_img (not always available).
    """
    hc_ious = hc_ei.get("ious")
    if hc_ious is None:
        return None
    hc_ious = np.array(hc_ious)  # (D, G)
    if d >= hc_ious.shape[0]:
        return None
    row = hc_ious[d]
    best = row.max() if row.size > 0 else 0.0
    if abs(best - thr) < BOUNDARY_EPS:
        return f"⚠ IoU={best:.8f} ≈ threshold — likely float jitter, not a real bug"
    return None


def _is_inert(ei):
    """True when this eval_img cannot influence any metric.

    hotcoco drives evaluation from the annotation index and skips (image,
    category, area) pairs with nothing to score, where pycocotools emits an entry
    regardless. Those entries are not always *empty* — a detection outside the
    area range still appears, with `dtIgnore` set — but they are inert: an ignored
    detection is neither a TP nor an FP, and an ignored GT never enters `num_gt`.
    Accumulation reaches the same numbers either way.

    So "present in pycocotools, missing from hotcoco" is only a divergence when
    the entry could actually have contributed. Flagging every skipped pair made
    level 2 fire on three fixtures whose metrics agree exactly, which is the kind
    of noise that gets a check switched off.
    """
    if ei is None:
        return True

    # pycocotools hands back numpy arrays here, so `or []` and bare truthiness
    # both raise "truth value of an empty array is ambiguous". Normalize first.
    def flatten(value):
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        out = []
        for v in value:
            out.extend(v if isinstance(v, (list, tuple)) else [v])
        return out

    gt_ids = flatten(ei.get("gtIds"))
    gt_ignore = flatten(ei.get("gtIgnore"))
    has_live_gt = any(not bool(g) for g in gt_ignore) if gt_ignore else bool(gt_ids)

    dt_ids = flatten(ei.get("dtIds"))
    dt_ignore = flatten(ei.get("dtIgnore"))
    has_live_dt = any(not bool(v) for v in dt_ignore) if dt_ignore else bool(dt_ids)

    return not (has_live_gt or has_live_dt)


def compare_eval_imgs(hc_ev, py_ev):
    """
    Compare eval_imgs from both tools. Returns list of dicts describing each
    divergent (image, category, aRng) pair with specific annotation-level details.
    """
    hc_idx = _index_eval_imgs(hc_ev.eval_imgs)
    py_idx = _index_eval_imgs(py_ev.evalImgs)
    # The grid both evaluators actually ran, so a non-default `iouThrs` labels
    # rows correctly instead of by a second copy of the default.
    iou_thrs = list(hc_ev.params.iouThrs)

    divergences = []
    for key in sorted(set(hc_idx) | set(py_idx)):
        image_id, category_id, aRng = key
        if key not in py_idx:
            issues = ["    pair present in hotcoco but missing from pycocotools"]
        elif key not in hc_idx:
            # Only a divergence if the entry could have contributed — see _is_inert.
            issues = [] if _is_inert(py_idx[key]) else ["    pair present in pycocotools but missing from hotcoco"]
        else:
            issues = _check_pair(hc_idx[key], py_idx[key], iou_thrs)
        if issues:
            divergences.append({"image_id": image_id, "category_id": category_id, "aRng": list(aRng), "issues": issues})

    return divergences


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_metric_failures(mismatches):
    """Print `helpers.MetricMismatch` records."""
    print("\n[LEVEL 1] METRIC DIVERGENCES:")
    for m in mismatches:
        print(f"  [{m.index}] {m.name:6s}: hotcoco={m.rs:.6f}  pycocotools={m.py:.6f}  diff={m.diff:.2e}")


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
    ap.add_argument(
        "--eval-imgs-only", action="store_true", help="Skip metric check; run only the eval_imgs comparison"
    )
    ap.add_argument("--metrics-only", action="store_true", help="Skip the per-match comparison")
    ap.add_argument("--max-divergences", type=int, default=20, help="Max eval_img pairs to print")
    args = ap.parse_args()

    fixture_path = args.fixture

    print(f"Fixture: {fixture_path}")
    print(f"IoU type: {args.iou_type}  metric threshold: {args.metric_thr}")

    gt_data, detections = load_fixture(fixture_path)
    with written_json(gt_data) as (gt_path,):
        hc_ev = run_hotcoco(gt_path, detections, args.iou_type)
        py_ev = run_pycocotools(gt_path, detections, args.iou_type)

    found_issue = False

    # Level 1. The comparison itself — the -1.0 "not computed" sentinel rule and
    # the length check — is `helpers.compare_metrics`, shared with
    # test_parity/fuzz_parity. Only the threshold is local: this harness is a
    # diagnostic tool with a deliberately looser default than the CI gate.
    if not args.eval_imgs_only:
        failures = compare_metrics(py_ev.stats, hc_ev.stats, hc_ev.metric_keys(), tolerance=args.metric_thr)
        if failures:
            print_metric_failures(failures)
            found_issue = True
        else:
            print("\n[LEVEL 1] Metrics OK — all within threshold")

    # Level 2 runs whether or not level 1 failed. It is uniquely good at a
    # matching divergence that cancels in the integral: two detections swapped
    # between images, or a crowd flag applied to the wrong annotation, can leave
    # AP identical to fifteen decimal places while every per-match decision
    # underneath is wrong. A check that only runs once something else has
    # already noticed is not a check.
    if not args.metrics_only:
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
