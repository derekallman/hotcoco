"""Build a lightweight TP/FP/FN lookup from COCOeval results.

Used by the browse viewer to color detections by evaluation status.
"""
from __future__ import annotations


def build_eval_index(coco_eval, iou_thr: float = 0.5) -> dict:
    """Extract per-annotation TP/FP/FN status from eval results.

    Parameters
    ----------
    coco_eval : COCOeval
        A COCOeval object that has already had ``evaluate()`` called.
    iou_thr : float
        IoU threshold for TP/FP classification (default 0.5).

    Returns
    -------
    dict
        Keys: ``dt_status``, ``gt_status``, ``dt_match``, ``gt_match``,
        ``img_summary``, ``iou_thr``.
    """
    eval_imgs = coco_eval.eval_imgs
    iou_thrs = list(coco_eval.params.iou_thrs)

    # Find the closest IoU threshold index
    t_idx = min(range(len(iou_thrs)), key=lambda i: abs(iou_thrs[i] - iou_thr))

    # "all" area range: [0, 1e10]
    all_rng = [0, 1e10]

    dt_status: dict[int, str] = {}   # ann_id -> "tp" | "fp"
    gt_status: dict[int, str] = {}   # ann_id -> "matched" | "fn"
    dt_match: dict[int, int] = {}    # dt_id -> gt_id (TP pairs)
    gt_match: dict[int, int] = {}    # gt_id -> dt_id (reverse)
    img_summary: dict[int, dict[str, int]] = {}

    for ei in eval_imgs:
        if ei is None:
            continue

        # Filter to "all" area range only
        a_rng = ei.get("aRng", [])
        if len(a_rng) >= 2 and not (a_rng[0] == all_rng[0] and a_rng[1] == all_rng[1]):
            continue

        img_id = ei["image_id"]
        dt_ids = ei.get("dtIds", [])
        gt_ids = ei.get("gtIds", [])
        dt_matches = ei.get("dtMatches", [])
        dt_ignore = ei.get("dtIgnore", [])
        gt_matched = ei.get("gtMatched", [])
        gt_ignore = ei.get("gtIgnore", [])

        if not dt_matches or t_idx >= len(dt_matches):
            # No threshold data — mark all non-ignored GTs as FN
            for g, gid in enumerate(gt_ids):
                if gid not in gt_status and not (g < len(gt_ignore) and gt_ignore[g]):
                    gt_status[gid] = "fn"
            continue

        matches_at_t = dt_matches[t_idx]
        ignore_at_t = dt_ignore[t_idx] if t_idx < len(dt_ignore) else []
        gt_matched_at_t = gt_matched[t_idx] if t_idx < len(gt_matched) else []

        tp = 0
        fp = 0
        fn = 0

        # Classify detections
        for d, did in enumerate(dt_ids):
            if did in dt_status:
                continue  # already classified from another area range
            ignored = d < len(ignore_at_t) and ignore_at_t[d]
            if ignored:
                continue
            matched_gt = matches_at_t[d] if d < len(matches_at_t) else 0
            if matched_gt > 0:
                dt_status[did] = "tp"
                dt_match[did] = matched_gt
                gt_match[matched_gt] = did
                tp += 1
            else:
                dt_status[did] = "fp"
                fp += 1

        # Classify ground truths
        for g, gid in enumerate(gt_ids):
            if gid in gt_status:
                continue
            if g < len(gt_ignore) and gt_ignore[g]:
                continue
            matched = g < len(gt_matched_at_t) and gt_matched_at_t[g]
            if matched:
                gt_status[gid] = "matched"
            else:
                gt_status[gid] = "fn"
                fn += 1

        # Accumulate per-image summary
        if img_id not in img_summary:
            img_summary[img_id] = {"tp": 0, "fp": 0, "fn": 0}
        img_summary[img_id]["tp"] += tp
        img_summary[img_id]["fp"] += fp
        img_summary[img_id]["fn"] += fn

    return {
        "dt_status": dt_status,
        "gt_status": gt_status,
        "dt_match": dt_match,
        "gt_match": gt_match,
        "img_summary": img_summary,
        "iou_thr": iou_thrs[t_idx],
    }
