"""Fuzz parity testing: hotcoco tracking vs TrackEval.

Generates edge-case MOT sequences and random fuzz sequences, comparing all
metrics between both implementations. Prints FOUND_BUG for any divergence.

Usage:
    uv run python scripts/fuzz_track_parity.py
"""

import sys
import traceback

sys.path.insert(0, "scripts")

import numpy as np
from test_track_parity import run_both

TOL = 1e-6
bugs_found = []
scenarios_run = 0


def compare(hc, hc_d, te_hota, te_clear, te_id, scenario_name, gt, dt):
    """Compare all metrics between hotcoco and TrackEval. Return list of bug dicts."""
    found = []

    # HOTA family
    for key in ["HOTA", "DetA", "AssA", "LocA"]:
        te_val = float(np.mean(te_hota[key]))
        hc_val = hc[key]
        if abs(hc_val - te_val) > TOL:
            found.append(
                {
                    "scenario": scenario_name,
                    "metric": key,
                    "hotcoco": hc_val,
                    "trackeval": te_val,
                    "diff": abs(hc_val - te_val),
                    "gt": gt,
                    "dt": dt,
                }
            )

    # CLEAR floats
    for hc_key, te_key in [("MOTA", "MOTA"), ("MOTP", "MOTP")]:
        hc_val = hc[hc_key]
        te_val = te_clear[te_key]
        if abs(hc_val - te_val) > TOL:
            found.append(
                {
                    "scenario": scenario_name,
                    "metric": hc_key,
                    "hotcoco": hc_val,
                    "trackeval": te_val,
                    "diff": abs(hc_val - te_val),
                    "gt": gt,
                    "dt": dt,
                }
            )

    # CLEAR integers
    for hc_key, te_key in [("IDSw", "IDSW"), ("Frag", "Frag"), ("MT", "MT"), ("ML", "ML")]:
        hc_val = int(hc[hc_key])
        te_val = int(te_clear[te_key])
        if hc_val != te_val:
            found.append(
                {
                    "scenario": scenario_name,
                    "metric": hc_key,
                    "hotcoco": hc_val,
                    "trackeval": te_val,
                    "diff": abs(hc_val - te_val),
                    "gt": gt,
                    "dt": dt,
                }
            )

    # Identity
    for key in ["IDF1", "IDP", "IDR"]:
        hc_val = hc[key]
        te_val = te_id[key]
        if abs(hc_val - te_val) > TOL:
            found.append(
                {
                    "scenario": scenario_name,
                    "metric": key,
                    "hotcoco": hc_val,
                    "trackeval": te_val,
                    "diff": abs(hc_val - te_val),
                    "gt": gt,
                    "dt": dt,
                }
            )

    return found


def run_scenario(name, gt, dt):
    """Run a single scenario and check for divergence."""
    global scenarios_run, bugs_found
    scenarios_run += 1
    try:
        hc, hc_d, te_hota, te_clear, te_id = run_both(gt, dt)
        found = compare(hc, hc_d, te_hota, te_clear, te_id, name, gt, dt)
        if found:
            bugs_found.extend(found)
            for b in found:
                print(
                    f"FOUND_BUG: {b['scenario']} | {b['metric']} | "
                    f"hc={b['hotcoco']} te={b['trackeval']} diff={b['diff']:.2e}"
                )
    except Exception as e:
        print(f"ERROR in {name}: {e}")
        traceback.print_exc()


# ===========================================================================
# Phase 2: Hand-crafted adversarial scenarios
# ===========================================================================

print("=" * 70)
print("Phase 2: Hand-crafted adversarial scenarios")
print("=" * 70)

# 1. Empty frames — frames with GT but no DT, or DT but no GT, or both empty
run_scenario("empty_both", {1: []}, {1: []})
run_scenario("empty_gt_frames", {1: [(1, 0, 0, 10, 10)], 2: []}, {1: [(1, 0, 0, 10, 10)]})
run_scenario("gt_only_some_frames", {1: [(1, 0, 0, 10, 10)], 2: [(1, 5, 0, 10, 10)]}, {2: [(1, 5, 0, 10, 10)]})
run_scenario("dt_extra_frame", {1: [(1, 0, 0, 10, 10)]}, {1: [(1, 0, 0, 10, 10)], 2: [(1, 50, 0, 10, 10)]})

# 2. Single-frame tracks
run_scenario("single_frame_single_track", {1: [(1, 0, 0, 10, 10)]}, {1: [(1, 0, 0, 10, 10)]})
run_scenario(
    "single_frame_multi_track",
    {1: [(1, 0, 0, 10, 10), (2, 50, 0, 10, 10)]},
    {1: [(1, 0, 0, 10, 10), (2, 50, 0, 10, 10)]},
)

# 3. All-same-ID — every DT box uses the same track_id
run_scenario(
    "all_same_dt_id",
    {f: [(1, 0, 0, 10, 10), (2, 50, 0, 10, 10)] for f in range(1, 6)},
    {f: [(1, 0, 0, 10, 10), (1, 50, 0, 10, 10)] for f in range(1, 6)},
)

# 4. Many ID switches — GT track 1 is matched to a different DT track every frame
run_scenario(
    "id_switch_every_frame",
    {f: [(1, 0, 0, 10, 10)] for f in range(1, 8)},
    {f: [(f, 0, 0, 10, 10)] for f in range(1, 8)},
)

# 5. Near-threshold IoU — boxes with IoU very close to 0.5
# IoU of two 10x10 boxes shifted by dx: intersection = (10-dx)*10, union = 2*100 - (10-dx)*10
# For IoU = 0.5: 10*(10-dx) / (200 - 10*(10-dx)) = 0.5 => 10-dx = 200/30 = 6.667 => dx = 3.333
run_scenario(
    "iou_just_above_0.5",
    {f: [(1, 0, 0, 10, 10)] for f in range(1, 6)},
    {f: [(1, 3.33, 0, 10, 10)] for f in range(1, 6)},
)  # IoU ~= 0.5008
run_scenario(
    "iou_just_below_0.5",
    {f: [(1, 0, 0, 10, 10)] for f in range(1, 6)},
    {f: [(1, 3.34, 0, 10, 10)] for f in range(1, 6)},
)  # IoU ~= 0.4993

# 6. Large track count — 20+ tracks across 50+ frames
gt_large = {}
dt_large = {}
for f in range(1, 51):
    gt_large[f] = [(tid, tid * 30.0, 0, 10, 10) for tid in range(1, 22)]
    dt_large[f] = [(tid, tid * 30.0 + 1.0, 0, 10, 10) for tid in range(1, 22)]
run_scenario("large_20tracks_50frames", gt_large, dt_large)

# 7. Fragmented tracks — GT track appears, disappears for several frames, reappears
gt_frag = {}
dt_frag = {}
for f in [1, 2, 3, 7, 8, 9, 15, 16]:
    gt_frag[f] = [(1, 0, 0, 10, 10)]
    dt_frag[f] = [(1, 0, 0, 10, 10)]
run_scenario("fragmented_track", gt_frag, dt_frag)

gt_frag2 = {}
dt_frag2 = {}
for f in range(1, 11):
    gt_frag2[f] = [(1, 0, 0, 10, 10)]
for f in [1, 2, 5, 6, 9, 10]:
    dt_frag2[f] = [(1, 0, 0, 10, 10)]
run_scenario("fragmented_dt_gaps", gt_frag2, dt_frag2)

# 8. Overlapping boxes within a frame — multiple GT or DT boxes at similar positions
run_scenario(
    "overlapping_gt_boxes",
    {1: [(1, 0, 0, 10, 10), (2, 2, 0, 10, 10)], 2: [(1, 0, 0, 10, 10), (2, 2, 0, 10, 10)]},
    {1: [(1, 0, 0, 10, 10), (2, 2, 0, 10, 10)], 2: [(1, 0, 0, 10, 10), (2, 2, 0, 10, 10)]},
)

run_scenario(
    "overlapping_dt_3_boxes",
    {1: [(1, 0, 0, 10, 10)], 2: [(1, 0, 0, 10, 10)]},
    {
        1: [(1, 0, 0, 10, 10), (2, 1, 0, 10, 10), (3, 2, 0, 10, 10)],
        2: [(1, 0, 0, 10, 10), (2, 1, 0, 10, 10), (3, 2, 0, 10, 10)],
    },
)

# 9. Sparse frames — frames 1, 100, 200 (large gaps in frame numbering)
run_scenario(
    "sparse_frames",
    {1: [(1, 0, 0, 10, 10)], 100: [(1, 50, 0, 10, 10)], 200: [(1, 100, 0, 10, 10)]},
    {1: [(1, 0, 0, 10, 10)], 100: [(1, 50, 0, 10, 10)], 200: [(1, 100, 0, 10, 10)]},
)

run_scenario(
    "sparse_frames_with_switch",
    {1: [(1, 0, 0, 10, 10)], 100: [(1, 50, 0, 10, 10)], 200: [(1, 100, 0, 10, 10)]},
    {1: [(1, 0, 0, 10, 10)], 100: [(2, 50, 0, 10, 10)], 200: [(3, 100, 0, 10, 10)]},
)

# 10. One-to-many confusion — one GT box overlaps with 3 DT boxes above threshold
run_scenario(
    "one_gt_three_dt",
    {f: [(1, 0, 0, 10, 10)] for f in range(1, 6)},
    {f: [(1, 0, 0, 10, 10), (2, 1, 0, 10, 10), (3, 2, 0, 10, 10)] for f in range(1, 6)},
)

# 11. Asymmetric tracks — GT has 10 tracks, DT has 2 (or vice versa)
run_scenario(
    "gt10_dt2",
    {f: [(tid, tid * 25.0, 0, 10, 10) for tid in range(1, 11)] for f in range(1, 6)},
    {f: [(1, 25.0, 0, 10, 10), (2, 50.0, 0, 10, 10)] for f in range(1, 6)},
)

run_scenario(
    "gt2_dt10",
    {f: [(1, 25.0, 0, 10, 10), (2, 50.0, 0, 10, 10)] for f in range(1, 6)},
    {f: [(tid, tid * 25.0, 0, 10, 10) for tid in range(1, 11)] for f in range(1, 6)},
)

# 12. Zero-area boxes — width or height is 0
run_scenario("zero_width_box", {1: [(1, 0, 0, 0, 10)]}, {1: [(1, 0, 0, 0, 10)]})

run_scenario("zero_height_box", {1: [(1, 0, 0, 10, 0)]}, {1: [(1, 0, 0, 10, 0)]})

# Extra adversarial cases
# 13. Duplicate track IDs in a single frame (same ID twice)
run_scenario(
    "duplicate_gt_id_in_frame",
    {1: [(1, 0, 0, 10, 10), (1, 50, 0, 10, 10)], 2: [(1, 0, 0, 10, 10), (1, 50, 0, 10, 10)]},
    {1: [(1, 0, 0, 10, 10), (2, 50, 0, 10, 10)], 2: [(1, 0, 0, 10, 10), (2, 50, 0, 10, 10)]},
)

# 14. Many-to-many: equal number of tracks, all overlapping
run_scenario(
    "many_to_many_overlap",
    {f: [(1, 0, 0, 20, 20), (2, 5, 0, 20, 20), (3, 10, 0, 20, 20)] for f in range(1, 8)},
    {f: [(1, 2, 0, 20, 20), (2, 7, 0, 20, 20), (3, 12, 0, 20, 20)] for f in range(1, 8)},
)

# 15. Track that appears in frame 1 only in GT, then DT has it in all frames
run_scenario("gt_short_dt_long", {1: [(1, 0, 0, 10, 10)]}, {f: [(1, 0, 0, 10, 10)] for f in range(1, 11)})

# 16. All FP — DT has tracks GT doesn't
run_scenario(
    "all_fp_different_location",
    {f: [(1, 0, 0, 10, 10)] for f in range(1, 6)},
    {f: [(1, 100, 0, 10, 10)] for f in range(1, 6)},
)

# 17. Two tracks crossing paths (GT1 goes left->right, GT2 goes right->left)
gt_cross = {}
dt_cross = {}
for f in range(1, 11):
    x1 = f * 5.0  # GT1 moves right
    x2 = 50.0 - f * 5.0  # GT2 moves left
    gt_cross[f] = [(1, x1, 0, 10, 10), (2, x2, 0, 10, 10)]
    dt_cross[f] = [(1, x1, 0, 10, 10), (2, x2, 0, 10, 10)]
run_scenario("crossing_tracks", gt_cross, dt_cross)

# 18. Track swap at crossing point
gt_swap = {}
dt_swap = {}
for f in range(1, 11):
    x1 = f * 5.0
    x2 = 50.0 - f * 5.0
    gt_swap[f] = [(1, x1, 0, 10, 10), (2, x2, 0, 10, 10)]
    # DT swaps IDs at frame 5
    if f < 5:
        dt_swap[f] = [(1, x1, 0, 10, 10), (2, x2, 0, 10, 10)]
    else:
        dt_swap[f] = [(2, x1, 0, 10, 10), (1, x2, 0, 10, 10)]
run_scenario("track_swap_at_cross", gt_swap, dt_swap)

# 19. Very tiny boxes
run_scenario(
    "tiny_boxes", {f: [(1, 0, 0, 0.01, 0.01)] for f in range(1, 4)}, {f: [(1, 0, 0, 0.01, 0.01)] for f in range(1, 4)}
)

# 20. Mixed: some frames matched, some only GT, some only DT
run_scenario(
    "mixed_coverage",
    {1: [(1, 0, 0, 10, 10)], 2: [(1, 5, 0, 10, 10)], 3: [(1, 10, 0, 10, 10)], 4: [(1, 15, 0, 10, 10)]},
    {
        2: [(1, 5, 0, 10, 10)],
        3: [(2, 10, 0, 10, 10)],  # ID switch
        5: [(1, 50, 0, 10, 10)],
    },
)  # extra frame


print(f"\nPhase 2 complete: {scenarios_run} hand-crafted scenarios")
handcrafted_count = scenarios_run

# ===========================================================================
# Phase 3: Randomized fuzzing
# ===========================================================================

print("\n" + "=" * 70)
print("Phase 3: Randomized fuzzing (200 sequences)")
print("=" * 70)

rng = np.random.RandomState(42)

for trial in range(200):
    num_gt_tracks = rng.randint(1, 10)
    num_dt_tracks = rng.randint(1, 10)
    num_frames = rng.randint(2, 20)

    # Generate base positions for each track
    gt_base_pos = {tid: (rng.uniform(0, 200), rng.uniform(0, 200)) for tid in range(1, num_gt_tracks + 1)}
    dt_base_pos = {}
    for tid in range(1, num_dt_tracks + 1):
        # Some DT tracks match GT positions, some are random
        if tid <= num_gt_tracks and rng.random() < 0.7:
            bx, by = gt_base_pos[tid]
            dt_base_pos[tid] = (bx + rng.uniform(-3, 3), by + rng.uniform(-3, 3))
        else:
            dt_base_pos[tid] = (rng.uniform(0, 200), rng.uniform(0, 200))

    gt = {}
    dt = {}

    box_w = rng.uniform(5, 30)
    box_h = rng.uniform(5, 30)

    for f in range(1, num_frames + 1):
        gt_boxes = []
        for tid in range(1, num_gt_tracks + 1):
            if rng.random() < 0.7:  # 70% chance of appearing
                bx, by = gt_base_pos[tid]
                # Small per-frame perturbation
                px = bx + rng.uniform(-2, 2) + f * rng.uniform(-1, 1)
                py = by + rng.uniform(-2, 2)
                gt_boxes.append((tid, px, py, box_w, box_h))
        if gt_boxes:
            gt[f] = gt_boxes

        dt_boxes = []
        for tid in range(1, num_dt_tracks + 1):
            if rng.random() < 0.7:
                bx, by = dt_base_pos[tid]
                px = bx + rng.uniform(-2, 2) + f * rng.uniform(-1, 1)
                py = by + rng.uniform(-2, 2)
                dt_boxes.append((tid, px, py, box_w, box_h))
        if dt_boxes:
            dt[f] = dt_boxes

    # Skip empty sequences
    if not gt and not dt:
        continue

    run_scenario(f"random_{trial}", gt, dt)

random_count = scenarios_run - handcrafted_count

# ===========================================================================
# Summary
# ===========================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if not bugs_found:
    print(f"\nRESULT: OK")
    print(f"Scenarios: {handcrafted_count} hand-crafted, {random_count} random")
    print("All scenarios passed.")
else:
    print(f"\nRESULT: FOUND_BUG")
    print(f"Total bugs found: {len(bugs_found)}")
    print(f"Scenarios: {handcrafted_count} hand-crafted, {random_count} random")

    # Deduplicate by (scenario, metric)
    seen = set()
    for b in bugs_found:
        key = (b["scenario"], b["metric"])
        if key in seen:
            continue
        seen.add(key)
        print(f"\n--- Bug ---")
        print(f"Scenario: {b['scenario']}")
        print(f"Metric: {b['metric']}")
        print(f"hotcoco: {b['hotcoco']}")
        print(f"TrackEval: {b['trackeval']}")
        print(f"Diff: {b['diff']:.2e}")
        # Print compact GT/DT
        gt_repr = {}
        for fid, boxes in sorted(b["gt"].items()):
            gt_repr[fid] = [(tid, round(x, 2), round(y, 2), round(w, 2), round(h, 2)) for tid, x, y, w, h in boxes]
        dt_repr = {}
        for fid, boxes in sorted(b["dt"].items()):
            dt_repr[fid] = [(tid, round(x, 2), round(y, 2), round(w, 2), round(h, 2)) for tid, x, y, w, h in boxes]
        print(f"GT frames: {gt_repr}")
        print(f"DT frames: {dt_repr}")
