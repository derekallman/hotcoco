"""Hypothesis-based parity fuzzer: hotcoco tracking vs TrackEval.

Property-based testing generates thousands of random tracking sequences,
comparing all metrics between both implementations. Hypothesis automatically
shrinks failing examples to minimal reproducers.

Usage:
    uv run pytest scripts/fuzz_track_hypothesis.py -v -x --tb=short
    uv run pytest scripts/fuzz_track_hypothesis.py -v -x --tb=short -s  # with print output
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import sys

sys.path.insert(0, "scripts")
from test_track_parity import run_both

TOL = 1e-6

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


# A frame with unique track IDs: sample N unique IDs, then generate boxes for each.
@st.composite
def unique_frame(draw):
    """Generate a frame with unique track IDs and well-separated boxes.

    Boxes are spaced by track ID to avoid exact overlaps between different
    tracks (which create ambiguous matchings that depend on solver tie-breaking).
    """
    n = draw(st.integers(min_value=0, max_value=8))
    if n == 0:
        return []
    ids = draw(st.lists(st.integers(min_value=1, max_value=20), min_size=n, max_size=n, unique=True))
    boxes = []
    for tid in ids:
        # Offset base position by track ID to avoid exact overlaps
        base_x = tid * 30.0
        base_y = tid * 30.0
        dx = draw(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))
        dy = draw(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))
        w = draw(st.floats(min_value=5, max_value=30, allow_nan=False, allow_infinity=False))
        h = draw(st.floats(min_value=5, max_value=30, allow_nan=False, allow_infinity=False))
        boxes.append((tid, base_x + dx, base_y + dy, w, h))
    return boxes


# A sequence: dict of frame_id -> frame, with 1-15 frames
sequence_strategy = st.dictionaries(st.integers(min_value=1, max_value=30), unique_frame(), min_size=1, max_size=15)


def filter_empty_sequence(seq):
    """Remove empty frames and return None if fully empty."""
    filtered = {k: v for k, v in seq.items() if v}
    return filtered if filtered else None


def dedup_track_ids(seq):
    """Remove duplicate track IDs within a frame (keep first per ID).

    Duplicate track IDs in a single frame are malformed tracking data.
    TrackEval handles them via NumPy's non-deterministic fancy indexing
    semantics, which are impossible to replicate exactly. We test only
    valid inputs where each track ID appears at most once per frame.
    """
    if seq is None:
        return None
    result = {}
    for fid, boxes in seq.items():
        seen = set()
        deduped = []
        for box in boxes:
            tid = box[0]
            if tid not in seen:
                seen.add(tid)
                deduped.append(box)
        if deduped:
            result[fid] = deduped
    return result if result else None


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=2000, deadline=None, suppress_health_check=[HealthCheck.too_slow], database=None)
@given(gt_seq=sequence_strategy, dt_seq=sequence_strategy)
def test_hota_parity(gt_seq, dt_seq):
    """HOTA, DetA, AssA, LocA must match TrackEval within tolerance."""
    gt = filter_empty_sequence(gt_seq)
    dt = filter_empty_sequence(dt_seq)
    if gt is None and dt is None:
        return
    gt = gt or {}
    dt = dt or {}

    hc, _, te_hota, te_clear, te_id = run_both(gt, dt)

    for key in ["HOTA", "DetA", "AssA", "LocA"]:
        te_val = float(np.mean(te_hota[key]))
        hc_val = hc[key]
        assert abs(hc_val - te_val) < TOL, (
            f"{key}: hotcoco={hc_val:.8f} vs TrackEval={te_val:.8f} (diff={abs(hc_val - te_val):.2e})"
        )


@settings(max_examples=2000, deadline=None, suppress_health_check=[HealthCheck.too_slow], database=None)
@given(gt_seq=sequence_strategy, dt_seq=sequence_strategy)
def test_clear_parity(gt_seq, dt_seq):
    """MOTA, MOTP, IDSw, Frag, MT, ML must match TrackEval."""
    gt = filter_empty_sequence(gt_seq)
    dt = filter_empty_sequence(dt_seq)
    if gt is None and dt is None:
        return

    gt = gt or {}
    dt = dt or {}

    hc, _, te_hota, te_clear, te_id = run_both(gt, dt)

    # Float metrics
    for hc_key, te_key in [("MOTA", "MOTA"), ("MOTP", "MOTP")]:
        hc_val = hc[hc_key]
        te_val = te_clear[te_key]
        assert abs(hc_val - te_val) < TOL, f"{hc_key}: hotcoco={hc_val:.8f} vs TrackEval={te_val:.8f}"

    # Integer metrics
    for hc_key, te_key in [("IDSw", "IDSW"), ("Frag", "Frag"), ("MT", "MT"), ("ML", "ML")]:
        hc_val = int(hc[hc_key])
        te_val = int(te_clear[te_key])
        assert hc_val == te_val, f"{hc_key}: hotcoco={hc_val} vs TrackEval={te_val}"


@settings(max_examples=2000, deadline=None, suppress_health_check=[HealthCheck.too_slow], database=None)
@given(gt_seq=sequence_strategy, dt_seq=sequence_strategy)
def test_identity_parity(gt_seq, dt_seq):
    """IDF1, IDP, IDR must match TrackEval within tolerance."""
    gt = filter_empty_sequence(gt_seq)
    dt = filter_empty_sequence(dt_seq)
    if gt is None and dt is None:
        return

    gt = gt or {}
    dt = dt or {}

    hc, _, te_hota, te_clear, te_id = run_both(gt, dt)

    for key in ["IDF1", "IDP", "IDR"]:
        hc_val = hc[key]
        te_val = te_id[key]
        assert abs(hc_val - te_val) < TOL, f"{key}: hotcoco={hc_val:.8f} vs TrackEval={te_val:.8f}"
