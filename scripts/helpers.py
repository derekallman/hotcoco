"""Shared helpers for the dev scripts here and the tests in `tests/`.

`pythonpath` in `pyproject.toml` puts this directory on the path for pytest, so
both import `helpers` by the same name and there is one copy of these.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "data"
# Tracked test oracles: the adversarial corpus, `val2017_expected.json`,
# `oid_tf_expected.json`. The `gen_*` scripts write here; the tests read here.
FIXTURES_DIR = WORKSPACE / "tests" / "fixtures"

# The val2017 ground-truth / result pairs every real-data script runs against,
# keyed by iou_type. Absolute paths derived from DATA_DIR: a script that spells
# these as relative strings works from the repo root and nowhere else, which is
# the failure DATA_DIR exists to prevent.
#
# `data/` is gitignored, so these files may not exist — callers that can survive
# a missing dataset should check `.exists()` rather than assume.
VAL2017 = {
    "bbox": {"gt": DATA_DIR / "annotations/instances_val2017.json", "dt": DATA_DIR / "bbox_val2017_results.json"},
    "segm": {"gt": DATA_DIR / "annotations/instances_val2017.json", "dt": DATA_DIR / "segm_val2017_results.json"},
    "keypoints": {
        "gt": DATA_DIR / "annotations/person_keypoints_val2017.json",
        "dt": DATA_DIR / "kpt_val2017_results.json",
    },
}

# ---------------------------------------------------------------------------
# COCO keypoint constants
# ---------------------------------------------------------------------------

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

# ---------------------------------------------------------------------------
# stdout suppression
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def suppress_output(*, stderr: bool = True):
    """Suppress stdout — and stderr unless ``stderr=False`` — at the fd level.

    Redirecting the file descriptors, not just `sys.stdout`, is what catches Rust
    `println!`. fd 1 alone is enough for pycocotools' chatter; it is not enough for
    hotcoco, whose `summarize()` writes comparability warnings to fd 2, deliberately
    bypassing `sys.stderr` so they survive redirection. A script that evaluates in a
    loop — `tests/test_parity_oid.py` runs 70 cases — otherwise buries its own result under one
    warning per case.
    """
    fds = (1, 2) if stderr else (1,)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(fd) for fd in fds]
    for fd in fds:
        os.dup2(devnull_fd, fd)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    if stderr:
        sys.stderr = io.StringIO()
    try:
        yield
    finally:
        for fd, saved_fd in zip(fds, saved):
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        os.close(devnull_fd)
        sys.stdout, sys.stderr = old_out, old_err


# ---------------------------------------------------------------------------
# Temp-JSON round-trip
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def written_json(*objects, quiet: bool = False):
    """Write each object to its own temp ``.json`` file and yield their paths.

    Always yields a tuple, one path per object, so the two-file GT/DT case reads
    ``as (gt_path, dt_path)`` and the single-file case ``as (path,)``. Files are
    removed on the way out even if the body raises.

    ``quiet=True`` also suppresses stdout for the duration of the body, which is
    what a differential run wants: `COCO(path)` chatters on both sides.
    """
    paths = []
    try:
        for obj in objects:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(obj, f)
                paths.append(f.name)
        if quiet:
            with suppress_output(stderr=False):
                yield tuple(paths)
        else:
            yield tuple(paths)
    finally:
        for path in paths:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Differential parity scaffold
# ---------------------------------------------------------------------------

# Floating-point noise only. Individual callers pass a looser `tolerance` when
# they are comparing against a reference that genuinely diverges (see
# tests/test_parity_lvis.py and adversarial_harness.py) — that per-script sizing is
# deliberate and lives at the call site.
TOLERANCE = 1e-10


class MetricMismatch(NamedTuple):
    """One metric where the two implementations disagree beyond tolerance."""

    index: int
    name: str
    py: float
    rs: float
    diff: float

    def line(self) -> str:
        return f"  [{self.index}] {self.name}: py={self.py:.15f} rs={self.rs:.15f} diff={self.diff:.2e}"


def metric_names_for(iou_type):
    """Canonical metric names from the Rust evaluator for a given iou_type.

    Asked of an empty evaluator, so it works before (or without) a run, and so
    the names come from the implementation rather than a list transcribed here
    that a new metric would silently outgrow.
    """
    from hotcoco import COCO, COCOeval

    return COCOeval(COCO(), COCO(), iou_type).metric_keys()


def compare_metrics(py_stats, rs_stats, metric_names, *, tolerance=TOLERANCE):
    """Compare two stats vectors. Returns the list of :class:`MetricMismatch`.

    Metrics that are the -1.0 "not computed for this configuration" sentinel on
    *both* sides are skipped: agreeing that a metric is undefined is a real
    assertion (0.5 against -1.0 is a mismatch here), but it exercises no
    arithmetic. Cases that care whether a run reached the numeric paths pin the
    sentinels themselves — see ``test_empty_gt`` and ``test_all_crowd`` — which is
    the stronger check, since it names *which* metrics must be undefined.
    """
    expected_len = len(metric_names)
    if len(py_stats) != expected_len:
        raise ValueError(f"reference returned {len(py_stats)} metrics, expected {expected_len}")
    if len(rs_stats) != expected_len:
        raise ValueError(f"hotcoco returned {len(rs_stats)} metrics, expected {expected_len}")

    mismatches = []
    for i in range(expected_len):
        py_val, rs_val = py_stats[i], rs_stats[i]
        if py_val == -1.0 and rs_val == -1.0:
            continue
        diff = abs(py_val - rs_val)
        if diff > tolerance:
            mismatches.append(MetricMismatch(i, metric_names[i], py_val, rs_val, diff))
    return mismatches


def assert_metrics_match(py_stats, rs_stats, iou_type, *, tolerance=TOLERANCE, on_mismatch=None):
    """Assert every metric matches within tolerance.

    ``on_mismatch`` is called with the list of :class:`MetricMismatch` before the
    assertion fires — the fuzzer uses it to save a reproducer to disk.
    """
    mismatches = compare_metrics(py_stats, rs_stats, metric_names_for(iou_type), tolerance=tolerance)
    if mismatches:
        if on_mismatch is not None:
            on_mismatch(mismatches)
        raise AssertionError(
            f"\n{iou_type} metric mismatch (tol={tolerance}):\n" + "\n".join(m.line() for m in mismatches)
        )


def reference_stats(gt_file, dt_file, iou_type):
    """pycocotools' 12 (or 10) summary numbers for a GT/DT file pair, quietly.

    The one definition of "what the reference says" for val2017: `parity.py`
    compares against it live and `gen_val2017_baseline.py` pins it.
    """
    from pycocotools.coco import COCO as PyCOCO
    from pycocotools.cocoeval import COCOeval as PyCOCOeval

    with suppress_output(stderr=False):
        gt = PyCOCO(str(gt_file))
        dt = gt.loadRes(str(dt_file))
        ev = PyCOCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return [float(v) for v in ev.stats]


def run_both(gt_dataset, dt_results, iou_type):
    """Evaluate one (GT dict, DT list) pair through both implementations.

    Returns ``(py_stats, rs_stats, rs_ev)``. The evaluator comes back because the
    fuzzer asserts hotcoco-only invariants on it; callers that only diff metrics
    unpack it as ``_``. Both tools read from files, so the pair is round-tripped
    through temp JSON — which is also what the real entry points do.
    """
    from hotcoco import COCO as RsCOCO  # noqa: PLC0415
    from hotcoco import COCOeval as RsCOCOeval  # noqa: PLC0415
    from pycocotools.coco import COCO as PyCOCO  # noqa: PLC0415
    from pycocotools.cocoeval import COCOeval as PyCOCOeval  # noqa: PLC0415

    with written_json(gt_dataset, dt_results, quiet=True) as (gt_path, dt_path):
        py_gt = PyCOCO(gt_path)
        py_dt = py_gt.loadRes(dt_path)
        py_ev = PyCOCOeval(py_gt, py_dt, iou_type)
        py_ev.evaluate()
        py_ev.accumulate()
        py_ev.summarize()
        py_stats = py_ev.stats.tolist()

        rs_gt = RsCOCO(gt_path)
        rs_dt = rs_gt.load_res(dt_path)
        rs_ev = RsCOCOeval(rs_gt, rs_dt, iou_type)
        rs_ev.evaluate()
        rs_ev.accumulate()
        rs_ev.summarize()
        rs_stats = rs_ev.stats

    return py_stats, rs_stats, rs_ev
