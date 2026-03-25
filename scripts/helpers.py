"""Shared helpers for hotcoco dev scripts (parity, bench, test, fuzz)."""

import contextlib
import io
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "data"

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

COCO_KPT_OKS_SIGMAS = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
]

# ---------------------------------------------------------------------------
# stdout suppression
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout at the file-descriptor level (catches Rust println! too)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(1)
    os.dup2(devnull_fd, 1)
    old_sys = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        os.dup2(old_fd, 1)
        os.close(old_fd)
        os.close(devnull_fd)
        sys.stdout = old_sys
