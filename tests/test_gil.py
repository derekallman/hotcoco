"""Whatever reads numpy memory holds the GIL, since another Python thread could
write a buffer read without it mid-count: the rule `crates/hotcoco-pyo3/src/mask.rs`
states for the encoders. A thread ticks every half millisecond beside each call; a
tick inside the call means the call let it run, so released the GIL while
reading the array. A new entry point that reads a numpy array in place, and runs
long enough to watch, needs a row in ``CASES``.
"""

import sys
import threading
import time

import numpy as np
import pytest
from hotcoco import metrics, primitives, semantic

RNG = np.random.default_rng(0)
# About 12 to 50 ms of counting each on an M1: long enough to see a release.
GT = RNG.integers(0, 4, (4096, 4096), dtype=np.uint8)
PRED = RNG.integers(0, 4, (4096, 4096), dtype=np.uint8)

# Each makes its call, with everything but the call itself done first: the
# interpreter may switch threads between any two bytecodes, so the timed window
# holds only the call into hotcoco.
CASES = {
    "SemanticEval.update": lambda: semantic.SemanticEval(4, boundary_dilation=None).update,
    "metrics.pixel_confusion": lambda: lambda gt, pred: metrics.pixel_confusion(gt, pred, num_classes=4),
    "primitives.segment_overlaps": lambda: primitives.segment_overlaps,
}


def ticks_inside(call):
    """The ticks a background thread made while ``call(GT, PRED)`` ran, leaving
    out the first and last 3 ms, where a thread switch can land either side."""
    stamps, stop = [], threading.Event()

    def tick():
        while not stop.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0.0005)

    interval = sys.getswitchinterval()
    # A thread waiting for the GIL gets it within 0.1 ms of a release.
    sys.setswitchinterval(1e-4)
    ticker = threading.Thread(target=tick)
    ticker.start()
    try:
        time.sleep(0.01)
        start = time.perf_counter()
        call(GT, PRED)
        end = time.perf_counter()
    finally:
        stop.set()
        ticker.join()
        sys.setswitchinterval(interval)
    assert end - start > 0.008, f"the call took {end - start:.4f} s, too short to see a release"
    return [s for s in stamps if start + 0.003 < s < end - 0.003]


def test_the_probe_sees_a_release():
    assert ticks_inside(lambda gt, pred: time.sleep(0.03))


@pytest.mark.parametrize("make", CASES.values(), ids=CASES.keys())
def test_reads_numpy_memory_holding_the_gil(make):
    assert not ticks_inside(make())
