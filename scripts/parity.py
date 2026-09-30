"""Parity check: hotcoco vs pycocotools on COCO val2017.

Uses third-party published fake result files from ppwwyyxx/cocoapi.
See docs/getting-started/installation.md for download instructions.

Tolerances are 1e-12 across the board — tight enough that only floating-point
noise passes. The threshold grids match `numpy.linspace` bit-for-bit and polygon
rasterization reproduces the reference's per-architecture rounding (FMA on
arm64, two roundings on x86-64), so the measured worst case across all 34
metrics is 3.7e-14. A gate has to be sized to what the code actually does, or
it stops being a gate.

Usage:
    uv run python scripts/parity.py
    just parity
"""

import json
import sys

from helpers import FIXTURES_DIR, VAL2017, compare_metrics, reference_stats, suppress_output
from hotcoco import COCO, COCOeval

# Files come from `helpers.VAL2017`; the tolerance is this script's own, and is
# the same for every iou_type — see the module docstring for how it was sized.
TOL = 1e-12


def run_hotcoco(gt_file, dt_file, iou_type):
    with suppress_output(stderr=False):
        gt = COCO(str(gt_file))
        dt = gt.load_res(str(dt_file))
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats, ev.metric_keys()


# A pinned baseline alongside the live comparison.
#
# The live run catches hotcoco drifting away from pycocotools. It cannot catch
# both of them drifting *together*: a pycocotools upgrade that changed a metric
# would move the reference and the comparison would stay green. The baseline was
# produced by pycocotools itself (see scripts/gen_val2017_baseline.py), so a
# mismatch means the reference moved and someone has to decide whether that is
# intended.
#
# Loaded unconditionally, and a missing or unreadable file is fatal: a baseline
# that disappears silently is worse than no baseline, because it reports the
# check it isn't running.
BASELINE_PATH = FIXTURES_DIR / "val2017_expected.json"
if not BASELINE_PATH.exists():
    sys.exit(f"missing baseline {BASELINE_PATH} — regenerate with scripts/gen_val2017_baseline.py")
try:
    BASELINE = json.loads(BASELINE_PATH.read_text())
    BASELINE_METRICS = BASELINE["metrics"]
    BASELINE_REF = BASELINE["reference"]
except (json.JSONDecodeError, KeyError) as exc:
    sys.exit(f"unreadable baseline {BASELINE_PATH}: {exc} — regenerate with scripts/gen_val2017_baseline.py")

all_pass = True

for iou_type, files in VAL2017.items():
    print(f"\n{'=' * 68}")
    print(f"  {iou_type}  (<= {TOL:.0e})")
    print(f"{'=' * 68}")
    print(f"  {'Metric':<8} {'pycocotools':>14} {'hotcoco':>14} {'diff':>12}  status")
    print(f"  {'-' * 58}")

    py = reference_stats(files["gt"], files["dt"], iou_type)
    rs, metric_names = run_hotcoco(files["gt"], files["dt"], iou_type)

    # The decision is `helpers.compare_metrics` — one owner for the length check
    # and the -1.0 sentinel rule. The table prints every row regardless.
    failed = {m.name for m in compare_metrics(py, rs, metric_names, tolerance=TOL)}
    for name, p, r in zip(metric_names, py, rs):
        print(f"  {name:<8} {p:>14.8f} {r:>14.8f} {abs(p - r):>12.2e}  {'FAIL' if name in failed else 'PASS'}")
    type_pass = not failed

    # Compare against the pinned reference values as well. An absent key is a
    # failure, not a skip — the alternative is a green run over an empty check.
    expected = BASELINE_METRICS.get(iou_type)
    if expected is None:
        print(f"\n  BASELINE: no pinned values for '{iou_type}' — nothing was checked against the pin. FAIL")
        print("    Regenerate with scripts/gen_val2017_baseline.py.")
        type_pass = False
    else:
        drift = compare_metrics(expected, py, metric_names, tolerance=TOL)
        if drift:
            print(f"\n  BASELINE DRIFT — pinned values were produced by {BASELINE_REF}:")
            for m in drift:
                print(f"    {m.name:<8} pinned={m.py:.8f}  reference now={m.rs:.8f}  diff={m.diff:.2e}")
            print("    The reference implementation moved. Confirm intended, then regenerate.")
            type_pass = False
        else:
            print(f"  {'(baseline)':<8} {len(expected)} pinned reference values match")

    all_pass &= type_pass
    print(f"\n  {'ALL PASS' if type_pass else 'SOME METRICS FAILED'}")

print(f"\n{'=' * 68}")
if all_pass:
    print("ALL METRICS PASS")
else:
    print("PARITY FAILURES DETECTED")
    sys.exit(1)
