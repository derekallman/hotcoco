#!/usr/bin/env python3
"""Parity check: hotcoco tide_errors() vs tidecv reference.

Usage:
    uv run python scripts/parity_tide.py

Requires: uv pip install tidecv

Runs two comparisons at pos_thr=0.5, bg_thr=0.1, and exits non-zero if either
fails (or if tidecv is missing):

1. **Crowd-free val2017** -- every image holding an iscrowd GT is dropped (411 of
   5,000), from both the GT file and the results. This is the real parity gate:
   all nine numbers (ap_base, FP, FN, Cls, Loc, Both, Dupe, Bkg, Miss) are held to
   +/-0.005; with no crowd regions the two tools classify the same detections the
   same way, and the error counts agree exactly.

2. **Full val2017** -- the same nine, at +/-0.005 except Loc and Miss, which are
   bounded deviations from crowd handling (below).

Crowd handling, the one deliberate deviation
--------------------------------------------
hotcoco follows COCO -- an iscrowd region takes part in matching and a detection
landing on one is ignored. tidecv removes crowd regions from the GT and ignores a
detection afterwards only if it is unmatched AND same-class AND clears IoA 0.5, so
it books as false positives some detections COCO-style evaluation ignores (on
val2017, ~617 more Loc errors and 573 fewer Miss). That puts full-set Loc ~0.010
below tidecv and Miss ~0.007 above it. Full write-up: the "Coming from tidecv"
section of docs/guide/diagnostics.md.

Do not relax a tolerance to make the full set pass without checking the
crowd-free gate first: a full-set agreement once came from two errors cancelling
(see the CHANGELOG), and the "global top-100 per image" explanation for the Miss
gap is false -- no val2017 image holds more than 76 detections.

ΔAP parity is not count parity: a low-ranked error barely moves the AP integral,
which is why the crowd-free run also prints both tools' counts.

tidecv API notes (v1.0.1):
  tide.run_thresholds[key] is a list of TIDERun objects (one per threshold).
  fix_main_errors()/fix_special_errors() return ΔAP in [0, 100]; divide by 100.
"""

import json
import sys
import tempfile
from pathlib import Path

from helpers import VAL2017

GT_PATH = Path(VAL2017["bbox"]["gt"])
DT_PATH = Path(VAL2017["bbox"]["dt"])

try:
    from tidecv import TIDE, datasets
    from tidecv.errors.main_errors import (
        BackgroundError,
        BoxError,
        ClassError,
        DuplicateError,
        FalseNegativeError,
        FalsePositiveError,
        MissedError,
        OtherError,
    )
except ImportError:
    # Exit non-zero: without the reference this script would print hotcoco's own
    # numbers and compare them to nothing. Exiting 0 made it look like a passing
    # parity check, which is how the TIDE gate went unverified for a whole
    # refactor of the TIDE classifier.
    print("tidecv not installed — cannot compare against the reference.")
    print("Install with: uv pip install tidecv")
    sys.exit(1)

from hotcoco import COCO, COCOeval  # noqa: E402

MAIN = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
TC_TYPES = {
    "Cls": ClassError,
    "Loc": BoxError,
    "Both": OtherError,
    "Dupe": DuplicateError,
    "Bkg": BackgroundError,
    "Miss": MissedError,
}
TOL = 0.005


def run_hotcoco(gt_path, dt_path):
    gt = COCO(str(gt_path))
    ev = COCOeval(gt, gt.load_res(str(dt_path)), "bbox")
    ev.evaluate()
    te = ev.tide_errors(pos_thr=0.5, bg_thr=0.1)
    return {"base": te["ap_base"], **te["delta_ap"]}, te["counts"]


def run_tidecv(gt_path, dt_path):
    tide = TIDE()
    tide.evaluate_range(datasets.COCO(str(gt_path)), datasets.COCOResult(str(dt_path)), mode=TIDE.BOX)
    run = list(tide.run_thresholds.values())[0][0]
    assert abs(run.pos_thresh - 0.5) < 1e-6, f"Expected pos_thresh=0.5, got {run.pos_thresh}"
    main, special = run.fix_main_errors(), run.fix_special_errors()
    delta = {"base": run.ap / 100.0}
    delta.update({k: main.get(t, 0.0) / 100.0 for k, t in TC_TYPES.items()})
    delta["FP"] = special.get(FalsePositiveError, 0.0) / 100.0
    delta["FN"] = special.get(FalseNegativeError, 0.0) / 100.0
    counts = {k: len(run.error_dict.get(t, [])) for k, t in TC_TYPES.items()}
    return delta, counts


def compare(label, gt_path, dt_path, tols):
    """Print one comparison table. `tols` overrides TOL per key, with the reason."""
    hc, hc_n = run_hotcoco(gt_path, dt_path)
    tc, tc_n = run_tidecv(gt_path, dt_path)
    print(f"\n=== {label} ===")
    ok = True
    for key in ["base", "FP", "FN", *MAIN]:
        tol, why = tols.get(key, (TOL, ""))
        diff = abs(hc[key] - tc[key])
        passed = diff <= tol
        ok &= passed
        counts = f"(n={hc_n[key]:5d} / {tc_n[key]:5d})" if key in MAIN else " " * 16
        note = f"  [±{tol:g}: {why}]" if why else ""
        print(
            f"  {key:4s}: hc={hc[key]:.4f}  tc={tc[key]:.4f}  {counts}  diff={diff:.4f}  "
            f"{'OK' if passed else 'FAIL'}{note}"
        )
    return ok


def crowd_free(tmp):
    """Write val2017 GT and results without any image that holds a crowd GT."""
    gt = json.loads(GT_PATH.read_text())
    crowd_imgs = {a["image_id"] for a in gt["annotations"] if a.get("iscrowd")}
    gt["images"] = [i for i in gt["images"] if i["id"] not in crowd_imgs]
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] not in crowd_imgs]
    dt = [d for d in json.loads(DT_PATH.read_text()) if d["image_id"] not in crowd_imgs]
    gt_out, dt_out = Path(tmp) / "gt.json", Path(tmp) / "dt.json"
    gt_out.write_text(json.dumps(gt))
    dt_out.write_text(json.dumps(dt))
    print(f"crowd-free subset: dropped {len(crowd_imgs)} images with an iscrowd GT")
    return gt_out, dt_out


with tempfile.TemporaryDirectory() as tmp:
    ok_free = compare("val2017, crowd-free images (all ±0.005)", *crowd_free(tmp), {})

ok_full = compare(
    "val2017, all images",
    GT_PATH,
    DT_PATH,
    {
        # Measured 2026-10: Loc 0.0098, Miss 0.0073. Bounds leave headroom for
        # the crowd deviation only; see the docstring.
        "Loc": (0.015, "crowd-handling deviation"),
        "Miss": (0.010, "crowd-handling deviation"),
    },
)

if not (ok_free and ok_full):
    print("\nSome values exceed tolerance.")
    sys.exit(1)

print("\nAll values within tolerance.")
