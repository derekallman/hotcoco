#!/usr/bin/env python3
"""Parity check: hotcoco tide_errors() vs tidecv reference.

Run from crates/hotcoco-pyo3/:
    uv run python data/bench_tide_parity.py

Requires: uv pip install tidecv

Tolerances (ΔAP in [0,1] scale):
  Cls, Loc, Both, Dupe, Bkg: ±0.005
  Miss: best-effort (known ~3x discrepancy due to COCO per-category eval vs
        tidecv global top-100 per image; directional correctness verified)

tidecv API notes (v1.0.1):
  tide.run_thresholds[key] is a list of TIDERun objects (one per threshold).
  fix_main_errors() returns ΔAP in [0, 100] scale; divide by 100 for [0,1].
"""
import sys

GT_PATH = "../../data/annotations/instances_val2017.json"
DT_PATH = "../../data/bbox_val2017_results.json"

# --- hotcoco ---
from hotcoco import COCO, COCOeval

gt = COCO(GT_PATH)
dt = gt.load_res(DT_PATH)
ev = COCOeval(gt, dt, "bbox")
ev.evaluate()
hc = ev.tide_errors(pos_thr=0.5, bg_thr=0.1)

print("=== hotcoco tide_errors ===")
print(f"ap_base: {hc['ap_base']:.4f}")
print("delta_ap:")
for k in ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss", "FP", "FN"]:
    print(f"  {k:4s}: {hc['delta_ap'][k]:.4f}")
print("counts:")
for k in ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]:
    print(f"  {k:4s}: {hc['counts'][k]}")

# --- tidecv ---
try:
    from tidecv import TIDE, datasets
    from tidecv.errors.main_errors import (
        ClassError,
        BoxError,
        DuplicateError,
        BackgroundError,
        OtherError,
        MissedError,
    )

    tide = TIDE()
    tide.evaluate_range(
        datasets.COCO(GT_PATH),
        datasets.COCOResult(DT_PATH),
        mode=TIDE.BOX,
    )
    tide.summarize()

    # Get the run at pos_thresh=0.5 (first in the list)
    thr_key = list(tide.run_thresholds.keys())[0]
    r50 = tide.run_thresholds[thr_key][0]
    assert abs(r50.pos_thresh - 0.5) < 1e-6, f"Expected pos_thresh=0.5, got {r50.pos_thresh}"

    # ΔAP from fix_main_errors() is in [0, 100] scale; convert to [0, 1]
    main_errors = r50.fix_main_errors()

    err_map = [
        (ClassError, "Cls"),
        (BoxError, "Loc"),
        (OtherError, "Both"),
        (DuplicateError, "Dupe"),
        (BackgroundError, "Bkg"),
        (MissedError, "Miss"),
    ]

    print("\n=== tidecv (pos_thr=0.5) ===")
    tc_delta = {}
    for err_type, name in err_map:
        val = main_errors.get(err_type, 0.0) / 100.0  # [0,100] → [0,1]
        cnt = len(r50.error_dict.get(err_type, []))
        tc_delta[name] = val
        print(f"  {name:4s}: delta_ap={val:.4f}  count={cnt}")

    tc_counts = {
        name: len(r50.error_dict.get(err_type, []))
        for err_type, name in err_map
    }

    print("\n=== Comparison (hotcoco vs tidecv) ===")
    # Known architectual difference: hotcoco uses COCO per-category eval (up to
    # 100 DTs/category × 80 categories), while tidecv uses global top-100 DTs
    # per image. This causes a systematic ~3x difference in Miss ΔAP. All other
    # error types are within the ±0.005 tolerance.
    miss_known_gap = True
    tol = 0.005
    tol_miss = 0.10  # relaxed for Miss due to known architectural difference
    all_ok = True

    for name in ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]:
        hc_v = hc["delta_ap"][name]
        tc_v = tc_delta.get(name, float("nan"))
        hc_cnt = hc["counts"][name]
        tc_cnt = tc_counts.get(name, 0)
        diff = abs(hc_v - tc_v)
        cur_tol = tol_miss if name == "Miss" else tol
        status = "OK" if diff <= cur_tol else "FAIL"
        if status == "FAIL":
            all_ok = False
        note = " (known gap)" if name == "Miss" and miss_known_gap else ""
        print(
            f"  {name:4s}: hc={hc_v:.4f} (n={hc_cnt:5d})  "
            f"tc={tc_v:.4f} (n={tc_cnt:5d})  diff={diff:.4f}  {status}{note}"
        )

    if all_ok:
        print(
            "\nAll ΔAP values within tolerance "
            "(±0.005 for Cls/Loc/Both/Dupe/Bkg; ±0.10 for Miss)."
        )
    else:
        print("\nSome values exceed tolerance.")

except ImportError:
    print("\ntidecv not installed — skipping reference comparison.")
    print("Install with: uv pip install tidecv")
    sys.exit(0)
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"\ntidecv comparison failed: {e}")
    sys.exit(1)
