"""Panoptic parity: hotcoco vs panopticapi on COCO panoptic val2017.

Runs both on the 5,000 val2017 images against the perturbed predictions
`scripts/download_panoptic.py` writes (`just download-panoptic`), and compares
through `helpers.pq_disagreements`, the same comparison `tests/test_parity_panoptic.py`
runs on synthetic data: per-category TP, FP, FN exactly, summed IoU to 1e-9,
and PQ, SQ, RQ, N for All, Things and Stuff to 1e-9.

Exits 1 on any disagreement, and prints both implementations' tables either way.
Needs panopticapi, which the dev extra installs from a pinned commit.

    uv run python scripts/parity_panoptic.py
    just parity-panoptic
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
import time

import numpy as np
from helpers import PANOPTIC_VAL2017, panopticapi_reference, pq_disagreements
from hotcoco import panoptic

TOL = 1e-9


def table(name: str, res: dict) -> str:
    lines = [f"{name}", f"{'':10}| {'PQ':>5}  {'SQ':>5}  {'RQ':>5} {'N':>5}", "-" * 38]
    for split in ("All", "Things", "Stuff"):
        r = res[split]
        lines.append(f"{split:10}| {100 * r['pq']:5.1f}  {100 * r['sq']:5.1f}  {100 * r['rq']:5.1f} {r['n']:5d}")
    return "\n".join(lines)


def main() -> int:
    paths = PANOPTIC_VAL2017
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        print("Missing panoptic data:", *missing, "Run `just download-panoptic` first.", sep="\n  ", file=sys.stderr)
        return 2

    print(f"panopticapi {importlib.metadata.version('panopticapi')}, numpy {np.__version__}")
    t0 = time.perf_counter()
    ref, ref_counts = panopticapi_reference(
        paths["gt"], paths["dt"], paths["gt_folder"], paths["dt_folder"], multi_core=True
    )
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    ev = panoptic.PanopticEval(paths["gt"], paths["dt"], gt_folder=paths["gt_folder"], pred_folder=paths["dt_folder"])
    ev.evaluate()
    got = ev.results()
    t_got = time.perf_counter() - t0

    print(table(f"panopticapi ({t_ref:.1f}s)", ref))
    print(table(f"hotcoco ({t_got:.1f}s)", got))
    category_ids = [c["id"] for c in json.loads(paths["gt"].read_text())["categories"]]
    evaluable = sum(1 for c in ref_counts.values() if c["tp"] + c["fp"] + c["fn"])
    print(f"\n{len(category_ids)} categories, {evaluable} evaluable")

    problems = pq_disagreements(ref, ref_counts, got, category_ids, tol=TOL)
    if problems:
        print(f"\nFAIL: {len(problems)} disagreement(s) above {TOL:g}:")
        for p in problems[:50]:
            print("  " + p)
        if len(problems) > 50:
            print(f"  ... {len(problems) - 50} more")
        return 1
    print(f"\nOK: counts exact and every score within {TOL:g} of panopticapi")
    return 0


if __name__ == "__main__":
    sys.exit(main())
