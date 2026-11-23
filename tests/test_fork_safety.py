"""A child that ``os.fork()`` makes inherits rayon's thread pool but none of its
threads, so a fan-out there queues work no thread will take and waits forever.
That is how a PyTorch ``DataLoader`` worker on Linux starts: forked from a
process that already loaded a dataset, then decoding masks. hotcoco spots such a
child by its process id and runs it on one thread instead. A child forked
before the parent's pool started starts a pool of its own, which the Rust test
in ``primitives/sim.rs`` covers.

Each family's fan-outs run here in a child forked after the parent's pool
started. ``tests/architecture.rs`` holds every fan-out in the source to the
same check; this file shows the check works, family by family. A new evaluator
or entry point that fans out needs a row in ``CASES``.

Every case runs in a subprocess that forks, so a hang is a timeout and a killed
child, not a stuck test run.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork() on this platform")

SETUP = textwrap.dedent(
    """
    import json, os, sys, time, traceback
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from hotcoco import COCO, COCOeval, compare, mask, panoptic, semantic, tracking

    tmp = Path(sys.argv[1])
    rng = np.random.default_rng(0)
    stack = np.asfortranarray(rng.random((480, 640, 8)) > 0.5).astype(np.uint8)
    rles = mask.encode(stack)

    def square(i, k=0):
        x, y = 5 + i % 7, 5 + k
        return [[x, y, x + 20, y, x + 20, y + 20, x, y + 20]]

    images = [{"id": i, "width": 64, "height": 64, "file_name": f"{i}.png"} for i in range(1, 9)]
    cats = [{"id": 1, "name": "a", "isthing": 1}, {"id": 2, "name": "b", "isthing": 1}]
    gt_anns = [
        {"id": i, "image_id": i, "category_id": 1 + i % 2, "bbox": [5 + i % 7, 5, 20, 20], "area": 400,
         "iscrowd": 0, "segmentation": square(i)}
        for i in range(1, 9)
    ]
    dts = [
        {"image_id": i, "category_id": 1 + i % 2, "bbox": [6 + i % 7, 5, 20, 18], "score": 0.5 + i / 20}
        for i in range(1, 9)
    ]

    def gt():
        return COCO({"images": images, "annotations": gt_anns, "categories": cats})

    def run(iou_type, dt=dts):
        g = gt()
        ev = COCOeval(g, g.load_res(dt), iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return ev

    # Past the JSON loader's 4 MiB block, so it parses one block while reading the next.
    big = tmp / "big.json"
    if not big.exists():
        anns = [
            {"id": i, "image_id": 1 + i % 8, "category_id": 1, "bbox": [1.5, 2.5, 3.5, 4.5], "area": 15.75,
             "iscrowd": 0, "segmentation": square(i)}
            for i in range(1, 60_001)
        ]
        big.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))

    for side in ("gt", "pred"):
        (tmp / side).mkdir(exist_ok=True)
        for i in range(4):
            Image.fromarray(rng.integers(0, 3, (32, 48), dtype=np.uint8)).save(tmp / side / f"{i}.png")

    (tmp / "labels").mkdir(exist_ok=True)
    (tmp / "labels" / "data.yaml").write_text("names:\\n  0: thing\\n")
    (tmp / "imgs").mkdir(exist_ok=True)
    for i in range(4):
        (tmp / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.25 0.25\\n")
        Image.new("RGB", (20 + i, 10)).save(tmp / "imgs" / f"{i}.png")

    # MOTChallenge's layout: gt/<seq>/gt/gt.txt beside gt/<seq>/seqinfo.ini, and pred/<seq>.txt.
    for seq in ("a", "b", "c"):
        (tmp / "mot" / "gt" / seq / "gt").mkdir(parents=True, exist_ok=True)
        (tmp / "mot" / "pred").mkdir(parents=True, exist_ok=True)
        rows = [[f, t, *(rng.random(2) * 50).round(2), 10, 10, 1, 1, 1] for f in range(1, 6) for t in range(1, 4)]
        (tmp / "mot" / "gt" / seq / "gt" / "gt.txt").write_text("".join(",".join(map(str, r)) + "\\n" for r in rows))
        (tmp / "mot" / "gt" / seq / "seqinfo.ini").write_text(f"[Sequence]\\nname={seq}\\nseqLength=5\\n")
        (tmp / "mot" / "pred" / f"{seq}.txt").write_text("".join(",".join(map(str, r[:7])) + "\\n" for r in rows))

    def tracking_run():
        ev = tracking.TrackingEval("MOT17")
        ev.update_folders(tmp / "mot" / "gt", tmp / "mot" / "pred")
        ev.compute()

    def semantic_run():
        ev = semantic.SemanticEval(["a", "b", "c"])
        ev.update_folders(tmp / "gt", tmp / "pred")
        ev.results()

    CASES = {
        "mask.decode and encode": lambda: mask.encode(np.asfortranarray(mask.decode(rles))),
        "COCO from a large file": lambda: COCO(str(big)),
        "load_res of masks": lambda: gt().load_res(
            [{"image_id": 1, "category_id": 1, "segmentation": r, "score": 0.5} for r in rles]
        ),
        "COCOeval bbox": lambda: run("bbox"),
        "COCOeval segm": lambda: run("segm", [{**d, "segmentation": square(d["image_id"], 1)} for d in dts]),
        "tide_errors": lambda: run("bbox").tide_errors(),
        "confusion_matrix": lambda: run("bbox").confusion_matrix(),
        "calibration": lambda: run("bbox").calibration(),
        "image_diagnostics": lambda: run("bbox").image_diagnostics(),
        "compare with bootstrap": lambda: compare(run("bbox"), run("bbox"), n_bootstrap=20),
        "PanopticEval": lambda: panoptic.PanopticEval(gt(), gt()).run(),
        "SemanticEval.update_folders": semantic_run,
        "TrackingEval.update_folders": tracking_run,
        "from_yolo with images_dir": lambda: COCO.from_yolo(str(tmp / "labels"), images_dir=str(tmp / "imgs")),
    }
    """
)

IN_FORKED_CHILDREN = SETUP + textwrap.dedent(
    """
    # 2.4 MB of pixels, past the parallel threshold: the parent's pool starts.
    assert mask.decode(rles).shape == (480, 640, 8)

    outcomes = {}
    for name, case in CASES.items():
        pid = os.fork()
        if pid == 0:
            try:
                case()
                os._exit(0)
            except BaseException:
                traceback.print_exc()
                os._exit(2)
        deadline = time.monotonic() + 30
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                code = os.waitstatus_to_exitcode(status)
                outcomes[name] = "ok" if code == 0 else f"exited {code}"
                break
            if time.monotonic() > deadline:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
                outcomes[name] = "hung"
                break
            time.sleep(0.02)
    print(json.dumps(outcomes))
    """
)


def run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    done = run_script(IN_FORKED_CHILDREN, str(tmp_path_factory.mktemp("fork")))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr


CASE_NAMES = [
    "mask.decode and encode",
    "COCO from a large file",
    "load_res of masks",
    "COCOeval bbox",
    "COCOeval segm",
    "tide_errors",
    "confusion_matrix",
    "calibration",
    "image_diagnostics",
    "compare with bootstrap",
    "PanopticEval",
    "SemanticEval.update_folders",
    "TrackingEval.update_folders",
    "from_yolo with images_dir",
]


def test_every_case_is_listed(outcomes):
    assert sorted(outcomes[0]) == sorted(CASE_NAMES)


@pytest.mark.parametrize("case", CASE_NAMES)
def test_a_forked_child_finishes(outcomes, case):
    got, stderr = outcomes
    assert got[case] == "ok", stderr
