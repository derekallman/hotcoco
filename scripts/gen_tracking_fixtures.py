"""gen_tracking_fixtures.py — freeze TrackEval's MOTChallenge scores as a test oracle.

Builds synthetic MOTChallenge sequences, writes them in the folder layout
TrackEval's `MotChallenge2DBox` reads, runs TrackEval's own `Evaluator` with the
HOTA, CLEAR, Identity, and Count metrics, and freezes the inputs and outputs to
`tests/fixtures/trackeval_expected.json`. The tracking tests then need neither
TrackEval nor its numpy below 1.24.

    just gen-tracking-fixtures

That recipe builds `.venv-trackeval` on first use (TrackEval pinned to the commit
below, numpy 1.23.5, scipy pinned) and runs this script with that environment's
Python. The script refuses to run against any other TrackEval commit.

Every case is a set of named sequences under one benchmark rule:

- `"MOT15"`, `"MOT16"`, `"MOT17"`, `"MOT20"`: TrackEval with `BENCHMARK` set to that
  name and `DO_PREPROC` on.
- `null`: `DO_PREPROC` off. TrackEval still drops a ground-truth row whose column 6
  truncates to 0 even then, so every `null` case keeps that column nonzero, which
  is the one setting where "no clean-up" and TrackEval agree.

Each case carries a check that the rule it is named for actually fired (a
distractor case must lose tracker boxes, an id-switch case must count switches),
so a case that silently stops exercising its rule fails here, not in review.

Rows are written to the text files and to the fixture from the same Python
values: integers as `str`, floats as `repr`, so both parse to the same doubles.
Rows keep their order inside a frame, because TrackEval builds each frame's
similarity matrix in file order and scipy's choice among tied assignments
depends on it (see the `tied_*` cases).

`error_cases` records inputs TrackEval refuses, with the exception it raises.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy

with contextlib.redirect_stdout(io.StringIO()):
    # TrackEval prints a notice about BURST's optional pycocotools on import.
    import trackeval
    from trackeval.datasets._base_dataset import _BaseDataset

TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "trackeval_expected.json"
EPS = float(np.finfo(float).eps)
SEED = 20261118


def _check_trackeval_commit():
    """Refuse to freeze numbers from a TrackEval other than the pinned commit."""
    from importlib.metadata import distribution

    direct = distribution("trackeval").read_text("direct_url.json")
    commit = json.loads(direct or "{}").get("vcs_info", {}).get("commit_id")
    if commit != TRACKEVAL_COMMIT:
        sys.exit(f"TrackEval commit is {commit!r}, expected {TRACKEVAL_COMMIT}; run `just setup-trackeval`")


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def box(x, y=0.0, w=10.0, h=20.0):
    return [float(x), float(y), float(w), float(h)]


def shifted(b, dx):
    """`b` moved right by `dx`. For a box 10 wide, IoU is (10 - dx) / (10 + dx)."""
    return [b[0] + dx, b[1], b[2], b[3]]


def gt(frame, tid, b, *, cls=1, zero_marked=1, vis=1.0, ncols=9):
    row = [frame, tid, *b, zero_marked, cls, vis]
    return row[:ncols] if ncols <= 9 else row + [-1] * (ncols - 9)


def pred(frame, tid, b, *, conf=0.9, cls=-1, ncols=10):
    return [frame, tid, *b, conf, cls, -1, -1][:ncols]


def seq(name, num_frames, gt_rows, pred_rows):
    return {"name": name, "num_frames": num_frames, "gt": gt_rows, "pred": pred_rows}


def iou(a, b):
    """TrackEval's own box IoU, so constructed edge values are exact."""
    return float(_BaseDataset._calculate_box_ious(np.array([a]), np.array([b]), box_format="xywh")[0, 0])


def box_with_iou(target):
    """A (ground truth, tracker) pair whose IoU, as TrackEval computes it, is exactly `target`.

    The tracker box is the ground-truth box cut short in height, so the IoU is
    close to h / H; stepping `h` one ulp at a time lands on the exact double.
    """
    for gw in (10.0, 7.0, 3.0, 13.0, 1.0, 6.0):
        for gh in (10.0, 7.0, 13.0, 9.0, 11.0, 3.0, 17.0, 6.0):
            g = [0.0, 0.0, gw, gh]
            up = down = target * gh
            for _ in range(400):
                for h in (up, down):
                    if iou(g, [0.0, 0.0, gw, h]) == target:
                        return g, [0.0, 0.0, gw, float(h)]
                up, down = np.nextafter(up, np.inf), np.nextafter(down, -np.inf)
    raise RuntimeError(f"no box pair found with IoU exactly {target!r}")


# ---------------------------------------------------------------------------
# Running TrackEval
# ---------------------------------------------------------------------------


def _fmt(v):
    return str(v) if isinstance(v, int) else repr(float(v))


def _write_rows(path, rows):
    with open(path, "w") as f:
        f.writelines(",".join(_fmt(v) for v in r) + "\n" for r in rows)


def run_trackeval(benchmark, sequences):
    """Run TrackEval's Evaluator on `sequences`; returns its result dict for the one tracker."""
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = os.path.join(tmp, "gt")
        trk_dir = os.path.join(tmp, "trackers")
        os.makedirs(os.path.join(trk_dir, "tracker", "data"))
        for s in sequences:
            os.makedirs(os.path.join(gt_dir, s["name"], "gt"))
            with open(os.path.join(gt_dir, s["name"], "seqinfo.ini"), "w") as f:
                f.write(f"[Sequence]\nname={s['name']}\nseqLength={s['num_frames']}\n")
            _write_rows(os.path.join(gt_dir, s["name"], "gt", "gt.txt"), s["gt"])
            _write_rows(os.path.join(trk_dir, "tracker", "data", s["name"] + ".txt"), s["pred"])
        seqmap = os.path.join(tmp, "seqmap.txt")
        with open(seqmap, "w") as f:
            f.write("name\n" + "".join(s["name"] + "\n" for s in sequences))

        eval_config = {
            "USE_PARALLEL": False,
            "BREAK_ON_ERROR": True,
            "RETURN_ON_ERROR": False,
            "LOG_ON_ERROR": None,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_EMPTY_CLASSES": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
        dataset_config = {
            "GT_FOLDER": gt_dir,
            "TRACKERS_FOLDER": trk_dir,
            "OUTPUT_FOLDER": os.path.join(tmp, "output"),
            "TRACKERS_TO_EVAL": ["tracker"],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": benchmark or "MOT17",
            "SPLIT_TO_EVAL": "train",
            "INPUT_AS_ZIP": False,
            "PRINT_CONFIG": False,
            "DO_PREPROC": benchmark is not None,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "TRACKER_DISPLAY_NAMES": None,
            "SEQMAP_FOLDER": None,
            "SEQMAP_FILE": seqmap,
            "SEQ_INFO": None,
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            "SKIP_SPLIT_FOL": True,
        }
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            evaluator = trackeval.Evaluator(eval_config)
            dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
            metrics = [
                trackeval.metrics.HOTA(),
                trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
                trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
            ]
            res, _ = evaluator.evaluate([dataset], metrics)
        return res["MotChallenge2DBox"]["tracker"]


with contextlib.redirect_stdout(io.StringIO()):
    _METRICS = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
        trackeval.metrics.Count(),
    ]
ALPHAS = [float(a) for a in _METRICS[0].array_labels]
INTEGER_FIELDS = {"Frames"}
for _m in _METRICS:
    INTEGER_FIELDS |= set(_m.integer_fields) | set(_m.integer_array_fields)


def flatten(per_metric):
    """One sequence's (or COMBINED_SEQ's) results, flat, under TrackEval's field names."""
    out = {}
    for fields in per_metric.values():
        for field, value in fields.items():
            assert field not in out, f"field {field} appears in two metrics"
            values = np.atleast_1d(np.asarray(value, dtype=float))
            assert np.isfinite(values).all(), f"{field} is not finite: {value}"
            if field in INTEGER_FIELDS:
                assert (values == np.round(values)).all(), f"{field} is not integral: {value}"
                values = [int(v) for v in values]
            else:
                values = [float(v) for v in values]
            if np.ndim(value) == 0:
                out[field] = values[0]
            else:
                assert len(values) == len(ALPHAS), field
                out[field] = values
    return out


def expected_for(benchmark, sequences):
    res = run_trackeval(benchmark, sequences)
    return {
        "combined": flatten(res["COMBINED_SEQ"]["pedestrian"]),
        "per_sequence": {s["name"]: flatten(res[s["name"]]["pedestrian"]) for s in sequences},
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

CASES = []


def case(name, benchmark):
    """Register a builder returning (sequences, check); check(expected, sequences) asserts."""

    def register(fn):
        CASES.append((name, benchmark, fn))
        return fn

    return register


def n_pred(sequences):
    return sum(len(s["pred"]) for s in sequences)


def n_gt(sequences):
    return sum(len(s["gt"]) for s in sequences)


# Classes 1 to 13 of MOTChallenge, one ground truth each in frame 1, with a
# tracker box at IoU 0.818 on every non-pedestrian one.
_ALL_CLASSES = [2, 7, 8, 12, 6, 3, 4, 5, 9, 10, 11, 13]


def _distractor_sequences():
    g, p = [], []
    for f in range(1, 5):
        g.append(gt(f, 1, box(0)))
        for k, c in enumerate(_ALL_CLASSES, start=2):
            g.append(gt(f, k, box(100 * k), cls=c, zero_marked=0))
    # Frame 1: every class gets a tracker box at IoU 0.818.
    p.append(pred(1, 1, box(0)))
    for k, _ in enumerate(_ALL_CLASSES, start=2):
        p.append(pred(1, 100 + k, shifted(box(100 * k), 1)))
    # Frame 2: static person (7) at IoU 0.43, below the clean-up threshold, so it
    # stays; non-motor vehicle (6) at IoU 0.818, removed under MOT20 only.
    p.append(pred(2, 1, box(0)))
    p.append(pred(2, 103, shifted(box(300), 4)))
    p.append(pred(2, 106, shifted(box(600), 1)))
    # Frame 3: a tracker box far from everything.
    p.append(pred(3, 1, box(0)))
    p.append(pred(3, 99, box(5000)))
    # Frame 4: no tracker boxes. Frame 5: nothing.
    return [seq("seq-a", 5, g, p)]


def _distractor_check(removed):
    def check(exp, seqs):
        assert exp["combined"]["Dets"] == n_pred(seqs) - removed, exp["combined"]["Dets"]

    return check


@case("distractors_mot16", "MOT16")
def _():
    return _distractor_sequences(), _distractor_check(4)


@case("distractors_mot17", "MOT17")
def _():
    return _distractor_sequences(), _distractor_check(4)


@case("distractors_mot20", "MOT20")
def _():
    """MOT20 also removes boxes matched to class 6, non-motor vehicle."""
    return _distractor_sequences(), _distractor_check(6)


@case("distractors_mot15", "MOT15")
def _():
    """MOT15 has no class clean-up: nothing removed, every ground-truth class kept."""
    seqs = _distractor_sequences()
    for s in seqs:  # zero_marked would still drop rows; keep every row live
        s["gt"] = [r[:6] + [1] + r[7:] for r in s["gt"]]

    def check(exp, seqs):
        assert exp["combined"]["Dets"] == n_pred(seqs)
        assert exp["combined"]["GT_Dets"] == n_gt(seqs)

    return seqs, check


@case("distractor_by_assignment", "MOT17")
def _():
    """The clean-up match is an optimal assignment, not greedy.

    Tracker box 1 overlaps the pedestrian (IoU 0.818) more than the static person
    (0.667), but pairing it with the static person lets tracker box 2 take the
    pedestrian (0.538): 1.205 beats 0.818, so box 1 is removed.
    """
    g = [gt(1, 1, box(0, w=10)), gt(1, 2, box(3, w=10), cls=7, zero_marked=0)]
    p = [pred(1, 1, box(1, w=10)), pred(1, 2, box(-3, w=10))]

    def check(exp, seqs):
        c = exp["combined"]
        assert c["Dets"] == 1 and c["CLR_TP"] == 1 and c["CLR_FP"] == 0, c

    return [seq("seq-a", 1, g, p)], check


def _edge_targets():
    half = 0.5
    below = [half]
    for _ in range(5):
        below.append(float(np.nextafter(below[-1], 0.0)))
    # below[1] is 0.5 - 2**-54; below[4] is exactly 0.5 - eps; below[5] is under it.
    assert below[4] == half - EPS and below[5] < half - EPS
    return below


@case("distractor_threshold_edge", "MOT17")
def _():
    """Clean-up zeroes similarity below 0.5 - eps: 0.5 and 0.5 - eps match, one ulp less does not."""
    targets = _edge_targets()
    chosen = [targets[0], targets[1], targets[4], targets[5]]
    g, p = [], []
    for f, t in enumerate(chosen, start=1):
        gbox, tbox = box_with_iou(t)
        g.append(gt(f, 1, box(1000)))
        g.append(gt(f, 2, gbox, cls=8, zero_marked=0))
        p.append(pred(f, 1, box(1000)))
        p.append(pred(f, 2, tbox))

    def check(exp, seqs):
        # Three removed (0.5, 0.5 - 1 ulp, 0.5 - eps); the box one ulp under 0.5 - eps stays.
        assert exp["combined"]["Dets"] == n_pred(seqs) - 3, exp["combined"]["Dets"]

    return [seq("seq-a", len(chosen), g, p)], check


@case("similarity_on_thresholds", None)
def _():
    """One pair per frame, its IoU sitting exactly on a threshold or one ulp under it.

    CLEAR and HOTA compare with `>= threshold - eps`; Identity compares with
    `>= 0.5` and no eps. 0.5 - 1 ulp and 0.5 - eps are TPs for CLEAR and HOTA at
    alpha 0.5 but not IDTPs.
    """
    edges = _edge_targets()
    targets = [1.0, edges[0], edges[1], edges[4], edges[5], 0.75, 0.3, 0.25, 0.15, 0.05, 0.95]
    for a in (ALPHAS[2], ALPHAS[9], ALPHAS[14], ALPHAS[18]):
        at = a - EPS
        targets += [at, float(np.nextafter(at, 0.0))]
    g, p = [], []
    for f, t in enumerate(targets, start=1):
        gbox, tbox = box_with_iou(t)
        g.append(gt(f, 1, gbox))
        p.append(pred(f, 1, tbox))

    def check(exp, seqs):
        c = exp["combined"]
        in_eps_band = sum(0.5 - EPS <= t < 0.5 for t in targets)
        assert in_eps_band >= 2 and c["CLR_TP"] - c["IDTP"] == in_eps_band, (c["CLR_TP"], c["IDTP"])
        for a, alpha in enumerate(ALPHAS):
            want = sum(t >= alpha - EPS for t in targets)
            assert c["HOTA_TP"][a] == want, (alpha, c["HOTA_TP"][a], want)

    return [seq("seq-a", len(targets), g, p)], check


def _zero_marked_sequences():
    g, p = [], []
    for f in range(1, 4):
        g.append(gt(f, 1, box(0), zero_marked=0))  # pedestrian, not evaluated
        g.append(gt(f, 2, box(100), cls=7, zero_marked=0))  # static person
        g.append(gt(f, 3, box(200)))
        g.append(gt(f, 4, box(300), zero_marked=0 if f == 2 else 1))
        p.append(pred(f, 11, shifted(box(0), 1)))
        p.append(pred(f, 12, shifted(box(100), 1)))
        p.append(pred(f, 13, box(200)))
        p.append(pred(f, 14, shifted(box(300), 1)))
    return [seq("seq-a", 3, g, p)]


@case("zero_marked", "MOT17")
def _():
    """A zero-marked pedestrian leaves the ground truth, but the tracker box on it
    stays and is a false positive; only a distractor class removes a tracker box."""

    def check(exp, seqs):
        c = exp["combined"]
        assert c["GT_Dets"] == 3 + 2 and c["Dets"] == 12 - 3 and c["CLR_FP"] >= 3, c

    return _zero_marked_sequences(), check


@case("zero_marked_mot15", "MOT15")
def _():
    """MOT15 applies zero_marked but no class rule: nothing is removed."""

    def check(exp, seqs):
        c = exp["combined"]
        assert c["GT_Dets"] == 3 + 2 and c["Dets"] == 12, c

    return _zero_marked_sequences(), check


@case("zero_marked_truncation", "MOT17")
def _():
    """zero_marked is column 6 cast with int(): 0.5, 0.99 and -0.5 all mean 0."""
    marks = [0.5, 0.99, -0.5, -1, 2, 1.5, 0, 1]
    g = [gt(1, k, box(100 * k), zero_marked=m) for k, m in enumerate(marks, start=1)]
    p = [pred(1, k, box(100 * k)) for k in range(1, len(marks) + 1)]

    def check(exp, seqs):
        assert exp["combined"]["GT_Dets"] == 4, exp["combined"]["GT_Dets"]

    return [seq("seq-a", 1, g, p)], check


@case("mot15_classes_ignored", "MOT15")
def _():
    """MOT15 keeps every ground-truth class, valid or not, as a pedestrian."""
    classes = [-1, 0, 1, 2, 7, 14, 99]
    g, p = [], []
    for f in range(1, 4):
        for k, c in enumerate(classes, start=1):
            g.append(gt(f, k, box(100 * k), cls=c, ncols=10))
            p.append(pred(f, 10 + k, shifted(box(100 * k), f - 1)))

    def check(exp, seqs):
        assert exp["combined"]["GT_Dets"] == n_gt(seqs) and exp["combined"]["Dets"] == n_pred(seqs)

    return [seq("seq-a", 3, g, p)], check


@case("no_preproc_classes_ignored", None)
def _():
    """With clean-up off, ground-truth class is never read: class 7 or 99 is a pedestrian."""
    classes = [1, 2, 6, 7, 8, 12, 14, 99]
    g, p = [], []
    for f in range(1, 4):
        for k, c in enumerate(classes, start=1):
            g.append(gt(f, k, box(100 * k), cls=c))
            p.append(pred(f, 10 + k, shifted(box(100 * k), 1)))

    def check(exp, seqs):
        c = exp["combined"]
        assert c["GT_Dets"] == n_gt(seqs) and c["Dets"] == n_pred(seqs) and c["CLR_TP"] == n_gt(seqs)

    return [seq("seq-a", 3, g, p)], check


@case("invalid_gt_class_without_tracker", "MOT17")
def _():
    """The 1..13 class check runs only in frames where both sides have rows.

    Classes 14, 0 and -1 sit in frames with no tracker boxes, so TrackEval raises
    nothing and drops them as non-pedestrians.
    """
    g = [gt(1, 1, box(0)), gt(2, 1, box(0)), gt(2, 2, box(100), cls=14), gt(3, 3, box(200), cls=0)]
    g.append(gt(3, 4, box(300), cls=-1))
    p = [pred(1, 1, box(0))]

    def check(exp, seqs):
        assert exp["combined"]["GT_Dets"] == 2, exp["combined"]["GT_Dets"]

    return [seq("seq-a", 3, g, p)], check


@case("empty_frames", "MOT17")
def _():
    """Frames with no ground truth, with no tracker boxes, with neither, and empty trailing frames."""
    g, p = [], []
    for f in (1, 2, 4, 6, 7, 8):  # no ground truth in 3, 5
        g.append(gt(f, 1, box(0)))
        g.append(gt(f, 2, box(100)))
    for f in (1, 2, 3, 6, 7, 8):  # no tracker boxes in 4, 5
        p.append(pred(f, 1, shifted(box(0), 0.5)))
        p.append(pred(f, 2, shifted(box(100), 2)))
    p.append(pred(3, 3, box(500)))

    def check(exp, seqs):
        c = exp["combined"]
        assert c["CLR_Frames"] == 10 and c["CLR_FP"] == 3 and c["CLR_FN"] == 2, c

    return [seq("seq-a", 10, g, p)], check


def _plain_sequence(name, frames=6, ids=(1, 2, 3)):
    g, p = [], []
    for f in range(1, frames + 1):
        for k, tid in enumerate(ids):
            g.append(gt(f, tid, box(100 * k)))
            if (f + k) % 4:
                p.append(pred(f, 10 + tid, shifted(box(100 * k), 0.4 * k + 0.1 * f)))
    return seq(name, frames, g, p)


@case("empty_tracker_sequence", "MOT17")
def _():
    """One sequence with an empty tracker file, combined with a normal one."""
    s = _plain_sequence("seq-b")
    s["pred"] = []

    def check(exp, seqs):
        b = exp["per_sequence"]["seq-b"]
        assert b["Dets"] == 0 and b["CLR_FN"] == n_gt([seqs[1]]) and b["CLR_Frames"] == 0, b

    return [_plain_sequence("seq-a"), s], check


@case("empty_gt_sequence", "MOT17")
def _():
    """One sequence with an empty ground-truth file, combined with a normal one."""
    s = _plain_sequence("seq-b")
    s["gt"] = []

    def check(exp, seqs):
        b = exp["per_sequence"]["seq-b"]
        assert b["GT_Dets"] == 0 and b["CLR_FP"] == n_pred([seqs[1]]) and b["CLR_Frames"] == 0, b

    return [_plain_sequence("seq-a"), s], check


@case("both_empty_sequence", "MOT17")
def _():
    s = seq("seq-b", 4, [], [])

    def check(exp, seqs):
        assert exp["per_sequence"]["seq-b"]["Dets"] == 0

    return [_plain_sequence("seq-a"), s], check


@case("only_empty_tracker", "MOT17")
def _():
    """The only sequence has no tracker boxes; COMBINED_SEQ is built from the early return."""
    s = _plain_sequence("seq-a")
    s["pred"] = []

    def check(exp, seqs):
        assert exp["combined"]["Dets"] == 0 and exp["combined"]["LocA"] == [1.0] * 19

    return [s], check


@case("only_empty_gt", None)
def _():
    s = _plain_sequence("seq-a")
    s["gt"] = []

    def check(exp, seqs):
        assert exp["combined"]["GT_Dets"] == 0 and exp["combined"]["CLR_FP"] == n_pred(seqs)

    return [s], check


@case("all_removed_by_cleanup", "MOT17")
def _():
    """Clean-up empties one side of a sequence, which then takes the early return.

    seq-a: every ground truth is a distractor, so no ground truth is left.
    seq-b: every tracker box sits on a distractor, so no tracker box is left.
    """
    a_g, a_p, b_g, b_p = [], [], [], []
    for f in range(1, 4):
        a_g.append(gt(f, 1, box(0), cls=7, zero_marked=0))
        a_g.append(gt(f, 2, box(100), cls=2, zero_marked=0))
        a_p.append(pred(f, 1, shifted(box(0), 1)))
        a_p.append(pred(f, 2, box(300)))  # unmatched, stays
        b_g.append(gt(f, 1, box(0)))
        b_g.append(gt(f, 2, box(100), cls=12, zero_marked=0))
        b_p.append(pred(f, 5, shifted(box(100), 1)))

    def check(exp, seqs):
        a, b = exp["per_sequence"]["seq-a"], exp["per_sequence"]["seq-b"]
        assert a["GT_Dets"] == 0 and a["Dets"] == 3 and b["Dets"] == 0 and b["GT_Dets"] == 3, (a, b)

    return [seq("seq-a", 3, a_g, a_p), seq("seq-b", 3, b_g, b_p), _plain_sequence("seq-c")], check


@case("id_switch", "MOT17")
def _():
    """Switches by swap, against the last match before a gap, and back and forth.

    gt 1, 2: their tracker ids swap at frame 5 (2 switches).
    gt 3: tracked by 20, absent frames 4-8, back tracked by 21 (1 switch after a gap).
    gt 4: tracked by 30, absent frames 4-8, back tracked by 30 (no switch; one fragment).
    gt 5: tracked by 40, by 41 in frame 5 only, by 40 again (2 switches).
    """
    g, p = [], []
    for f in range(1, 13):
        if f <= 8:
            g += [gt(f, 1, box(0)), gt(f, 2, box(100))]
            a, b = (10, 11) if f <= 4 else (11, 10)
            p += [pred(f, a, shifted(box(0), 0.5)), pred(f, b, shifted(box(100), 1))]
            g.append(gt(f, 5, box(400)))
            p.append(pred(f, 41 if f == 5 else 40, shifted(box(400), 0.2 * f)))
        if f <= 3 or f >= 9:
            g += [gt(f, 3, box(200)), gt(f, 4, box(300))]
            p += [pred(f, 20 if f <= 3 else 21, box(200)), pred(f, 30, shifted(box(300), 1.5))]

    def check(exp, seqs):
        assert exp["combined"]["IDSW"] == 5, exp["combined"]["IDSW"]

    return [seq("seq-a", 12, g, p)], check


@case("clear_memory_survives_empty_frame", "MOT17")
def _():
    """CLEAR's one-frame-back memory survives a frame it skips.

    In frame 3 ground truth A has two candidates: tracker 1 at IoU 0.6 (its match
    in frame 1) and tracker 2 at IoU 0.905. If frame 2 is skipped (no tracker
    boxes, no ground truth, or neither) the memory of frame 1 survives, tracker 1
    gets the 1000 bonus, and there is no switch and no new fragment. If frame 2
    has a tracker box elsewhere, the memory is cleared and tracker 2 wins.
    """
    a = box(0, w=10)
    far = box(500)

    def build(name, frame2_gt, frame2_pred):
        g = [gt(1, 1, a), *frame2_gt, gt(3, 1, a)]
        p = [pred(1, 1, a), *frame2_pred, pred(3, 1, shifted(a, 2.5)), pred(3, 2, shifted(a, 0.5))]
        return seq(name, 3, g, p)

    seqs = [
        build("no-gt-frame", [], [pred(2, 1, far)]),
        build("no-tracker-frame", [gt(2, 1, a)], []),
        build("nothing-frame", [], []),
        build("reset-frame", [gt(2, 1, a)], [pred(2, 3, far)]),
    ]

    def check(exp, seqs):
        s = exp["per_sequence"]
        for name in ("no-gt-frame", "no-tracker-frame", "nothing-frame"):
            assert s[name]["IDSW"] == 0 and s[name]["Frag"] == 0, (name, s[name]["IDSW"], s[name]["Frag"])
        assert s["reset-frame"]["IDSW"] == 1 and s["reset-frame"]["Frag"] == 1

    return seqs, check


@case("fragmentation", "MOT17")
def _():
    """Fragments, and mostly-tracked ratios sitting exactly on 0.8 and 0.2.

    A frame where a track is "untracked" still has its tracker box, moved to IoU
    0.333, so CLEAR does not skip the frame.
    gt 1: TTFFTTFT over 8 frames (3 runs, Frag 2; 5/8, partly tracked).
    gt 2: 4 of 5 (ratio 0.8 is not above 0.8: partly tracked; Frag 1).
    gt 3: 5 of 5 (mostly tracked). gt 4: 1 of 5 (ratio 0.2: partly tracked).
    gt 5: 1 of 10 (mostly lost). gt 6: 0 of 3 (mostly lost).
    """
    patterns = {1: "TTFFTTFT", 2: "TTFTT", 3: "TTTTT", 4: "FFTFF", 5: "FFFFTFFFFF", 6: "FFF"}
    g, p = [], []
    for f in range(1, 11):
        for tid, pat in patterns.items():
            if f <= len(pat):
                b = box(100 * tid)
                g.append(gt(f, tid, b))
                p.append(pred(f, 10 + tid, b if pat[f - 1] == "T" else shifted(b, 5)))

    def check(exp, seqs):
        c = exp["combined"]
        assert (c["Frag"], c["MT"], c["PT"], c["ML"]) == (3, 1, 3, 2), (c["Frag"], c["MT"], c["PT"], c["ML"])

    return [seq("seq-a", 10, g, p)], check


@case("tied_identical_boxes", "MOT17")
def _():
    """Identical boxes on both sides, so every assignment is co-optimal.

    scipy returns the identity pairing on an all-equal matrix, in file row
    order. The ground-truth rows swap order every frame, so HOTA pairs gt 1 with
    tracker 5 in odd frames and with tracker 6 in even ones, and AssA is 1/3, not
    1: a port that sorts rows by id inside a frame gets a different number.
    tie-then-split: frame 1 is a tie, after which the boxes separate the other
    way round, so the identity choice in frame 1 costs two id switches.
    """
    b = box(0)
    g, p = [], []
    for f in range(1, 7):
        order = (1, 2) if f % 2 else (2, 1)
        g += [gt(f, tid, b) for tid in order]
        p += [pred(f, 5, b), pred(f, 6, b)]
    tie2 = seq("tie-2x2", 6, g, p)

    g, p = [], []
    gt_orders = [(30, 10, 20), (10, 20, 30), (20, 30, 10), (30, 20, 10), (10, 30, 20)]
    trk_orders = [(9, 7, 8), (9, 8, 7), (7, 9, 8), (8, 7, 9), (7, 8, 9)]
    for f, (go, to) in enumerate(zip(gt_orders, trk_orders), start=1):
        g += [gt(f, tid, b) for tid in go]
        p += [pred(f, tid, b) for tid in to]
    tie3 = seq("tie-3x3", 5, g, p)

    g = [gt(1, 1, b), gt(1, 2, b)]
    p = [pred(1, 5, b), pred(1, 6, b)]
    for f in range(2, 5):
        g += [gt(f, 1, box(0)), gt(f, 2, box(100))]
        p += [pred(f, 5, box(100)), pred(f, 6, box(0))]
    split = seq("tie-then-split", 4, g, p)

    def check(exp, seqs):
        s = exp["per_sequence"]
        assert abs(s["tie-2x2"]["AssA"][0] - 1 / 3) < 1e-12, s["tie-2x2"]["AssA"][0]
        assert s["tie-2x2"]["IDSW"] == 0
        assert s["tie-then-split"]["IDSW"] == 2, s["tie-then-split"]["IDSW"]

    return [tie2, tie3, split], check


@case("zero_area_boxes", "MOT17")
def _():
    """A box whose area is at most eps has IoU 0, even with an identical box.

    Area is computed from the corners, ((x + w) - x) * ((y + h) - y), not w * h.
    """
    pairs = [
        (box(0, w=0.0), box(0, w=0.0)),
        (box(100, h=0.0), box(100, h=0.0)),
        (box(200), box(200, w=0.0)),
        (box(300, w=1e-9, h=1e-9), box(300, w=1e-9, h=1e-9)),  # area 1e-18 <= eps
        (box(400, w=1e-7, h=1e-7), box(400, w=1e-7, h=1e-7)),  # area ~1e-14 > eps: IoU 1
        (box(500), box(500)),
    ]
    g = [gt(f, f, a) for f, (a, _) in enumerate(pairs, start=1)]
    p = [pred(f, f, b) for f, (_, b) in enumerate(pairs, start=1)]

    def check(exp, seqs):
        c = exp["combined"]
        assert (c["CLR_TP"], c["CLR_FN"], c["CLR_FP"]) == (2, 4, 4), c

    return [seq("seq-a", len(pairs), g, p)], check


@case("negative_size_boxes", None)
def _():
    """A negative width or height gives a negative area, which is at most eps: IoU 0."""
    pairs = [
        (box(0, w=-10.0), box(0, w=-10.0)),
        (box(100), box(110, w=-10.0)),
        (box(200, h=-20.0), box(200, h=-20.0)),
        (box(300), box(300)),
    ]
    g = [gt(f, f, a) for f, (a, _) in enumerate(pairs, start=1)]
    p = [pred(f, f, b) for f, (_, b) in enumerate(pairs, start=1)]

    def check(exp, seqs):
        assert exp["combined"]["CLR_TP"] == 1, exp["combined"]["CLR_TP"]

    return [seq("seq-a", len(pairs), g, p)], check


@case("large_and_sparse_ids", "MOT17")
def _():
    """Ids are renumbered 0..n per sequence after clean-up.

    Tracker 888888 sits only on a distractor and is removed, so it is not counted
    in IDs; gt 55 is only ever zero-marked, so it is not counted in GT_IDs, and
    tracker 4242 on it stays as a false positive. Id 0 is a valid id.
    """
    gt_ids = [0, 3, 999999, 123456]
    trk_ids = [7, 500000, 1, 0]
    g, p = [], []
    for f in range(1, 5):
        for k, (gid, tid) in enumerate(zip(gt_ids, trk_ids)):
            g.append(gt(f, gid, box(100 * k)))
            if gid == 999999 and f >= 3:
                tid = 2_000_000
            p.append(pred(f, tid, shifted(box(100 * k), 0.3 * k)))
        g.append(gt(f, 77, box(1000), cls=7, zero_marked=0))
        p.append(pred(f, 888888, shifted(box(1000), 1)))
        g.append(gt(f, 55, box(1200), zero_marked=0))
        p.append(pred(f, 4242, box(1200)))

    def check(exp, seqs):
        c = exp["combined"]
        assert (c["GT_IDs"], c["IDs"], c["IDSW"]) == (4, 6, 1), (c["GT_IDs"], c["IDs"], c["IDSW"])

    return [seq("seq-a", 4, g, p)], check


@case("tracker_column_counts", "MOT17")
def _():
    """Tracker files of 7, 8, 9 and 10 columns; ground truth of 8, 9 and 10.

    With 7 columns every row is class 1. With 8 or more, column 7 is the class,
    and -1, 0 and 1 all pass: TrackEval only refuses a class above 1, and never
    filters tracker rows by class.
    """
    specs = [("cols-07", 7, 8, -1), ("cols-08", 8, 9, 1), ("cols-09", 9, 9, 0), ("cols-10", 10, 10, -1)]
    seqs = []
    for name, pcols, gcols, cls in specs:
        g, p = [], []
        for f in range(1, 5):
            for k in range(2):
                g.append(gt(f, k + 1, box(100 * k), ncols=gcols))
                p.append(pred(f, 5 + k, shifted(box(100 * k), 0.5 * k + 0.25 * f), cls=cls, ncols=pcols))
        seqs.append(seq(name, 4, g, p))

    def check(exp, seqs):
        assert exp["combined"]["Dets"] == n_pred(seqs)
        for s in seqs:
            assert len({len(r) for r in s["pred"]}) == 1

    return seqs, check


@case("multi_sequence", "MOT17")
def _():
    """Three sequences of very different size and quality, to test the combination.

    AssA, AssRe and AssPr combine weighted by HOTA_TP, and LocA by HOTA_TP too;
    a plain mean over sequences gives a different number here.
    """
    g, p = [], []
    for f in range(1, 31):
        for k in range(3):
            g.append(gt(f, k + 1, box(100 * k)))
            tid = 10 + k if not (k == 2 and f > 20) else 99
            p.append(pred(f, tid, shifted(box(100 * k), 0.1 * (f % 7))))
    big = seq("seq-a", 30, g, p)

    g, p = [], []
    for f in range(1, 11):
        for k in range(2):
            g.append(gt(f, k + 1, box(100 * k)))
            p.append(pred(f, 20 + (f + k) % 2, shifted(box(100 * k), 1.5)))
    poor = seq("seq-b", 10, g, p)

    g = [gt(f, 1, box(0)) for f in range(1, 4)]
    p = [pred(f, 1, shifted(box(0), 2)) for f in range(1, 4)] + [pred(f, 2 + f, box(300 + 50 * f)) for f in (1, 2, 3)]
    small = seq("seq-c", 3, g, p)

    def check(exp, seqs):
        c, s = exp["combined"], exp["per_sequence"]
        plain = np.mean([s[n]["AssA"][9] for n in s])
        assert abs(c["AssA"][9] - plain) > 1e-3, (c["AssA"][9], plain)

    return [big, poor, small], check


# ---------------------------------------------------------------------------
# Random cases
# ---------------------------------------------------------------------------


def simulate(rng, name, num_frames, num_tracks, *, class_weights, zero_marked_rate, pred_cols):
    """A crowd of linear tracks with noisy detections, misses, false positives and id swaps.

    Ground-truth rows are written id-major (as MOTChallenge gt.txt files are);
    tracker rows frame-major, shuffled inside each frame.
    """
    classes = list(class_weights)
    weights = [class_weights[c] for c in classes]
    gt_ids = rng.sample(range(1, 4000), num_tracks)
    tracks = []
    for gid in gt_ids:
        length = rng.randint(max(8, num_frames // 3), num_frames)
        start = rng.randint(1, num_frames - length + 1)
        cls = rng.choices(classes, weights)[0]
        w = rng.uniform(20, 90)
        static = cls in (7, 8)
        tracks.append(
            {
                "gid": gid,
                "cls": cls,
                "start": start,
                "end": start + length - 1,
                "x": rng.uniform(-20, 1800),
                "y": rng.uniform(0, 900),
                "w": w,
                "h": w * rng.uniform(1.8, 2.8),
                "vx": 0.0 if static else rng.gauss(0, 3),
                "vy": 0.0 if static else rng.gauss(0, 1),
                "zero_marked_track": cls != 1 and rng.random() < 0.8,
            }
        )

    next_id = [rng.randint(1, 40)]

    def fresh():
        next_id[0] += rng.randint(1, 6)
        return next_id[0]

    state = {t["gid"]: {"tid": fresh(), "occluded": 0, "detect": rng.random() < 0.9 or t["cls"] == 1} for t in tracks}
    gt_rows, pred_frames = [], {f: [] for f in range(1, num_frames + 1)}
    fp_tracks = []
    for f in range(1, num_frames + 1):
        alive = [t for t in tracks if t["start"] <= f <= t["end"]]
        if len(alive) >= 2 and rng.random() < 0.04:  # an id swap between two live tracks
            a, b = rng.sample(alive, 2)
            state[a["gid"]]["tid"], state[b["gid"]]["tid"] = state[b["gid"]]["tid"], state[a["gid"]]["tid"]
        for t in alive:
            dt = f - t["start"]
            x = t["x"] + t["vx"] * dt + rng.gauss(0, 0.5)
            y = t["y"] + t["vy"] * dt + rng.gauss(0, 0.3)
            gbox = [round(x, 2), round(y, 2), round(t["w"], 2), round(t["h"], 2)]
            zm = 0 if t["zero_marked_track"] or rng.random() < zero_marked_rate else 1
            gt_rows.append(gt(f, t["gid"], gbox, cls=t["cls"], zero_marked=zm, vis=round(rng.random(), 2)))
            st = state[t["gid"]]
            if st["occluded"] > 0:
                st["occluded"] -= 1
                if st["occluded"] == 0 and rng.random() < 0.4:
                    st["tid"] = fresh()  # re-acquired under a new id
                continue
            if rng.random() < 0.03:
                st["occluded"] = rng.randint(2, 12)
                continue
            if not st["detect"] or rng.random() < 0.08:
                continue
            loose = rng.random() < 0.3  # a loose box, so IoUs spread over the low alphas too
            s = rng.uniform(0.7, 1.3) if loose else rng.uniform(0.9, 1.1)
            pw, ph = t["w"] * s, t["h"] * rng.uniform(0.92, 1.08)
            px = x + rng.gauss(0, (0.3 if loose else 0.07) * t["w"]) - (pw - t["w"]) / 2
            py = y + rng.gauss(0, (0.15 if loose else 0.04) * t["h"]) - (ph - t["h"]) / 2
            pbox = [round(px, 2), round(py, 2), round(pw, 2), round(ph, 2)]
            pred_frames[f].append(pred(f, st["tid"], pbox, conf=round(rng.uniform(0.3, 1.0), 2), ncols=pred_cols))
        for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):  # new false-positive tracks
            fx, fy, fw = rng.uniform(0, 1800), rng.uniform(0, 900), rng.uniform(20, 80)
            if alive and rng.random() < 0.5:  # beside a real object, overlapping it a little
                t = rng.choice(alive)
                fw = t["w"]
                fx = t["x"] + t["vx"] * (f - t["start"]) + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9) * fw
                fy = t["y"] + t["vy"] * (f - t["start"])
            fp_tracks.append([fresh(), f + rng.randint(0, 5), fx, fy, fw])
        for fp in fp_tracks:
            if f <= fp[1]:
                w = fp[4]
                fbox = [
                    round(fp[2] + rng.gauss(0, 2), 2),
                    round(fp[3] + rng.gauss(0, 2), 2),
                    round(w, 2),
                    round(2.2 * w, 2),
                ]
                pred_frames[f].append(pred(f, fp[0], fbox, conf=round(rng.uniform(0.1, 0.6), 2), ncols=pred_cols))
        fp_tracks = [fp for fp in fp_tracks if f < fp[1]]
    gt_rows.sort(key=lambda r: (r[1], r[0]))
    pred_rows = []
    for f in range(1, num_frames + 1):
        rows = pred_frames[f]
        rng.shuffle(rows)
        pred_rows += rows
    return seq(name, num_frames, gt_rows, pred_rows)


def _random_check(benchmark):
    def check(exp, seqs):
        c = exp["combined"]
        assert max(c["HOTA"]) - min(c["HOTA"]) > 0.1, c["HOTA"]
        assert abs(c["HOTA"][0] - c["HOTA"][9]) > 0.02, "no pairs between IoU 0.05 and 0.5"
        assert c["IDSW"] > 0 and c["Frag"] > 0, (c["IDSW"], c["Frag"])
        if benchmark is not None:
            assert c["Dets"] < n_pred(seqs), "no tracker box was removed as a distractor"

    return check


@case("random_mot17", "MOT17")
def _():
    rng = random.Random(SEED + 17)
    weights = {1: 0.82, 2: 0.04, 7: 0.06, 8: 0.03, 12: 0.02, 3: 0.03}
    seqs = [
        simulate(rng, f"rand-{i:02d}", 100, 34, class_weights=weights, zero_marked_rate=0.02, pred_cols=10)
        for i in range(1, 4)
    ]
    return seqs, _random_check("MOT17")


@case("random_mot20", "MOT20")
def _():
    rng = random.Random(SEED + 20)
    weights = {1: 0.8, 6: 0.08, 7: 0.06, 2: 0.03, 13: 0.03}
    seqs = [
        simulate(rng, f"rand-{i:02d}", 60, 52, class_weights=weights, zero_marked_rate=0.02, pred_cols=9)
        for i in range(1, 3)
    ]

    def check(exp, seqs):
        _random_check("MOT20")(exp, seqs)
        as_mot17 = expected_for("MOT17", seqs)["combined"]["Dets"]
        assert exp["combined"]["Dets"] < as_mot17, "no tracker box was removed on class 6"

    return seqs, check


@case("random_no_preproc", None)
def _():
    rng = random.Random(SEED)
    seqs = [
        simulate(rng, f"rand-{i:02d}", 60, 24, class_weights={1: 1.0}, zero_marked_rate=0.0, pred_cols=7)
        for i in range(1, 3)
    ]
    return seqs, _random_check(None)


# ---------------------------------------------------------------------------
# Inputs TrackEval refuses
# ---------------------------------------------------------------------------

ERROR_CASES = []


def error_case(name, benchmark, sequences):
    ERROR_CASES.append((name, benchmark, sequences))


def _errors():
    b0, b1 = box(0), box(100)
    ok_g = [gt(1, 1, b0), gt(2, 1, b0)]
    ok_p = [pred(1, 1, b0), pred(2, 1, b0)]
    error_case("duplicate_tracker_id", "MOT17", [seq("seq-a", 2, ok_g, [pred(1, 5, b0), pred(1, 5, b1)])])
    error_case("duplicate_gt_id", "MOT17", [seq("seq-a", 2, [gt(1, 1, b0), gt(1, 1, b1)], ok_p)])
    error_case(
        "duplicate_gt_id_on_zero_marked_distractor",
        "MOT17",
        [seq("seq-a", 2, [gt(1, 1, b0), gt(1, 1, b1, cls=7, zero_marked=0)], ok_p)],
    )
    error_case("tracker_frame_zero", "MOT17", [seq("seq-a", 2, ok_g, [*ok_p, pred(0, 1, b0)])])
    error_case("tracker_frame_past_end", "MOT17", [seq("seq-a", 2, ok_g, [*ok_p, pred(3, 1, b0)])])
    error_case("gt_frame_past_end", "MOT17", [seq("seq-a", 2, [*ok_g, gt(3, 1, b0)], ok_p)])
    error_case("gt_frame_negative", "MOT17", [seq("seq-a", 2, [*ok_g, gt(-1, 1, b0)], ok_p)])
    for name, bench in (
        ("tracker_class_2", "MOT17"),
        ("tracker_class_2_mot15", "MOT15"),
        ("tracker_class_2_no_preproc", None),
    ):
        error_case(name, bench, [seq("seq-a", 2, ok_g, [pred(1, 1, b0), pred(2, 1, b0, cls=2)])])
    for cls in (14, 0, -1):
        error_case(
            f"gt_class_{cls}".replace("-", "minus_"),
            "MOT17",
            [seq("seq-a", 2, [gt(1, 1, b0), gt(2, 1, b0, cls=cls)], ok_p)],
        )
    error_case("gt_seven_columns", "MOT17", [seq("seq-a", 2, [r[:7] for r in ok_g], ok_p)])
    error_case("tracker_six_columns", "MOT17", [seq("seq-a", 2, ok_g, [r[:6] for r in ok_p])])
    error_case(
        "negative_ids_collide_after_renumbering",
        None,
        [seq("seq-a", 2, ok_g, [pred(1, -1, b0), pred(1, 3, b1), pred(2, -1, b0)])],
    )


def run_error_case(benchmark, sequences):
    try:
        run_trackeval(benchmark, sequences)
    except Exception as e:  # noqa: BLE001 - recording whatever TrackEval raises is the point
        return {"type": type(e).__name__, "message": str(e)}
    raise AssertionError("TrackEval accepted an input expected to fail")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _dump(obj, depth=0):
    """JSON with one row (or one per-threshold list) per line, so diffs stay readable."""
    pad, inner = " " * depth, " " * (depth + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{inner}{json.dumps(k)}: {_dump(v, depth + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list) and any(isinstance(v, (dict, list)) for v in obj):
        return "[\n" + ",\n".join(inner + _dump(v, depth + 1) for v in obj) + "\n" + pad + "]"
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def main():
    _check_trackeval_commit()
    cases = []
    seen = set()
    for name, benchmark, build in CASES:
        assert name not in seen, name
        seen.add(name)
        sequences, check = build()
        names = [s["name"] for s in sequences]
        assert names == sorted(names) and len(set(names)) == len(names), name
        if benchmark is None:
            for s in sequences:
                assert all(int(r[6]) != 0 for r in s["gt"]), f"{name}: a null case must keep column 6 nonzero"
        expected = expected_for(benchmark, sequences)
        check(expected, sequences)
        cases.append({"name": name, "benchmark": benchmark, "sequences": sequences, "expected": expected})
        c = expected["combined"]
        print(
            f"{name:36s} {benchmark or 'null':6s} rows {n_gt(sequences):5d}/{n_pred(sequences):5d}  "
            f"HOTA {np.mean(c['HOTA']):.4f}  MOTA {c['MOTA']:+.4f}  IDF1 {c['IDF1']:.4f}  "
            f"IDSW {c['IDSW']:3d}  Frag {c['Frag']:3d}  Dets {c['Dets']}"
        )

    # Cross-case checks: the distractor rules differ between benchmarks as claimed.
    by_name = {c["name"]: c["expected"]["combined"] for c in cases}
    assert by_name["distractors_mot16"] == by_name["distractors_mot17"]
    assert (
        by_name["distractors_mot20"]["Dets"]
        < by_name["distractors_mot17"]["Dets"]
        < by_name["distractors_mot15"]["Dets"]
    )

    _errors()
    errors = []
    for name, benchmark, sequences in ERROR_CASES:
        err = run_error_case(benchmark, sequences)
        errors.append({"name": name, "benchmark": benchmark, "sequences": sequences, "trackeval_error": err})
        print(f"error {name:40s} {err['type']}: {err['message'][:70]}")

    payload = {
        "trackeval_commit": TRACKEVAL_COMMIT,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "alphas": ALPHAS,
        "eps": EPS,
        "cases": cases,
        "error_cases": errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(_dump(payload) + "\n")
    print(f"wrote {len(cases)} cases and {len(errors)} error cases to {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
