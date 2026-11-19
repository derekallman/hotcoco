"""Tracking parity: hotcoco vs TrackEval, through the Python API.

The fixture (`fixtures/trackeval_expected.json`) holds MOTChallenge rows and
what TrackEval `12c8791` computed for them under numpy 1.23.5 and scipy 1.15.3;
`scripts/gen_tracking_fixtures.py` regenerates it (`just gen-tracking-fixtures`).
The Rust suite (`crates/hotcoco/tests/tracking.rs`) checks the core; this file
checks the same numbers through the bindings and the two input paths they add:

- `TrackingEval.update` on numpy arrays, one sequence per call, and on lists;
- `update_folders` on the MOTChallenge folder layout, written to a temp dir.

Every field TrackEval reports, combined and per sequence, is held to exact
equality: integers as ints, floats to the bit. The per-threshold fields are read
from `curves()`, the rest from `results()` and `sequence_results()`.

    uv run pytest tests/test_parity_tracking.py -v
"""

from __future__ import annotations

import copy
import json
import pickle

import numpy as np
import pytest
from helpers import FIXTURES_DIR
from hotcoco import tracking
from hotcoco.tracking import TrackingEval

FIXTURE = json.loads((FIXTURES_DIR / "trackeval_expected.json").read_text())
CASES = FIXTURE["cases"]
ERROR_CASES = FIXTURE["error_cases"]

# TrackEval raises on these and hotcoco does not, by decision: without a
# benchmark the class column is not read, and a prediction's confidence column
# is never read, so 6 columns are enough.
DELIBERATELY_ACCEPTED = {"tracker_class_2_no_preproc", "tracker_six_columns"}


def case_named(name):
    return next(c for c in CASES if c["name"] == name)


def evaluator(case, sequences=None, rows=np.asarray):
    ev = TrackingEval(case["benchmark"])
    for s in case["sequences"] if sequences is None else sequences:
        ev.update(s["name"], rows(s["gt"]), rows(s["pred"]), num_frames=s["num_frames"])
    return ev


def mismatches(at, scalars, curves, want):
    """Every field of ``want`` that ``scalars`` (for a number) or ``curves`` (for a
    per-threshold list) does not hold exactly, and every field of ours TrackEval
    does not report."""
    found = []
    for field, w in want.items():
        if isinstance(w, list):
            g = curves.get(field)
            same = g is not None and len(g) == len(w) and all(a == b for a, b in zip(g, w))
        else:
            g = scalars.get(field)
            # An int is an int and a float a float: `1 == 1.0`, so check the type too.
            same = type(g) is type(w) and g == w
        if not same:
            found.append(f"{at} {field}: {g!r}, expected {w!r}")
    ours = (set(scalars) | set(curves)) - {"alphas"}
    found += [f"{at}: {field} is not a field TrackEval reports" for field in sorted(ours - set(want))]
    return found


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_matches_trackeval_on_every_field(case):
    ev = evaluator(case)
    found = mismatches("combined", ev.results(), ev.curves(), case["expected"]["combined"])
    per_sequence = ev.sequence_results()
    assert list(per_sequence) == list(case["expected"]["per_sequence"])
    for seq, want in case["expected"]["per_sequence"].items():
        found += mismatches(seq, per_sequence[seq], ev.curves(seq), want)
    assert not found, "\n".join(found)


def test_the_comparison_tells_a_wrong_number_apart():
    """The gate is only as strong as its comparison: a float off by one ulp, an
    int returned as a float, and a field TrackEval does not have must each show."""
    case = CASES[0]
    ev = evaluator(case)
    want = case["expected"]["combined"]
    scalars, curves = ev.results(), ev.curves()
    assert mismatches("c", scalars, curves, want) == []
    assert mismatches("c", {**scalars, "MOTA": np.nextafter(scalars["MOTA"], 2.0)}, curves, want)
    assert mismatches("c", {**scalars, "IDSW": float(scalars["IDSW"])}, curves, want)
    assert mismatches("c", {**scalars, "extra": 1.0}, curves, want)
    bumped = {**curves, "HOTA": [*curves["HOTA"][:-1], np.nextafter(curves["HOTA"][-1], 2.0)]}
    assert mismatches("c", scalars, bumped, want)


def test_alphas_are_trackevals():
    ev = evaluator(CASES[0])
    assert ev.curves()["alphas"] == FIXTURE["alphas"]
    for seq in ev.sequence_results():
        assert ev.curves(seq)["alphas"] == FIXTURE["alphas"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_the_headline_is_the_mean_over_thresholds(case):
    """TrackEval's summary prints ``mean(HOTA)``, not ``sqrt(mean DetA * mean AssA)``."""
    results = evaluator(case).results()
    for field in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA"):
        assert results[field] == float(np.mean(case["expected"]["combined"][field])), field


def test_lists_score_as_arrays_do():
    for case in CASES:
        assert evaluator(case, rows=list).sequence_results() == evaluator(case).sequence_results(), case["name"]


@pytest.mark.parametrize("case", ERROR_CASES, ids=lambda c: c["name"])
def test_refuses_what_trackeval_refuses(case):
    def run():
        evaluator(case)

    if case["name"] in DELIBERATELY_ACCEPTED:
        run()
        return
    with pytest.raises(ValueError, match="seq-a"):
        run()


def test_the_deliberately_accepted_cases_are_still_in_the_fixture():
    assert DELIBERATELY_ACCEPTED <= {c["name"] for c in ERROR_CASES}


# --- the folder layout -----------------------------------------------------------


def write_rows(path, rows):
    path.write_text("".join(",".join(repr(v) for v in row) + "\n" for row in rows))


def write_benchmark(root, case):
    """The case as TrackEval lays a benchmark out: ``gt/<seq>/gt/gt.txt`` beside
    ``gt/<seq>/seqinfo.ini``, and ``pred/<seq>.txt``."""
    gt_root, pred_root = root / "gt", root / "pred"
    pred_root.mkdir(parents=True)
    for s in case["sequences"]:
        (gt_root / s["name"] / "gt").mkdir(parents=True)
        write_rows(gt_root / s["name"] / "gt" / "gt.txt", s["gt"])
        (gt_root / s["name"] / "seqinfo.ini").write_text(
            f"[Sequence]\nname={s['name']}\nframeRate=30\nseqLength={s['num_frames']}\n"
        )
        write_rows(pred_root / f"{s['name']}.txt", s["pred"])
    return gt_root, pred_root


@pytest.mark.parametrize("name", ["multi_sequence", "random_mot17", "tracker_column_counts"])
def test_folders_score_as_the_arrays_do(tmp_path, name):
    case = case_named(name)
    gt_root, pred_root = write_benchmark(tmp_path, case)
    ev = TrackingEval(case["benchmark"])
    assert ev.update_folders(gt_root, pred_root) == len(case["sequences"])
    arrays = evaluator(case)
    assert ev.sequence_results() == arrays.sequence_results()
    assert ev.results() == arrays.results()
    assert ev.curves() == arrays.curves()


def test_a_seqmap_picks_the_sequences(tmp_path):
    case = case_named("multi_sequence")
    gt_root, pred_root = write_benchmark(tmp_path, case)
    first = case["sequences"][0]
    seqmap = tmp_path / "seqmap.txt"
    seqmap.write_text(f"name\n{first['name']}\n")
    ev = TrackingEval(case["benchmark"])
    assert ev.update_folders(str(gt_root), str(pred_root), seqmap=str(seqmap)) == 1
    assert ev.sequence_results() == evaluator(case, [first]).sequence_results()
    # The sequence is now evaluated, so adding the whole benchmark is refused, and nothing is added.
    with pytest.raises(ValueError, match=first["name"]):
        ev.update_folders(gt_root, pred_root)
    assert list(ev.sequence_results()) == [first["name"]]


def test_a_missing_folder_or_file_is_file_not_found_naming_it(tmp_path):
    case = case_named("multi_sequence")
    gt_root, pred_root = write_benchmark(tmp_path, case)
    missing = tmp_path / "nowhere"
    gone = pred_root / f"{case['sequences'][-1]['name']}.txt"
    gone.unlink()
    for call, path in [
        (lambda: TrackingEval().update_folders(missing, pred_root), missing),
        (lambda: TrackingEval().update_folders(gt_root, pred_root), gone),
        (lambda: TrackingEval().update_folders(gt_root, pred_root, seqmap=missing), missing),
    ]:
        with pytest.raises(FileNotFoundError) as caught:
            call()
        assert str(path) in str(caught.value), caught.value


# --- shards, pickling, and call order ---------------------------------------------


def test_shards_merge_to_the_whole():
    case = case_named("random_mot17")
    whole = evaluator(case)
    shards = [pickle.loads(pickle.dumps(evaluator(case, [s]))) for s in case["sequences"]]  # the DDP round trip
    merged = shards[-1]
    for shard in shards[:-1]:
        merged.merge(shard)
    assert merged.sequence_results() == whole.sequence_results()
    assert merged.results() == whole.results()
    assert merged.curves() == whole.curves()
    assert merged.report() == whole.report()


def test_merge_refuses_another_configuration_a_shared_sequence_or_itself():
    case = case_named("multi_sequence")
    one, two = case["sequences"][:2]
    ev = evaluator(case, [one])
    with pytest.raises(ValueError, match="different benchmark or threshold"):
        ev.merge(evaluator({**case, "benchmark": "MOT20"}, [two]))
    other = TrackingEval(case["benchmark"], clear_threshold=0.6)
    other.update(two["name"], np.asarray(two["gt"]), np.asarray(two["pred"]), num_frames=two["num_frames"])
    with pytest.raises(ValueError, match="different benchmark or threshold"):
        ev.merge(other)
    with pytest.raises(ValueError, match=one["name"]):
        ev.merge(evaluator(case, [one]))
    with pytest.raises(ValueError, match="itself"):
        ev.merge(ev)
    assert list(ev.sequence_results()) == [one["name"]], "nothing added on an error"


def test_an_evaluator_pickles_and_copies():
    case = case_named("random_mot20")
    ev = TrackingEval(case["benchmark"], clear_threshold=0.4, identity_threshold=0.6)
    for s in case["sequences"]:
        ev.update(s["name"], np.asarray(s["gt"]), np.asarray(s["pred"]), num_frames=s["num_frames"])
    for back in (pickle.loads(pickle.dumps(ev)), copy.deepcopy(ev), copy.copy(ev)):
        assert (back.benchmark, back.clear_threshold, back.identity_threshold) == ("MOT20", 0.4, 0.6)
        assert back.sequence_results() == ev.sequence_results()
        assert back.results() == ev.results() and back.stats == ev.stats
    empty = pickle.loads(pickle.dumps(TrackingEval("mot15")))
    assert empty.benchmark == "MOT15" and empty.sequence_results() == {}


def test_reading_results_before_any_update_is_a_runtime_error():
    ev = TrackingEval()
    for call in (
        ev.results,
        lambda: ev.stats,
        ev.curves,
        lambda: ev.curves("seq-a"),
        ev.report,
        ev.summarize,
        ev.summary_lines,
    ):
        with pytest.raises(RuntimeError, match="update"):
            call()
    assert ev.sequence_results() == {}
    assert ev.reference_deviations() == [] and ev.provenance() == "parity_verified"


def test_an_unknown_sequence_is_a_key_error():
    ev = evaluator(case_named("multi_sequence"))
    with pytest.raises(KeyError, match="seq-z"):
        ev.curves("seq-z")


# --- the report and the printed tables ---------------------------------------------


def test_report_contract():
    case = case_named("multi_sequence")
    ev = evaluator(case)
    report = ev.report()
    assert report["task"] == "tracking"
    assert report["provenance"] == "parity_verified"
    assert len(tracking.METRIC_NAMES) == 44 and tracking.METRIC_NAMES[:3] == ["HOTA", "DetA", "AssA"]
    assert list(ev.results()) == tracking.METRIC_NAMES
    assert report["metrics"] == {k: float(v) for k, v in ev.results().items()}
    assert ev.stats == [float(v) for v in ev.results().values()]
    assert report["per_class"] == {}
    assert report["per_group"] == {
        seq: {k: float(v) for k, v in r.items() if k != "Frames"} for seq, r in ev.sequence_results().items()
    }
    assert report["curves"] == ev.curves()
    assert report["params"] == {
        "benchmark": "MOT17",
        "clear_threshold": 0.5,
        "identity_threshold": 0.5,
        "n_sequences": 3,
        "num_alphas": 19,
    }


def test_summary_tables(capsys):
    case = case_named("multi_sequence")
    ev = evaluator(case)
    lines = ev.summary_lines()
    for table in ("HOTA", "CLEAR", "Identity", "Count"):
        assert any(line.split()[0] == table for line in lines if line), table
    for s in case["sequences"]:
        assert any(line.startswith(s["name"]) for line in lines), s["name"]
    assert sum(line.startswith("COMBINED") for line in lines) == 4
    ev.summarize()
    assert capsys.readouterr().out.splitlines() == lines
