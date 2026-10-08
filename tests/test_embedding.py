"""What a library that embeds hotcoco (Lightning, torchmetrics) needs from it.

Three small contracts, none needing `data/`:

- A threshold grid built in `float32` is evaluated as given, as pycocotools
  evaluates it, and is not a deviation: no warning, and the run stays
  `parity_verified`.
- `summary_lines()` is the quiet path: it fills `stats` without printing or
  warning, and `reference_deviations()` still reports what a warning would have.
- `hotcoco.__version__` exists and names the binary that is loaded.
"""

import importlib.metadata
import warnings

import hotcoco
import numpy as np
import pytest
from helpers import float32_grids, grid_sensitive_records
from hotcoco import Params, StreamingEval

IOU_F32, REC_F32 = float32_grids()


def tie_sensitive_eval(n_gt=20, iou_thrs=None, rec_thrs=None):
    """``helpers.grid_sensitive_records`` streamed, then accumulated on the given grids."""
    se = StreamingEval([{"id": 1, "name": "a"}])
    images, gts, dts = grid_sensitive_records(n_gt)
    se.update(images, gts, dts)
    ev = se.finalize()
    if iou_thrs is not None:
        ev.params.iou_thrs = iou_thrs
    if rec_thrs is not None:
        ev.params.rec_thrs = rec_thrs
    ev.accumulate()
    return ev


def bits(array):
    return np.ascontiguousarray(array, dtype=np.float64).view(np.uint64).tolist()


def test_fixture_grids_really_drift():
    """A rounding test on a grid that was already exact would pass for nothing."""
    default = Params()
    for f32, exact in ((IOU_F32, default.iou_thrs), (REC_F32, default.rec_thrs)):
        drift = np.abs(np.array(f32) - np.array(exact)).max()
        assert 1e-9 < drift < 1e-6


class TestFloat32Grids:
    """torchmetrics hands every COCO backend `torch.linspace` grids. pycocotools
    evaluates them as given, so hotcoco must too, or one grid gives two answers.
    `test_parity.py` checks the arrays against pycocotools bit for bit."""

    def test_setters_store_the_grid_as_given(self):
        p = Params()
        p.iou_thrs = IOU_F32
        p.rec_thrs = REC_F32
        assert p.iou_thrs == IOU_F32
        assert p.rec_thrs == REC_F32

    def test_camel_case_aliases_store_it_too(self):
        p = Params()
        p.iouThrs = IOU_F32
        p.recThrs = REC_F32
        assert p.iouThrs == IOU_F32
        assert p.recThrs == REC_F32

    @pytest.mark.parametrize("n_gt", [20, 25, 50, 100])
    def test_float32_grids_change_the_numbers(self, n_gt):
        """Why the grid must be kept rather than snapped: recall `k / n_gt` is exactly a
        grid point, and a grid point one ulp higher excludes it, so 60-240 precision
        cells move here, by up to 0.33. pycocotools moves the same cells."""
        default = tie_sensitive_eval(n_gt)
        float32 = tie_sensitive_eval(n_gt, iou_thrs=IOU_F32, rec_thrs=REC_F32)
        assert bits(float32.eval["precision"]) != bits(default.eval["precision"])

    def test_float32_grids_are_not_a_deviation(self, capsys):
        ev = tie_sensitive_eval(20, iou_thrs=IOU_F32, rec_thrs=REC_F32)
        assert ev.reference_deviations() == []
        assert ev.provenance() == "parity_verified"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ev.summarize()
        capsys.readouterr()
        assert ev.stats[1] >= 0 and ev.stats[2] >= 0, "AP50 and AP75 are found: 0.5 and 0.75 are exact in float32"

    def test_a_real_deviation_still_warns(self, capsys):
        grid = Params().iou_thrs
        grid[3] += 1e-3
        ev = tie_sensitive_eval(20, iou_thrs=grid)
        assert any("iou_thrs differ" in d for d in ev.reference_deviations())
        with pytest.warns(UserWarning, match="iou_thrs differ"):
            ev.summarize()
        capsys.readouterr()


class TestQuietSummary:
    """`summary_lines()` already is the quiet switch; these pin it for embedders."""

    def test_summary_lines_fills_stats_without_printing_or_warning(self, capsys):
        ev = tie_sensitive_eval(20)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lines = ev.summary_lines()
        assert capsys.readouterr().out == ""
        assert len(lines) == 12
        assert np.asarray(ev.stats).shape == (12,)
        assert np.asarray(ev.stats)[0] > 0

    def test_the_quiet_path_keeps_the_comparability_signal(self, capsys):
        ev = tie_sensitive_eval(20)
        ev.params.max_dets = [1, 10, 50]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ev.summary_lines()
        assert capsys.readouterr().out == ""
        assert any("max_dets differ" in d for d in ev.reference_deviations())
        with pytest.warns(UserWarning, match="max_dets differ"):
            ev.summarize()
        capsys.readouterr()


class TestVersion:
    def test_matches_the_installed_distribution(self):
        assert hotcoco.__version__ == importlib.metadata.version("hotcoco")

    def test_is_part_of_the_public_surface(self):
        assert "__version__" in hotcoco.__all__
        assert isinstance(hotcoco.__version__, str)
