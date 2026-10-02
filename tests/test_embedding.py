"""What a library that embeds hotcoco (Lightning, torchmetrics) needs from it.

Three small contracts, none needing `data/`:

- A threshold grid built in `float32` reads back as the default grid, so the run
  gets the default numbers and no false "differs from default" warning.
- `summary_lines()` is the quiet path: it fills `stats` without printing or
  warning, and `reference_deviations()` still reports what a warning would have.
- `hotcoco.__version__` exists and names the binary that is loaded.
"""

import importlib.metadata
import warnings

import hotcoco
import numpy as np
import pytest
from hotcoco import Params, StreamingEval

# `torch.linspace(...)` computes in float32; `.tolist()` then reads it back as float64.
IOU_F32 = np.linspace(0.5, 0.95, 10, dtype=np.float32).astype(np.float64).tolist()
REC_F32 = np.linspace(0.0, 1.0, 101, dtype=np.float32).astype(np.float64).tolist()


def tie_sensitive_eval(n_gt=20, iou_thrs=None, rec_thrs=None):
    """One category with `n_gt` ground truths, true positives at triangular-number ranks.

    Recall climbs in exact steps of `1 / n_gt`, which land on recall-grid points,
    while precision falls (k / T_k), so a recall threshold that is one ulp too high
    picks a different precision. A flat precision curve would hide the effect.
    """
    se = StreamingEval([{"id": 1, "name": "a"}])
    images = [{"id": i, "width": 200, "height": 200} for i in range(1, n_gt + 1)]
    gts = [
        {"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400.0, "iscrowd": 0}
        for i in range(1, n_gt + 1)
    ]
    tp_ranks = {k * (k + 1) // 2 for k in range(1, n_gt) if k * (k + 1) // 2 <= n_gt}
    dts = [
        {
            "image_id": rank,
            "category_id": 1,
            "bbox": [10, 10, 20, 20] if rank in tp_ranks else [150, 150, 20, 20],
            "score": 1.0 - 0.001 * rank,
        }
        for rank in range(1, n_gt + 1)
    ]
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
    """A snap test on a grid that was already exact would pass for nothing."""
    default = Params()
    for f32, exact in ((IOU_F32, default.iou_thrs), (REC_F32, default.rec_thrs)):
        drift = np.abs(np.array(f32) - np.array(exact)).max()
        assert 1e-9 < drift < 1e-6


class TestFloat32Grids:
    def test_setters_snap_a_rounded_default_grid(self):
        p = Params()
        p.iou_thrs = IOU_F32
        p.rec_thrs = REC_F32
        assert p.iou_thrs == Params().iou_thrs
        assert p.rec_thrs == Params().rec_thrs

    def test_camel_case_aliases_snap_too(self):
        p = Params()
        p.iouThrs = IOU_F32
        p.recThrs = REC_F32
        assert p.iouThrs == Params().iou_thrs
        assert p.recThrs == Params().rec_thrs

    def test_a_real_deviation_is_kept_as_set(self):
        grid = Params().iou_thrs
        grid[3] += 1e-3
        p = Params()
        p.iou_thrs = grid
        assert p.iou_thrs == grid

    def test_tolerance_edges(self):
        inside, beyond = Params().rec_thrs, Params().rec_thrs
        inside[50] += 5e-7
        beyond[50] += 2e-6
        p = Params()
        p.rec_thrs = inside
        assert p.rec_thrs == Params().rec_thrs
        p.rec_thrs = beyond
        assert p.rec_thrs == beyond

    def test_another_length_is_never_snapped(self):
        p = Params()
        p.rec_thrs = REC_F32[:100]
        assert p.rec_thrs == REC_F32[:100]

    @pytest.mark.parametrize("n_gt", [20, 25, 50, 100])
    def test_float32_grids_give_the_default_numbers_bit_for_bit(self, n_gt):
        """The reason the setters snap rather than the warning check tolerating.

        Without the snap, a float32 recall grid changes 60-240 precision cells here by up
        to 0.33: recall `k / n_gt` is exactly a grid point, and a grid point one ulp higher
        excludes it. The default-grid run is the reference.
        """
        default = tie_sensitive_eval(n_gt)
        float32 = tie_sensitive_eval(n_gt, iou_thrs=IOU_F32, rec_thrs=REC_F32)
        assert bits(float32.eval["precision"]) == bits(default.eval["precision"])
        assert bits(float32.eval["recall"]) == bits(default.eval["recall"])

    def test_the_differential_can_fail(self):
        """Positive control: a grid shifted past the tolerance must change the arrays."""
        shifted = [x + 1e-3 for x in Params().rec_thrs]
        default = tie_sensitive_eval(20)
        other = tie_sensitive_eval(20, rec_thrs=shifted)
        assert bits(other.eval["precision"]) != bits(default.eval["precision"])

    def test_float32_grids_raise_no_deviation_and_no_warning(self, capsys):
        ev = tie_sensitive_eval(20, iou_thrs=IOU_F32, rec_thrs=REC_F32)
        assert ev.reference_deviations() == []
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ev.summarize()
        capsys.readouterr()

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
