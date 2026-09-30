"""Matplotlib import helpers, figure utilities, and shared plot primitives."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # numpy is imported lazily inside functions so plotting stays optional;
    # this branch exists only so annotations referring to `np` can be resolved.
    import numpy as np

_MPL_ERROR = "matplotlib is required for plotting. Install with: pip install hotcoco[plot]"

# TIDE's error taxonomy, in the order the paper presents it. Every surface that
# lists error types — the CLI table, the matplotlib bar chart, the dashboard —
# reads this so they cannot drift into disagreeing orders.
TIDE_ERROR_ORDER: tuple[str, ...] = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")


def _import_mpl():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        return matplotlib, plt, font_manager
    except ImportError:
        raise ImportError(_MPL_ERROR) from None


# ---------------------------------------------------------------------------
# Font registration
# ---------------------------------------------------------------------------

_FONT_FAMILY: list[str] | None = None


def _resolve_font_family() -> list[str]:
    global _FONT_FAMILY
    if _FONT_FAMILY is not None:
        return _FONT_FAMILY

    _, _, font_manager = _import_mpl()
    fonts_dir = Path(__file__).parent.parent / "_fonts"
    for ttf in fonts_dir.glob("*.ttf"):
        try:
            font_manager.fontManager.addfont(str(ttf))
        except Exception:
            pass

    # Only name families matplotlib can actually resolve. Listing a missing one
    # emits a `findfont` warning per text object — hundreds per figure — so the
    # preference order is filtered against what is installed or vendored rather
    # than asserted. tests/test_theme.py checks which face actually wins.
    preferred = ["IBM Plex Sans", "Helvetica Neue", "DejaVu Sans"]
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()
    resolved = [name for name in preferred if name in available]

    _FONT_FAMILY = resolved or ["DejaVu Sans"]
    return _FONT_FAMILY


# ---------------------------------------------------------------------------
# Figure / axes helpers
# ---------------------------------------------------------------------------


def _headroom_for_bar_labels(ax, values, axis: str = "x", frac: float = 0.14) -> None:
    """Leave room past the longest bar for its `bar_label` text.

    `bar_label(padding=...)` places the text outside the bar end but does not
    widen the axes, so the label on the longest bar runs into the margin and
    clips. The needed room depends on the rendered width of the digits, which
    changes with the body face — this keeps the figures font-independent rather
    than tuned to whichever face is vendored.
    """
    finite = [v for v in values if v is not None and v == v]
    if not finite:
        return
    lo, hi = min(0.0, min(finite)), max(0.0, max(finite))
    span = hi - lo
    if span <= 0:
        return
    pad = span * frac
    if axis == "x":
        ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)
    else:
        ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)


def _new_figure(figsize: tuple[float, float], ax=None, layout: str | None = "constrained"):
    _, plt, _ = _import_mpl()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, layout=layout)
    return fig, ax


def _place_title_and_subtitle(ax, title: str, subtitle: str) -> None:
    """Position a suptitle + subtitle above the axes, reserving space so they never overlap."""
    import matplotlib as _mpl

    fig = ax.figure
    title_size = 12
    sub_size = 9
    h = fig.get_figheight()

    # Convert font sizes from points (1/72 in) to figure-fraction
    # so spacing adapts to any figure height.
    title_frac = title_size / 72 / h
    sub_frac = sub_size / 72 / h
    inter_gap = 0.005
    bottom_pad = 0.008

    subtitle_y = 0.98 - title_frac - inter_gap
    axes_top = subtitle_y - sub_frac - bottom_pad

    # Both constrained and compressed layout engines support the rect
    # parameter; for other layouts we fall back to subplots_adjust.
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(rect=[0, 0, 1, axes_top])
    else:
        fig.subplots_adjust(top=axes_top - 0.02)

    fig.suptitle(title, fontsize=title_size, fontweight=700, y=0.98)
    fig.text(
        0.5,
        subtitle_y,
        subtitle,
        ha="center",
        va="top",
        fontsize=sub_size,
        fontweight=400,
        color=_mpl.rcParams.get("axes.labelcolor", "#666"),
    )


def _configure_axes(ax, title: str | None = None, subtitle: str | None = None, value_axis: str | None = "y"):
    """Set grid direction, title, and subtitle. Colors come from active rcParams."""
    if value_axis == "y":
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        ax.tick_params(axis="x", length=0)
    elif value_axis == "x":
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.tick_params(axis="y", length=0)
    else:
        ax.grid(False)

    if title:
        if subtitle:
            _place_title_and_subtitle(ax, title, subtitle)
        else:
            ax.set_title(title, fontsize=11, fontweight=500, pad=10)


def _save_and_return(fig, ax, save_path):
    if save_path is not None:
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig, ax


def _mask_invalid_prec(arr) -> "np.ndarray":
    """Return a copy of arr with COCO sentinel values (-1) replaced by NaN."""
    import numpy as np

    out = arr.copy()
    out[out < 0] = np.nan
    return out


def _report_curves(curves: dict) -> "tuple[np.ndarray, list[float], np.ndarray]":
    """Unpack ``report()["curves"]`` into ``(rec_thrs, iou_thrs, precision)``.

    ``precision`` has shape ``(T, R)``, one aggregate curve per IoU threshold in
    ascending threshold order — the curve keys are ``"pr@0.50"`` … ``"pr@0.95"``,
    and dict order is not threshold order.

    This is the boundary where Rust's ``-1.0`` sentinel ("not computed for this
    configuration") becomes NumPy's own missing value, NaN. Every renderer of
    these curves crosses that boundary here, once.
    """
    import numpy as np

    rec_thrs = np.asarray(curves["rec_thrs"], dtype=float)
    ordered = sorted(
        ((float(key[len("pr@") :]), values) for key, values in curves.items() if key.startswith("pr@")),
        key=lambda item: item[0],
    )
    iou_thrs = [thr for thr, _ in ordered]
    precision = _mask_invalid_prec(np.asarray([values for _, values in ordered], dtype=float))
    return rec_thrs, iou_thrs, precision


def _top_confusion_keep(matrix, n_cats: int, top_n: int | None = None) -> "list[int] | None":
    """Pick which confusion-matrix rows/columns to keep, by off-diagonal mass.

    *matrix* is the full ``(K + 1, K + 1)`` matrix, BG last. Returns the kept
    category indices in ascending order with the BG index appended, or ``None``
    when everything fits and the caller should not subset at all.

    ``top_n=None`` means "decide for me": above 30 categories a full matrix is
    unreadable, so it falls back to the 25 worst offenders.
    """
    import numpy as np

    if top_n is None and n_cats > 30:
        top_n = 25
    if top_n is None or top_n >= n_cats:
        return None

    cat_block = matrix[:n_cats, :n_cats]
    diag = np.diag(cat_block)
    confusion_mass = (cat_block.sum(axis=1) - diag) + (cat_block.sum(axis=0) - diag)
    top_indices = np.argsort(confusion_mass)[::-1][:top_n]
    return sorted(int(i) for i in top_indices) + [n_cats]


def _f1_peak(recall_pts, prec) -> "tuple[int, float] | None":
    """Return ``(index, f1)`` at the peak F1 of a PR curve, or ``None``.

    NaN precision entries (the -1 sentinel, already masked) are skipped; the
    index is into the arrays as passed.
    """
    import numpy as np

    f1 = 2 * prec * recall_pts / np.maximum(prec + recall_pts, 1e-8)
    if np.all(np.isnan(f1)):
        return None
    best = int(np.nanargmax(f1))
    return best, float(f1[best])


def _annotate_f1_peak(ax, recall_pts, prec, line, *, fill: bool = True):
    """Mark the F1 peak of a PR curve, optionally filling the area under it.

    ``fill=False`` for a multi-curve plot. The fill runs from the curve down to
    zero, so under the topmost curve of a sweep it tints every other curve's
    area as well and washes the whole plot rather than delimiting anything.
    """
    color = line.get_color()
    if fill:
        ax.fill_between(recall_pts, prec, alpha=0.15, color=color)
    peak = _f1_peak(recall_pts, prec)
    if peak is None:
        return
    best, f1_val = peak
    if prec[best] > 0:
        ax.plot(recall_pts[best], prec[best], "o", color=color, markersize=5, zorder=5)
        ax.annotate(
            f"F1={f1_val:.2f}", (recall_pts[best], prec[best]), textcoords="offset points", xytext=(5, 5), fontsize=8
        )
