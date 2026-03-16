"""Matplotlib import helpers, figure utilities, and shared plot primitives."""

from __future__ import annotations

from pathlib import Path

_MPL_ERROR = "matplotlib is required for plotting. Install with: pip install hotcoco[plot]"


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

    _FONT_FAMILY = ["Inter", "Helvetica Neue", "DejaVu Sans"]
    return _FONT_FAMILY


# ---------------------------------------------------------------------------
# Figure / axes helpers
# ---------------------------------------------------------------------------


def _new_figure(figsize: tuple[float, float], ax=None, layout: str | None = "constrained"):
    _, plt, _ = _import_mpl()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, layout=layout)
    return fig, ax


def _configure_axes(ax, title: str | None = None, subtitle: str | None = None, value_axis: str = "y"):
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
        full_title = f"{title}\n{subtitle}" if subtitle else title
        ax.set_title(full_title, fontsize=11, fontweight=500, pad=10)


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


def _annotate_f1_peak(ax, recall_pts, prec, line):
    """Fill under a PR curve and mark the F1 peak."""
    import numpy as np

    color = line.get_color()
    ax.fill_between(recall_pts, prec, alpha=0.15, color=color)
    f1 = 2 * prec * recall_pts / np.maximum(prec + recall_pts, 1e-8)
    if np.all(np.isnan(f1)):
        return
    best = int(np.nanargmax(f1))
    if prec[best] > 0:
        ax.plot(recall_pts[best], prec[best], "o", color=color, markersize=5, zorder=5)
        ax.annotate(
            f"F1={f1[best]:.2f}", (recall_pts[best], prec[best]), textcoords="offset points", xytext=(5, 5), fontsize=8
        )
