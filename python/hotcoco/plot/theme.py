"""Theme definitions, rcParams builder, and style() context manager."""

from __future__ import annotations

from contextlib import contextmanager

from .core import _import_mpl, _resolve_font_family

_THEMES: dict[str, dict] = {
    "cold-brew": {
        "series": [
            "#5E81AC", "#C47A52", "#5A9E78", "#D4A03E", "#9673A6",
            "#3D9B96", "#C46070", "#7B8C42", "#6E6EAA", "#B5694A",
        ],
        "chrome": {"text": "#28231F", "label": "#4A3F38", "tick": "#7A6E64", "grid": "#E8E2DA", "spine": "#D4CCC2"},
        "background": "#FAF7F4",
        "plot_bg": "#ffffff",
        "sequential": ["#F4F1EE", "#5E81AC", "#2A4060"],
        "cmap": "hotcoco_coldbrew",
    },
    "warm-slate": {
        "series": ["#5C7080", "#C46B50", "#2B7A8C", "#C9943E", "#8A5A90", "#5B7F63", "#3A7CA5", "#B07650"],
        "chrome": {"text": "#2C2420", "label": "#4A3F38", "tick": "#7A6E64", "grid": "#E8E2DA", "spine": "#D4CCC2"},
        "background": "#FAF7F4",
        "plot_bg": "#ffffff",
        "sequential": ["#ffffff", "#A8BFCA", "#5C7080", "#2B3D4A"],
        "cmap": "hotcoco_seq",
    },
    "scientific-blue": {
        "series": ["#1B4F8A", "#E84855", "#2E86AB", "#F4A261", "#457B9D", "#74C69D"],
        "chrome": {"text": "#1A2B3C", "label": "#2E3F52", "tick": "#6B7B8E", "grid": "#DDE4EF", "spine": "#C0CCE0"},
        "background": "#F2F5F9",
        "plot_bg": "#ffffff",
        "sequential": ["#ffffff", "#2E86AB", "#1B4F8A", "#0D2240"],
        "cmap": "hotcoco_sci",
    },
    "ember": {
        "series": ["#BF4E30", "#F2C14E", "#D4915E", "#2D3A3A", "#6B4C3B", "#E8D5B0"],
        "chrome": {"text": "#2D3A3A", "label": "#4A3520", "tick": "#8B7355", "grid": "#E6D9C5", "spine": "#D4C4A8"},
        "background": "#F5EFE6",
        "plot_bg": "#FFFDF8",
        "sequential": ["#ffffff", "#D4915E", "#BF4E30", "#6B2210"],
        "cmap": "hotcoco_ember",
    },
}

_CMAPS_REGISTERED: set[str] = set()


def _get_theme(name: str) -> dict:
    if name not in _THEMES:
        raise ValueError(f"Unknown theme {name!r}. Choose from: {list(_THEMES)}")
    return _THEMES[name]


def _ensure_cmap(theme: dict) -> None:
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    cmap_name = theme["cmap"]
    if cmap_name not in _CMAPS_REGISTERED:
        cmap = LinearSegmentedColormap.from_list(cmap_name, theme["sequential"])
        try:
            matplotlib.colormaps.register(cmap)
        except ValueError:
            pass
        _CMAPS_REGISTERED.add(cmap_name)


def _build_rc(theme_name: str = "cold-brew", paper_mode: bool = False) -> dict:
    from cycler import cycler

    t = _get_theme(theme_name)
    _ensure_cmap(t)
    bg = "#ffffff" if paper_mode else t["background"]
    plot_bg = t["background"] if paper_mode else t["plot_bg"]
    c = t["chrome"]
    return {
        "figure.facecolor": bg,
        "axes.facecolor": plot_bg,
        "axes.edgecolor": c["spine"],
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": c["grid"],
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=t["series"]),
        "xtick.color": c["tick"],
        "ytick.color": c["tick"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "axes.labelcolor": c["label"],
        "axes.labelsize": 11,
        "axes.labelpad": 8,
        "text.color": c["text"],
        "font.family": _resolve_font_family(),
        "legend.frameon": False,
        "image.cmap": t["cmap"],
    }


# ---------------------------------------------------------------------------
# Public palette constants (cold-brew defaults, kept for backwards compat)
# ---------------------------------------------------------------------------

_DEFAULT_THEME = _THEMES["cold-brew"]

SERIES_COLORS: list[str] = _DEFAULT_THEME["series"]
CHROME: dict[str, str] = {
    **_DEFAULT_THEME["chrome"],
    "background": _DEFAULT_THEME["background"],
    "plot_bg": _DEFAULT_THEME["plot_bg"],
}
SEQUENTIAL: list[str] = _DEFAULT_THEME["sequential"]


# ---------------------------------------------------------------------------
# Style context manager
# ---------------------------------------------------------------------------


@contextmanager
def style(theme: str = "cold-brew", paper_mode: bool = False):
    """Context manager that applies a hotcoco matplotlib theme.

    Parameters
    ----------
    theme : str
        ``"cold-brew"`` (default), ``"warm-slate"``, ``"scientific-blue"``, or ``"ember"``.
    paper_mode : bool
        White figure background with the theme tint on axes. Useful for
        LaTeX inclusion or PowerPoint embedding.

    Usage::

        with hotcoco.plot.style():
            fig, ax = pr_curve(ev)

        with hotcoco.plot.style(theme="scientific-blue", paper_mode=True):
            fig, ax = pr_curve(ev)

    All plot functions also accept ``theme`` and ``paper_mode`` directly,
    which is equivalent and more concise for single calls.
    """
    mpl, _, _ = _import_mpl()
    with mpl.rc_context(_build_rc(theme, paper_mode)):
        yield
