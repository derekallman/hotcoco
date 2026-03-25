"""Publication-quality plots for hotcoco evaluation results.

Requires matplotlib (install with ``pip install hotcoco[plot]``).

Apply the hotcoco visual style with the ``style()`` context manager::

    with hotcoco.plot.style():
        fig, ax = pr_curve(ev)
        fig, ax = confusion_matrix(ev.confusion_matrix())

Without the context, all plot functions work with plain matplotlib defaults.
Use your own ``rcParams`` or ``plt.style.use()`` to theme them as you like.

Pass ``ax=`` to draw on an existing axes, ``save_path=`` to write to disk.
"""

from .plots import (
    category_deltas,
    comparison_bar,
    confusion_matrix,
    per_category_ap,
    pr_curve,
    pr_curve_by_category,
    pr_curve_iou_sweep,
    pr_curve_top_n,
    reliability_diagram,
    tide_errors,
    top_confusions,
)
from .report import report
from .theme import CHROME, SEQUENTIAL, SERIES_COLORS, style

__all__ = [
    "style",
    "pr_curve",
    "pr_curve_iou_sweep",
    "pr_curve_by_category",
    "pr_curve_top_n",
    "confusion_matrix",
    "top_confusions",
    "per_category_ap",
    "tide_errors",
    "reliability_diagram",
    "comparison_bar",
    "category_deltas",
    "report",
    "SERIES_COLORS",
    "CHROME",
    "SEQUENTIAL",
]
