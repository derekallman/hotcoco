"""PDF evaluation report: report() and all _draw_* layout helpers."""

from __future__ import annotations

import re
from pathlib import Path

from .core import _import_mpl, _mask_invalid_prec, _resolve_font_family
from .data import PlotData
from .theme import CHROME, SERIES_COLORS

# ---------------------------------------------------------------------------
# Metric display helpers
# ---------------------------------------------------------------------------

def _metric_math(key: str) -> str:
    """Return a LaTeX display string for a metric key.

    Parses the key rather than looking it up in a static dict, so new
    metrics added in Rust automatically render correctly.

    Examples: ``"AP50"`` → ``$\\mathrm{AP}_{50}$``,
    ``"ARs@300"`` → ``$\\mathrm{AR}^{\\mathrm{S}}_{300}$``.
    """
    if key == "F1":
        return "F1"
    # @N pattern: AR@300, ARs@300, ARm@300, ARl@300
    m = re.match(r"^(AP|AR)([sml])?@(\d+)$", key, re.IGNORECASE)
    if m:
        base = m.group(1).upper()
        size = m.group(2)
        n = m.group(3)
        sup = rf"\mathrm{{{size.upper()}}}" if size else ""
        return rf"$\mathrm{{{base}}}{'^{' + sup + '}' if sup else ''}_{{{n}}}$"
    # Standard: AP, AP50, AP75, APs, APm, APl, APr, APc, APf, AR, AR1, …
    m = re.match(r"^(AP|AR)(\d+|[a-z])?$", key, re.IGNORECASE)
    if m:
        base = m.group(1).upper()
        suffix = m.group(2)
        if not suffix:
            return rf"$\mathrm{{{base}}}$"
        if suffix.isdigit():
            return rf"$\mathrm{{{base}}}_{{{suffix}}}$"
        if suffix.lower() in ("s", "m", "l"):
            return rf"$\mathrm{{{base}}}_{{\mathrm{{{suffix.upper()}}}}}$"
        # r, c, f (LVIS frequency groups) — keep lowercase
        return rf"$\mathrm{{{base}}}_{{\mathrm{{{suffix.lower()}}}}}$"
    return key


_SIZE_LABEL = {"s": "small", "m": "medium", "l": "large"}
_FREQ_DESC = {"r": "rare", "c": "common", "f": "frequent"}


def _area_desc(size_key: str, area_ranges: dict) -> str:
    """Human-readable area range description derived from area_ranges bounds."""
    label = _SIZE_LABEL.get(size_key.lower(), size_key)
    if label not in area_ranges:
        return label
    lo, hi = area_ranges[label]
    lo_px = round(lo ** 0.5) if lo > 0 else 0
    hi_px = round(hi ** 0.5) if hi < 1e9 else None
    if lo == 0 and hi_px:
        return f"area < {hi_px}\u00b2"
    if lo_px and hi_px:
        return f"{lo_px}\u00b2 \u2013 {hi_px}\u00b2"
    return f"area > {lo_px}\u00b2"


def _metric_desc(key: str, data: PlotData) -> str:
    """Generate a human-readable description for a metric key."""
    iou_thrs = data.iou_thresholds
    iou_all = f"IoU {iou_thrs[0]:.2f}:{iou_thrs[-1]:.2f}" if len(iou_thrs) > 1 else f"IoU {iou_thrs[0]:.2f}"

    # @N pattern
    m = re.match(r"^AR([sml])?@(\d+)$", key, re.IGNORECASE)
    if m:
        size, n = m.group(1), int(m.group(2))
        return _area_desc(size, data.area_ranges) if size else f"max {n} dets"

    m = re.match(r"^(AP|AR)(\d+|[a-z])?$", key, re.IGNORECASE)
    if not m:
        return key
    _, suffix = m.group(1), m.group(2)
    if not suffix:
        return iou_all
    if suffix.isdigit():
        n = int(suffix)
        iou_val = n / 100.0
        if iou_val in iou_thrs:
            return f"IoU {iou_val:.2f}"
        if n in data.max_dets:
            return f"max {n} det{'s' if n > 1 else ''}"
        return f"IoU {iou_val:.2f}"
    if suffix.lower() in _SIZE_LABEL:
        return _area_desc(suffix, data.area_ranges)
    return _FREQ_DESC.get(suffix.lower(), suffix)


def _build_metric_rows(data: PlotData) -> tuple[list, list, str]:
    """Derive (AP_ROWS, AR_ROWS, ar_kpi_key) from PlotData.

    Rows are (display_key, description, metric_key) tuples. Only metrics
    present in data.metrics are included, in canonical display order from Rust.
    """
    present = set(data.metrics)
    ordered = [k for k in data.metric_key_order if k in present]

    ap_rows = [(_k, _metric_desc(_k, data), _k) for _k in ordered if _k.startswith("AP")]
    ar_rows = [(_k, _metric_desc(_k, data), _k) for _k in ordered if not _k.startswith("AP")]

    # Determine the primary AR metric key for the KPI header
    ar_kpi_key = ar_rows[0][0] if ar_rows else "AR100"
    return ap_rows, ar_rows, ar_kpi_key

# ---------------------------------------------------------------------------
# Layout constants (inches, letter page)
# ---------------------------------------------------------------------------

_PAGE_W = 8.5
_MARGIN_H = 0.65
_MARGIN_V = 0.62
_HEADER_H = 0.38
_CTX_H = 0.65
_SECTION_H = 0.21
_GAP = 0.10
_CAP_H = 0.16
_ROW_H = 0.17
_CAT_HDR_H = 0.16
_CAT_ROW_H = 0.115
_MIN_BLOCK_H = 1.8  # minimum metrics block height — keeps the PR curve legible

# ---------------------------------------------------------------------------
# Report color palette
# ---------------------------------------------------------------------------

_RC = {
    "text": CHROME["text"],
    "label": CHROME["label"],
    "muted": "#9A9088",
    "very_muted": "#A09888",
    "section": SERIES_COLORS[0],
    "border": CHROME["grid"],
    "border_dk": CHROME["spine"],
    "block_bdr": "#EDE7DF",
    "kpi_bg": "#F0EBE4",
    "legend_edge": "#E4DED7",
    "pr_tick": "#7A6E64",
    "pr_50": SERIES_COLORS[2],
    "pr_75": SERIES_COLORS[0],
    "pr_mean": SERIES_COLORS[1],
    "cat_bar": SERIES_COLORS[0],
}

# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------


def _kpi_tile(ax, value: str, label: str, vc) -> None:
    ax.set_facecolor(_RC["kpi_bg"])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color(_RC["block_bdr"])
        spine.set_linewidth(0.6)
    ax.text(0.5, 0.62, value, fontsize=9, fontweight="bold", color=vc, ha="center", va="center", transform=ax.transAxes)
    ax.text(
        0.5,
        0.28,
        _metric_math(label),
        fontsize=8,
        color=_RC["muted"],
        ha="center",
        va="center",
        transform=ax.transAxes,
    )


def _draw_section_heading(ax, label: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        0.88,
        label,
        fontsize=7,
        fontweight="bold",
        color=_RC["section"],
        va="top",
        ha="left",
        transform=ax.transAxes,
    )
    ax.axhline(0.18, color=_RC["border_dk"], linewidth=0.75)


def _draw_table_caption(ax, label: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        1.0,
        label.upper(),
        fontsize=6,
        fontweight="bold",
        color=_RC["muted"],
        va="top",
        ha="left",
        transform=ax.transAxes,
    )
    ax.axhline(0.3, xmin=0, xmax=0.98, color=_RC["border_dk"], linewidth=0.4)


def _draw_metrics_table(ax, rows, metrics) -> None:
    ax.set_axis_off()
    ax.set_facecolor("none")
    cell_text = [
        [_metric_math(name_key), desc, f"{metrics.get(mkey, 0.0):.3f}"] for name_key, desc, mkey in rows
    ]
    tbl = ax.table(cellText=cell_text, colWidths=[0.19, 0.59, 0.22], bbox=[0, 0, 1, 1], cellLoc="left", edges="open")
    tbl.auto_set_font_size(False)
    n = len(rows)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("none")
        t = cell.get_text()
        if c == 0:
            t.set_fontsize(7)
            t.set_color(_RC["label"])
        elif c == 1:
            t.set_fontsize(6)
            t.set_color(_RC["muted"])
        else:
            t.set_fontsize(7)
            t.set_color(_RC["text"])
            t.set_ha("left")
        if r < n - 1:
            cell.visible_edges = "B"
            cell.set_edgecolor(_RC["border"])
            cell.set_linewidth(0.35)


def _draw_report_pr_curve(ax, recall_pts, pr50, pr75, pr_mean, metrics, *, is_oid=False) -> None:
    import numpy as np
    from matplotlib.lines import Line2D as _L2D

    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=5.5, colors=_RC["pr_tick"], length=2, width=0.5, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_RC["border_dk"])
        ax.spines[side].set_linewidth(0.6)

    for v in [0.25, 0.5, 0.75]:
        ax.axhline(v, color=_RC["border"], linewidth=0.4, zorder=0)
        ax.axvline(v, color=_RC["border"], linewidth=0.4, zorder=0)

    ax.fill_between(recall_pts, np.nan_to_num(pr50), alpha=0.10, color=_RC["pr_50"], zorder=1)
    ax.plot(recall_pts, pr50, color=_RC["pr_50"], lw=1.5, zorder=3)
    if not is_oid:
        ax.plot(recall_pts, pr75, color=_RC["pr_75"], lw=1.2, zorder=3)
        ax.plot(recall_pts, pr_mean, color=_RC["pr_mean"], lw=0.9, linestyle="--", zorder=3)

    valid = ~np.isnan(pr50)
    r_v, p_v = recall_pts[valid], pr50[valid]
    if len(r_v):
        f1 = 2 * p_v * r_v / np.maximum(p_v + r_v, 1e-8)
        best = int(np.argmax(f1))
        ax.plot(
            r_v[best],
            p_v[best],
            "o",
            color=_RC["pr_50"],
            markersize=3.5,
            markerfacecolor="none",
            markeredgewidth=1.2,
            zorder=5,
        )
        near_right = r_v[best] > 0.95
        ax.annotate(
            f"F1 {f1[best]:.3f}",
            (r_v[best], p_v[best]),
            xytext=(-5, 5) if near_right else (5, 5),
            textcoords="offset points",
            fontsize=5.5,
            color=_RC["pr_50"],
            ha="right" if near_right else "left",
        )

    ax.set_xlabel("Recall", fontsize=6, color=_RC["muted"], labelpad=1)
    ax.set_ylabel("Precision", fontsize=6, color=_RC["muted"], labelpad=3)

    if is_oid:
        handles = [
            _L2D([0], [0], color=_RC["pr_50"], lw=1.5, label=f"{metrics.get('AP', 0):.3f}  AP50"),
        ]
    else:
        handles = [
            _L2D([0], [0], color=_RC["pr_50"], lw=1.5, label=f"{metrics.get('AP50', 0):.3f}  AP50"),
            _L2D([0], [0], color=_RC["pr_75"], lw=1.2, label=f"{metrics.get('AP75', 0):.3f}  AP75"),
            _L2D([0], [0], color=_RC["pr_mean"], lw=0.9, ls="--", label=f"{metrics.get('AP', 0):.3f}  AP"),
        ]
    leg = ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=6,
        frameon=True,
        edgecolor=_RC["legend_edge"],
        fancybox=False,
        borderpad=0.5,
        labelspacing=0.25,
        handlelength=1.5,
        handletextpad=0.5,
    )
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.9)


def _draw_header(ax, title: str) -> None:
    import datetime

    ax.set_axis_off()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(
        0.0, 1.0, title, fontsize=13, fontweight="bold", color=_RC["text"], va="top", ha="left", transform=ax.transAxes
    )
    ax.text(
        1.0,
        1.0,
        "hotcoco",
        fontsize=10,
        fontweight="bold",
        color=_RC["section"],
        va="top",
        ha="right",
        transform=ax.transAxes,
    )
    ax.text(1.0, 0.38, date_str, fontsize=7.5, color=_RC["muted"], va="top", ha="right", transform=ax.transAxes)
    ax.axhline(0.10, color=_RC["border_dk"], linewidth=1.2)


def _draw_context_box(
    fig, gs_cell, gt_path, dt_path, iou_type, iou_str, max_dets, n_images, n_anns, n_cats, n_dets
) -> None:
    gs_ctx = gs_cell.subgridspec(1, 2, width_ratios=[11, 9], wspace=0.04)

    ax_txt = fig.add_subplot(gs_ctx[0])
    ax_txt.set_axis_off()
    ax_txt.set_facecolor("none")
    for i, (lbl, val) in enumerate(
        [
            ("GT", gt_path or "\u2014"),
            ("DT", dt_path or "\u2014"),
            ("Params", f"{iou_type}  \u00b7  IoU {iou_str}  \u00b7  max {max_dets[-1]} dets"),
        ]
    ):
        ry = 0.75 - i * 0.30
        ax_txt.text(
            0.04,
            ry,
            lbl,
            fontsize=7,
            fontweight="bold",
            color=_RC["very_muted"],
            va="center",
            transform=ax_txt.transAxes,
        )
        ax_txt.text(0.16, ry, val, fontsize=6.5, color=_RC["label"], va="center", transform=ax_txt.transAxes)

    gs_tiles = gs_ctx[1].subgridspec(1, 4, wspace=0.15)
    for i, (val, lbl) in enumerate(
        [
            (f"{n_images:,}", "images"),
            (f"{n_anns:,}", "annotations"),
            (str(n_cats), "categories"),
            (f"{n_dets:,}", "detections"),
        ]
    ):
        _kpi_tile(fig.add_subplot(gs_tiles[i]), val, lbl, _RC["text"])


def _draw_metrics_block(
    fig, gs_cell, AP_ROWS, AR_ROWS, metrics, recall_pts, pr50, pr75, pr_mean, f1_peak, ar_kpi_key="AR100", *, is_oid=False, block_h=0.0
) -> None:
    gs_met = gs_cell.subgridspec(1, 3, width_ratios=[9, 9, 4], wspace=0.1)

    if AR_ROWS:
        content_ratios = [_CAP_H, len(AP_ROWS) * _ROW_H, _CAP_H, len(AR_ROWS) * _ROW_H]
    else:
        content_ratios = [_CAP_H, len(AP_ROWS) * _ROW_H]
    sum_content = sum(content_ratios)
    # Spacer absorbs any extra height from the block_h minimum, keeping each row its natural size.
    leftover = max(0.0, block_h - sum_content)
    left_ratios = content_ratios + ([leftover] if leftover > 0 else [])
    gs_left = gs_met[0].subgridspec(len(left_ratios), 1, height_ratios=left_ratios, hspace=0)
    ax_ap_cap = fig.add_subplot(gs_left[0])
    ax_ap_tbl = fig.add_subplot(gs_left[1])
    for ax in (ax_ap_cap, ax_ap_tbl):
        ax.set_facecolor("none")
    _draw_table_caption(ax_ap_cap, "Average Precision")
    _draw_metrics_table(ax_ap_tbl, AP_ROWS, metrics)
    if AR_ROWS:
        ax_ar_cap = fig.add_subplot(gs_left[2])
        ax_ar_tbl = fig.add_subplot(gs_left[3])
        for ax in (ax_ar_cap, ax_ar_tbl):
            ax.set_facecolor("none")
        _draw_table_caption(ax_ar_cap, "Average Recall")
        _draw_metrics_table(ax_ar_tbl, AR_ROWS, metrics)

    gs_pr = gs_met[1].subgridspec(2, 1, height_ratios=[_CAP_H, block_h - _CAP_H], hspace=0.05)
    ax_pr_cap = fig.add_subplot(gs_pr[0])
    ax_pr_cur = fig.add_subplot(gs_pr[1])
    ax_pr_cap.set_facecolor("none")
    ax_pr_cur.set_facecolor("none")
    pos = ax_pr_cur.get_position()
    ax_pr_cur.set_position([pos.x0 + 0.025, pos.y0 + 0.012, pos.width - 0.025, pos.height - 0.012])
    _draw_table_caption(ax_pr_cap, "Precision\u2013Recall")
    _draw_report_pr_curve(ax_pr_cur, recall_pts, pr50, pr75, pr_mean, metrics, is_oid=is_oid)

    if is_oid:
        kpi_data = [
            (f"{metrics.get('AP', 0):.3f}", "AP", _RC["pr_50"]),
            (f"{f1_peak:.3f}", "F1", _RC["text"]),
        ]
    else:
        kpi_data = [
            (f"{metrics.get('AP', 0):.3f}", "AP", _RC["pr_mean"]),
            (f"{metrics.get('AP50', 0):.3f}", "AP50", _RC["pr_50"]),
            (f"{metrics.get(ar_kpi_key, 0):.3f}", ar_kpi_key, _RC["pr_75"]),
            (f"{f1_peak:.3f}", "F1", _RC["text"]),
        ]
    gs_kpi = gs_met[2].subgridspec(len(kpi_data), 1, hspace=0.15)
    for i, (val, lbl, vc) in enumerate(kpi_data):
        _kpi_tile(fig.add_subplot(gs_kpi[i]), val, lbl, vc)


def _draw_category_section(fig, gs_cell, cat_items, n_cols, rows_per_col, has_counts, ann_counts, img_counts, virtual_cats=None) -> None:
    from matplotlib.patches import Rectangle

    gs_cat = gs_cell.subgridspec(
        2, n_cols, height_ratios=[_CAT_HDR_H, rows_per_col * _CAT_ROW_H], hspace=0, wspace=0.04
    )

    raw_ratios = [2, 8, 4] + ([3, 3] if has_counts else [])
    col_fracs = [r / sum(raw_ratios) for r in raw_ratios]
    n_data_cols = len(col_fracs)
    data_aligns = ["center", "left", "left"] + (["right", "right"] if has_counts else [])
    hdr_labels = ["#", "Category", "AP"] + (["ann", "img"] if has_counts else [])

    virtual_cats = virtual_cats or set()

    for ci in range(n_cols):
        ax_hdr = fig.add_subplot(gs_cat[0, ci])
        ax_hdr.set_axis_off()
        cumx = 0.0
        for hci, (hlbl, ha) in enumerate(zip(hdr_labels, data_aligns)):
            fx = cumx + (col_fracs[hci] / 2 if ha == "center" else col_fracs[hci] if ha == "right" else 0)
            ax_hdr.text(
                fx,
                0.45,
                hlbl,
                fontsize=6,
                fontweight="bold",
                color=_RC["muted"],
                va="center",
                ha=ha,
                transform=ax_hdr.transAxes,
            )
            cumx += col_fracs[hci]
        ax_hdr.axhline(0.04, xmin=0, xmax=0.99, color=_RC["border_dk"], linewidth=0.45)

        ax_tbl = fig.add_subplot(gs_cat[1, ci])
        ax_tbl.set_xlim(0, 1)
        ax_tbl.set_ylim(0, 1)
        ax_tbl.set_axis_off()

        start = ci * rows_per_col
        col_items = cat_items[start : start + rows_per_col]
        actual_rows = len(col_items)
        row_h_ax = 1.0 / rows_per_col

        for ri, (cname, apv) in enumerate(col_items):
            yb = 1.0 - (ri + 1) * row_h_ax
            ax_tbl.add_patch(
                Rectangle(
                    (0.0, yb),
                    max(0.0, min(1.0, apv)),
                    row_h_ax,
                    transform=ax_tbl.transAxes,
                    facecolor=_RC["cat_bar"],
                    alpha=0.12,
                    clip_on=True,
                    zorder=0,
                )
            )
            if ri < actual_rows - 1:
                ax_tbl.axhline(yb, xmin=0.04, xmax=0.96, color=_RC["border"], linewidth=0.25)

        is_virtual_row = []
        cell_text = []
        for ri in range(rows_per_col):
            if ri < actual_rows:
                cname, apv = col_items[ri]
                is_virt = cname in virtual_cats
                is_virtual_row.append(is_virt)
                label = cname if len(cname) <= 17 else cname[:16] + "\u2026"
                if is_virt:
                    label += "*"
                row = [str(start + ri + 1), label, f"{apv:.3f}"]
                if has_counts:
                    row += [f"{ann_counts.get(cname, 0):,}", f"{img_counts.get(cname, 0):,}"]
            else:
                is_virtual_row.append(False)
                row = [""] * n_data_cols
            cell_text.append(row)

        tbl = ax_tbl.table(
            cellText=cell_text, colWidths=col_fracs, bbox=[0.0, 0.0, 1.0, 1.0], cellLoc="left", edges="open"
        )
        tbl.auto_set_font_size(False)
        tbl.set_zorder(3)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("none")
            cell.visible_edges = "open"
            t = cell.get_text()
            t.set_fontsize(6)
            t.set_ha(data_aligns[c] if c < n_data_cols else "left")
            if r >= actual_rows:
                cell.set_visible(False)
            elif is_virtual_row[r]:
                t.set_color(_RC["muted"])
            else:
                t.set_color([_RC["very_muted"], _RC["label"], _RC["text"], _RC["muted"], _RC["muted"]][c])

    if virtual_cats and any(cname in virtual_cats for cname, _ in cat_items):
        # Footnote in the bottom-left of the last column's axes.
        ax_tbl.text(
            0.0,
            -0.01,
            "* expanded via category hierarchy",
            fontsize=5,
            color=_RC["muted"],
            va="top",
            ha="left",
            transform=ax_tbl.transAxes,
        )


def _draw_footer(fig, page_h: float) -> None:
    import importlib.metadata

    from matplotlib.lines import Line2D

    try:
        ver = importlib.metadata.version("hotcoco")
        footer_text = f"hotcoco v{ver}  \u00b7  github.com/derekallman/hotcoco"
    except Exception:
        footer_text = "github.com/derekallman/hotcoco"

    lx = _MARGIN_H / _PAGE_W
    fy = _MARGIN_V * 0.45 / page_h
    line_y = fy + 7 / (page_h * 72) + 0.004
    fig.add_artist(
        Line2D(
            [lx, 1.0 - lx],
            [line_y, line_y],
            transform=fig.transFigure,
            color=_RC["border"],
            linewidth=0.5,
            solid_capstyle="butt",
        )
    )
    fig.text(
        1.0 - lx,
        fy,
        footer_text,
        fontsize=7,
        color=_RC["very_muted"],
        ha="right",
        va="bottom",
        transform=fig.transFigure,
    )


# ---------------------------------------------------------------------------
# report()
# ---------------------------------------------------------------------------


def report(
    coco_eval,
    *,
    save_path: str | Path,
    gt_path: str | None = None,
    dt_path: str | None = None,
    title: str | None = None,
) -> None:
    """Generate a PDF evaluation report.

    Parameters
    ----------
    coco_eval : COCOeval
        Must have ``run()`` called first.
    save_path : str or Path
        Output PDF path.
    gt_path : str, optional
        Ground-truth JSON path to display in the run context block.
    dt_path : str, optional
        Detections JSON path to display in the run context block.
    title : str, optional
        Report title shown in the header. Auto-detected from eval mode if not given.
    """
    import math

    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    mpl, plt, _ = _import_mpl()
    family = _resolve_font_family()

    data = PlotData.from_coco_eval(coco_eval, per_class=True)
    metrics = data.metrics
    per_class = data.per_class or {}

    a_idx = data.area_idx("all")
    m_idx = data.max_det_idx(None)

    def _mprec(t):
        return np.nanmean(_mask_invalid_prec(data.precision[t, :, :, a_idx, m_idx]), axis=1)

    t50 = data.nearest_iou_idx(0.50)
    t75 = data.nearest_iou_idx(0.75)
    all_prec = np.array([_mprec(t) for t in range(len(data.iou_thresholds))])
    pr_mean = np.nanmean(all_prec, axis=0)
    pr50 = all_prec[t50]
    pr75 = all_prec[t75]

    try:
        n_images = len(coco_eval.coco_gt.get_img_ids())
        n_anns = len(coco_eval.coco_gt.get_ann_ids())
    except Exception:
        n_images = n_anns = 0
    try:
        n_dets = len(coco_eval.coco_dt.get_ann_ids())
    except Exception:
        n_dets = 0

    n_cats = len(data.cat_ids)
    iou_thrs = data.iou_thresholds
    max_dets = data.max_dets
    iou_str = f"{iou_thrs[0]:.2f}" if len(iou_thrs) == 1 else f"{iou_thrs[0]:.2f}\u2013{iou_thrs[-1]:.2f}"

    cat_items = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    ann_counts: dict[str, int] = {}
    img_counts: dict[str, int] = {}
    virtual_cats: set[str] = set(getattr(coco_eval, "virtual_cat_names", []))
    try:
        for cid, cname in data.cat_names.items():
            ann_counts[cname] = len(coco_eval.coco_gt.get_ann_ids(cat_ids=[cid]))
            img_counts[cname] = len(coco_eval.coco_gt.get_img_ids(cat_ids=[cid]))
    except Exception:
        pass

    has_counts = bool(ann_counts)

    valid = ~np.isnan(pr50)
    if valid.any():
        r_v, p_v = data.recall_pts[valid], pr50[valid]
        f1_peak = float(np.max(2 * p_v * r_v / np.maximum(p_v + r_v, 1e-8)))
    else:
        f1_peak = 0.0

    is_lvis = data.eval_mode == "lvis"
    is_kpts = data.iou_type == "keypoints"
    is_oid = data.eval_mode == "openimages"

    if title is None:
        if is_oid:
            title = "Open Images Evaluation Report"
        elif is_lvis:
            title = "LVIS Evaluation Report"
        elif is_kpts:
            title = "Keypoints Evaluation Report"
        else:
            title = "COCO Evaluation Report"

    AP_ROWS, AR_ROWS, ar_kpi_key = _build_metric_rows(data)

    # TODO: pagination not implemented — report() produces a single variable-height page.
    # For very large category counts (hundreds), consider splitting into multiple fixed-height
    # pages via a _draw_continuation_page() helper. Implement when requested.
    n_cols = 3
    rows_per_col = math.ceil(n_cats / n_cols) if n_cats > 0 else 1
    n_captions = 2 if AR_ROWS else 1
    block_h = max(
        n_captions * _CAP_H + (len(AP_ROWS) + len(AR_ROWS)) * _ROW_H + _GAP * 1.5,
        _MIN_BLOCK_H,
    )
    cat_h = _CAT_HDR_H + rows_per_col * _CAT_ROW_H

    # Single source of truth: row heights drive both page_h and height_ratios.
    # Row index names match the unpacked constants below.
    _row_heights = [
        _HEADER_H,    # _R_HEADER
        _GAP * 0.5,   # _R_GAP1
        _CTX_H,       # _R_CTX
        _GAP * 0.8,   # _R_GAP2
        _SECTION_H,   # _R_SEC1
        block_h,      # _R_METRICS
        _GAP * 0.6,   # _R_GAP3
        _SECTION_H,   # _R_SEC2
        cat_h,        # _R_CATS
    ]
    _R_HEADER, _R_GAP1, _R_CTX, _R_GAP2, _R_SEC1, _R_METRICS, _R_GAP3, _R_SEC2, _R_CATS = range(9)
    page_h = sum(_row_heights) + 2 * _MARGIN_V

    fig = plt.figure(figsize=(_PAGE_W, page_h))
    fig.patch.set_facecolor("#FFFFFF")

    try:
        with mpl.rc_context({"font.family": family}):
            gs = fig.add_gridspec(
                len(_row_heights),
                1,
                height_ratios=_row_heights,
                hspace=0,
                left=_MARGIN_H / _PAGE_W,
                right=1 - _MARGIN_H / _PAGE_W,
                top=1 - _MARGIN_V / page_h,
                bottom=_MARGIN_V / page_h,
            )

            _draw_header(fig.add_subplot(gs[_R_HEADER]), title)
            _draw_context_box(
                fig, gs[_R_CTX], gt_path, dt_path, data.iou_type, iou_str, max_dets, n_images, n_anns, n_cats, n_dets
            )
            _draw_section_heading(fig.add_subplot(gs[_R_SEC1]), "SUMMARY METRICS")
            _draw_metrics_block(
                fig, gs[_R_METRICS], AP_ROWS, AR_ROWS, metrics,
                data.recall_pts, pr50, pr75, pr_mean, f1_peak,
                ar_kpi_key=ar_kpi_key, is_oid=is_oid, block_h=block_h
            )
            _draw_section_heading(fig.add_subplot(gs[_R_SEC2]), "PER-CATEGORY AP  \u00b7  SORTED DESCENDING")
            _draw_category_section(
                fig, gs[_R_CATS], cat_items, n_cols, rows_per_col,
                has_counts, ann_counts, img_counts, virtual_cats=virtual_cats,
            )
            _draw_footer(fig, page_h)

        with PdfPages(str(save_path)) as pdf:
            pdf.savefig(fig)
    finally:
        plt.close(fig)
