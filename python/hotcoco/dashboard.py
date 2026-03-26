"""Interactive eval dashboard — Plotly charts for the browse server."""

from __future__ import annotations

import numpy as np

from .plot.data import PlotData

# ── Theme constants matching browse CSS tokens ───────────────────────
_BG_SURFACE = "#1c1916"
_BG_ELEVATED = "#252119"
_TEXT_PRIMARY = "#ede8e3"
_TEXT_SECONDARY = "#9a918a"
_TEXT_TERTIARY = "#6b6259"
_BORDER_SUBTLE = "#2a2520"
_ACCENT = "#d4a574"

_EVAL_TP = "#22c55e"
_EVAL_FP = "#ef4444"
_EVAL_FN = "#3b82f6"

_COLORWAY = [
    _ACCENT, "#78b4ff", _EVAL_TP, _EVAL_FP, "#a78bfa",
    "#f59e0b", "#ec4899", "#14b8a6", "#f97316", "#6366f1",
]

_FONT_BODY = "DM Sans, -apple-system, BlinkMacSystemFont, sans-serif"
_FONT_MONO = "JetBrains Mono, ui-monospace, SFMono-Regular, monospace"


def _dark_layout(**overrides):
    """Return a Plotly layout dict matching the browse dark theme.

    We set all dark-theme colors manually instead of using
    ``template="plotly_dark"`` because the built-in template embeds
    defaults for every trace type (scatter3d, scattergeo, mesh3d, etc.).
    The cartesian partial bundle doesn't include those trace modules, so
    Plotly.js errors out trying to register them and nothing renders.
    """
    import plotly.graph_objects as go

    _axis = dict(
        gridcolor=_BORDER_SUBTLE,
        zerolinecolor=_BORDER_SUBTLE,
        linecolor=_BORDER_SUBTLE,
        tickcolor=_TEXT_TERTIARY,
        tickfont=dict(color=_TEXT_SECONDARY),
        titlefont=dict(color=_TEXT_PRIMARY),
    )
    base = dict(
        template={},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_BG_ELEVATED,
        font=dict(family=_FONT_BODY, color=_TEXT_PRIMARY, size=13),
        hoverlabel=dict(bgcolor=_BG_SURFACE, font_color=_TEXT_PRIMARY, bordercolor=_BORDER_SUBTLE),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=_TEXT_TERTIARY, activecolor=_ACCENT),
        colorway=_COLORWAY,
        margin=dict(l=60, r=20, t=20, b=40),
        xaxis=_axis,
        yaxis=_axis,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_PRIMARY)),
    )
    base.update(overrides)
    return go.Layout(**base)


def _to_html(fig, div_id, *, post_script=None):
    """Render a Plotly figure to an HTML fragment."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"responsive": True, "displaylogo": False},
        post_script=post_script,
    )


# ── KPI tiles ────────────────────────────────────────────────────────

def kpi_tiles(coco_eval) -> list[dict]:
    """Extract headline metrics for KPI tile display.

    Returns a list of {key, value} dicts in display order.
    Picks AP, AP50, AP75, and the first AR metric (AR100 for bbox/segm,
    AR10 for keypoints) as the 4 most useful headlines.
    """
    metrics = coco_eval.get_results()
    keys = coco_eval.metric_keys()

    # Pick the 4 most informative headline metrics
    preferred = ["AP", "AP50", "AP75", "AR100", "AR10", "AR1"]
    headline_keys = [k for k in preferred if k in metrics][:4]

    return [{"key": k, "value": metrics.get(k, 0.0)} for k in headline_keys]


# ── PR Curves ────────────────────────────────────────────────────────

def chart_pr_curves(coco_eval) -> str:
    """IoU-sweep PR curves with hover showing threshold values."""
    import plotly.graph_objects as go

    data = PlotData.from_coco_eval(coco_eval)
    a_idx = data.area_idx("all")
    m_idx = data.max_det_idx(None)

    fig = go.Figure(layout=_dark_layout(
        title=None,
        xaxis=dict(title="Recall", range=[0, 1], gridcolor=_BORDER_SUBTLE),
        yaxis=dict(title="Precision", range=[0, 1], gridcolor=_BORDER_SUBTLE),
        height=440,
        legend=dict(font=dict(size=11)),
    ))

    for t_idx, iou_thr in enumerate(data.iou_thresholds):
        prec_raw = data.precision[t_idx, :, :, a_idx, m_idx]  # (R, K)
        # Mask invalid (-1 sentinel) then mean across categories
        prec_masked = np.where(prec_raw < 0, np.nan, prec_raw)
        prec_mean = np.nanmean(prec_masked, axis=1)

        fig.add_trace(go.Scatter(
            x=data.recall_pts.tolist(),
            y=prec_mean.tolist(),
            mode="lines",
            name=f"IoU={iou_thr:.2f}",
            line=dict(width=2.5 if t_idx == 0 else 1.5),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra>IoU=%{fullData.name}</extra>",
        ))

    # Add diagonal reference
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=_TEXT_TERTIARY, width=1, dash="dot"))

    return _to_html(fig, "pr-curves")


# ── Per-Category AP ──────────────────────────────────────────────────

def chart_per_category_ap(coco_eval) -> str:
    """Per-category AP as a native HTML leaderboard with expand/collapse."""
    from html import escape

    results = coco_eval.results(per_class=True)
    per_class = results.get("per_class", {})
    if not per_class:
        return "<p style='color: #9a918a; text-align: center; padding: 40px;'>No per-class data available.</p>"

    items = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    mean_ap = sum(v for _, v in items) / len(items) if items else 0
    max_ap = max(v for _, v in items) if items else 1
    total = len(items)
    collapsed_n = 25

    rows = []
    for rank, (name, ap) in enumerate(items, 1):
        pct = (ap / max_ap * 100) if max_ap > 0 else 0
        esc_name = escape(name)
        above_mean = "above" if ap >= mean_ap else "below"
        hidden = ' hidden' if rank > collapsed_n and total > collapsed_n else ''
        rows.append(
            f'<a class="cat-row{hidden}" href="/?categories={esc_name}" data-rank="{rank}">'
            f'<span class="cat-rank">{rank}</span>'
            f'<span class="cat-name">{esc_name}</span>'
            f'<span class="cat-bar-wrap">'
            f'<span class="cat-bar {above_mean}" style="width:{pct:.1f}%"></span>'
            f'</span>'
            f'<span class="cat-ap">{ap:.3f}</span>'
            f'</a>'
        )

    toggle_html = ""
    if total > collapsed_n:
        toggle_html = (
            f'<button class="cat-toggle" id="cat-ap-toggle" onclick="toggleCatAP()">'
            f'<span class="cat-toggle-text">Show all {total} categories</span>'
            f'<span class="cat-toggle-icon">\u25BE</span>'
            f'</button>'
        )

    mean_line = (
        f'<div class="cat-mean">'
        f'<span class="cat-mean-label">Mean AP</span>'
        f'<span class="cat-mean-value">{mean_ap:.3f}</span>'
        f'</div>'
    )

    script = (
        '<script>'
        'function toggleCatAP(){'
        '  var rows=document.querySelectorAll(".cat-row.hidden");'
        '  var btn=document.getElementById("cat-ap-toggle");'
        '  var txt=btn.querySelector(".cat-toggle-text");'
        '  var icon=btn.querySelector(".cat-toggle-icon");'
        '  if(rows.length>0){'
        '    document.querySelectorAll(".cat-row").forEach(function(r){r.classList.remove("hidden")});'
        f'    txt.textContent="Show top {collapsed_n}";'
        '    icon.textContent="\u25B4";'
        '  }else{'
        f'    document.querySelectorAll(".cat-row").forEach(function(r,i){{if(i>={collapsed_n})r.classList.add("hidden")}});'
        f'    txt.textContent="Show all {total} categories";'
        '    icon.textContent="\u25BE";'
        '    document.getElementById("cat-ap-list").scrollIntoView({behavior:"smooth",block:"start"});'
        '  }'
        '}'
        '</script>'
    )

    return (
        f'{mean_line}'
        f'<div class="cat-ap-list" id="cat-ap-list">'
        + "\n".join(rows)
        + f'</div>'
        + toggle_html
        + script
    )


# ── Confusion Matrix ─────────────────────────────────────────────────

def chart_confusion_matrix(coco_eval, iou_thr=0.5) -> str:
    """Interactive confusion matrix heatmap."""
    import plotly.graph_objects as go

    cm = coco_eval.confusion_matrix(iou_thr=iou_thr)
    raw_matrix = np.asarray(cm["matrix"], dtype=float)
    norm_matrix = np.asarray(cm["normalized"], dtype=float)
    cat_names = list(cm["cat_names"])
    K = len(cat_names)
    labels = cat_names + ["BG"]

    # Auto top-N for large category sets
    top_n = 25 if K > 30 else None
    data = norm_matrix

    if top_n is not None and top_n < K:
        cat_block = data[:K, :K]
        row_total = cat_block.sum(axis=1)
        diag = np.diag(cat_block)
        col_total = cat_block.sum(axis=0)
        confusion_mass = (row_total - diag) + (col_total - diag)
        top_indices = np.argsort(confusion_mass)[::-1][:top_n]
        keep = sorted(top_indices.tolist()) + [len(labels) - 1]
        data = data[np.ix_(keep, keep)]
        labels = [labels[i] for i in keep]

    n = len(labels)

    # Build customdata: gt_name for each cell (row label)
    customdata = [[labels[i] for _ in range(n)] for i in range(n)]

    # Hover text with counts
    hover_text = []
    for i in range(n):
        row = []
        for j in range(n):
            val = data[i, j]
            row.append(f"GT: {labels[i]}<br>Pred: {labels[j]}<br>Rate: {val:.3f}")
        hover_text.append(row)

    # Show cell text only for small matrices
    show_text = n <= 20
    text = [[f"{data[i][j]:.2f}" if data[i][j] > 0.01 else "" for j in range(n)] for i in range(n)] if show_text else None

    fig = go.Figure(layout=_dark_layout(
        title=None,
        xaxis=dict(title="Predicted", tickangle=45, gridcolor=_BORDER_SUBTLE, tickfont=dict(color=_TEXT_SECONDARY, size=11)),
        yaxis=dict(title="Ground Truth", autorange="reversed", gridcolor=_BORDER_SUBTLE, tickfont=dict(color=_TEXT_SECONDARY, size=11)),
        height=max(550, 24 * n + 120),
        margin=dict(l=120, r=40, t=20, b=100),
        dragmode=False,
    ))

    fig.add_trace(go.Heatmap(
        z=data.tolist(),
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}" if show_text else None,
        textfont=dict(size=max(7, min(11, 200 // n))),
        customdata=customdata,
        hovertext=hover_text,
        hoverinfo="text",
        colorscale=[[0, _BG_ELEVATED], [0.5, "#8B6914"], [1, _ACCENT]],
        colorbar=dict(title="Rate", tickfont=dict(color=_TEXT_SECONDARY)),
        zmin=0,
        zmax=1,
    ))

    click_script = """
    var plotDiv = document.getElementById('{plot_id}');
    plotDiv.on('plotly_click', function(data) {
        var pt = data.points[0];
        var gtCat = pt.y;
        if (gtCat === 'BG') return;
        window.location.href = '/?categories=' + encodeURIComponent(gtCat) + '&eval_filter=has_errors';
    });
    """

    return _to_html(fig, "confusion-matrix", post_script=click_script)


# ── TIDE Errors ──────────────────────────────────────────────────────

def chart_tide_errors(coco_eval) -> str:
    """TIDE error breakdown as native HTML bars."""
    tide = coco_eval.tide_errors()
    delta_ap = tide["delta_ap"]
    ap_base = tide["ap_base"]
    counts = tide.get("counts", {})

    error_types = [
        ("Cls", "Classification", "Predicted wrong class"),
        ("Loc", "Localization", "Poor bounding box overlap"),
        ("Both", "Cls + Loc", "Wrong class and poor overlap"),
        ("Dupe", "Duplicate", "Redundant detection of same object"),
        ("Bkg", "Background", "Detection on background region"),
        ("Miss", "Missed", "Failed to detect a ground truth"),
    ]

    max_val = max((delta_ap.get(k, 0.0) for k, _, _ in error_types), default=0.001) or 0.001

    rows = []
    for key, label, desc in error_types:
        val = delta_ap.get(key, 0.0)
        count = counts.get(key, 0)
        pct = (val / max_val * 100) if max_val > 0 else 0
        rows.append(
            f'<div class="tide-row">'
            f'<span class="tide-label" title="{desc}">{label}</span>'
            f'<span class="tide-bar-wrap">'
            f'<span class="tide-bar" style="width:{pct:.1f}%"></span>'
            f'</span>'
            f'<span class="tide-delta">{val:.4f}</span>'
            f'<span class="tide-count">{count:,}</span>'
            f'</div>'
        )

    return (
        f'<div class="tide-header">'
        f'<span class="tide-header-label">\u0394AP impact if error type were fixed</span>'
        f'</div>'
        + "\n".join(rows)
        + f'<div class="tide-footer">'
        f'<span class="tide-footer-label">Baseline AP @ IoU=0.50</span>'
        f'<span class="tide-footer-value">{ap_base:.3f}</span>'
        f'</div>'
    )


# ── Calibration ──────────────────────────────────────────────────────

def chart_calibration(coco_eval) -> str:
    """Reliability diagram with ECE/MCE annotation."""
    import plotly.graph_objects as go

    cal = coco_eval.calibration()
    bins = cal["bins"]
    ece = cal["ece"]
    mce = cal["mce"]

    midpoints = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
    accuracies = [b["avg_accuracy"] for b in bins]
    counts = [b["count"] for b in bins]
    bin_width = (bins[0]["bin_upper"] - bins[0]["bin_lower"]) * 0.85 if bins else 0.085

    # Filter to non-empty bins
    mid_ne, acc_ne, cnt_ne = [], [], []
    for m, a, c in zip(midpoints, accuracies, counts):
        if c > 0:
            mid_ne.append(m)
            acc_ne.append(a)
            cnt_ne.append(c)

    fig = go.Figure(layout=_dark_layout(
        title=None,
        xaxis=dict(title="Confidence", range=[0, 1], gridcolor=_BORDER_SUBTLE),
        yaxis=dict(title="Accuracy", range=[0, 1], gridcolor=_BORDER_SUBTLE),
        height=420,
    ))

    # Perfect calibration diagonal
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=_TEXT_TERTIARY, width=1, dash="dot"))

    # Accuracy bars
    fig.add_trace(go.Bar(
        x=mid_ne,
        y=acc_ne,
        width=bin_width,
        marker=dict(color=_ACCENT, opacity=0.7),
        name="Accuracy",
        customdata=cnt_ne,
        hovertemplate="Confidence: %{x:.2f}<br>Accuracy: %{y:.3f}<br>Count: %{customdata}<extra></extra>",
    ))

    # ECE/MCE annotation
    fig.add_annotation(
        x=0.95, y=0.05, xref="paper", yref="paper",
        text=f"ECE = {ece:.4f}<br>MCE = {mce:.4f}",
        showarrow=False, align="right",
        font=dict(size=12, family=_FONT_MONO, color=_TEXT_PRIMARY),
        bgcolor=_BG_SURFACE, bordercolor=_BORDER_SUBTLE, borderwidth=1, borderpad=6,
    )

    return _to_html(fig, "calibration")


# ── F1 Distribution ──────────────────────────────────────────────────

def chart_f1_distribution(coco_eval, iou_thr=0.5) -> str:
    """Histogram of per-image F1 scores, colored by error profile."""
    import plotly.graph_objects as go

    diag = coco_eval.image_diagnostics(iou_thr=iou_thr)
    img_summary = diag.get("img_summary", {})
    if not img_summary:
        return "<p style='color: #9a918a; text-align: center; padding: 40px;'>No image diagnostics available.</p>"

    # Desaturated versions of eval status colors for chart readability
    profile_colors = {
        "perfect": "#2d8a4e",    # muted green
        "fp_heavy": "#c0392b",   # muted red
        "fn_heavy": "#2e6da4",   # muted blue
        "mixed": "#7c6f94",      # muted purple
    }

    # Group F1 scores by error profile
    by_profile: dict[str, list[float]] = {}
    for s in img_summary.values():
        profile = s.get("error_profile", "mixed")
        by_profile.setdefault(profile, []).append(s.get("f1", 0.0))

    fig = go.Figure(layout=_dark_layout(
        title=None,
        xaxis=dict(title="F1 Score", range=[0, 1.05], gridcolor=_BORDER_SUBTLE),
        yaxis=dict(title="Image Count", gridcolor=_BORDER_SUBTLE),
        barmode="stack",
        height=360,
        legend=dict(font=dict(size=11)),
    ))

    for profile in ["perfect", "fp_heavy", "fn_heavy", "mixed"]:
        scores = by_profile.get(profile, [])
        if not scores:
            continue
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=20,
            name=profile.replace("_", " ").title(),
            marker=dict(color=profile_colors.get(profile, _TEXT_TERTIARY)),
            hovertemplate="F1: %{x:.2f}<br>Count: %{y}<extra>%{fullData.name}</extra>",
        ))

    return _to_html(fig, "f1-dist")


# ── Label Errors Table ───────────────────────────────────────────────

def label_errors_table(coco_eval, iou_thr=0.5, top_n=20) -> list[dict]:
    """Top suspected label errors for HTML table rendering."""
    diag = coco_eval.image_diagnostics(iou_thr=iou_thr)
    errors = diag.get("label_errors", [])
    return errors[:top_n]


# ── Orchestrator ─────────────────────────────────────────────────────

def build_dashboard(coco_eval, slices=None) -> dict:
    """Compute all dashboard data at once. Returns dict for template rendering."""
    # Grab contextual info for template card subtitles
    tide = coco_eval.tide_errors()

    result = {
        "kpi": kpi_tiles(coco_eval),
        "pr_curves_html": chart_pr_curves(coco_eval),
        "per_cat_ap_html": chart_per_category_ap(coco_eval),
        "confusion_html": chart_confusion_matrix(coco_eval),
        "tide_html": chart_tide_errors(coco_eval),
        "calibration_html": chart_calibration(coco_eval),
        "f1_dist_html": chart_f1_distribution(coco_eval),
        "label_errors": label_errors_table(coco_eval),
        "iou_type": coco_eval.params.iou_type,
        "num_categories": len(coco_eval.params.cat_ids),
        "num_images": len(coco_eval.params.img_ids),
        "tide_ap_base": tide["ap_base"],
    }

    if slices:
        try:
            result["slice_data"] = coco_eval.slice_by(slices)
        except Exception:
            result["slice_data"] = None
    else:
        result["slice_data"] = None

    return result
