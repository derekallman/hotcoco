# plot

```python
from hotcoco.plot import (
    report,
    pr_curve, pr_curve_iou_sweep, pr_curve_by_category, pr_curve_top_n,
    confusion_matrix, top_confusions, per_category_ap, tide_errors,
    style, SERIES_COLORS, CHROME, SEQUENTIAL,
)
```

Requires `pip install hotcoco[plot]` (matplotlib >= 3.5).

All functions share these common parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `theme` | `str` | Visual theme: `"warm-slate"` (default), `"scientific-blue"`, or `"ember"`. |
| `paper_mode` | `bool` | Set both figure and axes background to white. Useful for LaTeX / PowerPoint. Default `False`. |
| `ax` | <code>Axes &#124; None</code> | Draw on an existing axes. If `None`, creates a new figure. |
| `save_path` | <code>str &#124; Path &#124; None</code> | Save figure to this path (150 DPI). |

All functions return `(Figure, Axes)`.

---

## `pr_curve_iou_sweep`

```python
pr_curve_iou_sweep(
    coco_eval, *,
    iou_thrs=None, area_rng="all", max_det=None,
    theme="warm-slate", paper_mode=False, ax=None, save_path=None,
)
```

Plot one precision-recall curve per IoU threshold, with precision averaged across all categories. The primary line (lowest IoU) gets an under-fill and F1 peak annotation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `iou_thrs` | <code>list[float] &#124; None</code> | IoU thresholds to include. Default: all thresholds in params. |
| `area_rng` | `str` | Area range: `"all"`, `"small"`, `"medium"`, `"large"`. Default `"all"`. |
| `max_det` | <code>int &#124; None</code> | Max detections. Default: last entry in `params.max_dets`. |

---

## `pr_curve_by_category`

```python
pr_curve_by_category(
    coco_eval, cat_id, *,
    iou_thr=0.5, area_rng="all", max_det=None,
    theme="warm-slate", paper_mode=False, ax=None, save_path=None,
)
```

Plot the precision-recall curve for a single category at a fixed IoU threshold, with an F1 peak annotation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `cat_id` | `int` | Category ID to plot. |
| `iou_thr` | `float` | IoU threshold. Default `0.5`. |
| `area_rng` | `str` | Area range. Default `"all"`. |
| `max_det` | <code>int &#124; None</code> | Max detections. Default: last entry in `params.max_dets`. |

---

## `pr_curve_top_n`

```python
pr_curve_top_n(
    coco_eval, *,
    cat_ids=None, top_n=10, iou_thr=0.5, area_rng="all", max_det=None,
    theme="warm-slate", paper_mode=False, ax=None, save_path=None,
)
```

Plot precision-recall curves for multiple categories on one axes. When `cat_ids` is omitted, selects the top `top_n` categories by AP automatically.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `cat_ids` | <code>list[int] &#124; None</code> | Categories to plot. Default: top `top_n` by AP. |
| `top_n` | `int` | Number of top categories when `cat_ids` is omitted. Default `10`. |
| `iou_thr` | `float` | IoU threshold. Default `0.5`. |
| `area_rng` | `str` | Area range. Default `"all"`. |
| `max_det` | <code>int &#124; None</code> | Max detections. Default: last entry in `params.max_dets`. |

---

## `pr_curve`

```python
pr_curve(
    coco_eval, *,
    iou_thrs=None, cat_id=None, cat_ids=None,
    iou_thr=None, top_n=10, area_rng="all", max_det=None,
    theme="warm-slate", paper_mode=False, ax=None, save_path=None,
)
```

Convenience dispatcher — inspects the arguments and calls the appropriate named function. Prefer calling the named functions directly for clarity.

- No `cat_id` or `cat_ids` → `pr_curve_iou_sweep`
- `cat_id` set → `pr_curve_by_category`
- `cat_ids` or `iou_thr` set → `pr_curve_top_n`

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `iou_thrs` | <code>list[float] &#124; None</code> | IoU thresholds to plot (IoU sweep mode). |
| `cat_id` | <code>int &#124; None</code> | Single category to plot. |
| `cat_ids` | <code>list[int] &#124; None</code> | Categories to compare. |
| `iou_thr` | <code>float &#124; None</code> | Fixed IoU for single/multi-category modes. Default 0.50. |
| `top_n` | `int` | Top N categories by AP. Default 10. |
| `area_rng` | `str` | Area range: `"all"`, `"small"`, `"medium"`, `"large"`. |
| `max_det` | <code>int &#124; None</code> | Max detections. Default: last in params. |

---

## `confusion_matrix`

```python
confusion_matrix(
    cm_dict, *,
    normalize=True, top_n=None,
    group_by=None, cat_groups=None,
    theme="warm-slate", paper_mode=False, ax=None, save_path=None,
)
```

Plot a confusion matrix heatmap.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm_dict` | `dict` | Output of `coco_eval.confusion_matrix()`. |
| `normalize` | `bool` | Row-normalize values. Default `True`. |
| `top_n` | <code>int &#124; None</code> | Show only top N categories by confusion mass. Auto-set to 25 when K > 30. |
| `group_by` | <code>str &#124; None</code> | `"supercategory"` to aggregate by group. |
| `cat_groups` | <code>dict &#124; None</code> | Group name → list of category names. Required with `group_by`. |

---

## `top_confusions`

```python
top_confusions(
    cm_dict, *,
    top_n=20,
    theme="warm-slate", paper_mode=False, ax=None, save_path=None,
)
```

Plot the top N misclassifications as horizontal bars. Shows "ground truth → prediction" pairs sorted by count.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm_dict` | `dict` | Output of `coco_eval.confusion_matrix()`. |
| `top_n` | `int` | Number of confusions to show. Default 20. |

---

## `per_category_ap`

```python
per_category_ap(
    results_dict, *,
    top_n=20, bottom_n=5,
    theme="warm-slate", paper_mode=False, ax=None, save_path=None,
)
```

Plot per-category AP as horizontal bars with a mean AP reference line.

| Parameter | Type | Description |
|-----------|------|-------------|
| `results_dict` | `dict` | Output of `coco_eval.results(per_class=True)`. |
| `top_n` | `int` | Top categories to show. Default 20. |
| `bottom_n` | `int` | Bottom categories to show. Default 5. |

---

## `tide_errors`

```python
tide_errors(
    tide_dict, *,
    theme="warm-slate", paper_mode=False, ax=None, save_path=None,
)
```

Plot TIDE error breakdown as horizontal bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tide_dict` | `dict` | Output of `coco_eval.tide_errors()`. |

---

## `report`

```python
report(
    coco_eval, *,
    save_path,
    gt_path=None,
    dt_path=None,
    title="COCO Evaluation Report",
)
```

Generate a publication-quality single-page PDF report. Requires `pip install hotcoco[plot]`.

The report contains:

- **Header** — title and timestamp
- **Run context** — GT/DT file paths, eval params, and dataset statistics (images, annotations, categories, detections)
- **Summary metrics** — AP and AR tables with a PR-curve panel and KPI tiles
- **Per-category AP** — bar chart sorted descending, three columns

The metric rows adapt automatically to the evaluation mode:

| Mode | AP rows | AR rows |
|------|---------|---------|
| `bbox` / `segm` | AP AP50 AP75 APs APm APl | AR1 AR10 AR100 ARs ARm ARl |
| `keypoints` | AP AP50 AP75 APm APl | AR AR50 AR75 ARm ARl |
| LVIS | AP AP50 AP75 APs APm APl APr APc APf | AR@300 ARs@300 ARm@300 ARl@300 |

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `save_path` | <code>str &#124; Path</code> | Output PDF path. |
| `gt_path` | <code>str &#124; None</code> | Ground-truth JSON path shown in the run context block. |
| `dt_path` | <code>str &#124; None</code> | Detections JSON path shown in the run context block. |
| `title` | `str` | Report title shown in the header. Default `"COCO Evaluation Report"`. |

Returns `None`. Raises on I/O error or if `run()` was not called first.

---

## Themes

Three built-in themes:

| Theme | Character |
|-------|-----------|
| `"warm-slate"` | Default. Warm off-white background, terracotta + slate series colors. |
| `"scientific-blue"` | Cool/academic. Light blue-grey background, navy + red anchor colors. |
| `"ember"` | Warm/editorial. Parchment background, rust + copper + amber palette. |

Pass `paper_mode=True` to set figure and axes backgrounds to white, keeping all other theme colors intact. Useful when embedding plots in LaTeX documents or PowerPoint slides.

```python
# Academic paper
fig, ax = pr_curve(ev, theme="scientific-blue", paper_mode=True, save_path="pr.pdf")

# Warm editorial style
fig, ax = per_category_ap(results, theme="ember", save_path="ap.png")
```

Use the `style()` context manager to apply a theme to your own matplotlib code:

```python
from hotcoco.plot import style

with style(theme="scientific-blue", paper_mode=True):
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    fig.savefig("custom.pdf")
```

## Color palette

The `warm-slate` theme constants are available for custom plots:

```python
from hotcoco.plot import SERIES_COLORS, CHROME, SEQUENTIAL
```

- `SERIES_COLORS` — 8 data series colors (warm slate, terracotta, teal, gold, plum, sage, ocean, copper)
- `CHROME` — non-data element colors (text, label, tick, grid, spine, background)
- `SEQUENTIAL` — 4-stop colormap for heatmaps (off-white → steel-blue → slate → deep slate)
