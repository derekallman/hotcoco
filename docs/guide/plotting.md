# Plotting

hotcoco includes publication-quality plotting for evaluation results.
Install the optional dependency:

```bash
pip install hotcoco[plot]
```

All plot functions return `(Figure, Axes)` for further customization,
accept an optional `ax` to draw on existing axes, and accept `save_path`
to write directly to disk.

## PDF evaluation report

`report()` runs a full evaluation and saves a self-contained, single-page PDF —
useful for archiving results or sharing with collaborators.

```python
from hotcoco.plot import report

gt = hotcoco.COCO("instances_val2017.json")
dt = gt.load_res("bbox_results.json")
ev = hotcoco.COCOeval(gt, dt, "bbox")
ev.run()

report(ev, save_path="report.pdf", gt_path="instances_val2017.json", dt_path="bbox_results.json")
```

The report includes a run context block (dataset paths, eval params, image/annotation counts),
a full metrics table, precision-recall curves at IoU 0.50, 0.75, and the mean, F1 peak,
and a per-category AP bar chart sorted from best to worst.

Works with all three evaluation modes — hotcoco automatically selects the right metric rows
for each:

| Mode | Rows |
|------|------|
| `bbox` / `segm` | AP, AP50, AP75, APs, APm, APl · AR1, AR10, AR100, ARs, ARm, ARl |
| `keypoints` | AP, AP50, AP75, APm, APl · AR, AR50, AR75, ARm, ARl |
| LVIS | AP, AP50, AP75, APs, APm, APl, APr, APc, APf · AR@300, ARs@300, ARm@300, ARl@300 |

Or from the CLI (requires `pip install hotcoco[plot]`):

```bash
coco eval --gt instances_val2017.json --dt bbox_results.json --report report.pdf
```

---

## Quick start

```python
import hotcoco
from hotcoco.plot import pr_curve, per_category_ap

gt = hotcoco.COCO("instances_val2017.json")
dt = gt.load_res("detections.json")
ev = hotcoco.COCOeval(gt, dt, "bbox")
ev.run()

fig, ax = pr_curve(ev, save_path="pr.png")
fig, ax = per_category_ap(ev.results(per_class=True), save_path="ap.png")
```

## Available plots

### Precision-recall curves

```python
from hotcoco.plot import pr_curve

# IoU sweep (default) — one line per IoU threshold
fig, ax = pr_curve(ev)

# Single category
fig, ax = pr_curve(ev, cat_id=1)

# Top 10 categories at IoU=0.50
fig, ax = pr_curve(ev, iou_thr=0.50, top_n=10)
```

### Per-category AP

```python
from hotcoco.plot import per_category_ap

results = ev.results(per_class=True)
fig, ax = per_category_ap(results)
```

Shows horizontal bars sorted by AP with a mean AP reference line.
When there are many categories, the top 20 and bottom 5 are shown
with a visual break.

### Confusion matrix

```python
from hotcoco.plot import confusion_matrix

fig, ax = confusion_matrix(ev.confusion_matrix())
```

For datasets with many categories (>30), the matrix auto-filters to
the 25 most confused categories. You can also aggregate by supercategory:

```python
# Build supercategory groups from the dataset
cats = gt.load_cats(gt.get_cat_ids())
groups = {}
for c in cats:
    groups.setdefault(c["supercategory"], []).append(c["name"])

fig, ax = confusion_matrix(ev.confusion_matrix(), group_by="supercategory", cat_groups=groups)
```

### Top confusions

```python
from hotcoco.plot import top_confusions

fig, ax = top_confusions(ev.confusion_matrix())
```

A bar chart of the most common misclassifications — more readable than
a full heatmap when you have many categories.

### TIDE error breakdown

```python
from hotcoco.plot import tide_errors

fig, ax = tide_errors(ev.tide_errors())
```

Shows the six TIDE error types (Cls, Loc, Both, Dupe, Bkg, Miss)
as horizontal bars with their delta-AP values.

### Model comparison

```python
from hotcoco import compare
from hotcoco.plot import comparison_bar, category_deltas

result = compare(ev_a, ev_b, n_bootstrap=1000)
fig, ax = comparison_bar(result)           # grouped bar chart of all metrics
fig, ax = category_deltas(result, top_k=10) # per-category AP delta bars
```

`comparison_bar` shows Model A vs Model B side by side for each metric, with
bootstrap CI error bars when available. `category_deltas` shows per-category
AP deltas sorted by magnitude — green for improvements, red for regressions.

## Themes

Every plot function accepts a `theme` argument:

| Theme | Description |
|-------|-------------|
| `"warm-slate"` | Default. Warm off-white background, terracotta + slate palette. |
| `"scientific-blue"` | Cool/academic. Blue-grey background, navy + red anchor colors. |
| `"ember"` | Warm/editorial. Parchment background, rust + copper + amber palette. |

```python
fig, ax = pr_curve(ev, theme="scientific-blue")
fig, ax = per_category_ap(results, theme="ember")
```

Add `paper_mode=True` to force white backgrounds — useful when embedding in LaTeX or PowerPoint:

```python
fig, ax = pr_curve(ev, theme="scientific-blue", paper_mode=True, save_path="pr.pdf")
```

To apply a theme to your own matplotlib code, use the `style()` context manager:

```python
from hotcoco.plot import style
import matplotlib.pyplot as plt

with style(theme="warm-slate", paper_mode=True):
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    fig.savefig("custom.pdf")
```

To use a completely different style (seaborn, corporate rcParams, etc.), simply don't call
hotcoco plot functions inside a `style()` context — the default matplotlib style applies:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
fig, ax = plt.subplots()
# your own plotting code here
```

## Composing plots

Pass an existing `ax` to draw on a subplot:

```python
import matplotlib.pyplot as plt
from hotcoco.plot import pr_curve, per_category_ap

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
pr_curve(ev, ax=ax1)
per_category_ap(ev.results(per_class=True), ax=ax2)
fig.savefig("dashboard.png", dpi=150)
```
