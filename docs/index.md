---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# hotcoco

<p class="hero-tagline">
Perception evaluation for Python, written in Rust.
</p>

<p class="hero-sub">
hotcoco evaluates perception models, starting with detection: boxes, masks, keypoints, and oriented boxes on the COCO, LVIS, and Open Images protocols. It's a drop-in replacement for pycocotools — same numbers to double precision, up to 84× faster — and it includes the analysis you'd otherwise need separate tools for: TIDE error analysis, confusion matrices, calibration, model comparison, and a dataset browser.
</p>

<div class="hero-actions" markdown>

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[API Reference](api/coco.md){ .md-button }
[Open Notebook](https://github.com/derekallman/hotcoco/blob/main/examples/coco_evaluation_101.ipynb){ .md-button }

</div>

</div>

## Quick start

```bash
pip install hotcoco
```

=== "Python"

    ```python
    from hotcoco import COCO, COCOeval

    coco_gt = COCO("instances_val2017.json")
    coco_dt = coco_gt.load_res("detections.json")

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.run()
    ```

=== "Drop-in replacement"

    ```python
    from hotcoco import init_as_pycocotools
    init_as_pycocotools()

    # All pycocotools imports now resolve to hotcoco
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ```

=== "CLI"

    ```bash
    coco eval --gt instances_val2017.json --dt detections.json --iou-type bbox
    ```

=== "Rust"

    ```rust
    use hotcoco::{COCO, COCOeval};
    use hotcoco::params::IouType;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("instances_val2017.json"))?;
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<strong>Evaluate</strong>
<p>COCO, LVIS, and Open Images protocols over boxes, masks, keypoints, and oriented boxes. <code>init_as_pycocotools()</code> patches existing pycocotools imports in place — no code changes.</p>
</div>

<div class="feature-card" markdown>
<strong>Diagnose</strong>
<p>TIDE error analysis, confusion matrices, confidence calibration, and per-image label-error detection — see what's actually costing you AP.</p>
</div>

<div class="feature-card" markdown>
<strong>Explore your data</strong>
<p>Browse any COCO dataset in a local web UI with annotation overlays, run a dataset healthcheck, and convert between COCO, YOLO, Pascal VOC, CVAT, DOTA, and Open Images CSV.</p>
</div>

<div class="feature-card" markdown>
<strong>Compose</strong>
<p>IoU kernels, matchers, and metric functions are public and work on plain numpy arrays. Every evaluation returns the same report shape, and each number is marked as standard or a hotcoco extension.</p>
</div>

</div>

## Performance

Bbox evaluation on COCO val2017 takes **0.07s**; pycocotools takes 5.31s. The
precision, recall, and scores arrays are bit-identical to pycocotools', and every
summary metric matches to the limit of double precision.

<figure markdown>
![Grouped bar chart of evaluation wall clock for bbox, segm and keypoints across the five libraries](assets/benchmark-speed.png#only-light)
![Grouped bar chart of evaluation wall clock for bbox, segm and keypoints across the five libraries](assets/benchmark-speed-dark.png#only-dark)
<figcaption>COCO val2017, 36,781 detections.</figcaption>
</figure>

Full tables, hardware, memory, and parity verification are in [Benchmarks](benchmarks.md).

## Error analysis

mAP tells you that your model misses; it doesn't tell you what to fix. hotcoco computes
the breakdowns from the same evaluation pass: which error types cost the most AP (TIDE),
which categories get confused with each other, whether the confidence scores are
calibrated, and which images the model does worst on.

<div class="figure-gallery" markdown>

<figure markdown>
![Row-normalized confusion matrix showing which COCO categories get mistaken for each other](assets/confusion-matrix.png#only-light)
![Row-normalized confusion matrix showing which COCO categories get mistaken for each other](assets/confusion-matrix-dark.png#only-dark)
<figcaption>Which categories the model confuses, and how much of the loss is background
rather than a mix-up. See the <a href="guide/diagnostics/#confusion-matrix">confusion matrix guide</a>.</figcaption>
</figure>

<figure markdown>
![Per-category AP as horizontal bars, best to worst](assets/per-category-ap.png#only-light)
![Per-category AP as horizontal bars, best to worst](assets/per-category-ap-dark.png#only-dark)
<figcaption>The 0.70 mean AP spans 0.96 (bed) to 0.14 (sports ball). See
<a href="guide/results/#extracting-per-category-ap">per-category AP</a>.</figcaption>
</figure>

</div>

## The dataset browser

`coco.browse()` opens a local web UI that shows every image with its annotations
overlaid, one color per category. Filter by category, zoom in, and scan a split for
labeling problems without opening files one by one. Pass `eval=` and the same UI adds
an interactive dashboard — PR curves, confusion matrix, TIDE errors, and per-image
scores next to the images they come from.

<figure class="screenshot" markdown>
![The hotcoco dataset browser: a category filter sidebar beside a grid of thumbnails with colored annotation overlays](assets/browse-ui.webp)
<figcaption>Every thumbnail is drawn with its annotations already overlaid. See the
<a href="guide/browse/">dataset browser guide</a>.</figcaption>
</figure>

## Use the metrics directly

The metric functions don't require an evaluator — they're plain functions over arrays:

```python
import numpy as np
from hotcoco import metrics

scores  = np.array([0.9, 0.8, 0.7, 0.6])
matched = np.array([True, False, True, True])

metrics.average_precision(scores, matched, num_gt=4)  # 0.6287
metrics.calibration_error(scores, matched)            # (ece, mce)
```

`COCOeval` calls these same functions internally, so numbers you compute by hand match
what `summarize()` prints.

Panoptic and tracking are next, on the same engine —
see the [roadmap](https://github.com/derekallman/hotcoco/blob/main/ROADMAP.md).
