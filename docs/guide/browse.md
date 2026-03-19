# Dataset Browser

hotcoco ships a lightweight visual browser for COCO datasets — no FiftyOne, no heavy
dependencies. Two entry points:

- **`coco.browse()`** — launches inline in Jupyter; opens a local server otherwise
- **`coco explore`** — CLI subcommand for standalone use

Both require the `[browse]` optional extra:

```bash
pip install hotcoco[browse]
```

---

## Quick start

```python
from hotcoco import COCO

coco = COCO("instances_val2017.json", image_dir="/data/coco/val2017/")
coco.browse()
```

That's it. A local server starts and a browser tab opens showing a scrollable grid
of annotated thumbnails with a lightbox for full-resolution detail.

From the command line:

```bash
coco explore \
    --gt instances_val2017.json \
    --images /data/coco/val2017/
```

---

## UI overview

```
┌──────────────────────────────────────────────────────────────────┐
│  hotcoco · dataset browser                                       │
├──────────────┬───────────────────────────────────────────────────┤
│  FILTERS     │  Thumbnail grid                                   │
│              │                                                   │
│  Category    │  [img] [img] [img] [img]                          │
│  [dropdown]  │  [img] [img] [img] [img]                          │
│              │  [img] [img] [img] [img]                          │
│  Min Score   │                ↓ infinite scroll                   │
│  [slider]    │                                                   │
│              │                                                   │
│  [Shuffle ⇄] │                                                   │
│  24 of 5000  │                                                   │
├──────────────┴───────────────────────────────────────────────────┤
│                                                                   │
│  Lightbox (click any thumbnail):                                 │
│  ┌─────────────────────────────┬──────────────┐                  │
│  │  Full-res image + canvas    │  Annotations  │                 │
│  │  overlay (bbox, segm, kpts) │  list sidebar │                 │
│  │                             │              │                  │
│  │  [☑ Boxes] [☑ Segm] [☑ KP] │  ● person GT │                 │
│  │  [☑ GT] [☑ DT]             │  ● car   0.92│                  │
│  └─────────────────────────────┴──────────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

**Sidebar controls:**

| Control | What it does |
|---------|-------------|
| Category dropdown | Filter to images containing the selected categories (multi-select) |
| Min Score slider | Filter detections below a confidence threshold (when DT loaded) |
| Shuffle | Randomise the display order |
| N of M images | Live image count for the current filter |

**Thumbnail grid:**

- Annotated thumbnails with bounding boxes rendered server-side
- Infinite scroll — new batches load automatically as you scroll down
- Click any thumbnail to open the lightbox

**Lightbox:**

- Full-resolution image with a canvas overlay for annotations
- Hover any annotation to highlight it — the sidebar and canvas stay in sync
- Toggle layers (Boxes / Segments / Keypoints) and sources (GT / DT) instantly
- Scroll to zoom, drag to pan, double-click to reset
- Arrow keys navigate between images; Escape closes the lightbox

---

## Viewing detections

Pass detection results to `browse()` and hotcoco overlays your model's predictions
alongside ground truth:

```python
from hotcoco import COCO

coco = COCO("instances_val2017.json", image_dir="/data/coco/val2017/")

# Pass a path — auto-loaded via load_res()
coco.browse(dt="bbox_results.json")

# Or pass a COCO object you already have
results = coco.load_res("bbox_results.json")
coco.browse(dt=results)
```

From the command line:

```bash
coco explore \
    --gt instances_val2017.json \
    --images /data/coco/val2017/ \
    --dt bbox_results.json
```

**What you get with `dt` loaded:**

| Feature | Description |
|---------|-------------|
| GT labels | Ground truth annotations with solid bounding boxes |
| DT labels | Detection predictions with dashed bounding boxes |
| Score display | Confidence score shown on each detection label |
| Sources toggle | Show/hide ground truth and detections independently |
| Min confidence slider | Filter detections below a score threshold (0–1) |

GT and DT use the same per-category color palette so you can compare spatially.
When no DT is loaded the browser behaves exactly as before (no slider, no source toggles).

!!! tip
    Detections often lack segmentation masks. When `segm` is selected but a detection
    has no mask, the browser falls back to its bounding box automatically.

---

## `image_dir`

The browser needs to know where your images live. Pass it at construction:

```python
coco = COCO("annotations.json", image_dir="/data/images/")
coco.browse()
```

Or set it after the fact:

```python
coco = COCO("annotations.json")
coco.image_dir = "/data/images/"
coco.browse()
```

Or pass it directly to `browse()` (overrides `image_dir` on the object):

```python
coco.browse(image_dir="/different/path/")
```

`image_dir` is propagated automatically through `filter`, `split`, `sample`,
and `load_res`, so a filtered subset keeps the same path:

```python
people = coco.filter(cat_ids=[1])
people.browse()  # image_dir carried over from coco
```

---

## Annotation rendering

| Annotation type | How it's rendered |
|----------------|-------------------|
| Bounding box | Canvas overlay — solid stroke for GT, dashed for DT |
| Segmentation | Canvas polygon fill + stroke |
| Keypoints | Dots + skeleton lines on canvas |

All annotations are rendered client-side on an HTML Canvas overlay, so they stay
crisp at any zoom level. Colors are assigned per category deterministically — the
same category always gets the same color across all images.

If an image file is missing from `image_dir`, a gray placeholder is shown instead
of raising an error.

---

## Responsive layout

The browser adapts to different viewport sizes, making it suitable for Jupyter
IFrames, side-by-side notebook layouts, and standalone browser windows:

- **Wide (800px+):** Sidebar on the left, gallery grid on the right
- **Medium (480–800px):** Toolbar mode — controls collapse to a compact horizontal bar
- **Narrow (<480px):** Minimal toolbar, smaller thumbnails, full-screen lightbox

The lightbox switches between side-by-side (image + info panel) and stacked
(image above, annotations below) layout depending on available width.

---

## CLI options

```bash
coco explore \
    --gt <annotations.json> \
    --images <images_dir/> \
    [--dt <results.json>] \
    [--batch-size 12] \
    [--port 7860]
```

---

## Advanced: `create_app`

For full control over launching, use `create_app` directly:

```python
from hotcoco import COCO
from hotcoco.server import create_app, run_server

coco = COCO("annotations.json", image_dir="/data/images/")
app = create_app(coco, batch_size=24)
run_server(app, port=7861, open_browser=True)
```

`create_app` returns a FastAPI app. You can mount it inside a larger application
or run it with any ASGI server.
