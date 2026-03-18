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

That's it. A Gradio app opens in your browser (or inline in Jupyter) showing a
scrollable grid of annotated images.

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
│  coco.browse()                                                   │
│  Dataset browser · powered by hotcoco                           │
├──────────────┬───────────────────────────┬──────────────────────┤
│  FILTERS     │  Image grid               │  Detail              │
│              │                           │                      │
│  Category    │  [img] [img] [img] [img]  │  (click any image)   │
│  [dropdown]  │  [img] [img] [img] [img]  │                      │
│              │  [img] [img] [img] [img]  │  Full-res image      │
│  Show:       │                           │  with native mask,   │
│  ☑ bbox      │                           │  bbox, and label     │
│  ☑ segm      │                           │  overlays + legend   │
│  ☑ keypoints │                           │                      │
│              │                           │                      │
│  [Shuffle ⇄] │                           │                      │
│  12 of 500   │  [Load more]              │                      │
├──────────────┴───────────────────────────┴──────────────────────┤
│                    hotcoco dataset browser                       │
└──────────────────────────────────────────────────────────────────┘
```

**Sidebar controls:**

| Control | What it does |
|---------|-------------|
| Category dropdown | Filter to images containing the selected categories |
| bbox / segm / keypoints | Toggle which annotation layers are rendered |
| Shuffle ⇄ | Randomise the display order |
| N of M images | Live image count for the current filter |

**Image grid:**

- Clean thumbnail grid for fast navigation
- Click any thumbnail to load the full-resolution image in the detail panel
- **Load more** appends the next batch without losing what's already loaded

**Annotation detail panel:**

- Renders at full resolution using Gradio's native `AnnotatedImage` component
- Segmentation masks and bounding boxes are rendered by the browser — no PIL scaling artifacts
- A color-coded legend lists every category present in the image
- Toggling annotation type checkboxes updates the open detail view instantly

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

| Annotation type | Detail panel |
|----------------|-------------|
| Segmentation mask | Native browser overlay — no PIL scaling artifacts |
| Bounding box | Native browser overlay (shown when no mask, or mask disabled) |
| Keypoints | Dots + skeleton lines drawn on the image |

Masks and bounding boxes are rendered by Gradio's `AnnotatedImage` component directly
in the browser, so they stay crisp at any zoom level. A per-category color legend
appears below the image automatically.

Colors are assigned per category deterministically — the same category always gets the
same color across all images in the session.

If an image file is missing from `image_dir`, a gray placeholder is shown instead
of raising an error.

---

## CLI options

```bash
coco explore \
    --gt <annotations.json> \
    --images <images_dir/> \
    [--batch-size 12] \
    [--port 7860] \
    [--share]
```

`--share` creates a public Gradio link — useful on Colab or remote servers where
`localhost` isn't accessible.

---

## Advanced: `build_app`

For full control over launching, call `build_app` directly and call `.launch()` yourself:

```python
from hotcoco import COCO, browse

coco = COCO("annotations.json", image_dir="/data/images/")
app = browse.build_app(coco, batch_size=24)
app.launch(server_port=7861, share=True)
```

`build_app` returns a `gr.Blocks` object that has not been launched yet. This lets
you embed it inside a larger Gradio app, configure authentication, or pass any
`launch()` option.
