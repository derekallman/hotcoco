# STYLE.md

Rules for user-facing text: docs, CLI output, error messages, and docstrings.
This file targets **consistency between human and AI-authored contributions.**

## Terminology

Use the abbreviated form in technical/API contexts, the spelled-out form in introductory prose.

| Canonical (technical) | Spelled-out (intro/README) | Never use |
|-----------------------|----------------------------|-----------|
| bbox | bounding box | bounding-box, BBox |
| segm | segmentation | seg |
| keypoints | keypoint evaluation | kpt (except `kpt_oks_sigmas`) |
| IoU | Intersection over Union | iou (prose), IOU |
| AP | Average Precision | ap (prose) |
| AR | Average Recall | ar (prose) |
| RLE | Run-Length Encoding | rle (prose) |
| GT | ground truth | ground-truth (hyphenated) |
| DT | detection / detections | det |
| COCO | COCO dataset / COCO format | Coco, coco (prose) |

**Rule of thumb:** If it's a parameter value or API name, use the short form (`"bbox"`, `"segm"`, `"keypoints"`). If it's a sentence explaining something to a new user, spell it out on first use, then abbreviate.

## CLI Output

Both the Rust CLI (`coco-eval`) and the Python CLI (`coco`) follow the same conventions:

- **Status lines:** `Verb  object  (timing)` — e.g., `Loaded  ground truth val.json  0.42s`
- **Colors:** green = verb/success, dim = filenames/secondary info, red = errors
- **No periods** at the end of status messages
- **Verbs are capitalized:** Loaded, Evaluated, Saved, Written
- **Filenames are dimmed**, not quoted

## Error Messages

- **No periods** at the end
- **Start with uppercase** for standalone messages: `"Cannot use both oid_style and lvis_style"`
- **Start with lowercase** after a prefix: `"load_res: numpy array must have 6 or 7 columns"`
- **Use abbreviations** (bbox, RLE, IoU) — error messages target developers, not beginners
- **Include the actual value** when helpful: `"expected 6 or 7 columns, got {ncols}"`

## Documentation

- **Voice:** Second person, imperative. "Set `iou_type` to..." not "The user should set..."
- **Passive is OK for definitions:** "IoU is computed as..."
- **Code examples use snake_case API** (`coco_gt`, `get_ann_ids`), not camelCase aliases
- **Python-first:** Lead with Python usage, mention Rust alternative second
- **One blank line** between sections in narrative docs

## Python Docstrings (numpydoc)

All `#[doc = "..."]` strings in PyO3 bindings use numpydoc format:

```
Short summary line.

Longer description if needed.

Parameters
----------
name : type
    Description as a sentence fragment, no trailing period
name : type, optional
    Description. Default ``value``.

Returns
-------
type
    Description.

Raises
------
ExceptionType
    When this happens.

Example
-------
::

    code here
```

- **Parameter descriptions** are sentence fragments, no trailing period
- **Use double backticks** for Python identifiers in docstrings: `` ``COCO`` ``, `` ``None`` ``
- **Default values** noted inline: `"Default ``0.5``."`

## Brand

- **hotcoco** — always lowercase, never HotCoco or HOTCOCO
- **pycocotools** — always lowercase (matching upstream)
- **COCO** — uppercase when referring to the dataset or format
