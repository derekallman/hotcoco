# Migrating from pycocotools

hotcoco is a drop-in replacement for pycocotools. This guide covers the two migration paths and the few differences to be aware of.

## Option 1: Change your imports

If you control the code that imports pycocotools, swap the import paths:

```python
# Before
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util

# After
from hotcoco import COCO, COCOeval
from hotcoco import mask as mask_util
```

Everything else stays the same. The classes, methods, and return types are identical.

## Option 2: Zero-code drop-in

If pycocotools is imported by a library you don't control (e.g. mmdet, detectron2), call `init_as_pycocotools()` once at startup:

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()

# All pycocotools imports now resolve to hotcoco
from pycocotools.coco import COCO          # → hotcoco.COCO
from pycocotools.cocoeval import COCOeval  # → hotcoco.COCOeval
from pycocotools import mask               # → hotcoco.mask
```

This patches `sys.modules` so that `pycocotools`, `pycocotools.coco`, `pycocotools.cocoeval`, and `pycocotools.mask` all resolve to their hotcoco equivalents.

!!! tip
    Call `init_as_pycocotools()` before any `pycocotools` imports. The best place is the top of your entry point script.

## Method naming

Both camelCase and snake_case names are supported:

| pycocotools (camelCase) | hotcoco (snake_case) | Notes |
|------------------------|----------------------|-------|
| `getAnnIds()` | `get_ann_ids()` | Both work |
| `getCatIds()` | `get_cat_ids()` | Both work |
| `getImgIds()` | `get_img_ids()` | Both work |
| `loadAnns()` | `load_anns()` | Both work |
| `loadCats()` | `load_cats()` | Both work |
| `loadImgs()` | `load_imgs()` | Both work |
| `loadRes()` | `load_res()` | Both work |
| `annToRLE()` | `ann_to_rle()` | Both work |
| `annToMask()` | `ann_to_mask()` | Both work |

The same applies to `Params` properties: `maxDets` / `max_dets`, `catIds` / `cat_ids`, `imgIds` / `img_ids`, `iouThrs` / `iou_thrs`, `recThrs` / `rec_thrs`, `areaRng` / `area_rng`, `areaRngLbl` / `area_rng_lbl`, `useCats` / `use_cats`, `kptOksSigmas` / `kpt_oks_sigmas`.

And `mask` functions: `toBbox` / `to_bbox`, `frPoly` / `fr_poly`, `frBbox` / `fr_bbox`, `frPyObjects` / `fr_py_objects`.

## Return types

hotcoco returns plain Python dicts and lists, matching pycocotools:

```python
coco = COCO("instances_val2017.json")
anns = coco.load_anns([101])
print(type(anns[0]))  # <class 'dict'>
```

Annotation dicts have the same keys: `id`, `image_id`, `category_id`, `bbox`, `area`, `segmentation`, `iscrowd`, etc.

## What about `params`?

Accessing `ev.params` returns a **clone** of the internal parameters, consistent with pycocotools. Modify it and assign it back, or modify it before calling `evaluate()`:

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.params.cat_ids = [1, 2, 3]      # Modify before evaluate()
ev.params.max_dets = [1, 10, 100]
ev.evaluate()
ev.accumulate()
ev.summarize()
```

## Known differences

| Behavior | pycocotools | hotcoco |
|----------|-------------|-----------|
| Print on load | Prints "loading annotations..." to stdout | Silent |
| `COCO()` with no args | Creates empty instance with print statements | Creates empty instance silently |
| Annotation IDs | Requires unique positive integers | Also accepts 0-based IDs |
| Performance | Single-threaded C + Python | Multi-threaded Rust |

## Metric parity

All 34 COCO metrics match pycocotools within tolerance — your AP and AR scores won't change. Verified on COCO val2017: bbox ≤1e-4, segmentation ≤2e-4, keypoints exact.

See [Benchmarks](../benchmarks.md) for detailed parity verification.
