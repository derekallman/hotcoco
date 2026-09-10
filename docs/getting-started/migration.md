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

Everything else stays the same.

## Option 2: Zero-code drop-in

If pycocotools is imported by a library you don't control, such as mmdet or detectron2, call `init_as_pycocotools()` once at startup:

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()

# All pycocotools imports now resolve to hotcoco
from pycocotools.coco import COCO          # → hotcoco.COCO
from pycocotools.cocoeval import COCOeval  # → hotcoco.COCOeval
from pycocotools import mask               # → hotcoco.mask
```

How the patch works, where to call it, and per-framework notes are in [Framework integrations](../guide/frameworks.md).

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

The same applies to `Params` properties: `maxDets` / `max_dets`, `catIds` / `cat_ids`, `imgIds` / `img_ids`, `iouThrs` / `iou_thrs`, `recThrs` / `rec_thrs`, `areaRng` / `area_rng`, `areaRngLbl` / `area_rng_lbl`, `useCats` / `use_cats`. OKS sigmas have one spelling, `kpt_oks_sigmas` — pycocotools also spells that attribute in snake_case.

And `mask` functions: `toBbox` / `to_bbox`, `frPoly` / `fr_poly`, `frBbox` / `fr_bbox`, `frPyObjects` / `fr_py_objects`.

## Return types

hotcoco returns plain Python dicts and lists, matching pycocotools:

```python
coco = COCO("instances_val2017.json")
anns = coco.load_anns([101])
print(type(anns[0]))  # <class 'dict'>
```

Annotation dicts carry the same keys — `id`, `image_id`, `category_id`, `bbox`, `area`, `segmentation`, `iscrowd` — plus any custom fields your dataset defines.

## Getters return copies — assign back to apply

The data lives in Rust, so `ev.params`, `coco.dataset`, `ev.coco_gt`, and
`ev.coco_dt` return **copies** on each access. Attribute assignment
(`ev.params.cat_ids = [...]`) works — the setter routes the change back to Rust —
but mutating a *container inside* a copy is a no-op:

```python
ev.params.cat_ids = [1, 2, 3]       # applied — attribute assignment works
ev.params.maxDets.append(200)       # no-op — mutates a temporary list

md = ev.params.maxDets              # instead: pull, edit, assign back
md.append(200)
ev.params.maxDets = md
```

The same idiom applies to the dataset dict. pycocotools' in-memory construction
flow works because assigning `coco.dataset` re-indexes:

```python
d = coco.dataset
d["annotations"].append(new_ann)
coco.dataset = d                    # replaces contents and rebuilds the index
coco.createIndex()                  # supported, but a formality after assignment
```

Editing annotations that are already there does not need the whole dataset:

```python
coco.set_ann_field("area", {ann_id: mask_area})   # one field, many annotations
coco.update_anns([edited_ann, ...])               # whole annotations, by id
```

See [`set_ann_field`](../api/coco.md#set_ann_field) and
[`update_anns`](../api/coco.md#update_anns) for what each one raises and when it
re-indexes.

## Known differences

| Behavior | pycocotools | hotcoco |
|----------|-------------|-----------|
| Print on load | Prints "loading annotations..." to stdout | No progress output; loader warnings go to stderr and `coco.load_warnings` |
| `COCO()` with no args | Creates empty instance with print statements | Creates empty instance silently |
| Annotation IDs | Requires unique positive integers | Also accepts 0-based IDs |
| `getAnnIds(areaRng=...)` on annotations missing `area` | Raises `KeyError` | Excludes them from the query |
| Mutating `coco.dataset` / `ev.params` internals in place | Mutates shared state | No-op on a copy — [assign back to apply](#getters-return-copies-assign-back-to-apply) |
| Performance | Single-threaded C + Python | Multi-threaded Rust |

## Metric parity

Every COCO metric matches pycocotools to floating-point precision — the measured differences on COCO val2017, and a script to check your own data, are in [Metric parity](../benchmarks.md#metric-parity).

## Rust: module paths renamed in 1.0

For Rust users upgrading from 0.x, these module paths moved (crate-root re-exports
like `hotcoco::COCOeval` and `hotcoco::Hierarchy` are unchanged, so most code
needs no edits):

| Pre-1.0 module path | 1.0 |
|---|---|
| `hotcoco::eval` | `hotcoco::detection` |
| `hotcoco::hierarchy` | `hotcoco::detection::hierarchy` |
| `hotcoco::healthcheck` | `hotcoco::quality::healthcheck` |
| `hotcoco::types::{SummaryStats, CategoryStats, DatasetStats}` | `hotcoco::quality` |
| `hotcoco::primitives::counts` | `hotcoco::metrics::counts` |

There are no compatibility aliases for the old module paths. The Python API is
unaffected.
