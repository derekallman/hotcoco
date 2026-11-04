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
| `getAnnIds()` | `get_ann_ids()` | Both work; only `getAnnIds` takes `iscrowd=0`/`1` and `areaRng=[]` |
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

## Argument types

The pycocotools-named functions accept the argument types pycocotools accepts:

- `COCO()` takes a `str` or any `os.PathLike`, such as a `pathlib.Path` — see the [constructor](../api/coco.md#constructor).
- Id-list arguments of the query and load methods take an `int` or any iterable of ints, such as a `set` or `dict.keys()` — see [Methods](../api/coco.md#methods).
- `getAnnIds` takes `iscrowd=0`/`1` and `areaRng=[]` — see [`get_ann_ids`](../api/coco.md#get_ann_ids).
- `mask.merge` takes `intersect=1` — see [`merge`](../api/mask.md#merge).
- `mask.iou` takes boxes as well as RLEs, so a `COCOeval` subclass whose `computeIoU` calls `maskUtils.iou` on boxes works under `init_as_pycocotools()` — see [`iou`](../api/mask.md#iou).

`get_ann_ids` and [`hotcoco.primitives`](../api/primitives.md) keep stricter, typed signatures.

## The dataset is live, params containers are copies

`coco.dataset` is the dataset dict itself: the same object on every access, and
the one you assigned if you built the `COCO` by assignment (`COCO(d)` reads `d`
and does not keep it). pycocotools' in-memory idioms work as they do there:

```python
coco = COCO()
coco.dataset = d                    # d is the dataset now; coco.dataset is d
coco.dataset["annotations"].append(new_ann)
coco.createIndex()                  # re-indexes from the dict, as in pycocotools

for ann in dt.dataset["annotations"]:
    ann["area"] = ann["area_segm"]  # torchmetrics' bbox+segm switch
COCOeval(gt, dt, "segm")            # sees the edit: the dict is read back first
```

Every method that needs the data reads the dict back before it runs, so
`createIndex()` is needed only after an edit made through a reference kept from
before the last call. The read-back is one parse of the dict, tens of
milliseconds for 40k annotations, and a bad edit raises there what assigning the
dict would. For a field edit by id, [`update_anns`](../api/coco.md#update_anns)
skips the parse. An evaluator built before an edit keeps the data it was built
with: hotcoco resolves the datasets when `COCOeval` is constructed, pycocotools
inside `evaluate()`.

`ev.params` is different. Attribute assignment (`ev.params.cat_ids = [...]`)
works, because the setter routes the change back to Rust, but mutating a
*container inside* it is a no-op on a copy:

```python
ev.params.cat_ids = [1, 2, 3]       # applied — attribute assignment works
ev.params.maxDets.append(200)       # no-op — mutates a temporary list

md = ev.params.maxDets              # instead: pull, edit, assign back
md.append(200)
ev.params.maxDets = md
```

`ev.coco_gt` and `ev.coco_dt` return a new object that shares the evaluator's
data without copying it; an edit to that object, through `dataset` or
`update_anns`, lands on a private copy and never reaches the evaluator.

## Known differences

| Behavior | pycocotools | hotcoco |
|----------|-------------|-----------|
| Print on load | Prints "loading annotations..." to stdout | No progress output; loader warnings go to stderr and `coco.load_warnings` |
| `COCO()` with no args | Creates empty instance with print statements | Creates empty instance silently |
| Annotation IDs | Requires unique positive integers | Also accepts 0-based IDs |
| `getAnnIds(areaRng=...)` on annotations missing `area` | Raises `KeyError` | Excludes them from the query |
| Evaluating annotations missing `area` | Raises `KeyError` | Derives it in the evaluator's copy: the mask's pixel count, then the box's `w × h`, for ground truth; the box first for detections, as `load_res` does. The dataset keeps its missing areas |
| Mutating `ev.params` containers in place | Mutates shared state | No-op on a copy — [assign back to apply](#the-dataset-is-live-params-containers-are-copies) |
| Evaluating a detection with a NaN score | Evaluates, with an undefined ranking | `load_res()` and `evaluate()` raise `ValueError` naming the detection |
| `summarize()` or `run()` with non-default params | Prints the table only | Also emits a [`ReferenceDeviationWarning`](../guide/results.md#filtering-the-warnings) per deviation |
| Summary stats when `maxDets` is not `[1, 10, 100]` | Reads AP (`stats[0]`) at 100, or reports `-1` when 100 is not a cap; reads the other AP lines and the last four AR lines at the third cap | Reads every AP line and the last four AR lines at the largest cap, as faster-coco-eval does. With three caps only `stats[0]` differs: `maxDets=[1, 10, 500]` reports AP at 500 where pycocotools reports `-1` |
| A key or string holding a lone surrogate, such as `"\ud800"` | Kept; `json.dump` writes it back escaped | Raises `UnicodeEncodeError` or `ValueError` — text is stored as UTF-8, which has no form for a lone surrogate |
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
