# COCO

Load and query COCO-format datasets.

=== "Python"

    ```python
    from hotcoco import COCO

    coco = COCO("instances_val2017.json")
    ```

=== "Rust"

    ```rust
    use hotcoco::COCO;
    use std::path::Path;

    let coco = COCO::new(Path::new("instances_val2017.json"))?;
    ```

---

## Constructor

=== "Python"

    ```python
    COCO(annotation_file: str | dict | None = None, *, image_dir: str | None = None)
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `annotation_file` | <code>str &#124; dict &#124; None</code> | `None` | Path to a COCO JSON file, an in-memory dataset dict, or `None` for an empty instance. |
    | `image_dir` | <code>str &#124; None</code> | `None` | Root directory for image files. Used by `browse()` and `coco explore`. Can also be set afterwards via `coco.image_dir = "..."`. |

=== "Rust"

    ```rust
    COCO::new(annotation_file: &Path) -> Result<Self, Box<dyn Error>>
    COCO::from_dataset(dataset: Dataset) -> Self
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `annotation_file` | `&Path` | Path to a COCO JSON annotation file |
    | `dataset` | `Dataset` | A pre-built `Dataset` struct (for `from_dataset`) |

---

## Properties

### `image_dir`

Root directory for image files, used by `browse()` and `coco explore`.
Set at construction time or assign directly. Propagated automatically
through `filter`, `split`, `sample`, and `load_res`.

```python
# At construction
coco = COCO("instances_val2017.json", image_dir="/data/coco/images")

# After construction
coco.image_dir = "/data/coco/images"
print(coco.image_dir)  # "/data/coco/images"
```

---

### `dataset`

The full dataset with `images`, `annotations`, and `categories`. Writable in
Python: assigning a dataset dict replaces the contents and rebuilds the index.
Reading it returns a copy, so edit the dict and assign it back — see
[Getters return copies — assign back to apply](../getting-started/migration.md#getters-return-copies-assign-back-to-apply).
To edit annotations that are already there, [`set_ann_field`](#set_ann_field) and
[`update_anns`](#update_anns) do it without a full rebuild.

Keys outside the COCO schema (custom metadata on images, annotations, or
categories) are preserved through load, dataset ops, and `save` — see
[The COCO format](../getting-started/coco-format.md#unknown-keys-are-preserved).

=== "Python"

    ```python
    coco = COCO("instances_val2017.json")
    print(len(coco.dataset["images"]))       # 5000
    print(len(coco.dataset["annotations"]))  # 36781
    ```

=== "Rust"

    ```rust
    let coco = COCO::new(Path::new("instances_val2017.json"))?;
    println!("{}", coco.dataset.images.len());       // 5000
    println!("{}", coco.dataset.annotations.len());  // 36781
    ```

---

### `load_warnings`

The warnings the loader printed to stderr, kept inspectable afterwards; empty
for a clean load. What the loader tolerates and flags is listed under
[Loading quirks worth knowing](../getting-started/coco-format.md#loading-quirks-worth-knowing).

```python
coco = COCO("annotations.json")
for w in coco.load_warnings:
    print("loader:", w)
```

---

## Methods

### `get_ann_ids`

Get annotation IDs matching the given filters. All filters are ANDed together.

=== "Python"

    ```python
    get_ann_ids(
        img_ids: int | list[int] = [],
        cat_ids: int | list[int] = [],
        area_rng: list[float] | None = None,
        iscrowd: bool | None = None,
    ) -> list[int]
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `img_ids` | <code>int &#124; list[int]</code> | `[]` | Filter by image IDs (empty = all) |
    | `cat_ids` | <code>int &#124; list[int]</code> | `[]` | Filter by category IDs (empty = all) |
    | `area_rng` | <code>list[float] &#124; None</code> | `None` | Filter by area range `[min, max]` |
    | `iscrowd` | <code>bool &#124; None</code> | `None` | Filter by crowd flag |

    ```python
    ann_ids = coco.get_ann_ids(img_ids=[42], cat_ids=[1])
    ann_ids = coco.get_ann_ids(42)   # a bare id works, as in pycocotools
    ```

=== "Rust"

    ```rust
    fn get_ann_ids(
        &self,
        img_ids: &[u64],
        cat_ids: &[u64],
        area_rng: Option<[f64; 2]>,
        is_crowd: Option<bool>,
    ) -> Vec<u64>
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `img_ids` | `&[u64]` | Filter by image IDs (empty = all) |
    | `cat_ids` | `&[u64]` | Filter by category IDs (empty = all) |
    | `area_rng` | `Option<[f64; 2]>` | Filter by area range `[min, max]` |
    | `is_crowd` | `Option<bool>` | Filter by crowd flag |

    ```rust
    let ann_ids = coco.get_ann_ids(&[42], &[1], None, None);
    ```

---

### `get_cat_ids`

Get category IDs matching the given filters.

=== "Python"

    ```python
    get_cat_ids(
        cat_nms: list[str] = [],
        sup_nms: list[str] = [],
        cat_ids: list[int] = [],
    ) -> list[int]
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `cat_nms` | `list[str]` | `[]` | Filter by category names |
    | `sup_nms` | `list[str]` | `[]` | Filter by supercategory names |
    | `cat_ids` | `list[int]` | `[]` | Filter by category IDs |

    ```python
    cat_ids = coco.get_cat_ids(cat_nms=["person", "dog"])
    ```

=== "Rust"

    ```rust
    fn get_cat_ids(&self, cat_nms: &[&str], sup_nms: &[&str], cat_ids: &[u64]) -> Vec<u64>
    ```

    ```rust
    let cat_ids = coco.get_cat_ids(&["person", "dog"], &[], &[]);
    ```

---

### `get_img_ids`

Get image IDs matching the given filters.

=== "Python"

    ```python
    get_img_ids(
        img_ids: list[int] = [],
        cat_ids: list[int] = [],
    ) -> list[int]
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `img_ids` | `list[int]` | `[]` | Filter by image IDs |
    | `cat_ids` | `list[int]` | `[]` | Filter by category IDs (images containing these categories) |

    ```python
    img_ids = coco.get_img_ids(cat_ids=[1])
    ```

=== "Rust"

    ```rust
    fn get_img_ids(&self, img_ids: &[u64], cat_ids: &[u64]) -> Vec<u64>
    ```

    ```rust
    let img_ids = coco.get_img_ids(&[], &[1]);
    ```

---

### `load_anns`

Load annotations by their IDs.

=== "Python"

    ```python
    load_anns(ids: list[int]) -> list[dict]
    ```

    Returns annotation dicts with keys like `id`, `image_id`, `category_id`, `bbox`, `area`, `segmentation`, `iscrowd`.

    ```python
    anns = coco.load_anns([101, 102, 103])
    print(anns[0]["bbox"])  # [x, y, width, height]
    ```

=== "Rust"

    ```rust
    fn load_anns(&self, ids: &[u64]) -> Vec<&Annotation>
    ```

    Returns references to `Annotation` structs.

    ```rust
    let anns = coco.load_anns(&[101, 102, 103]);
    println!("{:?}", anns[0].bbox);  // [x, y, width, height]
    ```

---

### `load_cats`

Load categories by their IDs.

=== "Python"

    ```python
    load_cats(ids: list[int]) -> list[dict]
    ```

    Returns category dicts with keys `id`, `name`, `supercategory`.

    ```python
    cats = coco.load_cats([1, 2, 3])
    print(cats[0]["name"])  # "person"
    ```

=== "Rust"

    ```rust
    fn load_cats(&self, ids: &[u64]) -> Vec<&Category>
    ```

    ```rust
    let cats = coco.load_cats(&[1, 2, 3]);
    println!("{}", cats[0].name);  // "person"
    ```

---

### `load_imgs`

Load images by their IDs.

=== "Python"

    ```python
    load_imgs(ids: list[int]) -> list[dict]
    ```

    Returns image dicts with keys like `id`, `file_name`, `width`, `height`.

    ```python
    imgs = coco.load_imgs([42])
    print(f"{imgs[0]['width']}x{imgs[0]['height']}")
    ```

=== "Rust"

    ```rust
    fn load_imgs(&self, ids: &[u64]) -> Vec<&Image>
    ```

    ```rust
    let imgs = coco.load_imgs(&[42]);
    println!("{}x{}", imgs[0].width, imgs[0].height);
    ```

---

### `load_res`

Load detection results into a new `COCO` object. Images and categories are
copied from the ground truth. Missing fields are computed from the detection
type:

| Detection type | Auto-computed fields |
|---------------|---------------------|
| bbox | `area` from bbox, polygon `segmentation` from bbox |
| segm | `area` from RLE mask |
| keypoints | `area` from keypoint extent bbox |
| obb | `area` from width × height, axis-aligned `bbox` enclosing the rotated box |

=== "Python"

    ```python
    load_res(res: str | list[dict] | np.ndarray) -> COCO
    ```

    Three input formats are accepted:

    **JSON file path:**
    ```python
    coco_dt = coco_gt.load_res("detections.json")
    ```

    **List of dicts (in-memory results):**
    ```python
    coco_dt = coco_gt.load_res([
        {"image_id": 42, "category_id": 1, "bbox": [10, 20, 100, 80], "score": 0.95},
    ])
    ```

    **NumPy array** — shape `(N, 7)` with columns `[image_id, x, y, w, h, score, category_id]`,
    or `(N, 6)` with `category_id` defaulting to `1`. Array must be `float64`.
    Matches pycocotools `loadNumpyAnnotations` convention:
    ```python
    arr = np.array([[42, 10, 20, 100, 80, 0.95, 1]], dtype=np.float64)
    coco_dt = coco_gt.load_res(arr)
    ```

=== "Rust"

    ```rust
    // From a file
    fn load_res(&self, res_file: &Path) -> Result<COCO, Box<dyn Error>>

    // From in-memory annotations
    fn load_res_anns(&self, anns: Vec<Annotation>) -> Result<COCO, Box<dyn Error>>
    ```

    ```rust
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;
    let coco_dt = coco_gt.load_res_anns(my_annotations)?;
    ```

!!! tip
    A result carrying both `segmentation` and `keypoints` is treated as a segmentation result, matching pycocotools precedence.

---

### `set_ann_field`

Set one field on the named annotations, keeping the indices current. Every other
field of each annotation is carried over, so a partial edit cannot drop the rest
of the record.

A field outside the COCO schema is a custom key. Setting one the annotations
already carry works like any other field; adding a new one needs `create=True`,
so a misspelled schema field — `"Area"`, `"iscrowed"` — raises instead of quietly
landing beside the field you meant to change.

This is what evaluating one dataset under several IoU types needs: each
annotation's active `area` follows the box for `bbox` and the mask for `segm`.

=== "Python"

    ```python
    set_ann_field(field: str, values: dict[int, Any], *, create: bool = False) -> None
    ```

    | Parameter | Type | Description |
    |---|---|---|
    | `field` | `str` | Annotation key to set, for example `"area"`. Cannot be `"id"`. |
    | `values` | `dict[int, Any]` | Annotation ID to new value. |
    | `create` | `bool` | Allow `field` to be a custom key the annotations do not have yet. Default `False`. |

    ```python
    mask_areas = {ann["id"]: mask.area(coco.ann_to_rle(ann)) for ann in coco.dataset["annotations"]}
    coco.set_ann_field("area", mask_areas)
    ```

=== "Rust"

    Python only. In Rust, edit `coco.dataset.annotations` and call
    `create_index()`, or use `update_anns` below.

Raises `KeyError` if an annotation ID is not in the dataset, or if `field` is
neither a COCO field nor a custom key already on the annotation while `create` is
`False`; `TypeError` if a value does not fit the field, as `{1: "big"}` does not
fit `"area"`; and `ValueError` for `field="id"`. Nothing is written in any of
those cases.

---

### `update_anns`

Replace whole annotations, matched by `id`, keeping the indices current. The
targeted counterpart to assigning [`dataset`](#dataset): it edits the annotations
you name instead of rebuilding everything. Ids do not move, so only a
replacement that changes an `image_id` or a `category_id` costs a re-index.

Each dict **replaces** its annotation rather than merging into it — keys you
leave out come back as their defaults. Use `set_ann_field` to change one field
and keep the rest.

=== "Python"

    ```python
    update_anns(anns: list[dict]) -> None
    ```

    | Parameter | Type | Description |
    |---|---|---|
    | `anns` | `list[dict]` | Annotation dicts, each with an `id` already in the dataset. |

    ```python
    anns = coco.dataset["annotations"]
    for ann in anns:
        ann["area"] = ann["bbox"][2] * ann["bbox"][3]
    coco.update_anns(anns)
    ```

=== "Rust"

    ```rust
    fn update_anns(&mut self, anns: Vec<Annotation>) -> Result<(), UnknownAnnIds>
    ```

Raises `KeyError` if a dict has no `id`, or names an `id` the dataset does not
have; `TypeError` if the argument is not a list or an element is not a dict; and
`ValueError` if a dict is missing a required field or holds a value that does not
fit it, the same errors assigning `dataset` raises. Nothing is written in any of
those cases. In a dataset with duplicate annotation IDs, the last occurrence is
the one replaced — the record the ID lookup holds.

!!! tip
    A `COCOeval` copies both datasets when it is constructed, so an evaluator
    built before the mutation keeps evaluating the old annotations. Mutate
    first, then construct the evaluator.

---

### `ann_to_rle`

Convert an annotation to RLE format.

=== "Python"

    ```python
    ann_to_rle(ann: dict) -> dict
    ```

    Returns an RLE dict with `"size"` (`[h, w]`) and `"counts"` (`bytes`) — the
    same format `mask.encode` and pycocotools produce.

    ```python
    ann = coco.load_anns([101])[0]
    rle = coco.ann_to_rle(ann)
    print(rle["size"])           # [height, width]
    print(type(rle["counts"]))   # <class 'bytes'>
    ```

=== "Rust"

    ```rust
    fn ann_to_rle(&self, ann: &Annotation) -> Option<Rle>
    ```

    Returns an `Rle` struct with `h`, `w`, and `counts` fields.

    ```rust
    let ann = &coco.load_anns(&[101])[0];
    if let Some(rle) = coco.ann_to_rle(ann) {
        println!("{}x{}", rle.h, rle.w);
    }
    ```

---

### `ann_to_mask`

Convert an annotation to a binary mask.

=== "Python"

    ```python
    ann_to_mask(ann: dict) -> numpy.ndarray
    ```

    Returns a binary mask of shape (h, w), dtype `uint8`.

    ```python
    ann = coco.load_anns([101])[0]
    mask = coco.ann_to_mask(ann)
    print(mask.shape)  # (height, width)
    ```

=== "Rust"

    ```rust
    fn ann_to_mask(&self, ann: &Annotation) -> Option<Vec<u8>>
    ```

    Returns a flat `Vec<u8>` in column-major order (h * w pixels).

    ```rust
    let ann = &coco.load_anns(&[101])[0];
    if let Some(mask) = coco.ann_to_mask(ann) {
        println!("pixels: {}", mask.len());
    }
    ```

---

### `stats`

Compute dataset health-check statistics: annotation counts, image dimensions,
annotation area distribution, and per-category breakdowns.

=== "Python"

    ```python
    stats() -> dict
    ```

    Returns a dict with the following structure:

    | Key | Type | Description |
    |-----|------|-------------|
    | `image_count` | `int` | Total number of images |
    | `annotation_count` | `int` | Total number of annotations |
    | `category_count` | `int` | Number of categories |
    | `crowd_count` | `int` | Number of crowd annotations (`iscrowd=1`) |
    | `per_category` | `list[dict]` | Per-category stats, sorted by `ann_count` descending |
    | `image_width` | `dict` | Width summary stats (`min`, `max`, `mean`, `median`) |
    | `image_height` | `dict` | Height summary stats |
    | `annotation_area` | `dict` | Area summary stats |

    Each `per_category` entry has keys `id`, `name`, `ann_count`, `img_count`, `crowd_count`.

    ```python
    s = coco.stats()
    print(s["image_count"])        # 5000
    print(s["annotation_count"])   # 36781

    for cat in s["per_category"][:5]:
        print(f"{cat['name']}: {cat['ann_count']} annotations")
    ```

=== "Rust"

    ```rust
    fn stats(&self) -> DatasetStats
    ```

    Returns a `DatasetStats` struct with fields mirroring the Python dict.

    ```rust
    let s = coco.stats();
    println!("{} images", s.image_count);
    println!("{} annotations", s.annotation_count);
    for cat in &s.per_category {
        println!("{}: {} anns", cat.name, cat.ann_count);
    }
    ```

---

### `healthcheck`

Validate a dataset before training or evaluation.

=== "Python"

    ```python
    healthcheck(dt: COCO | None = None) -> dict
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `dt` | <code>COCO &#124; None</code> | `None` | Detections. When given, also runs GT/DT compatibility checks. |

    **Returns** a dict with `errors` and `warnings` (each a `list[dict]` with `code`,
    `message`, and context fields) plus a `summary` dict of dataset counts and the
    category `imbalance_ratio`.

    ```python
    report = coco.healthcheck()
    for f in report["errors"]:
        print(f"[{f['code']}] {f['message']}")
    ```

=== "Rust"

    ```rust
    fn healthcheck(&self) -> HealthReport
    fn healthcheck_compatibility(&self, dt: &COCO) -> HealthReport
    ```

    The Python method dispatches to `healthcheck_compatibility` when `dt` is given.

Four layers run in order — structural, quality, distribution, and, with `dt`,
GT/DT compatibility. The [healthcheck guide](../guide/datasets.md#healthcheck)
lists what each layer catches.

---

### `browse`

Launch an interactive dataset browser. Requires `pip install hotcoco[browse]`.

```python
browse(
    image_dir: str | None = None,
    dt: COCO | str | None = None,
    iou_type: str = "bbox",
    iou_thr: float = 0.5,
    eval: COCOeval | None = None,
    slices: dict[str, list[int]] | str | None = None,
    batch_size: int = 12,
    port: int = 7860,
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_dir` | <code>str &#124; None</code> | `None` | Image directory. Overrides `self.image_dir` if given. |
| `dt` | <code>COCO &#124; str &#124; None</code> | `None` | Detection results to overlay. Pass a `COCO` object (from `load_res()`) or a path string (auto-loaded). |
| `iou_type` | `str` | `"bbox"` | Similarity used to match detections against ground truth in the browser. |
| `iou_thr` | `float` | `0.5` | IoU threshold for the TP/FP/FN coloring. |
| `eval` | <code>COCOeval &#124; None</code> | `None` | An evaluated `COCOeval`. Enables the eval dashboard tab — PR curves, confusion matrix, TIDE errors, calibration, per-image F1. |
| `slices` | <code>dict &#124; str &#124; None</code> | `None` | Named image subsets for the dashboard's slice breakdown, as a mapping or a path to a JSON file. |
| `batch_size` | `int` | `12` | Number of images loaded per batch. |
| `port` | `int` | `7860` | Local server port. |

```python
coco = COCO("instances_val2017.json", image_dir="/data/coco/images")
coco.browse(dt="bbox_results.json")
```

Passing `eval=` without `dt=` shows the dashboard but leaves the gallery without
detection overlays — pass both.

Raises `ValueError` if `image_dir` is `None` and `self.image_dir` is also `None`.
Raises `ImportError` if browse dependencies are not installed.

See the [Dataset browser guide](../guide/browse.md) for a full walkthrough.

---

## Dataset operations

The following methods reshape or subset a dataset, returning a new `COCO` object.
The original is never modified. See the [Dataset operations guide](../guide/datasets.md)
for worked examples.

---

### `filter`

Subset the dataset by category, by image, by annotation area, or by any combination of the three. All criteria are ANDed.

=== "Python"

    ```python
    filter(
        cat_ids: list[int] | None = None,
        img_ids: list[int] | None = None,
        area_rng: list[float] | None = None,
        drop_empty_images: bool = True,
    ) -> COCO
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `cat_ids` | <code>list[int] &#124; None</code> | `None` | Keep only these category IDs |
    | `img_ids` | <code>list[int] &#124; None</code> | `None` | Keep only these image IDs |
    | `area_rng` | <code>list[float] &#124; None</code> | `None` | Area range `[min, max]` (inclusive) |
    | `drop_empty_images` | `bool` | `True` | Remove images with no matching annotations |

    ```python
    person_id = coco.get_cat_ids(cat_nms=["person"])[0]
    people = coco.filter(cat_ids=[person_id])
    medium  = coco.filter(area_rng=[1024.0, 9216.0])
    ```

=== "Rust"

    ```rust
    fn filter(
        &self,
        cat_ids: Option<&[u64]>,
        img_ids: Option<&[u64]>,
        area_rng: Option<[f64; 2]>,
        drop_empty_images: bool,
    ) -> Dataset
    ```

    Returns a `Dataset`; wrap with `COCO::from_dataset()` to re-index.

    ```rust
    let people = COCO::from_dataset(coco.filter(Some(&[1]), None, None, true));
    ```

---

### `merge`

Merge a list of datasets into one. All datasets must share the same category
taxonomy. Image and annotation IDs are remapped to be globally unique.

Raises `ValueError` (Python) or returns `Err` (Rust) if taxonomies differ.

=== "Python"

    ```python
    COCO.merge(datasets: list[COCO]) -> COCO  # classmethod
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `datasets` | `list[COCO]` | Two or more `COCO` objects with identical category sets |

    ```python
    batch1 = COCO("batch1.json")
    batch2 = COCO("batch2.json")
    combined = COCO.merge([batch1, batch2])
    ```

=== "Rust"

    ```rust
    fn merge(datasets: &[&Dataset]) -> Result<Dataset, String>
    ```

    ```rust
    let combined = COCO::from_dataset(
        COCO::merge(&[&ds1, &ds2]).expect("incompatible taxonomies")
    );
    ```

---

### `split`

Split the dataset into train/val (or train/val/test) subsets. Images are shuffled
deterministically; annotations follow their images. All splits share the full
category list.

=== "Python"

    ```python
    split(
        val_frac: float = 0.2,
        test_frac: float | None = None,
        seed: int = 42,
    ) -> tuple[COCO, COCO] | tuple[COCO, COCO, COCO]
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `val_frac` | `float` | `0.2` | Fraction of images for validation |
    | `test_frac` | <code>float &#124; None</code> | `None` | Fraction for a test set; omit (`None`) for a two-way split. `0.0` returns a three-way split with an empty test set. |
    | `seed` | `int` | `42` | Random seed for reproducibility |

    ```python
    train, val = coco.split(val_frac=0.2)
    train, val, test = coco.split(val_frac=0.15, test_frac=0.15)
    ```

=== "Rust"

    ```rust
    fn split(
        &self,
        val_frac: f64,
        test_frac: Option<f64>,
        seed: u64,
    ) -> (Dataset, Dataset, Option<Dataset>)
    ```

    ```rust
    let (train, val, _) = coco.split(0.2, None, 42);
    let train = COCO::from_dataset(train);
    ```

---

### `sample`

Draw a random subset of images with their annotations. The sample is deterministic
for the same seed.

=== "Python"

    ```python
    sample(
        n: int | None = None,
        frac: float | None = None,
        seed: int = 42,
    ) -> COCO
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `n` | <code>int &#124; None</code> | `None` | Exact number of images to sample |
    | `frac` | <code>float &#124; None</code> | `None` | Fraction of images to sample |
    | `seed` | `int` | `42` | Random seed for reproducibility |

    Provide either `n` or `frac`, not both.

    ```python
    subset = coco.sample(n=500, seed=0)
    subset = coco.sample(frac=0.1, seed=0)
    ```

=== "Rust"

    ```rust
    fn sample(&self, n: Option<usize>, frac: Option<f64>, seed: u64) -> Dataset
    ```

    ```rust
    let subset = COCO::from_dataset(coco.sample(Some(500), None, 0));
    ```

---

### `save`

Serialize the dataset to a COCO-format JSON file.

=== "Python"

    ```python
    save(path: str) -> None
    ```

    ```python
    coco.filter(cat_ids=[1]).sample(n=500, seed=0).save("person_sample.json")
    ```

=== "Rust"

    `save` is a Python-only convenience method. In Rust, serialize with `serde_json`:

    ```rust
    use std::fs::File;
    use std::io::BufWriter;

    let file = BufWriter::new(File::create("output.json")?);
    serde_json::to_writer_pretty(file, &coco.dataset)?;
    ```

---

## Format conversion {#convert}

All ten converters share one contract:

- **Malformed input is an error, not a skip.** Parse failures raise `ValueError`
  naming the file and line/position; filesystem problems raise `IOError`. Records
  the target format cannot express (an annotation with no bbox in a
  bbox-only format, say) are skipped and **counted** in the returned stats dict
  under a `skipped_<reason>` key — nothing vanishes uncounted.
- **Missing image dimensions are an error wherever geometry must scale.**
  `to_yolo`, `from_yolo`, and `to_oid` need real `width`/`height` (YOLO and Open
  Images store normalized coordinates) and raise `ValueError` without them. Two
  documented exceptions: `from_oid` without `images_dir` keeps boxes normalized
  against a 1×1 image, and DOTA works in absolute pixels so dimensions are
  metadata only.
- **`file_name` is never invented.** Formats that record it (CVAT, VOC) round-trip
  it verbatim; formats keyed by file stem (YOLO, DOTA, Open Images) import the bare
  stem with no fabricated extension.
- **Exports fail loudly on ambiguity.** Two images whose file stems collide
  (`train/img.jpg` and `val/img.jpg` both writing `img.txt`) raise instead of
  silently overwriting, as does an annotation referencing a `category_id` missing
  from `categories`.

### `to_yolo`

Export the dataset to YOLO label format.

=== "Python"

    ```python
    to_yolo(output_dir: str) -> dict
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `output_dir` | `str` | Directory to write label files and `data.yaml`. Created if it doesn't exist. |

    Writes one `<stem>.txt` per image (normalized `class_idx cx cy w h` lines) and a
    `data.yaml` with `nc` and `names` (names containing commas are quoted). Returns
    a stats dict:

    | Key | Type | Description |
    |-----|------|-------------|
    | `images` | `int` | Number of images processed |
    | `annotations` | `int` | Number of label lines written |
    | `skipped_crowd` | `int` | Crowd annotations skipped |
    | `skipped_no_bbox` | `int` | Annotations without a bbox skipped |

    ```python
    coco = COCO("instances_val2017.json")
    stats = coco.to_yolo("labels/val2017/")
    print(stats)
    # {'images': 5000, 'annotations': 36781, 'skipped_crowd': 12, 'skipped_no_bbox': 0}
    ```

    Raises `ValueError` if any image lacks `width`/`height` — see the
    [converter contract](#convert).

=== "Rust"

    ```rust
    use hotcoco::convert::{coco_to_yolo, YoloStats};
    use std::path::Path;

    let stats: YoloStats = coco_to_yolo(&coco.dataset, Path::new("labels/"))?;
    println!("{} annotations written", stats.annotations);
    ```

---

### `from_yolo`

Load a YOLO label directory as a COCO dataset. Class method.

=== "Python"

    ```python
    COCO.from_yolo(
        yolo_dir: str,
        images_dir: str | None = None,
    ) -> COCO
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `yolo_dir` | `str` | *required* | Directory containing `.txt` label files and `data.yaml` |
    | `images_dir` | <code>str &#124; None</code> | `None` | Source image directory; used by Pillow to read `width`/`height`. Requires `pip install Pillow`. |

    ```python
    coco = COCO.from_yolo("labels/val2017/", images_dir="images/val2017/")
    coco.save("reconstructed.json")
    ```

    `data.yaml` `names` is accepted in all three common forms — the flow list
    (`names: [a, b]`), the block list, and the Ultralytics index-keyed dict
    (`names:\n  0: person`).

    Raises `ValueError` for an image whose dimensions cannot be determined — see
    the [converter contract](#convert). Raises `ImportError` if `images_dir` is
    given but Pillow is not installed.

=== "Rust"

    ```rust
    use hotcoco::convert::yolo_to_coco;
    use std::collections::HashMap;
    use std::path::Path;

    let dims: HashMap<String, (u32, u32)> = HashMap::new(); // or populate from image headers
    let dataset = yolo_to_coco(Path::new("labels/"), &dims)?;
    let coco = hotcoco::COCO::from_dataset(dataset);
    ```

!!! tip
    See the [Format Conversion guide](../guide/datasets.md#convert) for a full
    worked example including a round-trip and CLI usage.

### `to_voc`

Export the dataset to Pascal VOC annotation format.

=== "Python"

    ```python
    to_voc(output_dir: str) -> dict
    ```

    Writes one XML file per image into `output_dir/Annotations/`, plus `labels.txt`.
    Coordinates use VOC's 1-based inclusive convention (`xmin = x + 1`,
    `xmax = x + w`, rounded to integers); COCO `iscrowd` exports as
    `<difficult>1</difficult>`. Returns a stats dict with keys: `images`,
    `annotations`, `crowd_as_difficult`, `skipped_no_bbox`.

    ```python
    coco = COCO("instances_val2017.json")
    stats = coco.to_voc("voc_output/")
    ```

=== "Rust"

    ```rust
    use hotcoco::convert::{coco_to_voc, VocStats};
    use std::path::Path;

    let stats: VocStats = coco_to_voc(&coco.dataset, Path::new("voc_output/"))?;
    ```

### `from_voc`

Load a Pascal VOC annotation directory as a COCO dataset.

=== "Python"

    ```python
    COCO.from_voc(voc_dir: str) -> COCO
    ```

    Scans `voc_dir/Annotations/` for `.xml` files (falls back to `voc_dir/` directly).
    Image dimensions come from each XML's `<size>` element. Coordinates can be
    integers or floats and are converted from VOC's 1-based inclusive convention
    (`x = xmin − 1`, `w = xmax − xmin + 1` — the exact inverse of `to_voc`);
    `<difficult>1</difficult>` imports as `iscrowd`. `<truncated>` has no COCO
    counterpart and is dropped.

    ```python
    coco = COCO.from_voc("VOCdevkit/VOC2012/")
    coco.save("voc2012_as_coco.json")
    ```

=== "Rust"

    ```rust
    use hotcoco::convert::voc_to_coco;
    use std::path::Path;

    let dataset = voc_to_coco(Path::new("VOCdevkit/VOC2012/"))?;
    let coco = hotcoco::COCO::from_dataset(dataset);
    ```

### `to_cvat`

Export the dataset to CVAT for Images 1.1 XML format.

=== "Python"

    ```python
    to_cvat(output_path: str) -> dict
    ```

    Writes a single XML file. Bboxes become `<box>`, polygons become `<polygon>`.
    Returns a stats dict with keys: `images`, `boxes`, `polygons`,
    `skipped_no_geometry`, `skipped_degenerate` (polygons with fewer than three
    points).

    ```python
    coco = COCO("instances_val2017.json")
    stats = coco.to_cvat("annotations.xml")
    ```

=== "Rust"

    ```rust
    use hotcoco::convert::{coco_to_cvat, CvatStats};
    use std::path::Path;

    let stats: CvatStats = coco_to_cvat(&coco.dataset, Path::new("annotations.xml"))?;
    ```

### `from_cvat`

Load a CVAT for Images 1.1 XML file as a COCO dataset.

=== "Python"

    ```python
    COCO.from_cvat(cvat_path: str) -> COCO
    ```

    Reads a single XML file. Supports `<box>` and `<polygon>` elements, in both
    the self-closing form and the open/close-pair form CVAT writes when a shape
    carries `<attribute>` children. Unsupported shapes (`<polyline>`, `<points>`,
    `<cuboid>`) and degenerate polygons are skipped and reported with a
    `UserWarning` naming the count — they don't stop the conversion.

    ```python
    coco = COCO.from_cvat("annotations.xml")
    coco.save("cvat_as_coco.json")
    ```

=== "Rust"

    ```rust
    use hotcoco::convert::cvat_to_coco;
    use std::path::Path;

    let dataset = cvat_to_coco(Path::new("annotations.xml"))?;
    let coco = hotcoco::COCO::from_dataset(dataset);
    ```

### `to_dota`

Export oriented bounding boxes to DOTA label format.

=== "Python"

    ```python
    to_dota(output_dir: str) -> dict
    ```

    Writes one `.txt` per image: 8 corner coordinates, category name, difficulty
    flag (COCO `iscrowd` exports as difficulty `1`, and imports back as
    `iscrowd`). Only annotations carrying an `obb` are written. Returns a stats
    dict with keys: `images`, `annotations`, `skipped_no_obb`.

    ```python
    stats = coco.to_dota("labelTxt/")
    ```

=== "Rust"

    ```rust
    use hotcoco::convert::{coco_to_dota, DotaStats};
    use std::path::Path;

    let stats: DotaStats = coco_to_dota(&coco.dataset, Path::new("labelTxt/"))?;
    ```

### `from_dota`

Load a DOTA label directory as a COCO dataset with oriented boxes.

=== "Python"

    ```python
    COCO.from_dota(
        label_dir: str,
        images_dir: str | None = None,
        categories: list[str] | None = None,
    ) -> COCO
    ```

    Each annotation gets both an `obb` and its axis-aligned `bbox` envelope.
    `images_dir` only fills `width`/`height` on the image records (they stay `0`
    without it). Without `categories`, category names are discovered from the
    label files and sorted.

    ```python
    coco = COCO.from_dota("labelTxt/", images_dir="images/")
    ```

=== "Rust"

    ```rust
    use hotcoco::convert::dota_to_coco;
    use std::collections::HashMap;
    use std::path::Path;

    let dims: HashMap<String, (u32, u32)> = HashMap::new();
    let dataset = dota_to_coco(Path::new("labelTxt/"), None, &dims)?;
    ```

### `to_oid`

Export the dataset to Open Images challenge CSV format.

=== "Python"

    ```python
    to_oid(output_csv: str) -> dict
    ```

    Writes `ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf` — Open
    Images puts `XMax` before `YMin` — with coordinates normalized to `[0, 1]`.
    A `Score` column is added when any annotation carries a score, so detection
    files round-trip too. Returns a stats dict with keys: `images`,
    `annotations`, `group_of`, `skipped_no_bbox`.

    ```python
    stats = coco.to_oid("boxes.csv")
    ```

=== "Rust"

    ```rust
    use hotcoco::convert::{coco_to_oid, OidStats};
    use std::path::Path;

    let stats: OidStats = coco_to_oid(&coco.dataset, Path::new("boxes.csv"))?;
    ```

### `from_oid`

Load an Open Images annotation CSV as a COCO dataset.

=== "Python"

    ```python
    COCO.from_oid(
        csv_path: str,
        class_descriptions: str | None = None,
        images_dir: str | None = None,
    ) -> COCO
    ```

    Reads the full V6 layout and the challenge subset alike — columns are
    resolved by name, not position. `IsGroupOf` becomes the `is_group_of`
    annotation field. `class_descriptions` resolves `LabelName` MIDs such as
    `/m/0cmf2` to names such as `Beer`; without it, category names stay as MIDs.

    Without `images_dir`, boxes stay in `[0, 1]` against a 1×1 image; see the
    [conversion guide](../guide/datasets.md#open-images) for what that does and
    does not affect.

    ```python
    gt = COCO.from_oid(
        "challenge-2019-validation-detection-bbox.csv",
        class_descriptions="class-descriptions-boxable.csv",
    )
    ```

=== "Rust"

    ```rust
    use hotcoco::convert::oid_to_coco;
    use std::collections::HashMap;
    use std::path::Path;

    let dims: HashMap<String, (u32, u32)> = HashMap::new();
    let dataset = oid_to_coco(Path::new("boxes.csv"), None, &dims)?;
    ```

### `load_res_oid`

Load Open Images detections as a result `COCO`, aligned to this dataset.

=== "Python"

    ```python
    load_res_oid(csv_path: str, class_descriptions: str | None = None) -> COCO
    ```

    The Open Images counterpart to [`load_res`](#load_res). `ImageID` is matched
    against image file-name stems and `LabelName` against category names, so pass
    the same `class_descriptions` used for the ground truth. A detection naming an
    unknown image or category raises rather than being skipped.

    ```python
    dt = gt.load_res_oid("predictions.csv")
    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ```

=== "Rust"

    ```rust
    use hotcoco::convert::oid_results_to_anns;
    use std::path::Path;

    let anns = oid_results_to_anns(&gt.dataset, Path::new("predictions.csv"), None)?;
    let dt = gt.load_res_anns(anns)?;
    ```
