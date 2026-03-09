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
    COCO(annotation_file: str | dict | None = None)
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `annotation_file` | <code>str &#124; dict &#124; None</code> | `None` | Path to a COCO JSON file, an in-memory dataset dict, or `None` for an empty instance. |

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

### `dataset`

The full dataset with `images`, `annotations`, and `categories`.

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

## Methods

### `get_ann_ids`

Get annotation IDs matching the given filters. All filters are ANDed together.

=== "Python"

    ```python
    get_ann_ids(
        img_ids: list[int] = [],
        cat_ids: list[int] = [],
        area_rng: list[float] | None = None,
        iscrowd: bool | None = None,
    ) -> list[int]
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `img_ids` | `list[int]` | `[]` | Filter by image IDs (empty = all) |
    | `cat_ids` | `list[int]` | `[]` | Filter by category IDs (empty = all) |
    | `area_rng` | <code>list[float] &#124; None</code> | `None` | Filter by area range `[min, max]` |
    | `iscrowd` | <code>bool &#124; None</code> | `None` | Filter by crowd flag |

    ```python
    ann_ids = coco.get_ann_ids(img_ids=[42], cat_ids=[1])
    ```

    !!! note "camelCase alias"
        Also available as `getAnnIds()`.

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

    !!! note "camelCase alias"
        Also available as `getCatIds()`.

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

    !!! note "camelCase alias"
        Also available as `getImgIds()`.

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

    !!! note "camelCase alias"
        Also available as `loadAnns()`.

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

    !!! note "camelCase alias"
        Also available as `loadCats()`.

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

    !!! note "camelCase alias"
        Also available as `loadImgs()`.

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

Load detection results into a new `COCO` object. Images and categories are copied from the ground truth. Missing fields (`area`, `segmentation`) are computed automatically.

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

    !!! note "camelCase alias"
        Also available as `loadRes()`.

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
    `load_res` automatically computes missing fields: `area` from bounding boxes or segmentation masks, and polygon segmentations from bbox results. This matches pycocotools behavior.

---

### `ann_to_rle`

Convert an annotation to RLE format.

=== "Python"

    ```python
    ann_to_rle(ann: dict) -> dict
    ```

    Returns an RLE dict with `"counts"` (str) and `"size"` ([h, w]).

    ```python
    ann = coco.load_anns([101])[0]
    rle = coco.ann_to_rle(ann)
    print(rle.keys())  # dict_keys(['counts', 'size'])
    ```

    !!! note "camelCase alias"
        Also available as `annToRLE()`.

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

    !!! note "camelCase alias"
        Also available as `annToMask()`.

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

## Dataset Operations

The following methods reshape or subset a dataset, returning a new `COCO` object.
The original is never modified. See the [Dataset Operations guide](../guide/datasets.md)
for worked examples.

---

### `filter`

Subset the dataset by category, image, and/or annotation area. All criteria are ANDed.

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
    | `test_frac` | <code>float &#124; None</code> | `None` | Fraction for a test set; omit for a two-way split |
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

## Format Conversion

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
    `data.yaml` with `nc` and `names`. Returns a stats dict:

    | Key | Type | Description |
    |-----|------|-------------|
    | `images` | `int` | Number of images processed |
    | `annotations` | `int` | Number of label lines written |
    | `skipped_crowd` | `int` | Crowd annotations skipped |
    | `missing_bbox` | `int` | Annotations without a bbox skipped |

    ```python
    coco = COCO("instances_val2017.json")
    stats = coco.to_yolo("labels/val2017/")
    print(stats)
    # {'images': 5000, 'annotations': 36781, 'skipped_crowd': 12, 'missing_bbox': 0}
    ```

    Raises `RuntimeError` if any image has `width == 0` or `height == 0`.

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
    # Without image dims (width/height stored as 0)
    coco = COCO.from_yolo("labels/val2017/")

    # With real image dimensions
    coco = COCO.from_yolo("labels/val2017/", images_dir="images/val2017/")
    coco.save("reconstructed.json")
    ```

    Raises `ImportError` if `images_dir` is given but Pillow is not installed.

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
