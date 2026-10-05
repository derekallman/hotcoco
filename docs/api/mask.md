# mask

Low-level mask operations on Run-Length Encoded (RLE) binary masks.

=== "Python"

    ```python
    from hotcoco import mask
    ```

=== "Rust"

    ```rust
    use hotcoco::mask;
    ```

For background on RLE and usage patterns, see the [Mask operations](../guide/masks.md) guide.

Two conventions hold for every function here, both matching pycocotools:

- **Arrays are Fortran-order** (column-major). Mask arrays are returned
  Fortran-order; C-order input is accepted and transposed internally.
- **Return types are pycocotools' types** — `counts` is `bytes`, batch areas are
  `uint32`, boxes are `float64`.

The functions that have a camelCase name in `pycocotools.mask` are available
under both spellings; everything else has one. camelCase aliases: see the
[alias table](../getting-started/migration.md#method-naming).

---

## Functions

### `encode`

Encode a binary mask to RLE.

=== "Python"

    ```python
    encode(mask: numpy.ndarray) -> dict | list[dict]
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `mask` | `numpy.ndarray` | 2-D `(H, W)` or 3-D `(H, W, N)`, dtype `uint8`, `bool`, or `int8`. Any memory layout — C-order, Fortran-order, or a sliced view. |

    **Returns:**

    - 2-D input → `dict` with `"size"` (`[H, W]`) and `"counts"` (`bytes`)
    - 3-D input → `list[dict]` of *N* RLE dicts

    ```python
    rle = mask.encode(m)   # m: (H, W) uint8 or bool array
    # {"size": [100, 100], "counts": b"..."}
    ```

    `bool` masks are accepted as well as `uint8`, which pycocotools does not do:
    torch-side code stores masks as `bool`, so requiring a cast would break the
    drop-in path for no gain. Wider dtypes raise a `TypeError` naming the dtype and the cast to apply.

=== "Rust"

    ```rust
    fn encode(mask: &[u8], h: u32, w: u32) -> Rle
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `mask` | `&[u8]` | Binary mask in column-major order (h * w pixels) |
    | `h` | `u32` | Height |
    | `w` | `u32` | Width |

    **Returns:** `Rle`

    ```rust
    let rle = mask::encode(&pixels, 100, 100);
    ```

---

### `decode`

Decode an RLE to a binary mask.

=== "Python"

    ```python
    decode(rle: dict | list[dict]) -> numpy.ndarray
    ```

    | Input | Returns |
    |-------|---------|
    | Single dict | `(H, W)` uint8 array |
    | List of *N* dicts | `(H, W, N)` uint8 array |

    ```python
    m = mask.decode(rle)          # (H, W)
    m3 = mask.decode([r1, r2])    # (H, W, 2)
    ```

=== "Rust"

    ```rust
    fn decode(rle: &Rle) -> Vec<u8>
    ```

    **Returns:** `Vec<u8>` — Flat binary mask in column-major order.

    ```rust
    let pixels = mask::decode(&rle);
    ```

---

### `area`

Compute the area (number of foreground pixels) of RLE mask(s).

=== "Python"

    ```python
    area(rle: dict | list[dict]) -> int | numpy.ndarray
    ```

    | Input | Returns |
    |-------|---------|
    | Single dict | `int` |
    | List of dicts | `numpy.ndarray` of uint32 |

    ```python
    a = mask.area(rle)        # scalar
    areas = mask.area(rles)   # array
    ```

=== "Rust"

    ```rust
    fn area(rle: &Rle) -> u64
    ```

    ```rust
    let a = mask::area(&rle);
    ```

---

### `to_bbox`

Convert RLE mask(s) to bounding box(es).

=== "Python"

    ```python
    to_bbox(rle: dict | list[dict]) -> numpy.ndarray
    ```

    | Input | Returns |
    |-------|---------|
    | Single dict | `numpy.ndarray` of shape `(4,)`, float64 |
    | List of *N* dicts | `numpy.ndarray` of shape `(N, 4)`, float64 |

    Values are `[x, y, width, height]`.

    ```python
    bbox = mask.to_bbox(rle)      # shape (4,)
    bboxes = mask.to_bbox(rles)   # shape (N, 4)
    ```

=== "Rust"

    ```rust
    fn to_bbox(rle: &Rle) -> [f64; 4]
    ```

    **Returns:** `[x, y, width, height]`

    ```rust
    let bbox = mask::to_bbox(&rle);
    ```

---

### `merge`

Merge multiple RLE masks. Union by default, intersection if `intersect=True`.

=== "Python"

    ```python
    merge(rles: list[dict], intersect: int | bool = False) -> dict
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `rles` | `list[dict]` | | List of RLE dicts to merge |
    | `intersect` | `int \| bool` | `False` | If true (`True` or `1`), compute intersection instead of union |

    ```python
    merged = mask.merge([rle1, rle2])
    intersected = mask.merge([rle1, rle2], intersect=True)
    ```

=== "Rust"

    ```rust
    fn merge(rles: &[Rle], intersect: bool) -> Rle
    ```

    ```rust
    let merged = mask::merge(&[rle1, rle2], false);
    let intersected = mask::merge(&[rle1, rle2], true);
    ```

---

### `iou`

Compute pairwise IoU between two lists of RLE masks, or two lists of boxes.

=== "Python"

    ```python
    iou(
        dt: list[dict] | numpy.ndarray | list[list[float]],
        gt: list[dict] | numpy.ndarray | list[list[float]],
        iscrowd: list[bool] | list[int] | numpy.ndarray,
    ) -> numpy.ndarray
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `dt` | `list[dict]` or boxes | Detection RLE dicts, or `[x, y, width, height]` boxes |
    | `gt` | `list[dict]` or boxes | Ground truth RLE dicts, or boxes |
    | `iscrowd` | `list[bool] \| list[int]` | Per-GT crowd flag; `0`/`1` and numpy arrays work |

    **Returns:** `numpy.ndarray` of shape `(len(dt), len(gt))`, dtype float64.

    ```python
    ious = mask.iou(dt_rles, gt_rles, [False] * len(gt_rles))
    box_ious = mask.iou(dt_boxes, gt_boxes, [0] * len(gt_boxes))
    ```

    `dt` and `gt` are sorted the way `pycocotools.mask.iou` sorts them:

    - A numpy array is boxes, shape `(N, 4)`, of any numeric dtype.
    - A list of dicts is RLEs. A single RLE dict is also accepted.
    - A list of 4-element rows is boxes.
    - An empty list takes the other side's kind.

    `dt` and `gt` must be the same kind; one of each raises `TypeError`. Boxes
    go to [`bbox_iou`](#bbox_iou) and RLEs to
    [`primitives.mask_iou`](primitives.md#mask_iou) — this function only adds
    the dispatch.

=== "Rust"

    ```rust
    fn iou(dt: &[Rle], gt: &[Rle], iscrowd: &[bool]) -> Vec<Vec<f64>>
    ```

    **Returns:** `Vec<Vec<f64>>` of shape D x G.

    ```rust
    let ious = mask::iou(&dt_rles, &gt_rles, &vec![false; gt_rles.len()]);
    ```

`iscrowd` selects the crowd convention per GT — defined under
[`primitives.bbox_iou`](primitives.md#bbox_iou). Two masks of different sizes
score `-1`, as in pycocotools.

---

### `bbox_iou`

Compute pairwise IoU between two lists of bounding boxes. This is the same
object as [`primitives.bbox_iou`](primitives.md#bbox_iou), which owns its
reference; it has no pycocotools counterpart.

=== "Python"

    ```python
    bbox_iou(dt: numpy.ndarray | list[list[float]], gt: numpy.ndarray | list[list[float]], iscrowd: list[bool]) -> numpy.ndarray
    ```

    Bounding boxes are `[x, y, width, height]`: an `(N, 4)` array or a list of
    4-element rows.

    **Returns:** `numpy.ndarray` of shape `(len(dt), len(gt))`, dtype float64.

    ```python
    ious = mask.bbox_iou(dt_boxes, gt_boxes, [False] * len(gt_boxes))
    ```

=== "Rust"

    ```rust
    fn bbox_iou(dt: &[[f64; 4]], gt: &[[f64; 4]], iscrowd: &[bool]) -> Vec<Vec<f64>>
    ```

    ```rust
    let ious = mask::bbox_iou(&dt_boxes, &gt_boxes, &vec![false; gt_boxes.len()]);
    ```

---

### `frPyObjects`

Encode segmentation objects to RLEs. This is pycocotools' universal entry point for converting any segmentation format to compressed RLE.

=== "Python"

    ```python
    frPyObjects(seg, h: int, w: int) -> dict | list[dict]
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `seg` | `list[list[float]]` | List of polygon coordinate lists → list of RLE dicts. The first entry decides, as in pycocotools: 4 values means every entry is an `[x, y, w, h]` box; more means every entry is a polygon, and a later 4-value entry is a degenerate polygon (area 0), not a box |
    | | `ndarray` of shape `(N, 4)` | Boxes `[x, y, w, h]` → list of RLE dicts |
    | | `dict` | Single uncompressed RLE dict → single RLE dict |
    | | `list[dict]` | List of uncompressed RLE dicts → list of RLE dicts |
    | `h` | `int` | Image height |
    | `w` | `int` | Image width |

    ```python
    # Polygons
    rles = mask.frPyObjects([[x1,y1,x2,y2,...]], 480, 640)

    # Uncompressed RLE dict
    rle = mask.frPyObjects({"size": [480, 640], "counts": [0, 5, 100, ...]}, 480, 640)
    ```

---

### `fr_poly`

Rasterize a polygon to an RLE mask.

=== "Python"

    ```python
    fr_poly(xy: list[float], h: int, w: int) -> dict
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `xy` | `list[float]` | Flat list of coordinates `[x1, y1, x2, y2, ...]` |
    | `h` | `int` | Image height |
    | `w` | `int` | Image width |

    ```python
    rle = mask.fr_poly([10, 10, 50, 10, 50, 50, 10, 50], 100, 100)
    ```

=== "Rust"

    ```rust
    fn fr_poly(xy: &[f64], h: u32, w: u32) -> Rle
    ```

    ```rust
    let rle = mask::fr_poly(&[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0], 100, 100);
    ```

---

### `fr_bbox`

Convert a bounding box to an RLE mask.

=== "Python"

    ```python
    fr_bbox(bb: list[float], h: int, w: int) -> dict
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `bb` | `list[float]` | Bounding box `[x, y, width, height]` |
    | `h` | `int` | Image height |
    | `w` | `int` | Image width |

    ```python
    rle = mask.fr_bbox([10, 10, 40, 40], 100, 100)
    ```

=== "Rust"

    ```rust
    fn fr_bbox(bb: &[f64; 4], h: u32, w: u32) -> Rle
    ```

    ```rust
    let rle = mask::fr_bbox(&[10.0, 10.0, 40.0, 40.0], 100, 100);
    ```

---

### `rle_to_string`

Encode an RLE to its compact LEB128 string representation.

=== "Python"

    ```python
    rle_to_string(rle: dict) -> str
    ```

    ```python
    s = mask.rle_to_string(rle)
    ```

=== "Rust"

    ```rust
    fn rle_to_string(rle: &Rle) -> String
    ```

    ```rust
    let s = mask::rle_to_string(&rle);
    ```

---

### `rle_from_string`

Decode an LEB128 string to an RLE.

=== "Python"

    ```python
    rle_from_string(s: str, h: int, w: int) -> dict
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `s` | `str` | LEB128-encoded RLE string |
    | `h` | `int` | Image height |
    | `w` | `int` | Image width |

    ```python
    rle = mask.rle_from_string(s, 100, 100)
    ```

=== "Rust"

    ```rust
    fn rle_from_string(s: &str, h: u32, w: u32) -> Result<Rle, String>
    ```

    ```rust
    let rle = mask::rle_from_string(&s, 100, 100).unwrap();
    ```
