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

For background on RLE and usage patterns, see the [Mask Operations](../guide/masks.md) guide.

!!! tip "pycocotools drop-in"
    The Python `mask` module is a drop-in replacement for `pycocotools.mask`.
    All functions accept and return the same types — `encode` returns
    `{"size": [h, w], "counts": b"..."}`, `decode` returns Fortran-order arrays,
    and batch functions accept single values or lists.

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
    | `mask` | `numpy.ndarray` | 2-D `(H, W)` or 3-D `(H, W, N)`, dtype `uint8` |

    **Returns:**

    - 2-D input → `dict` with `"size"` (`[H, W]`) and `"counts"` (`bytes`)
    - 3-D input → `list[dict]` of *N* RLE dicts

    Accepts both Fortran-order (pycocotools convention) and C-order arrays.

    ```python
    import numpy as np
    from hotcoco import mask

    # Single mask (Fortran-order, matching pycocotools)
    m = np.zeros((100, 100), dtype=np.uint8, order="F")
    m[10:50, 20:80] = 1
    rle = mask.encode(m)
    # {"size": [100, 100], "counts": b"..."}

    # Batch of N masks
    m3 = np.zeros((100, 100, 3), dtype=np.uint8, order="F")
    m3[10:50, 20:80, 0] = 1
    rles = mask.encode(m3)  # list of 3 RLE dicts

    # C-order also works (auto-transposed internally)
    m_c = np.zeros((100, 100), dtype=np.uint8)
    m_c[10:50, 20:80] = 1
    rle = mask.encode(m_c)
    ```

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
    | Single dict | `(H, W)` uint8 Fortran-order array |
    | List of *N* dicts | `(H, W, N)` uint8 Fortran-order array |

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
    | Single dict | `int` (uint64) |
    | List of dicts | `numpy.ndarray` of uint64 |

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

    !!! note "camelCase alias"
        Also available as `toBbox()`.

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
    merge(rles: list[dict], intersect: bool = False) -> dict
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `rles` | `list[dict]` | | List of RLE dicts to merge |
    | `intersect` | `bool` | `False` | If `True`, compute intersection instead of union |

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

Compute pairwise IoU between two lists of RLE masks.

=== "Python"

    ```python
    iou(dt: list[dict], gt: list[dict], iscrowd: list[bool]) -> numpy.ndarray
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `dt` | `list[dict]` | Detection RLE dicts |
    | `gt` | `list[dict]` | Ground truth RLE dicts |
    | `iscrowd` | `list[bool]` | Per-GT crowd flag |

    **Returns:** `numpy.ndarray` of shape `(len(dt), len(gt))`, dtype float64.

    ```python
    ious = mask.iou(dt_rles, gt_rles, [False] * len(gt_rles))
    ```

=== "Rust"

    ```rust
    fn iou(dt: &[Rle], gt: &[Rle], iscrowd: &[bool]) -> Vec<Vec<f64>>
    ```

    **Returns:** `Vec<Vec<f64>>` of shape D x G.

    ```rust
    let ious = mask::iou(&dt_rles, &gt_rles, &vec![false; gt_rles.len()]);
    ```

When `iscrowd[j]` is `true`, uses `intersection / area(dt)` instead of standard IoU for GT `j`.

---

### `bbox_iou`

Compute pairwise IoU between two lists of bounding boxes.

=== "Python"

    ```python
    bbox_iou(dt: list[list[float]], gt: list[list[float]], iscrowd: list[bool]) -> numpy.ndarray
    ```

    Bounding boxes are `[x, y, width, height]`.

    **Returns:** `numpy.ndarray` of shape `(len(dt), len(gt))`, dtype float64.

    ```python
    ious = mask.bbox_iou(dt_boxes, gt_boxes, [False] * len(gt_boxes))
    ```

    !!! note "camelCase alias"
        Also available as `bboxIou()`.

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
    | `seg` | `list[list[float]]` | List of polygon coordinate lists → list of RLE dicts |
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

    !!! note "snake_case alias"
        Also available as `fr_py_objects()`.

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

    !!! note "camelCase alias"
        Also available as `frPoly()`.

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

    !!! note "camelCase alias"
        Also available as `frBbox()`.

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

    !!! note "camelCase alias"
        Also available as `rleToString()`.

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

    !!! note "camelCase alias"
        Also available as `rleFromString()`.

=== "Rust"

    ```rust
    fn rle_from_string(s: &str, h: u32, w: u32) -> Result<Rle, String>
    ```

    ```rust
    let rle = mask::rle_from_string(&s, 100, 100).unwrap();
    ```
