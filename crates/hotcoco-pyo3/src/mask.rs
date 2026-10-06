//! Python bindings for `hotcoco.mask` — the `pycocotools.mask` mirror.
//!
//! This is the compatibility surface: pycocotools' leniency (type dispatch in
//! `iou`, `0`/`1` for a flag) lives here and only here, as thin wrappers over
//! the strict kernels in [`crate::primitives`]. `mask.bbox_iou` is the primitive
//! itself, registered under a second name — one-way path sugar, never the
//! definition.

use hotcoco_core::mask as rmask;
use numpy::ndarray::{ArrayView2, Axis};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{
    Flag, RleDict, boxes_arg, compressed_rle_to_py, extract_coco_rle, extract_rle_list, f64_array,
    py_to_rle, rle_result, rle_to_coco_py, type_name,
};
use crate::primitives::{box_iou_matrix, rle_iou_matrix};
use crate::to_pyerr;
use std::time::{Duration, Instant};

/// Transpose between row-major (numpy) and column-major (hotcoco) mask layouts.
///
/// With `(h, w)`: reads row-major `src[y * w + x]` → writes column-major `dst[y + h * x]`.
/// For the reverse direction (column-major → row-major), swap the arguments: call with `(w, h)`.
pub(crate) fn transpose_mask(src: &[u8], h: usize, w: usize) -> Vec<u8> {
    debug_assert_eq!(src.len(), h * w);
    let mut dst = vec![0u8; h * w];
    transpose_into(src, w, &mut dst, h, h, w);
    dst
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

/// Encode a binary mask to RLE in pycocotools format.
///
/// Parameters
/// ----------
/// mask : numpy.ndarray
///     2-D ``(H, W)`` → returns a single RLE dict.
///     3-D ``(H, W, N)`` → returns a list of *N* RLE dicts.
///     Accepts any memory layout, and the ``uint8``, ``bool``, and ``int8``
///     dtypes.
///
/// Returns
/// -------
/// dict or list[dict]
///     ``{"size": [H, W], "counts": b"..."}`` matching pycocotools.
#[pyfunction]
#[pyo3(text_signature = "(mask)")]
pub fn encode(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mask = as_uint8(mask)?;
    match mask.ndim() {
        2 => encode_2d(py, mask.cast()?),
        3 => encode_3d(py, mask.cast()?),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "mask must be 2-D (H, W) or 3-D (H, W, N)",
        )),
    }
}

/// The mask as a numpy `uint8` array, or the `TypeError` naming what it is
/// instead.
///
/// `uint8` passes through. `bool` and `int8` are one byte wide too, so a
/// `view` relabels them without a copy — shape and strides carry over, a
/// sliced view stays a view — and the core encoder's own rule, any nonzero
/// byte is foreground, does the rest (`-1` is `255`). The one-byte limit is
/// deliberate: a wider dtype is an error naming the dtype and the cast, so
/// nothing is silently truncated.
///
/// numpy's own extraction failure reads `'ndarray' object is not an instance
/// of 'ndarray'`, which names the same type twice and never mentions the
/// dtype — the one thing the caller has to change. Anything that is not a
/// numpy array (a list, or a torch tensor passed by mistake) is named by its
/// type instead. That includes an array-like with a numpy dtype, such as a
/// JAX or CuPy array, which a cast would not help.
fn as_uint8<'py>(mask: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyUntypedArray>> {
    let py = mask.py();
    let got = match mask.cast::<PyUntypedArray>() {
        Ok(arr) => {
            // Kind and item size are fields of the dtype struct. `dtype.name`
            // is a Python function several frames deep: reading it cost
            // 0.5–0.9 µs a call, more than encoding an 8×8 mask takes without
            // it, and TorchMetrics calls `encode` once per mask. Among numpy's
            // dtypes, only uint8 is kind `u` at one byte, and only bool and
            // int8 are kinds `b` and `i` at one byte.
            let dtype = arr.dtype();
            match (dtype.kind(), dtype.itemsize()) {
                (b'u', 1) => return Ok(arr.clone()),
                // A dtype object, not the string `"uint8"`, which numpy would
                // parse on every call.
                (b'b' | b'i', 1) => {
                    let view =
                        arr.call_method1(pyo3::intern!(py, "view"), (numpy::dtype::<u8>(py),))?;
                    return Ok(view.cast_into()?);
                }
                _ => format!(
                    "{}; cast it with mask.astype(numpy.uint8)",
                    dtype.getattr(pyo3::intern!(py, "name"))?
                ),
            }
        }
        Err(_) => type_name(mask),
    };
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "encode(): mask must be a numpy array with dtype uint8, bool, or int8, got {got}"
    )))
}

/// Column-major (Fortran-order) bytes of one `(H, W)` view that is not
/// Fortran-contiguous — [`encode_view`] encodes those from their own buffer.
///
/// A C-order mask is the one common case, and gets [`transpose_mask`]'s
/// blocked transpose: about 45 µs for a 426×640 mask, against about 100 µs
/// for ndarray's strided copy. Any other layout (a sliced or strided view)
/// still takes that copy, which is about 3× faster than iterating the
/// transposed view element by element — the reason this is not
/// `t.iter().copied().collect()`.
fn col_major(view: ArrayView2<'_, u8>) -> Vec<u8> {
    if let Some(row_major) = view.to_slice() {
        let (h, w) = view.dim();
        return transpose_mask(row_major, h, w);
    }
    let t = view.t();
    let std = t.as_standard_layout();
    std.as_slice()
        .map_or_else(|| std.iter().copied().collect(), <[u8]>::to_vec)
}

/// Write the `rows × cols` byte matrix `src` into `dst` transposed: element
/// `(r, c)`, read from `src[r * src_stride + c]`, lands at
/// `dst[c * dst_stride + r]`.
///
/// Works in 8×8 blocks — eight 8-byte rows in, eight 8-byte columns out — so
/// both sides move a word at a time and the destination's cache lines are
/// reused by the next block down instead of each byte landing on its own
/// line. A stack of C-order masks reads every byte from a different line
/// otherwise, which made it about 7× slower than a Fortran-order stack. The
/// ragged right and bottom edges go byte by byte.
fn transpose_into(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    rows: usize,
    cols: usize,
) {
    let (rows8, cols8) = (rows - rows % 8, cols - cols % 8);
    for r0 in (0..rows8).step_by(8) {
        for c0 in (0..cols8).step_by(8) {
            let mut block = [0u64; 8];
            for (i, row) in block.iter_mut().enumerate() {
                let at = (r0 + i) * src_stride + c0;
                *row = load_le(&src[at..at + 8]);
            }
            for (j, col) in transpose_8x8(block).into_iter().enumerate() {
                let at = (c0 + j) * dst_stride + r0;
                dst[at..at + 8].copy_from_slice(&col.to_le_bytes());
            }
        }
        for c in cols8..cols {
            for r in r0..r0 + 8 {
                dst[c * dst_stride + r] = src[r * src_stride + c];
            }
        }
    }
    for r in rows8..rows {
        for c in 0..cols {
            dst[c * dst_stride + r] = src[r * src_stride + c];
        }
    }
}

/// Eight bytes as a word, first byte in the low bits on every platform.
fn load_le(bytes: &[u8]) -> u64 {
    let mut word = [0; 8];
    word.copy_from_slice(bytes);
    u64::from_le_bytes(word)
}

/// Transpose an 8×8 byte matrix held as eight little-endian rows.
///
/// Three swap rounds, each exchanging the off-diagonal blocks of every
/// diagonal block pair: 4×4 blocks, then 2×2, then single bytes.
fn transpose_8x8(mut m: [u64; 8]) -> [u64; 8] {
    for (shift, mask) in [
        (32, 0x0000_0000_FFFF_FFFF_u64),
        (16, 0x0000_FFFF_0000_FFFF),
        (8, 0x00FF_00FF_00FF_00FF),
    ] {
        let step = shift / 8;
        for i in (0..8).filter(|i| i & step == 0) {
            let t = ((m[i] >> shift) ^ m[i + step]) & mask;
            m[i] ^= t << shift;
            m[i + step] ^= t;
        }
    }
    m
}

// The GIL rule for the encoders below: whatever reads numpy memory holds the
// GIL, because a buffer read without it can be written by another Python
// thread mid-scan. That covers every 2-D mask, a strided stack, a
// Fortran-order stack for up to `IN_PLACE_BUDGET` (each encoded straight from
// the numpy buffer), and the copy that moves a stack out of numpy memory. The
// only release is for encoding and compressing such a copy, which no Python
// code can reach: the rest of a Fortran-order stack once its budget is spent,
// and a C-order stack's transpose from `DETACH_MIN_BYTES` up.
//
// For one mask a release would buy nothing and risk the convoy effect: it
// scans in about 10 µs (426×640), while getting the GIL back from a busy
// Python thread after a release measured about 2.5 ms per mask, waiting out
// switch intervals. pycocotools holds the GIL too.

/// The stack size in bytes (`h * w * n`) from which a C-order `(H, W, N)`
/// encode releases the GIL to encode and compress its transposed copy.
///
/// A release costs the caller a wait only when the call would otherwise end
/// within CPython's switch interval (5 ms by default). A thread kept waiting
/// longer has already asked for the GIL, and the caller hands it over on
/// return either way. On an M1, the transpose alone holds the GIL for 8–14 ms
/// at 16 MiB, across mask sizes from 128×128 to 1024×1024, so from here a
/// release adds no wait and lets that thread run during the encode. A smaller
/// stack can finish within the interval: 8 MiB of 1024×1024 masks takes 3 ms,
/// and releasing for it cost a waiting caller 6.6 ms more. The encode is a few
/// percent of the call for typical masks but most of it for masks with many
/// runs: a 1024×1024×100 noise stack held the GIL for 318 ms before this
/// release and 82 ms after.
const DETACH_MIN_BYTES: usize = 16 << 20;

/// How long a Fortran-order `(H, W, N)` stack is encoded in place, under the
/// GIL, before the rest is copied out and encoded without it: CPython's
/// default switch interval.
///
/// A size threshold suits a Fortran-order stack poorly, because its copy is a
/// memcpy, far quicker than encoding masks with many runs and slower than
/// encoding masks with few. Copying from 16 MiB up cost a 426×640×400 stack of
/// val2017 masks about 21 µs per mask against about 8 µs in place, and 16 MiB
/// of ellipse masks 7.5 ms against 0.6 ms with a busy thread alongside. Typical
/// masks finish within the budget and never copy; a 1024×1024×100 noise stack
/// held the GIL for 255 ms in place and about 11 ms with the budget.
const IN_PLACE_BUDGET: Duration = Duration::from_millis(5);

/// Encode one `(H, W)` view, from its own buffer when it is Fortran-contiguous
/// — TorchMetrics' call, one `np.asfortranarray` mask at a time — and through
/// [`col_major`] otherwise.
fn encode_view(view: ArrayView2<'_, u8>) -> hotcoco_core::error::Result<hotcoco_core::Rle> {
    let (h, w) = view.dim();
    // The transpose of a Fortran-order view is standard layout, and its slice
    // is the mask's own buffer: already the column-major bytes `encode` takes.
    match view.t().to_slice() {
        Some(col_major) => rmask::encode(col_major, h as u32, w as u32),
        None => rmask::encode(&col_major(view), h as u32, w as u32),
    }
}

fn encode_2d(py: Python<'_>, mask: &Bound<'_, PyArray2<u8>>) -> PyResult<Py<PyAny>> {
    let arr = mask.readonly();
    let rle = encode_view(arr.as_array()).map_err(to_pyerr)?;
    rle_to_coco_py(py, &rle)
}

fn encode_3d(py: Python<'_>, mask: &Bound<'_, PyArray3<u8>>) -> PyResult<Py<PyAny>> {
    let arr = mask.readonly();
    let view = arr.as_array();
    let (h, w, n) = view.dim();
    let hw = h * w;
    let counts = if let Some(stack) = view.reversed_axes().to_slice() {
        // Fortran order: slice `i` is the contiguous block `i * h * w ..`,
        // already column-major. ndarray counts every zero-size array as
        // contiguous, so those all land here and the C-order offsets below
        // never index an empty buffer.
        encode_fortran_stack(py, stack, h, w, n)
    } else if let Some(row_major) = view.to_slice().filter(|_| n >= 8) {
        // C order: one blocked pass to Fortran order. For each column `x`,
        // the `(h, n)` matrix of `(y, i)` bytes sits at `x * n` with row
        // stride `w * n`, and its transpose belongs at `x * h` with row stride
        // `h * w` — slice `i`'s column `x`. With fewer than eight slices there
        // is no 8×8 block along `n`, every byte would take the ragged-edge
        // loop, and slice by slice below is faster: the lone slice of an
        // `(h, w, 1)` stack is C-contiguous, so it still gets the blocked
        // transpose.
        let mut stack = vec![0; hw * n];
        for x in 0..w {
            transpose_into(&row_major[x * n..], w * n, &mut stack[x * h..], hw, h, n);
        }
        if stack.len() >= DETACH_MIN_BYTES {
            py.detach(|| encode_slices(&stack, h, w, n))
        } else {
            encode_slices(&stack, h, w, n)
        }
    } else {
        (0..n)
            .map(|i| {
                let rle = encode_view(view.index_axis(Axis(2), i))?;
                Ok(rmask::rle_to_string(&rle))
            })
            .collect()
    }
    .map_err(to_pyerr)?;
    let list = PyList::empty(py);
    for counts in &counts {
        list.append(compressed_rle_to_py(py, h as u32, w as u32, counts)?)?;
    }
    Ok(list.into_any().unbind())
}

/// [`encode_slices`] on a Fortran-order stack still in numpy memory: in place
/// under the GIL for up to [`IN_PLACE_BUDGET`], then on a copy of the slices
/// left, with the GIL released.
fn encode_fortran_stack(
    py: Python<'_>,
    stack: &[u8],
    h: usize,
    w: usize,
    n: usize,
) -> hotcoco_core::error::Result<Vec<String>> {
    let hw = h * w;
    let start = Instant::now();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if start.elapsed() >= IN_PLACE_BUDGET {
            let rest = stack[i * hw..].to_vec();
            out.extend(py.detach(|| encode_slices(&rest, h, w, n - i))?);
            break;
        }
        out.extend(encode_slices(&stack[i * hw..(i + 1) * hw], h, w, 1)?);
    }
    Ok(out)
}

/// The compressed `counts` string of each `h × w` column-major block of a
/// Fortran-order `(h, w, n)` stack. Each block's run list is dropped as soon
/// as it is compressed, so the runs of only one mask are held at a time.
fn encode_slices(
    stack: &[u8],
    h: usize,
    w: usize,
    n: usize,
) -> hotcoco_core::error::Result<Vec<String>> {
    let hw = h * w;
    (0..n)
        .map(|i| {
            let rle = rmask::encode(&stack[i * hw..(i + 1) * hw], h as u32, w as u32)?;
            Ok(rmask::rle_to_string(&rle))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

/// Decode RLE to a binary mask.
///
/// Parameters
/// ----------
/// rle : dict or list[dict]
///     Single RLE dict → ``(H, W)`` uint8 Fortran-order array.
///     List of *N* RLE dicts → ``(H, W, N)`` uint8 Fortran-order array.
///
/// Returns
/// -------
/// numpy.ndarray
#[pyfunction]
#[pyo3(text_signature = "(rle)")]
pub fn decode(py: Python<'_>, rle: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(dict) = rle.cast::<PyDict>() {
        // Single RLE → (H, W) Fortran-order
        let r = py_to_rle(dict)?;
        let col_major = py.detach(|| rmask::decode(&r));
        let h = r.h as usize;
        let w = r.w as usize;
        // col_major is already in Fortran order — create array and set flag
        let flat = PyArray1::from_vec(py, col_major);
        let arr2d = flat.reshape_with_order([h, w], numpy::npyffi::NPY_ORDER::NPY_FORTRANORDER)?;
        Ok(arr2d.into_any().unbind())
    } else {
        // List of RLEs → (H, W, N) Fortran-order
        let list: Vec<Bound<'_, PyAny>> = rle.extract()?;
        if list.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty RLE list"));
        }
        let rles: Vec<hotcoco_core::Rle> = list
            .iter()
            .map(|item| extract_coco_rle(item))
            .collect::<PyResult<_>>()?;

        let h = rles[0].h as usize;
        let w = rles[0].w as usize;
        let n = rles.len();

        // Build (H, W, N) Fortran-order: for each slice, decode gives
        // column-major data. In Fortran order for 3D, axis 0 varies fastest,
        // so memory layout is: all (h*w) of slice 0, then slice 1, etc.
        let data = py.detach(|| {
            let mut data = Vec::with_capacity(h * w * n);
            for r in &rles {
                data.extend_from_slice(&rmask::decode(r));
            }
            data
        });
        let flat = PyArray1::from_vec(py, data);
        let arr3d =
            flat.reshape_with_order([h, w, n], numpy::npyffi::NPY_ORDER::NPY_FORTRANORDER)?;
        Ok(arr3d.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// area
// ---------------------------------------------------------------------------

/// Compute the area (number of foreground pixels) of RLE mask(s).
///
/// Parameters
/// ----------
/// rle : dict or list[dict]
///     Single RLE dict → scalar uint64.
///     List of RLE dicts → numpy uint32 array, matching pycocotools.
#[pyfunction]
#[pyo3(text_signature = "(rle)")]
pub fn area(py: Python<'_>, rle: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(dict) = rle.cast::<PyDict>() {
        let a = rle_result(RleDict::read(dict)?.view()?.area())?;
        return Ok(a.into_pyobject(py)?.into_any().unbind());
    }
    // `uint32`, matching pycocotools' array dtype exactly — the parity suite
    // checks dtypes, not just values. (The scalar path above hands back a
    // Python int, which has no dtype to match.)
    let areas = each_rle(rle, rmask::areas)?;
    let arr = PyArray1::from_vec(py, areas.into_iter().map(|a| a as u32).collect());
    Ok(arr.into_any().unbind())
}

/// Run a core batch over a list of RLE dicts, in order, raising the first
/// RLE's error.
///
/// The views borrow `counts` strings from the Python objects, which this call
/// keeps alive and holds the GIL over, so the batch can decode them on
/// rayon's workers: they read the bytes and never touch the interpreter.
fn each_rle<T>(
    obj: &Bound<'_, PyAny>,
    batch: impl FnOnce(&[rmask::RleRef<'_>]) -> Vec<Result<T, hotcoco_core::Error>>,
) -> PyResult<Vec<T>> {
    let items: Vec<Bound<'_, PyAny>> = obj.extract()?;
    let read = items
        .iter()
        .map(|item| RleDict::read(item.cast::<PyDict>()?))
        .collect::<PyResult<Vec<_>>>()?;
    let views = read
        .iter()
        .map(RleDict::view)
        .collect::<PyResult<Vec<_>>>()?;
    batch(&views).into_iter().map(rle_result).collect()
}

// ---------------------------------------------------------------------------
// to_bbox / toBbox
// ---------------------------------------------------------------------------

/// Compute bounding box(es) from RLE mask(s).
///
/// Parameters
/// ----------
/// rle : dict or list[dict]
///     Single RLE dict → numpy float64(4,).
///     List of RLE dicts → numpy float64(N, 4).
#[pyfunction]
#[pyo3(text_signature = "(rle)")]
pub fn to_bbox(py: Python<'_>, rle: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(dict) = rle.cast::<PyDict>() {
        let bbox = rle_result(RleDict::read(dict)?.view()?.to_bbox())?;
        return Ok(PyArray1::from_vec(py, bbox.to_vec()).into_any().unbind());
    }
    let boxes = each_rle(rle, rmask::bboxes)?;
    let n = boxes.len();
    f64_array(py, boxes.into_iter().flatten().collect(), [n, 4])
}

/// Alias for `to_bbox` matching pycocotools naming.
#[pyfunction]
#[pyo3(name = "toBbox")]
pub fn to_bbox_camel(py: Python<'_>, rle: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    to_bbox(py, rle)
}

// ---------------------------------------------------------------------------
// merge
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (rles, intersect = Flag(false)))]
pub fn merge(py: Python<'_>, rles: &Bound<'_, PyAny>, intersect: Flag) -> PyResult<Py<PyAny>> {
    let rle_vec = extract_rle_list(rles)?;
    let result = rmask::merge(&rle_vec, intersect.0).map_err(to_pyerr)?;
    rle_to_coco_py(py, &result)
}

// ---------------------------------------------------------------------------
// iou
// ---------------------------------------------------------------------------

/// One side of a `pycocotools.mask.iou` call, after its type dispatch.
enum IouInput {
    Rles(Vec<hotcoco_core::Rle>),
    Boxes(Vec<[f64; 4]>),
}

impl IouInput {
    /// No inputs of the same kind — what an empty list becomes, since it has
    /// no kind of its own and takes the other side's.
    fn empty_like(&self) -> Self {
        match self {
            IouInput::Rles(_) => IouInput::Rles(Vec::new()),
            IouInput::Boxes(_) => IouInput::Boxes(Vec::new()),
        }
    }
}

/// Sort an `iou` argument the way pycocotools' `_preproc` does: a numpy array
/// is boxes (any numeric dtype), a list of dicts is RLEs, a list of 4-element
/// rows is boxes. A single RLE dict is also taken, as it was before boxes
/// were. `None` is an empty list, whose kind the caller takes from the other
/// side.
fn classify_iou_input(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<IouInput>> {
    if obj.cast::<numpy::PyUntypedArray>().is_ok() {
        return Ok(Some(IouInput::Boxes(boxes_arg(obj, name)?)));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return Ok(Some(IouInput::Rles(vec![py_to_rle(dict)?])));
    }
    let unrecognized = || {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a list of RLE dicts, a list of [x, y, w, h] boxes, \
             or an (N, 4) array, got {}",
            type_name(obj)
        ))
    };
    // Collected once and parsed from here, so a generator argument works.
    let items: Vec<Bound<'_, PyAny>> = obj
        .try_iter()
        .and_then(Iterator::collect)
        .map_err(|_| unrecognized())?;
    if items.is_empty() {
        return Ok(None);
    }
    if items.iter().all(|item| item.cast::<PyDict>().is_ok()) {
        let rles = items
            .iter()
            .map(extract_coco_rle)
            .collect::<PyResult<_>>()?;
        return Ok(Some(IouInput::Rles(rles)));
    }
    // A list that is neither RLEs nor 4-element rows is unrecognized input,
    // as pycocotools' `_preproc` reports it — not a malformed box array.
    let rows = PyList::new(obj.py(), &items)?;
    let boxes = boxes_arg(&rows, name).map_err(|_| unrecognized())?;
    Ok(Some(IouInput::Boxes(boxes)))
}

/// The `pycocotools.mask.iou` mirror: RLEs or boxes, decided by type.
///
/// The dispatch is pycocotools' and lives only here — `hotcoco.primitives`
/// keeps one input type per kernel, and this forwards to those kernels.
#[pyfunction]
#[pyo3(text_signature = "(dt, gt, iscrowd)")]
pub fn iou(
    py: Python<'_>,
    dt: &Bound<'_, PyAny>,
    gt: &Bound<'_, PyAny>,
    iscrowd: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    use IouInput::{Boxes, Rles};
    let (dt, gt) = (classify_iou_input(dt, "dt")?, classify_iou_input(gt, "gt")?);
    // Two empty lists take the RLE kernel; either kernel gives a (0, 0) result.
    let dt = dt.unwrap_or_else(|| gt.as_ref().map_or(Rles(Vec::new()), IouInput::empty_like));
    let gt = gt.unwrap_or_else(|| dt.empty_like());
    match (dt, gt) {
        (Rles(d), Rles(g)) => rle_iou_matrix(py, &d, &g, iscrowd),
        (Boxes(d), Boxes(g)) => box_iou_matrix(py, &d, &g, iscrowd),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "dt and gt must be the same kind: both RLE dicts or both boxes",
        )),
    }
}

// ---------------------------------------------------------------------------
// fr_poly / frPoly
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(text_signature = "(xy, h, w)")]
pub fn fr_poly(py: Python<'_>, xy: Vec<f64>, h: u32, w: u32) -> PyResult<Py<PyAny>> {
    let rle = rmask::fr_poly(&xy, h, w).map_err(to_pyerr)?;
    rle_to_coco_py(py, &rle)
}

/// Alias for `fr_poly` matching pycocotools naming.
#[pyfunction]
#[pyo3(name = "frPoly")]
pub fn fr_poly_camel(py: Python<'_>, xy: Vec<f64>, h: u32, w: u32) -> PyResult<Py<PyAny>> {
    fr_poly(py, xy, h, w)
}

// ---------------------------------------------------------------------------
// fr_bbox / frBbox
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(text_signature = "(bb, h, w)")]
pub fn fr_bbox(py: Python<'_>, bb: [f64; 4], h: u32, w: u32) -> PyResult<Py<PyAny>> {
    let rle = rmask::fr_bbox(&bb, h, w).map_err(to_pyerr)?;
    rle_to_coco_py(py, &rle)
}

/// Alias for `fr_bbox` matching pycocotools naming.
#[pyfunction]
#[pyo3(name = "frBbox")]
pub fn fr_bbox_camel(py: Python<'_>, bb: [f64; 4], h: u32, w: u32) -> PyResult<Py<PyAny>> {
    fr_bbox(py, bb, h, w)
}

// ---------------------------------------------------------------------------
// rle_to_string / rle_from_string
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(text_signature = "(rle)")]
pub fn rle_to_string(rle: &Bound<'_, PyDict>) -> PyResult<String> {
    let rle = py_to_rle(rle)?;
    Ok(rmask::rle_to_string(&rle))
}

#[pyfunction]
#[pyo3(text_signature = "(s, h, w)")]
pub fn rle_from_string(py: Python<'_>, s: &str, h: u32, w: u32) -> PyResult<Py<PyAny>> {
    let rle = rle_result(rmask::rle_from_string(s, h, w))?;
    rle_to_coco_py(py, &rle)
}

// ---------------------------------------------------------------------------
// frPyObjects / fr_py_objects
// ---------------------------------------------------------------------------

/// Encode segmentation objects to RLEs (pycocotools compatibility).
///
/// Parameters
/// ----------
/// seg : list[list[float]] | numpy.ndarray | dict | list[dict]
///     - List (or 2-D array) of boxes ``[x, y, w, h]`` → list of RLE dicts.
///     - List of flattened polygons ``[x1, y1, x2, y2, ...]`` → list of RLE dicts.
///       The first entry's length decides for the whole list, as in
///       pycocotools: 4 values means every entry is a box, more than 4 means
///       every entry is a polygon (a later 4-value entry is then a two-point
///       polygon with area 0).
///     - Single RLE dict (compressed or uncompressed) → one RLE dict.
///     - List of uncompressed RLE dicts → list of RLE dicts.
/// h : int
///     Image height.
/// w : int
///     Image width.
///
/// Returns
/// -------
/// dict or list[dict]
///     A single RLE dict when ``seg`` is a dict, a list of RLE dicts when it
///     is a list — dict in, dict out, exactly as pycocotools does it.
#[pyfunction]
#[pyo3(name = "frPyObjects", text_signature = "(seg, h, w)")]
pub fn fr_py_objects(
    py: Python<'_>,
    seg: &Bound<'_, PyAny>,
    h: u32,
    w: u32,
) -> PyResult<Py<PyAny>> {
    // Case 1: single dict (uncompressed or compressed RLE) → single dict
    if let Ok(dict) = seg.cast::<PyDict>() {
        let rle = py_to_rle(dict)?;
        return rle_to_coco_py(py, &rle);
    }

    // Case 2: list — could be list of polygons or list of dicts
    let items: Vec<Bound<'_, PyAny>> = seg.extract()?;
    if items.is_empty() {
        return Ok(PyList::empty(py).into_any().unbind());
    }

    // Check first element to determine type
    let first = &items[0];
    if first.cast::<PyDict>().is_ok() {
        // List of RLE dicts
        let list = PyList::empty(py);
        for item in &items {
            let rle = extract_coco_rle(item)?;
            list.append(rle_to_coco_py(py, &rle)?)?;
        }
        Ok(list.into_any().unbind())
    } else {
        // List (or ndarray) of coordinate sequences. pycocotools decides box vs
        // polygon *once*, from the first entry's length, and applies that to every
        // entry: `[[1,1,8,1,8,8], [5,5,6,6]]` is two polygons, the second a
        // degenerate two-point one with area 0 — not a polygon and a box.
        //
        // Deliberate deviations, both strictly more permissive: pycocotools' box
        // path requires a numpy array and raises `TypeError` on a list of lists,
        // and it sends *every* ndarray to the box path whatever its width (reading
        // a wider array's raw memory four values at a time). Here a list of boxes
        // works, and an ndarray dispatches on its row length like a list does.
        let first_len = first.len()?;
        let boxes = match first_len {
            4 => true,
            n if n > 4 => false,
            n => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "frPyObjects: the first entry must be a box [x, y, w, h] (4 values) \
                     or a flattened polygon [x1, y1, x2, y2, ...] (more than 4); got {n}"
                )));
            }
        };
        let list = PyList::empty(py);
        if boxes {
            // One read of the whole list or array. When it fails on an entry
            // that is not 4 values, the error says so in pycocotools' terms.
            let bbs = boxes_arg(seg, "seg").map_err(|err| {
                match items
                    .iter()
                    .find_map(|item| item.len().ok().filter(|&n| n != 4))
                {
                    Some(n) => pyo3::exceptions::PyValueError::new_err(format!(
                        "frPyObjects: the first entry is a box, so every entry must be \
                         [x, y, w, h] (4 values); got {n}"
                    )),
                    None => err,
                }
            })?;
            for bb in &bbs {
                list.append(rle_to_coco_py(
                    py,
                    &rmask::fr_bbox(bb, h, w).map_err(to_pyerr)?,
                )?)?;
            }
        } else {
            for item in &items {
                // Polygon mode, as pycocotools' `frPoly`: any length, `len // 2`
                // points; fewer than three rasterize to an empty mask.
                let coords: Vec<f64> = item.extract()?;
                let rle = rmask::fr_poly(&coords, h, w).map_err(to_pyerr)?;
                list.append(rle_to_coco_py(py, &rle)?)?;
            }
        }
        Ok(list.into_any().unbind())
    }
}

/// Snake-case alias for `frPyObjects`.
///
/// The explicit `name` is load-bearing: PyO3 falls back to the Rust identifier,
/// so without it the snake_case spelling this alias exists to provide is the one
/// spelling not reachable from Python.
#[pyfunction]
#[pyo3(name = "fr_py_objects", text_signature = "(seg, h, w)")]
pub fn fr_py_objects_snake(
    py: Python<'_>,
    seg: &Bound<'_, PyAny>,
    h: u32,
    w: u32,
) -> PyResult<Py<PyAny>> {
    fr_py_objects(py, seg, h, w)
}
