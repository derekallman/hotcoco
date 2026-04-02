use hotcoco_core::mask as rmask;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{py_to_rle, rle_to_coco_py};

/// Transpose between row-major (numpy) and column-major (hotcoco) mask layouts.
///
/// With `(h, w)`: reads row-major `src[y * w + x]` → writes column-major `dst[y + h * x]`.
/// For the reverse direction (column-major → row-major), swap the arguments: call with `(w, h)`.
pub(crate) fn transpose_mask(src: &[u8], h: usize, w: usize) -> Vec<u8> {
    debug_assert_eq!(src.len(), h * w);
    let mut dst = vec![0u8; h * w];
    for y in 0..h {
        for x in 0..w {
            dst[y + h * x] = src[y * w + x];
        }
    }
    dst
}

/// Extract a single RLE dict from a Python object (dict with "size"+"counts"
/// or "h"+"w"+"counts").
fn extract_coco_rle(obj: &Bound<'_, PyAny>) -> PyResult<hotcoco_core::Rle> {
    let dict = obj.cast::<PyDict>()?;
    py_to_rle(dict)
}

/// Extract a list of RLE dicts from a Python object. Accepts either a single
/// dict or a list of dicts.
fn extract_rle_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<hotcoco_core::Rle>> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        Ok(vec![py_to_rle(dict)?])
    } else {
        let list: Vec<Bound<'_, PyAny>> = obj.extract()?;
        list.iter().map(|item| extract_coco_rle(item)).collect()
    }
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
///     Accepts both Fortran-order (pycocotools convention) and C-order arrays.
///
/// Returns
/// -------
/// dict or list[dict]
///     ``{"size": [H, W], "counts": b"..."}`` matching pycocotools.
#[pyfunction]
#[pyo3(text_signature = "(mask)")]
pub fn encode(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let ndim: usize = mask.getattr("ndim")?.extract()?;
    match ndim {
        2 => encode_2d(py, mask),
        3 => encode_3d(py, mask),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "mask must be 2-D (H, W) or 3-D (H, W, N)",
        )),
    }
}

fn encode_2d(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr: PyReadonlyArray2<u8> = mask.extract()?;
    let shape = arr.shape();
    let h = shape[0];
    let w = shape[1];

    // Check if Fortran-order (column-major)
    let is_fortran: bool = mask.getattr("flags")?.getattr("f_contiguous")?.extract()?;

    let col_major = if is_fortran {
        // Already column-major — use raw data directly
        arr.as_slice()?.to_vec()
    } else {
        // C-order — transpose to column-major
        let slice = arr.as_slice()?;
        transpose_mask(slice, h, w)
    };

    let rle = rmask::encode(&col_major, h as u32, w as u32);
    rle_to_coco_py(py, &rle)
}

fn encode_3d(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr: PyReadonlyArray3<u8> = mask.extract()?;
    let shape = arr.shape();
    let h = shape[0];
    let w = shape[1];
    let n = shape[2];

    let raw = arr.as_array();
    let list = PyList::empty(py);
    for i in 0..n {
        // Build column-major data: iterate column-by-column (x then y)
        let slice_2d = raw.index_axis(numpy::ndarray::Axis(2), i);
        let mut col_major = Vec::with_capacity(h * w);
        for x in 0..w {
            for y in 0..h {
                col_major.push(slice_2d[[y, x]]);
            }
        }
        let rle = rmask::encode(&col_major, h as u32, w as u32);
        list.append(rle_to_coco_py(py, &rle)?)?;
    }
    Ok(list.into_any().unbind())
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
        let col_major = rmask::decode(&r);
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
        let mut data = Vec::with_capacity(h * w * n);
        for r in &rles {
            data.extend_from_slice(&rmask::decode(r));
        }
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
///     List of RLE dicts → numpy uint64 array.
#[pyfunction]
#[pyo3(text_signature = "(rle)")]
pub fn area(py: Python<'_>, rle: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(dict) = rle.cast::<PyDict>() {
        let r = py_to_rle(dict)?;
        let a = rmask::area(&r);
        Ok(a.into_pyobject(py)?.into_any().unbind())
    } else {
        let rles = extract_rle_list(rle)?;
        let areas: Vec<u64> = rles.iter().map(rmask::area).collect();
        let arr = PyArray1::from_vec(py, areas);
        Ok(arr.into_any().unbind())
    }
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
        let r = py_to_rle(dict)?;
        let bb = rmask::to_bbox(&r);
        let arr = PyArray1::from_vec(py, bb.to_vec());
        Ok(arr.into_any().unbind())
    } else {
        let rles = extract_rle_list(rle)?;
        let n = rles.len();
        let mut data = Vec::with_capacity(n * 4);
        for r in &rles {
            data.extend_from_slice(&rmask::to_bbox(r));
        }
        let flat = PyArray1::from_vec(py, data);
        let arr2d = flat.reshape([n, 4])?;
        Ok(arr2d.into_any().unbind())
    }
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
#[pyo3(signature = (rles, intersect = false))]
pub fn merge(py: Python<'_>, rles: &Bound<'_, PyAny>, intersect: bool) -> PyResult<Py<PyAny>> {
    let rle_vec = extract_rle_list(rles)?;
    let result = rmask::merge(&rle_vec, intersect);
    rle_to_coco_py(py, &result)
}

fn check_iscrowd_len(iscrowd_len: usize, gt_len: usize) -> PyResult<()> {
    if iscrowd_len != gt_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "iscrowd length ({iscrowd_len}) must equal gt length ({gt_len})"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// iou
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(text_signature = "(dt, gt, iscrowd)")]
pub fn iou(
    py: Python<'_>,
    dt: &Bound<'_, PyAny>,
    gt: &Bound<'_, PyAny>,
    iscrowd: Vec<bool>,
) -> PyResult<Py<PyAny>> {
    let dt_rles = extract_rle_list(dt)?;
    let gt_rles = extract_rle_list(gt)?;
    check_iscrowd_len(iscrowd.len(), gt_rles.len())?;
    let result = rmask::iou(&dt_rles, &gt_rles, &iscrowd);
    let d = dt_rles.len();
    let g = gt_rles.len();
    let mut flat = Vec::with_capacity(d * g);
    for row in &result {
        flat.extend_from_slice(row);
    }
    let arr = PyArray1::from_vec(py, flat);
    let arr2d = arr.reshape([d, g])?;
    Ok(arr2d.into_any().unbind())
}

#[pyfunction]
#[pyo3(text_signature = "(dt, gt, iscrowd)")]
pub fn bbox_iou(
    py: Python<'_>,
    dt: Vec<[f64; 4]>,
    gt: Vec<[f64; 4]>,
    iscrowd: Vec<bool>,
) -> PyResult<Py<PyAny>> {
    check_iscrowd_len(iscrowd.len(), gt.len())?;
    let result = rmask::bbox_iou(&dt, &gt, &iscrowd);
    let d = dt.len();
    let g = gt.len();
    let mut flat = Vec::with_capacity(d * g);
    for row in &result {
        flat.extend_from_slice(row);
    }
    let arr = PyArray1::from_vec(py, flat);
    let arr2d = arr.reshape([d, g])?;
    Ok(arr2d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// fr_poly / frPoly
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(text_signature = "(xy, h, w)")]
pub fn fr_poly(py: Python<'_>, xy: Vec<f64>, h: u32, w: u32) -> PyResult<Py<PyAny>> {
    let rle = rmask::fr_poly(&xy, h, w);
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
    let rle = rmask::fr_bbox(&bb, h, w);
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
    let rle = rmask::rle_from_string(s, h, w)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    rle_to_coco_py(py, &rle)
}

// ---------------------------------------------------------------------------
// frPyObjects / fr_py_objects
// ---------------------------------------------------------------------------

/// Encode segmentation objects to RLEs (pycocotools compatibility).
///
/// Parameters
/// ----------
/// seg : list[list[float]] | dict | list[dict]
///     - List of polygon coordinate lists → list of RLE dicts.
///     - Single uncompressed RLE dict → list of 1 RLE dict.
///     - List of uncompressed RLE dicts → list of RLE dicts.
/// h : int
///     Image height.
/// w : int
///     Image width.
///
/// Returns
/// -------
/// list[dict]
///     List of RLE dicts in pycocotools format.
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
        // List of polygon coordinate lists
        let list = PyList::empty(py);
        for item in &items {
            let coords: Vec<f64> = item.extract()?;
            let rle = rmask::fr_poly(&coords, h, w);
            list.append(rle_to_coco_py(py, &rle)?)?;
        }
        Ok(list.into_any().unbind())
    }
}

/// Snake-case alias for `frPyObjects`.
#[pyfunction]
pub fn fr_py_objects_snake(
    py: Python<'_>,
    seg: &Bound<'_, PyAny>,
    h: u32,
    w: u32,
) -> PyResult<Py<PyAny>> {
    fr_py_objects(py, seg, h, w)
}
