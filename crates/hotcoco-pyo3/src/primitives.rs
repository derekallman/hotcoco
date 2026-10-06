//! Python bindings for `hotcoco::primitives` — the matching kernels.
//!
//! The strict, typed layer: `bbox_iou` takes boxes, `mask_iou` takes RLEs, and
//! neither guesses which it was handed. `hotcoco.mask` mirrors
//! `pycocotools.mask` on top of these — its `iou` dispatches on input type the
//! way pycocotools does, then forwards here. The dispatch never moves down.

use numpy::PyArray1;
use pyo3::prelude::*;

use hotcoco_core::Rle;
use hotcoco_core::primitives::{assign, sim};

use crate::convert::{
    boxes_arg, check_parallel, extract_rle_list, f64_matrix, f64_matrix_arg, flag_vec,
};

/// `iscrowd` as one flag per ground truth — see [`flag_vec`] for the accepted
/// spellings.
fn extract_iscrowd(obj: &Bound<'_, PyAny>, gt_len: usize) -> PyResult<Vec<bool>> {
    let crowd = flag_vec(obj, "iscrowd")?;
    check_parallel(crowd.len(), gt_len, "iscrowd", "gt")?;
    Ok(crowd)
}

/// The RLE IoU kernel as a `(D, G)` `float64` array. Shared by
/// `primitives.mask_iou` and the RLE branch of `mask.iou`.
pub(crate) fn rle_iou_matrix(
    py: Python<'_>,
    dt: &[Rle],
    gt: &[Rle],
    iscrowd: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let iscrowd = extract_iscrowd(iscrowd, gt.len())?;
    // All inputs are owned by now; the O(D*G) kernel runs GIL-free like the
    // COCOeval paths.
    let result = py.detach(|| sim::mask_iou(dt, gt, &iscrowd));
    f64_matrix(py, &result, [dt.len(), gt.len()])
}

/// The box IoU kernel as a `(D, G)` `float64` array. Shared by
/// `primitives.bbox_iou` and the box branch of `mask.iou`.
pub(crate) fn box_iou_matrix(
    py: Python<'_>,
    dt: &[[f64; 4]],
    gt: &[[f64; 4]],
    iscrowd: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let iscrowd = extract_iscrowd(iscrowd, gt.len())?;
    let result = sim::bbox_iou(dt, gt, &iscrowd);
    f64_matrix(py, &result, [dt.len(), gt.len()])
}

#[pyfunction]
#[pyo3(text_signature = "(dt, gt, iscrowd)")]
#[doc = "Pairwise IoU between two sets of ``[x, y, w, h]`` boxes.

Args:
    dt: ``(D, 4)`` array of any integer or float dtype, or a sequence of
        4-element rows.
    gt: ``(G, 4)`` array of any integer or float dtype, or a sequence of
        4-element rows.
    iscrowd: One flag per ``gt`` box. A crowd box scores intersection over the
        detection's area (IoA) instead of IoU.

Returns:
    numpy.ndarray: ``float64``, shape ``(D, G)``.
"]
pub fn bbox_iou(
    py: Python<'_>,
    dt: &Bound<'_, PyAny>,
    gt: &Bound<'_, PyAny>,
    iscrowd: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let dt = boxes_arg(dt, "dt")?;
    let gt = boxes_arg(gt, "gt")?;
    box_iou_matrix(py, &dt, &gt, iscrowd)
}

#[pyfunction]
#[pyo3(text_signature = "(dt, gt, iscrowd)")]
#[doc = "Pairwise IoU between two sets of RLE masks.

Args:
    dt: Sequence of RLE dicts.
    gt: Sequence of RLE dicts.
    iscrowd: One flag per ``gt`` mask. A crowd mask scores intersection over
        the detection's area (IoA) instead of IoU.

Returns:
    numpy.ndarray: ``float64``, shape ``(D, G)``.
"]
pub fn mask_iou(
    py: Python<'_>,
    dt: &Bound<'_, PyAny>,
    gt: &Bound<'_, PyAny>,
    iscrowd: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let dt = extract_rle_list(dt)?;
    let gt = extract_rle_list(gt)?;
    rle_iou_matrix(py, &dt, &gt, iscrowd)
}

/// Reject a cost matrix holding a NaN — an unsolvable matrix is an error
/// rather than an arbitrary assignment. `lsap`-specific: whether NaN is
/// meaningful is a property of the assignment problem, not of extracting a
/// 2-D array, so this stays local rather than living in `f64_matrix_arg`.
fn check_no_nan(flat: &[f64]) -> PyResult<()> {
    if flat.iter().any(|v| v.is_nan()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cost contains NaN; the assignment is undefined",
        ));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(
    signature = (cost, maximize=false),
    text_signature = "(cost, maximize=False)"
)]
#[doc = "Optimal one-to-one assignment on a rectangular cost matrix.

Solves the linear sum assignment problem — the same thing
``scipy.optimize.linear_sum_assignment`` solves, and a semantic port of it, so
results agree including on ties. Use it when greedy matching is not good enough:
tracking associates detections to tracks this way, and it is what HOTA and MOTA
are defined against.

Args:
    cost: 2-D ``numpy.ndarray`` of costs, or a nested sequence (list of
        lists), ``cost[i][j]`` for row ``i`` and column ``j``. Rows need not
        equal columns; the smaller side bounds the assignment. An array of
        any integer or float dtype is the fast path — it is read in one
        pass, any strides; other arrays and nested sequences are converted
        element by element.
    maximize: Maximize total value instead of minimizing total cost. Pass
        ``True`` when the matrix holds similarities (IoU) rather than costs.

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: ``(row_ind, col_ind)``, the matched
    pairs. ``cost[row_ind[k]][col_ind[k]]`` is the k-th matched entry, and
    ``len(row_ind) == min(n_rows, n_cols)``.

Raises:
    ValueError: If ``cost`` is ragged or holds a NaN — an unsolvable matrix is
        an error rather than an arbitrary assignment.

Example:
    >>> from hotcoco import primitives
    >>> rows, cols = primitives.lsap([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> list(zip(rows.tolist(), cols.tolist()))   # total cost 1 + 2 + 2 = 5
    [(0, 1), (1, 0), (2, 2)]
"]
fn lsap(py: Python<'_>, cost: &Bound<'_, PyAny>, maximize: bool) -> PyResult<Py<PyAny>> {
    let (flat, nr, nc) = f64_matrix_arg(cost, "cost")?;
    check_no_nan(&flat)?;

    let (rows, cols) = assign::lsap(&flat, nr, nc, maximize);
    // `usize` implements `numpy::Element`, so these go straight out — remapping to
    // u64 first would allocate two more Vecs per call, and this is the primitive a
    // tracking loop calls once per frame.
    let rows = PyArray1::from_vec(py, rows);
    let cols = PyArray1::from_vec(py, cols);
    Ok((rows, cols).into_pyobject(py)?.into_any().unbind())
}

/// Build the `hotcoco.primitives` submodule.
pub fn register(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "primitives")?;
    m.add_function(wrap_pyfunction!(lsap, &m)?)?;
    m.add_function(wrap_pyfunction!(bbox_iou, &m)?)?;
    m.add_function(wrap_pyfunction!(mask_iou, &m)?)?;
    Ok(m)
}
