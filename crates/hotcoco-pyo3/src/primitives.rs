//! Python bindings for `hotcoco::primitives` — the matching kernels.
//!
//! The strict, typed layer: `bbox_iou` takes boxes, `mask_iou` takes RLEs, and
//! neither guesses which it was handed. `hotcoco.mask` mirrors
//! `pycocotools.mask` on top of these — its `iou` dispatches on input type the
//! way pycocotools does, then forwards here. The dispatch never moves down.
//!
//! The matchers take arrays in the caller's order and return indices or ids
//! in it: the ordering each kernel's contract asks for — ranked detections,
//! ignored ground truths last — is made here, the way the family drivers make
//! it, so a Python caller cannot hand a kernel the wrong order.

use numpy::{PyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use hotcoco_core::Rle;
use hotcoco_core::metrics::counts::score_order;
use hotcoco_core::metrics::semantic::{Label, LabelMap, LabelPairOp};
use hotcoco_core::panoptic::{MatchIds, SegmentInfo, labels_from_ids, match_image};
use hotcoco_core::params::default_iou_thrs;
use hotcoco_core::primitives::greedy::{
    self, GroupOfBoxes, GtMasks, ThreshMatrix, coco_match_floor,
};
use hotcoco_core::primitives::panoptic::{Overlaps, Side};
use hotcoco_core::primitives::{assign, sim};

use crate::convert::{
    boxes_arg, check_parallel, dict_list, extract_int, extract_rle_list, f64_matrix_arg, f64_vec,
    field_error, flag_vec, label_pair_or_lists, map_to_dict, py_to_segment, record_list,
    set_segment_matches, shaped_array, type_name,
};
use crate::to_pyerr;

/// `iscrowd` as one flag per ground truth — see [`flag_vec`] for the accepted
/// spellings; `None` is no crowd region.
fn extract_iscrowd(obj: Option<&Bound<'_, PyAny>>, gt_len: usize) -> PyResult<Vec<bool>> {
    let Some(obj) = obj else {
        return Ok(vec![false; gt_len]);
    };
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
    iscrowd: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let iscrowd = extract_iscrowd(iscrowd, gt.len())?;
    // All inputs are owned by now; the O(D*G) kernel runs GIL-free like the
    // COCOeval paths.
    let flat = py.detach(|| sim::mask_iou_flat(dt, gt, &iscrowd));
    shaped_array(py, flat, [dt.len(), gt.len()])
}

/// The box IoU kernel as a `(D, G)` `float64` array. Shared by
/// `primitives.bbox_iou` and the box branch of `mask.iou`.
pub(crate) fn box_iou_matrix(
    py: Python<'_>,
    dt: &[[f64; 4]],
    gt: &[[f64; 4]],
    iscrowd: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let iscrowd = extract_iscrowd(iscrowd, gt.len())?;
    let flat = sim::bbox_iou_flat(dt, gt, &iscrowd);
    shaped_array(py, flat, [dt.len(), gt.len()])
}

#[pyfunction]
#[pyo3(signature = (dt, gt, iscrowd=None))]
#[pyo3(text_signature = "(dt, gt, iscrowd=None)")]
#[doc = "Pairwise IoU between two sets of ``[x, y, w, h]`` boxes.

Args:
    dt: ``(D, 4)`` array of any integer or float dtype, or a sequence of
        4-element rows.
    gt: ``(G, 4)`` array of any integer or float dtype, or a sequence of
        4-element rows.
    iscrowd: One flag per ``gt`` box, or ``None`` for no crowd box. A crowd
        box scores intersection over the detection's area (IoA) instead of
        IoU, so leave this out only when no ground truth is a crowd region.

Returns:
    numpy.ndarray: ``float64``, shape ``(D, G)``.
"]
pub fn bbox_iou(
    py: Python<'_>,
    dt: &Bound<'_, PyAny>,
    gt: &Bound<'_, PyAny>,
    iscrowd: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let dt = boxes_arg(dt, "dt")?;
    let gt = boxes_arg(gt, "gt")?;
    box_iou_matrix(py, &dt, &gt, iscrowd)
}

#[pyfunction]
#[pyo3(signature = (dt, gt, iscrowd=None))]
#[pyo3(text_signature = "(dt, gt, iscrowd=None)")]
#[doc = "Pairwise IoU between two sets of RLE masks.

Args:
    dt: Sequence of RLE dicts.
    gt: Sequence of RLE dicts.
    iscrowd: One flag per ``gt`` mask, or ``None`` for no crowd mask. A crowd
        mask scores intersection over the detection's area (IoA) instead of
        IoU, so leave this out only when no ground truth is a crowd region.

Returns:
    numpy.ndarray: ``float64``, shape ``(D, G)``.
"]
pub fn mask_iou(
    py: Python<'_>,
    dt: &Bound<'_, PyAny>,
    gt: &Bound<'_, PyAny>,
    iscrowd: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let dt = extract_rle_list(dt)?;
    let gt = extract_rle_list(gt)?;
    rle_iou_matrix(py, &dt, &gt, iscrowd)
}

/// Reject a NaN in `values`, the argument `name`: a NaN cost, similarity, or
/// score leaves `what` undefined, so it is an error rather than an arbitrary
/// answer. Whether NaN is meaningful is a property of the problem, not of
/// extracting an array, so this stays with the matchers rather than living in
/// the readers.
fn check_no_nan(values: &[f64], name: &str, what: &str) -> PyResult<()> {
    if values.iter().any(|v| v.is_nan()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} contains NaN; {what} is undefined"
        )));
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
    check_no_nan(&flat, "cost", "the assignment")?;

    let (rows, cols) = assign::lsap(&flat, nr, nc, maximize);
    // `usize` implements `numpy::Element`, so these go straight out — remapping to
    // u64 first would allocate two more Vecs per call, and this is the primitive a
    // tracking loop calls once per frame.
    let rows = PyArray1::from_vec(py, rows);
    let cols = PyArray1::from_vec(py, cols);
    Ok((rows, cols).into_pyobject(py)?.into_any().unbind())
}

/// `iou_thrs` as thresholds, and whether it was one number — which drops the
/// threshold axis from every output, as a scalar `q` does in `numpy.quantile`.
fn thresholds_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<(Vec<f64>, bool)> {
    let Some(obj) = obj else {
        return Ok((default_iou_thrs(), false));
    };
    let scalar = match obj.cast::<numpy::PyUntypedArray>() {
        Ok(arr) => arr.ndim() == 0,
        Err(_) => obj.try_iter().is_err(),
    };
    if !scalar {
        return Ok((f64_vec(obj, "iou_thrs")?, false));
    }
    let t = obj.extract::<f64>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "iou_thrs must be a float or a sequence of floats, got {}",
            type_name(obj)
        ))
    })?;
    Ok((vec![t], true))
}

/// `greedy_match`'s per-ground-truth flag columns — `iscrowd`, `gt_ignore`,
/// and `group_of` — each read when given and filled with `false` otherwise,
/// all of one length: `sim`'s column count, or, when `sim` is `[]` and has
/// none, the length of the first flag column given. The columns given must
/// agree with each other and with `sim`.
fn gt_flags_args(
    iscrowd: Option<&Bound<'_, PyAny>>,
    gt_ignore: Option<&Bound<'_, PyAny>>,
    group_of: Option<&Bound<'_, PyAny>>,
    sim_shape: (usize, usize),
) -> PyResult<GtFlags> {
    let read = |obj: Option<&Bound<'_, PyAny>>, name| obj.map(|o| flag_vec(o, name)).transpose();
    let given = [
        ("iscrowd", read(iscrowd, "iscrowd")?),
        ("gt_ignore", read(gt_ignore, "gt_ignore")?),
        ("group_of", read(group_of, "group_of")?),
    ];
    let mut named = given
        .iter()
        .filter_map(|(name, f)| f.as_ref().map(|f| (*name, f)));
    let first = named.next();
    for (name, flags) in named {
        let (first_name, first_flags) = first.expect("a second named flag follows a first");
        check_parallel(flags.len(), first_flags.len(), name, first_name)?;
    }
    let g = match (sim_shape, first) {
        ((0, 0), Some((_, flags))) => flags.len(),
        ((_, g), _) => g,
    };
    let [iscrowd, gt_ignore, group_of] = given.map(|(name, flags)| match flags {
        Some(flags) => check_parallel(flags.len(), g, name, "sim's columns").map(|()| flags),
        None => Ok(vec![false; g]),
    });
    Ok(GtFlags {
        iscrowd: iscrowd?,
        gt_ignore: gt_ignore?,
        group_of: group_of?,
    })
}

/// `greedy_match`'s three per-ground-truth flag columns, each of length `g`.
struct GtFlags {
    iscrowd: Vec<bool>,
    gt_ignore: Vec<bool>,
    group_of: Vec<bool>,
}

/// A flat `[T * n]` buffer as a `(T, n)` array, or as `(n,)` when the caller
/// passed one threshold as a number.
fn per_threshold<T: numpy::Element>(
    py: Python<'_>,
    values: Vec<T>,
    t: usize,
    n: usize,
    scalar: bool,
) -> PyResult<Py<PyAny>> {
    if scalar {
        return Ok(PyArray1::from_vec(py, values).into_any().unbind());
    }
    shaped_array(py, values, [t, n])
}

#[pyfunction]
#[pyo3(
    signature = (sim, scores, iou_thrs=None, *, gt_ignore=None, iscrowd=None, dt_ignore=None, group_of=None),
    text_signature = "(sim, scores, iou_thrs=None, *, gt_ignore=None, iscrowd=None, dt_ignore=None, group_of=None)"
)]
#[doc = "COCO's greedy matching of detections to ground truths, pycocotools-exact.

The matcher behind ``COCOeval.evaluate()``, on arrays you supply. At each
threshold, detections take turns from the highest score down, ties in row
order as pycocotools' stable sort leaves them. Each takes the unmatched
ground truth it overlaps most, at or above the threshold; among equal
similarities the later column wins, as pycocotools' ``>=`` update does. An
ignored ground truth is a candidate only when no other one clears the
threshold. The matching is greedy, not optimal — that is the protocol; use
:func:`lsap` for an optimal assignment.

Args:
    sim: ``(D, G)`` similarity of each detection to each ground truth —
        :func:`bbox_iou` or :func:`mask_iou` output, or any array of any
        integer or float dtype, or a nested sequence.
    scores: ``(D,)`` detection confidence, which sets the matching order.
    iou_thrs: One threshold, or a sequence of them. Defaults to COCO's ten,
        0.50 to 0.95. A threshold of 1.0 is matched as ``1 - 1e-10``, as in
        pycocotools.
    gt_ignore: ``(G,)`` flags for ground truths that may absorb a detection
        but never count — outside the area range being scored, say.
    iscrowd: ``(G,)`` crowd flags. A crowd ground truth absorbs any number of
        detections, and is ignored whatever ``gt_ignore`` says, as in
        pycocotools. Pass the flags ``sim`` was computed with.
    dt_ignore: ``(D,)`` flags for detections ignored when nothing matches
        them — outside the area range being scored, say. A detection that
        matches takes its ground truth's flag instead, as in ``COCOeval``.
    group_of: ``(G,)`` Open Images group-of flags. A group-of box is worth one
        ground truth: the best-scoring detection inside it is a true positive,
        every other detection inside it is ignored, and an undetected box is
        a miss — the Challenge protocol, as ``COCOeval(oid_style=True)``
        matches it. Inside means intersection over the detection's area, the
        crowd measure, so compute ``sim`` with these flags as ``iscrowd``. A
        ground truth flagged here is held out of the matching above and takes
        detections it left unmatched; one flagged both crowd and group-of is
        a group-of box; one flagged here and in ``gt_ignore`` is held out of
        the count too, so the detections inside it are all ignored.

An image with no detections has a ``sim`` of ``(0, G)``. As ``[]`` it has no
columns to count, so the flags, when given, say how many ground truths there
are; without them, pass ``numpy.zeros((0, G))``, or ``gt_matches`` has no
entries to report the misses in.

Returns:
    dict: Three arrays, indexed like ``sim``'s rows and columns, with a leading
    threshold axis — dropped when ``iou_thrs`` is a single number:

    - ``dt_matches``: ``int64`` ``(T, D)``, the column each detection matched,
      or ``-1``.
    - ``gt_matches``: ``int64`` ``(T, G)``, the row that matched each ground
      truth, or ``-1``; for a crowd ground truth, the last one to, and for a
      group-of box the one credited with it.
    - ``dt_ignore``: ``bool`` ``(T, D)``, ``evalImgs``' ``dtIgnore``:
      detections matched to an ignored or crowd ground truth, every detection
      on a group-of box but its first, and unmatched ones flagged in the
      ``dt_ignore`` argument. These are neither true nor false positives.

    A detection is a true positive where ``dt_matches >= 0`` and not
    ``dt_ignore``, and a false positive where ``dt_matches == -1`` and not
    ``dt_ignore``.

Raises:
    ValueError: If ``sim``, ``scores``, or ``iou_thrs`` holds a NaN, or a
        length disagrees with ``sim``'s shape.

Example:
    >>> from hotcoco import primitives
    >>> sim = [[0.9, 0.6], [0.8, 0.0]]
    >>> m = primitives.greedy_match(sim, [0.5, 0.7], 0.5)
    >>> m['dt_matches'].tolist()   # row 1 scores higher and takes column 0 first
    [1, 0]
"]
fn greedy_match(
    sim: &Bound<'_, PyAny>,
    scores: &Bound<'_, PyAny>,
    iou_thrs: Option<&Bound<'_, PyAny>>,
    gt_ignore: Option<&Bound<'_, PyAny>>,
    iscrowd: Option<&Bound<'_, PyAny>>,
    dt_ignore: Option<&Bound<'_, PyAny>>,
    group_of: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let py = sim.py();
    let (flat, d, nc) = f64_matrix_arg(sim, "sim")?;
    check_no_nan(&flat, "sim", "the matching")?;
    let scores = f64_vec(scores, "scores")?;
    check_parallel(scores.len(), d, "scores", "sim's rows")?;
    check_no_nan(&scores, "scores", "the matching order")?;
    let (thrs, scalar) = thresholds_arg(iou_thrs)?;
    check_no_nan(&thrs, "iou_thrs", "the matching")?;
    let GtFlags {
        iscrowd,
        gt_ignore,
        group_of,
    } = gt_flags_args(iscrowd, gt_ignore, group_of, (d, nc))?;
    let g = gt_ignore.len();
    let dt_seed = match dt_ignore {
        Some(obj) => {
            let flags = flag_vec(obj, "dt_ignore")?;
            check_parallel(flags.len(), d, "dt_ignore", "sim's rows")?;
            flags
        }
        None => vec![false; d],
    };
    // A group-of box is ignored for the greedy phases and held out of the
    // second; its own pass is the one route to it, as in the detection driver.
    let ignored: Vec<bool> = (0..g)
        .map(|i| gt_ignore[i] || iscrowd[i] || group_of[i])
        .collect();
    let any_group_of = group_of.contains(&true);
    let t = thrs.len();

    let (dt_matches, gt_matches, dt_ignore) = py.detach(|| {
        // The order the matcher's contract asks for, made the way COCOeval
        // makes it: detections ranked by score with ties kept in row order,
        // ground truths stably partitioned non-ignored first.
        let dt_order = score_order(&scores);
        let gt_order: Vec<usize> = (0..g)
            .filter(|&i| !ignored[i])
            .chain((0..g).filter(|&i| ignored[i]))
            .collect();
        let num_not_ignored = ignored.iter().filter(|&&x| !x).count();
        let mut ranked = Vec::with_capacity(d * g);
        for &di in &dt_order {
            let row = &flat[di * g..(di + 1) * g];
            ranked.extend(gt_order.iter().map(|&gi| row[gi]));
        }
        let crowd_ranked: Vec<bool> = gt_order.iter().map(|&gi| iscrowd[gi]).collect();
        let floors: Vec<f64> = thrs.iter().map(|&t| coco_match_floor(t)).collect();
        // Open Images only: the group-of columns in rank order, their inverse
        // for the matcher's second phase, and which of them the count keeps.
        let group_of_ranked: Vec<bool> = gt_order.iter().map(|&gi| group_of[gi]).collect();
        let not_group_of: Vec<bool> = group_of_ranked.iter().map(|&x| !x).collect();
        let counted: Vec<bool> = gt_order.iter().map(|&gi| !gt_ignore[gi]).collect();
        let masks = GtMasks {
            rematchable: Some(&crowd_ranked),
            phase2_eligible: any_group_of.then_some(not_group_of.as_slice()),
        };
        let mut m = greedy::greedy_match_masked(&ranked, d, g, num_not_ignored, masks, &floors);
        let mut credited = ThreshMatrix::default();
        if any_group_of {
            let boxes = GroupOfBoxes {
                group_of: &group_of_ranked,
                counted: &counted,
            };
            greedy::match_group_of(&ranked, d, g, boxes, &floors, &mut m, &mut credited);
        }

        let mut dt_matches = vec![-1i64; t * d];
        let mut gt_matches = vec![-1i64; t * g];
        // Seeded per threshold, as COCOeval seeds `dtIgnore`; a match
        // overwrites it with the ground truth's flag.
        let mut dt_ignore = dt_seed.repeat(t);
        for (ti, row) in m.dt_gt.iter_rows().enumerate() {
            for (rank, (&di, &matched)) in dt_order.iter().zip(row).enumerate() {
                let Some(pos) = matched else {
                    continue;
                };
                let gi = gt_order[pos];
                dt_matches[ti * d + di] = gi as i64;
                let credited_here = any_group_of && credited[(ti, rank)];
                dt_ignore[ti * d + di] = pos >= num_not_ignored && !credited_here;
                // Rank order, so a crowd ground truth keeps the last detection;
                // a group-of box keeps the one credited with it.
                if credited_here || !group_of_ranked[pos] {
                    gt_matches[ti * g + gi] = di as i64;
                }
            }
        }
        (dt_matches, gt_matches, dt_ignore)
    });

    let out = PyDict::new(py);
    out.set_item("dt_matches", per_threshold(py, dt_matches, t, d, scalar)?)?;
    out.set_item("gt_matches", per_threshold(py, gt_matches, t, g, scalar)?)?;
    out.set_item("dt_ignore", per_threshold(py, dt_ignore, t, d, scalar)?)?;
    Ok(out.into_any().unbind())
}

/// A label as a 32-bit segment id, or `None` when it is negative or wider.
fn segment_label<L: Label>(label: L) -> Option<u32> {
    label.index().and_then(|v| u32::try_from(v).ok())
}

/// The overlap histogram of one label-map pair, for [`LabelMap::with_pair`];
/// on a label no 32-bit segment id can be, the side, the pixel, and the value.
struct CountOverlaps;

impl LabelPairOp for CountOverlaps {
    type Output = Result<Overlaps, (&'static str, usize, Option<u64>)>;

    fn run<G: Label, P: Label>(self, gt: &[G], pred: &[P]) -> Self::Output {
        Overlaps::compute_labels(gt, pred, segment_label, segment_label).map_err(|(side, i)| {
            match side {
                Side::Gt => ("gt", i, gt[i].index()),
                Side::Pred => ("pred", i, pred[i].index()),
            }
        })
    }
}

#[pyfunction]
#[pyo3(text_signature = "(gt, pred)")]
#[doc = "Pixel counts of every ``(gt_id, pred_id)`` pair in two panoptic label maps.

The similarity step of panoptic quality: everything PQ needs about geometry is
in this table. It is panopticapi's ``gt_pred_map``, which panopticapi builds
with ``np.unique(gt * OFFSET + pred, return_counts=True)``.

Args:
    gt: Ground-truth segment ids, ``0`` for void — an integer or bool array of
        any shape, or a nested list. A COCO panoptic PNG holds colors, not
        ids: decode it with ``R + 256 G + 256² B`` (panopticapi's ``rgb2id``)
        first.
    pred: Predicted segment ids, the same shape.

Returns:
    dict: ``{(gt_id, pred_id): pixels}`` for every pair that shares a pixel,
    void included, ordered by ``(gt_id, pred_id)``.

Raises:
    TypeError: If a map is not an integer or bool array.
    ValueError: If the shapes differ, or an id is negative or does not fit in
        32 bits.

Example:
    >>> import numpy as np
    >>> from hotcoco import primitives
    >>> gt = np.array([[1, 1, 0], [2, 2, 2]])
    >>> pred = np.array([[5, 5, 5], [5, 6, 6]])
    >>> primitives.segment_overlaps(gt, pred)
    {(0, 5): 1, (1, 5): 2, (2, 5): 1, (2, 6): 2}
"]
fn segment_overlaps(
    py: Python<'_>,
    gt: &Bound<'_, PyAny>,
    pred: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let (g, p) = label_pair_or_lists(gt, pred)?;
    let (gt, pred) = (g.map(), p.map());
    let overlaps = py
        .detach(|| LabelMap::with_pair(&gt, &pred, CountOverlaps))
        .map_err(|(side, i, value)| {
            pyo3::exceptions::PyValueError::new_err(match value {
                Some(v) => {
                    format!("{side} pixel {i} has label {v}; segment ids must fit in 32 bits")
                }
                None => format!("{side} pixel {i} has a negative label"),
            })
        })?;
    let out = map_to_dict(py, overlaps.iter().map(|(g, p, n)| ((g, p), n)))?;
    Ok(out.into_any().unbind())
}

/// `overlaps` as the histogram it spells: `{(gt_id, pred_id): pixels}`, each
/// entry read like a record's field, so a bad one is a `ValueError` naming it.
fn overlaps_arg(obj: &Bound<'_, PyAny>) -> PyResult<Overlaps> {
    let dict = obj.cast::<PyDict>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "overlaps must be a dict of {{(gt_id, pred_id): pixels}}, as segment_overlaps returns, got {}",
            type_name(obj)
        ))
    })?;
    let py = obj.py();
    let mut counts = Vec::with_capacity(dict.len());
    for (key, value) in dict {
        let entry = || -> PyResult<(u32, u32, u64)> {
            let (g, p): (Bound<'_, PyAny>, Bound<'_, PyAny>) = key.extract().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("key must be a (gt_id, pred_id) pair")
            })?;
            Ok((extract_int(&g)?, extract_int(&p)?, extract_int(&value)?))
        };
        let entry = entry().map_err(|e| {
            let key = key
                .repr()
                .map_or_else(|_| "?".to_owned(), |r| r.to_string());
            field_error(py, format_args!("overlaps[{key}]"), e)
        })?;
        counts.push(entry);
    }
    // Every area the matcher takes is a sum of these counts, so a total that
    // fits in 64 bits keeps every one of them exact.
    if counts
        .iter()
        .try_fold(0u64, |total, &(_, _, n)| total.checked_add(n))
        .is_none()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "overlaps counts more pixels than 2**64 - 1, more than any pair of label maps holds",
        ));
    }
    Ok(Overlaps::from_counts(counts))
}

/// A `segments_info` list: a list or tuple of dicts, each read by the record
/// rules ([`py_to_segment`]).
fn segments_arg(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<SegmentInfo>> {
    let list = record_list(obj).ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a list of segments_info dicts, got {}",
            type_name(obj)
        ))
    })?;
    dict_list(&list, name, py_to_segment)
}

#[pyfunction]
#[pyo3(text_signature = "(gt_segments, pred_segments, overlaps)")]
#[doc = "Match one image's segments by panoptic quality's rules.

A ground-truth and a predicted segment match when their categories agree, the
ground truth is not crowd, and their IoU is above 0.5 — with the prediction's
pixels on void left out of the union. When areas are the pixel counts and no
segment is listed with the void id 0, a match above 0.5 is unique, so there is
nothing to rank or assign. A stated area below a segment's pixels, or the void
id listed as a ground truth, can match a segment twice, and panopticapi counts
both.
An unmatched prediction is ignored, rather than a
false positive, when more than half of it lies on void and on the crowd region
of its own category. These are panopticapi's rules, and ``PanopticEval`` runs
this same function on every image.

Args:
    gt_segments: The image's ground-truth ``segments_info``: dicts with ``id``
        and ``category_id``, optionally ``iscrowd`` and ``area``. ``area``
        defaults to the segment's pixels in ``overlaps``; panopticapi uses the
        JSON's when it states one, and takes it at its word — a stated area
        below the pixels found can give an IoU above 1. ``bbox`` and
        ``segmentation`` are checked by the rules a ``COCO`` dict's annotations
        are, and other keys are dropped. An id listed twice keeps its last entry at its first
        position, as panopticapi's dict over the list does.
    pred_segments: The predicted ``segments_info``, the same shape. A
        prediction's area is always its pixel count in ``overlaps``, and a
        prediction is never crowd.
    overlaps: ``{(gt_id, pred_id): pixels}``, as :func:`segment_overlaps`
        returns — or panopticapi's own ``gt_pred_map``.

Returns:
    dict: Segment ids, by outcome:

    - ``matched``: ``(gt_id, pred_id, iou)`` per matched pair, ordered by
      ground-truth id — the order ``PanopticEval`` sums their IoU in.
    - ``missed``: unmatched ground truths, crowd excluded.
    - ``false_positives``: unmatched predictions that count against the model.
    - ``ignored``: unmatched predictions that count for nothing.

    The last three keep ``segments_info`` order. Per category, ``tp`` is the
    number of matched pairs, ``fp`` of false positives, ``fn`` of misses, and
    ``iou_sum`` the matched IoUs summed — the input to
    ``metrics.panoptic_quality``.

Raises:
    ValueError: If a segment dict is malformed; if a predicted id has pixels
        in ``overlaps`` but no entry, or an entry but no pixels, both errors in
        panopticapi too; or if a matched pair has no union, so its IoU is not a
        number — a stated ``area`` below the segment's pixels does that, as
        can the void id 0 listed as a segment, which panopticapi scores as one.
    TypeError: If ``overlaps`` is not a dict.

Example:
    >>> import numpy as np
    >>> from hotcoco import primitives
    >>> gt = np.array([[1, 1, 1, 1], [2, 2, 0, 0]])
    >>> pred = np.array([[7, 7, 7, 0], [8, 8, 8, 8]])
    >>> overlaps = primitives.segment_overlaps(gt, pred)
    >>> primitives.match_segments(
    ...     [{\"id\": 1, \"category_id\": 1}, {\"id\": 2, \"category_id\": 2}],
    ...     [{\"id\": 7, \"category_id\": 1}, {\"id\": 8, \"category_id\": 2}],
    ...     overlaps,
    ... )
    {'matched': [(1, 7, 0.75), (2, 8, 1.0)], 'missed': [], 'false_positives': [], 'ignored': []}
"]
fn match_segments(
    py: Python<'_>,
    gt_segments: &Bound<'_, PyAny>,
    pred_segments: &Bound<'_, PyAny>,
    overlaps: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let gt_infos = segments_arg(gt_segments, "gt_segments")?;
    let pred_infos = segments_arg(pred_segments, "pred_segments")?;
    let overlaps = overlaps_arg(overlaps)?;
    // The maps are painted with ids, as PanopticEval's PNG files are; a spare
    // label for an id past 32 bits must also miss every label in the maps.
    let gt = labels_from_ids(&gt_infos, |l| overlaps.iter().any(|(g, _, _)| g == l));
    let pred = labels_from_ids(&pred_infos, |l| overlaps.iter().any(|(_, p, _)| p == l));
    let m = match_image(&gt, &pred, &overlaps).map_err(to_pyerr)?;
    let out = PyDict::new(py);
    set_segment_matches(&out, &MatchIds::from_indices(&m, &gt, &pred))?;
    Ok(out.into_any().unbind())
}

/// Build the `hotcoco.primitives` submodule.
pub fn register(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "primitives")?;
    m.add_function(wrap_pyfunction!(lsap, &m)?)?;
    m.add_function(wrap_pyfunction!(bbox_iou, &m)?)?;
    m.add_function(wrap_pyfunction!(mask_iou, &m)?)?;
    m.add_function(wrap_pyfunction!(greedy_match, &m)?)?;
    m.add_function(wrap_pyfunction!(segment_overlaps, &m)?)?;
    m.add_function(wrap_pyfunction!(match_segments, &m)?)?;
    Ok(m)
}
