//! Python bindings for `hotcoco::metrics` — metric functions over flat arrays.
//!
//! These take lists or numpy arrays and return plain Python values. Nothing here
//! needs a `COCOeval`, which is the point: the same `average_precision` that
//! backs COCO's AP is callable on any `(scores, matched)` pair you have.

use pyo3::prelude::*;
use pyo3::types::PyList;

use hotcoco_core::metrics::panoptic::PqCounts;
use hotcoco_core::metrics::semantic::PixelConfusion;
use hotcoco_core::metrics::{calibration as rcal, confusion as rconf, counts as rcounts};
use hotcoco_core::params::default_rec_thrs;

use crate::convert::{
    Labels, bool_vec, calibration_bin_to_py, check_parallel, confusion_counts_to_py, f64_vec,
    label_pair, with_label_pair,
};
use crate::to_pyerr;

#[pyfunction]
#[pyo3(
    signature = (scores, matched, num_gt, ignored=None, rec_thrs=None),
    text_signature = "(scores, matched, num_gt, ignored=None, rec_thrs=None)"
)]
#[doc = "Average precision from per-prediction scores and match flags.

Sorts by score descending, classifies each prediction as a true or false
positive, and interpolates precision onto a fixed recall grid (PASCAL VOC
interpolation, exactly as pycocotools does it).

Args:
    scores: Confidence per prediction, in any order.
    matched: Whether each prediction is correct. Same length as ``scores``.
    num_gt: Total ground truths, including any never predicted. This is the
        recall denominator, so it must count *all* of them — not just the
        matched ones — or recall is overstated.
    ignored: Optional mask; ignored predictions count as neither TP nor FP.
    rec_thrs: Recall grid. Defaults to COCO's 101 points, 0.00 to 1.00.

Returns:
    float: AP in [0, 1]. Returns 0.0 when there are no predictions or no
    ground truth — if you want a different convention for the empty case
    (some metrics call an empty image perfect), branch before calling.

Example:
    >>> from hotcoco import metrics
    >>> round(metrics.average_precision([0.9, 0.8, 0.3], [True, False, True], num_gt=2), 4)
    0.835
"]
fn average_precision(
    scores: &Bound<'_, PyAny>,
    matched: &Bound<'_, PyAny>,
    num_gt: usize,
    ignored: Option<&Bound<'_, PyAny>>,
    rec_thrs: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let scores = f64_vec(scores, "scores")?;
    let matched = bool_vec(matched, "matched")?;
    let ignored = ignored.map(|o| bool_vec(o, "ignored")).transpose()?;
    let rec_thrs = rec_thrs.map(|o| f64_vec(o, "rec_thrs")).transpose()?;
    check_parallel(scores.len(), matched.len(), "scores", "matched")?;
    if let Some(ig) = &ignored {
        check_parallel(ig.len(), scores.len(), "ignored", "scores")?;
    }
    let thrs = rec_thrs.unwrap_or_else(default_rec_thrs);
    Ok(rcounts::average_precision(
        &scores,
        &matched,
        ignored.as_deref(),
        num_gt,
        &thrs,
    ))
}

#[pyfunction]
#[pyo3(
    signature = (tp_cum, fp_cum, num_gt, rec_thrs=None),
    text_signature = "(tp_cum, fp_cum, num_gt, rec_thrs=None)"
)]
#[doc = "Precision interpolated onto a recall grid, from cumulative TP/FP counts.

Lower level than :func:`average_precision` — use it when you already hold
cumulative counts, or want the curve rather than the scalar.

Args:
    tp_cum: Cumulative true positives, over predictions sorted by descending
        score. Must already be prefix-summed.
    fp_cum: Cumulative false positives, same ordering and length.
    num_gt: Total ground truths (the recall denominator).
    rec_thrs: Recall grid. Defaults to COCO's 101 points.

Returns:
    tuple: ``(final_recall, points)`` where ``points`` is a list of
    ``(threshold_index, precision, rank)``. Recall thresholds the predictions
    never reach are omitted rather than reported as zero, so ``points`` can be
    shorter than ``rec_thrs``.
"]
fn precision_recall_curve(
    py: Python<'_>,
    tp_cum: &Bound<'_, PyAny>,
    fp_cum: &Bound<'_, PyAny>,
    num_gt: usize,
    rec_thrs: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let tp_cum = f64_vec(tp_cum, "tp_cum")?;
    let fp_cum = f64_vec(fp_cum, "fp_cum")?;
    let rec_thrs = rec_thrs.map(|o| f64_vec(o, "rec_thrs")).transpose()?;
    check_parallel(tp_cum.len(), fp_cum.len(), "tp_cum", "fp_cum")?;
    let thrs = rec_thrs.unwrap_or_else(default_rec_thrs);
    let (final_recall, curve) = rcounts::precision_recall_curve(&tp_cum, &fp_cum, num_gt, &thrs);

    // Emit the same `(threshold_index, precision, rank)` tuples as before the
    // core function grew named `PrPoint` fields — the Python surface is stable.
    let points = PyList::empty(py);
    for p in curve {
        points.append((p.rec_thr_idx, p.precision, p.detection_rank))?;
    }
    Ok((final_recall, points)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

#[pyfunction]
#[pyo3(
    signature = (scores, matched, n_bins=10),
    text_signature = "(scores, matched, n_bins=10)"
)]
#[doc = "Reliability bins: predicted confidence against observed accuracy.

The data behind a reliability diagram. Buckets predictions into equal-width
confidence bins and reports, per bin, the mean confidence claimed and the
fraction actually correct. A perfectly calibrated model has the two equal in
every bin.

Args:
    scores: Confidence per prediction, in [0, 1].
    matched: Whether each prediction is correct.
    n_bins: Number of equal-width bins. Default 10.

Returns:
    list[dict]: One dict per bin with keys ``bin_lower``, ``bin_upper``,
    ``avg_confidence``, ``avg_accuracy``, ``count``. Empty bins are included
    with ``count = 0`` so the list always has ``n_bins`` entries and plots
    without gaps.
"]
fn calibration_curve(
    py: Python<'_>,
    scores: &Bound<'_, PyAny>,
    matched: &Bound<'_, PyAny>,
    n_bins: usize,
) -> PyResult<Py<PyAny>> {
    let scores = f64_vec(scores, "scores")?;
    let matched = bool_vec(matched, "matched")?;
    check_parallel(scores.len(), matched.len(), "scores", "matched")?;
    let bins = rcal::calibration_curve(&scores, &matched, n_bins);

    let out = PyList::empty(py);
    for bin in &bins {
        out.append(calibration_bin_to_py(py, bin)?)?;
    }
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(
    signature = (scores, matched, n_bins=10),
    text_signature = "(scores, matched, n_bins=10)"
)]
#[doc = "Expected and Maximum Calibration Error.

Args:
    scores: Confidence per prediction, in [0, 1].
    matched: Whether each prediction is correct.
    n_bins: Number of equal-width bins. Default 10.

Returns:
    tuple[float, float]: ``(ece, mce)``. ECE is the occupancy-weighted mean gap
    between confidence and accuracy — the headline number. MCE is the worst
    single bin's gap, which catches a badly calibrated region that ECE averages
    away. Both are 0.0 for empty input.

Example:
    >>> from hotcoco import metrics
    >>> # Always claims 0.9, right half the time.
    >>> ece, mce = metrics.calibration_error([0.9] * 100, [True] * 50 + [False] * 50)
    >>> round(ece, 3)
    0.4
"]
fn calibration_error(
    scores: &Bound<'_, PyAny>,
    matched: &Bound<'_, PyAny>,
    n_bins: usize,
) -> PyResult<(f64, f64)> {
    let scores = f64_vec(scores, "scores")?;
    let matched = bool_vec(matched, "matched")?;
    check_parallel(scores.len(), matched.len(), "scores", "matched")?;
    let bins = rcal::calibration_curve(&scores, &matched, n_bins);
    Ok(rcal::calibration_error(&bins))
}

#[pyfunction]
#[pyo3(
    signature = (gt, dt, num_classes),
    text_signature = "(gt, dt, num_classes)"
)]
#[doc = "Confusion counts over matched ground-truth/prediction pairs.

Unlike ``sklearn.metrics.confusion_matrix``, this handles predictions that
match nothing and ground truths that go unpredicted — the normal case in
detection and tracking. Pass ``None`` for the missing side, and it lands in the
background row or column.

Args:
    gt: Ground-truth class index per match record, or ``None`` for a spurious
        prediction.
    dt: Predicted class index per match record, or ``None`` for a missed
        ground truth. Same length as ``gt``.
    num_classes: Number of real classes. Index ``num_classes`` is background.

Returns:
    numpy.ndarray: ``(num_classes + 1, num_classes + 1)`` array of uint64.
    Rows are ground truth, columns are predictions. The last row holds false
    positives, the last column false negatives. Class indices outside
    ``range(num_classes)`` are dropped rather than raising.

Example:
    >>> from hotcoco import metrics
    >>> m = metrics.confusion_matrix([0, 1, None], [0, None, 1], num_classes=2)
    >>> m[0, 0], m[1, 2], m[2, 1]   # correct, missed, spurious
    (1, 1, 1)
"]
fn confusion_matrix(
    py: Python<'_>,
    gt: Vec<Option<usize>>,
    dt: Vec<Option<usize>>,
    num_classes: usize,
) -> PyResult<Py<PyAny>> {
    check_parallel(gt.len(), dt.len(), "gt", "dt")?;
    let flat = rconf::confusion_matrix(&gt, &dt, num_classes);
    confusion_counts_to_py(py, flat, num_classes + 1)
}

#[pyfunction]
#[pyo3(text_signature = "(iou_sum, tp, fp, fn_)")]
#[doc = "Panoptic quality from match counts: ``(PQ, SQ, RQ)``.

The formulas behind ``hotcoco.panoptic``, for counts produced by your own
matcher::

    PQ = iou_sum / (tp + 0.5 fp + 0.5 fn)
    SQ = iou_sum / tp             (0.0 when tp == 0)
    RQ = tp / (tp + 0.5 fp + 0.5 fn)

Args:
    iou_sum: Summed IoU of the matched pairs.
    tp: Matched ground-truth segments.
    fp: Predicted segments that matched nothing and were not ignored.
    fn_: Ground-truth segments (not crowd) that nothing matched. Spelled with
        a trailing underscore because ``fn`` is a keyword in Rust, where the
        same counts live.

Returns:
    tuple: ``(pq, sq, rq)`` in [0, 1], or ``(-1.0, -1.0, -1.0)`` when
    ``tp + fp + fn == 0`` — nothing to score, which panopticapi leaves out of
    its averages. Average per-category results with ``is_computed`` as the
    filter to get panopticapi's ``All``/``Things``/``Stuff`` numbers.

Example:
    >>> from hotcoco import metrics
    >>> metrics.panoptic_quality(1.6, tp=2, fp=1, fn_=1)
    (0.5333333333333333, 0.8, 0.6666666666666666)
"]
fn panoptic_quality(iou_sum: f64, tp: u64, fp: u64, fn_: u64) -> (f64, f64, f64) {
    let s = PqCounts {
        iou: iou_sum,
        tp,
        fp,
        fn_,
    }
    .scores_or_missing();
    (s.pq, s.sq, s.rq)
}

#[pyfunction]
#[pyo3(
    signature = (gt, pred, num_classes, *, ignore_index=None),
    text_signature = "(gt, pred, num_classes, *, ignore_index=None)"
)]
#[doc = "The pixel confusion matrix of one label-map pair: a ``uint64`` array of
shape ``(num_classes, num_classes)``, row the ground-truth class, column the
predicted class.

The matrix behind ``hotcoco.semantic``, for a loop that keeps its own: sum the
arrays across images, then hand the total to :func:`mean_iou`.

Args:
    gt: Ground-truth class ids, an integer array of any shape.
    pred: Predicted class ids, the same shape.
    num_classes: Every label must be below this, except ``ignore_index`` in ``gt``.
    ignore_index: A ground-truth label whose pixels are left out on both sides.

Example:
    >>> from hotcoco import metrics
    >>> m = metrics.pixel_confusion([[0, 1], [1, 255]], [[0, 1], [0, 1]], num_classes=2, ignore_index=255)
    >>> m.tolist()
    [[1, 0], [1, 1]]
"]
fn pixel_confusion(
    py: Python<'_>,
    gt: &Bound<'_, PyAny>,
    pred: &Bound<'_, PyAny>,
    num_classes: usize,
    ignore_index: Option<u64>,
) -> PyResult<Py<PyAny>> {
    // Lists are welcome here as they are everywhere in `metrics`; the label
    // readers want an array, and `asarray` hands an array back unchanged.
    let asarray = py.import("numpy")?.getattr("asarray")?;
    let as_array =
        |obj: &Bound<'_, PyAny>| -> PyResult<Py<PyAny>> { Ok(asarray.call1((obj,))?.unbind()) };
    let (gt, pred) = (as_array(gt)?, as_array(pred)?);
    let (g, p) = label_pair(gt.bind(py), pred.bind(py))?;
    let mut m = PixelConfusion::new(num_classes);
    with_label_pair!(&g, &p, |g, p| m.update(g, p, ignore_index)).map_err(to_pyerr)?;
    confusion_counts_to_py(py, m.into_counts(), num_classes)
}

#[pyfunction]
#[doc = "mIoU, fwIoU, mAcc, and aAcc from a pixel confusion matrix.

Takes a square integer array — row the ground-truth class, column the
predicted class, as :func:`pixel_confusion` and ``SemanticEval.confusion()``
produce — and returns a dict::

    {\"mIoU\": 0.62, \"fwIoU\": 0.81, \"mAcc\": 0.74, \"aAcc\": 0.90,
     \"IoU\": [0.9, 0.5, ...], \"Acc\": [0.95, 0.6, ...]}   # per class, by id

mmsegmentation's conventions: a class on neither side has no IoU, one with no
ground truth has no accuracy, and either is ``-1.0`` here and left out of its
mean. A class predicted but never in the ground truth scores IoU ``0`` and
counts toward ``mIoU``.

Example:
    >>> from hotcoco import metrics
    >>> metrics.mean_iou([[3, 1], [0, 2]])[\"mIoU\"]
    0.7083333333333333
"]
fn mean_iou(py: Python<'_>, confusion: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (counts, rows, cols) = crate::convert::u64_matrix_arg(confusion, "confusion")?;
    if rows != cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "confusion must be square, got {rows}×{cols}"
        )));
    }
    let m = PixelConfusion::from_counts(rows, counts).map_err(to_pyerr)?;
    let s = m.scores();
    let d = pyo3::types::PyDict::new(py);
    d.set_item("mIoU", s.miou)?;
    d.set_item("fwIoU", s.fwiou)?;
    d.set_item("mAcc", s.macc)?;
    d.set_item("aAcc", s.aacc)?;
    d.set_item("IoU", s.per_class.iter().map(|c| c.iou).collect::<Vec<_>>())?;
    d.set_item("Acc", s.per_class.iter().map(|c| c.acc).collect::<Vec<_>>())?;
    Ok(d.into_any().unbind())
}

#[pyfunction]
#[doc = "Whether a metric value was actually computed.

hotcoco reports ``-1.0`` for a metric that was *not computed for this
configuration* — an area range with no ground truth, a category absent from the
split — never as a low score. This is the one place that convention is decided;
call it rather than re-deriving ``v >= 0.0`` by hand.

Example:
    >>> from hotcoco import metrics
    >>> [v for v in report['per_class']['dog'].values() if metrics.is_computed(v)]
"]
fn is_computed(v: f64) -> bool {
    hotcoco_core::metrics::is_computed(v)
}

#[pyfunction]
#[doc = "Whether a metric value is the 'not computed' sentinel (``-1.0``).

The negation of :func:`is_computed`, provided so filters read in either
direction without a hand-rolled comparison."]
fn is_missing(v: f64) -> bool {
    hotcoco_core::metrics::is_missing(v)
}

/// Build the `hotcoco.metrics` submodule.
pub fn register(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "metrics")?;
    m.add_function(wrap_pyfunction!(average_precision, &m)?)?;
    m.add_function(wrap_pyfunction!(precision_recall_curve, &m)?)?;
    m.add_function(wrap_pyfunction!(calibration_curve, &m)?)?;
    m.add_function(wrap_pyfunction!(calibration_error, &m)?)?;
    m.add_function(wrap_pyfunction!(confusion_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(panoptic_quality, &m)?)?;
    m.add_function(wrap_pyfunction!(pixel_confusion, &m)?)?;
    m.add_function(wrap_pyfunction!(mean_iou, &m)?)?;
    m.add_function(wrap_pyfunction!(is_computed, &m)?)?;
    m.add_function(wrap_pyfunction!(is_missing, &m)?)?;
    Ok(m)
}
