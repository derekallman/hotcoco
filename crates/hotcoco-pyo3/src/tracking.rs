//! Python bindings for `hotcoco::tracking` — HOTA, CLEAR, and Identity on
//! MOTChallenge 2D boxes.
//!
//! `TrackingEval` is fed one sequence at a time: each `update` evaluates its
//! sequence and keeps only the result, so the whole state is the
//! configuration plus a result per sequence. It pickles and `merge`s, as
//! `SemanticEval` does, which is what a sharded run needs.

use std::path::PathBuf;

use numpy::{PyArrayDescrMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use hotcoco_core::tracking::{Benchmark, METRIC_NAMES, TrackingEval, TrackingResult};

use crate::convert::{extract_int, f64_matrix_arg, is_bool, with_context};
use crate::{ReferenceDeviationWarning, print_lines, provenance_str, serde_to_py, to_pyerr, warn};

/// A result's [`METRIC_NAMES`] values, plus `Frames` for one sequence, under
/// TrackEval's names. A count is a Python `int`: whatever the core serializes
/// as an integer, so the binding keeps no list of which fields are counts.
fn summary_dict<'py>(py: Python<'py>, r: &TrackingResult) -> PyResult<Bound<'py, PyDict>> {
    let fields = serde_json::to_value(r).map_err(|e| to_pyerr(hotcoco_core::Error::Json(e)))?;
    let d = PyDict::new(py);
    for (name, value) in METRIC_NAMES.iter().zip(r.stats()) {
        match fields.get(name).and_then(serde_json::Value::as_u64) {
            Some(count) => d.set_item(name, count)?,
            None => d.set_item(name, value)?,
        }
    }
    if let Some(frames) = r.count.frames {
        d.set_item("Frames", frames)?;
    }
    Ok(d)
}

#[doc = "Multi-object tracking evaluation: HOTA, CLEAR (MOTA, MOTP), and Identity
(IDF1) against TrackEval, on MOTChallenge 2D boxes.

Parameters
----------
benchmark : str or None, default None
    The MOTChallenge clean-up to apply: ``\"MOT15\"``, ``\"MOT16\"``,
    ``\"MOT17\"``, or ``\"MOT20\"``, in any case. ``None`` evaluates every row
    as given, which is what a dataset of your own wants. An unknown name
    raises ``ValueError``.
clear_threshold : float, default 0.5
    The IoU at which CLEAR counts a match, TrackEval's ``THRESHOLD`` option.
identity_threshold : float, default 0.5
    The same for Identity. Either threshold outside 0 to 1 raises
    ``ValueError``.

Example
-------
::

    import numpy as np
    from hotcoco.tracking import TrackingEval

    ev = TrackingEval(\"MOT17\")
    gt = np.loadtxt(\"MOT17-02-FRCNN/gt/gt.txt\", delimiter=\",\", ndmin=2)
    pred = np.loadtxt(\"MyTracker/data/MOT17-02-FRCNN.txt\", delimiter=\",\", ndmin=2)
    ev.update(\"MOT17-02-FRCNN\", gt, pred, num_frames=600)
    ev.summarize()                  # TrackEval's HOTA, CLEAR, Identity, and Count tables
    ev.results()[\"HOTA\"]

A whole benchmark in the folder layout TrackEval reads::

    ev.update_folders(\"gt/mot_challenge/MOT17-train\", \"trackers/mot_challenge/MOT17-train/MyTracker/data\")
"]
#[pyclass(name = "TrackingEval", module = "hotcoco.tracking")]
pub(crate) struct PyTrackingEval {
    inner: TrackingEval,
}

impl PyTrackingEval {
    /// `RuntimeError`, the core's `InvalidState`, before any sequence is added.
    fn check_updated(&self) -> PyResult<()> {
        self.inner.results().map(drop).map_err(to_pyerr)
    }
}

#[pymethods]
impl PyTrackingEval {
    #[new]
    #[pyo3(signature = (benchmark=None, *, clear_threshold=0.5, identity_threshold=0.5))]
    fn new(
        benchmark: Option<&str>,
        clear_threshold: f64,
        identity_threshold: f64,
    ) -> PyResult<Self> {
        let benchmark = benchmark
            .map(str::parse::<Benchmark>)
            .transpose()
            .map_err(to_pyerr)?;
        let inner =
            TrackingEval::new(benchmark, clear_threshold, identity_threshold).map_err(to_pyerr)?;
        Ok(PyTrackingEval { inner })
    }

    #[doc = "Evaluate one sequence and keep its result.

``gt`` and ``pred`` are 2-D arrays of MOTChallenge rows, one box per row —
``frame, id, x, y, w, h, conf, class, visibility``, as ``gt.txt`` and a
tracker's file hold them — as a numpy array of any integer or float dtype or
a list of rows. Ground truth needs the first 8 columns with a benchmark and 6
without; predictions need 6, and a prediction row without a class column is a
pedestrian. Frames count from 1 to ``num_frames``, which the rows cannot say:
a frame with no boxes still counts. An empty array is a side with no boxes.

The rows are copied, and the evaluation runs without the GIL. Raises
``ValueError``, naming the sequence, for a name already evaluated, too few
columns, a frame outside ``1..num_frames``, a ``num_frames`` too large to
index, a frame or track id that is fractional or negative, a box or confidence
that is not finite, a box whose far edge or area passes the largest float, a
track id twice in one frame, or a benchmark's class rule: a prediction class
above 1, or a ground-truth class outside 1..13 in a frame with predictions.
Nothing is kept then."]
    #[pyo3(signature = (name, gt, pred, *, num_frames))]
    fn update(
        &mut self,
        py: Python<'_>,
        name: &str,
        gt: &Bound<'_, PyAny>,
        pred: &Bound<'_, PyAny>,
        num_frames: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let num_frames: usize =
            extract_int(num_frames).map_err(|e| with_context(py, "num_frames", e))?;
        let (gt, gt_ncols) = track_rows(gt, "gt")?;
        let (pred, pred_ncols) = track_rows(pred, "pred")?;
        let inner = &mut self.inner;
        py.detach(|| inner.update(name, &gt, gt_ncols, &pred, pred_ncols, num_frames))
            .map_err(to_pyerr)
    }

    #[doc = "Evaluate every sequence of a benchmark laid out as TrackEval reads it,
in parallel without the GIL. Returns how many sequences were evaluated.

Sequence ``s`` is ``{gt_folder}/{s}/gt/gt.txt``, its frame count the
``seqLength`` in ``{gt_folder}/{s}/seqinfo.ini``, against the predictions in
``{pred_folder}/{s}.txt``. The sequences are the ones a TrackEval ``seqmap``
file names — a header line, then a name per line — or, without one, every
folder in ``gt_folder`` whose name does not start with a dot. A file is
comma-separated, or whitespace-separated on a line with no comma.

Raises ``FileNotFoundError`` (an ``OSError``) naming a folder or file that is
missing, and ``ValueError`` for a line that is not numbers, a ``seqinfo.ini``
with no ``seqLength``, a sequence already evaluated, or anything ``update``
refuses. Nothing is kept then."]
    #[pyo3(signature = (gt_folder, pred_folder, *, seqmap=None))]
    fn update_folders(
        &mut self,
        py: Python<'_>,
        gt_folder: PathBuf,
        pred_folder: PathBuf,
        seqmap: Option<PathBuf>,
    ) -> PyResult<usize> {
        let inner = &mut self.inner;
        py.detach(|| inner.update_folders(&gt_folder, &pred_folder, seqmap.as_deref()))
            .map_err(to_pyerr)
    }

    #[doc = "Add another evaluator's sequences to this one — the shards of a
distributed run. Raises ``ValueError`` if the two differ in benchmark or
thresholds, share a sequence name, or are the same evaluator; nothing is
added then."]
    fn merge(slf: &Bound<'_, Self>, other: &Bound<'_, PyTrackingEval>) -> PyResult<()> {
        // Borrowing one object both ways is a `RuntimeError` from the cell,
        // which names neither the argument nor what is wrong with it.
        if slf.is(other) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "an evaluator cannot merge with itself; merge takes the other shards",
            ));
        }
        slf.borrow_mut()
            .inner
            .merge(&other.borrow().inner)
            .map_err(to_pyerr)
    }

    #[getter]
    #[doc = "The benchmark whose clean-up this run applies, such as ``\"MOT17\"``, or
``None``."]
    fn benchmark(&self) -> Option<&'static str> {
        self.inner.benchmark().map(Benchmark::as_str)
    }

    #[getter]
    fn clear_threshold(&self) -> f64 {
        self.inner.clear_threshold()
    }

    #[getter]
    fn identity_threshold(&self) -> f64 {
        self.inner.identity_threshold()
    }

    #[getter]
    #[doc = "The combined run's 44 values as floats, in ``tracking.METRIC_NAMES``
order: ``results()`` without the names. Raises ``RuntimeError`` before any
sequence is added."]
    fn stats(&self) -> PyResult<Vec<f64>> {
        Ok(self.inner.stats().map_err(to_pyerr)?.to_vec())
    }

    #[doc = "Print TrackEval's four tables — HOTA, CLEAR, Identity, and Count — with a
row per sequence and a ``COMBINED`` row, scores in percent. Each entry of
``reference_deviations()`` is emitted first as a ``ReferenceDeviationWarning``.
Raises ``RuntimeError`` before any sequence is added."]
    fn summarize(&self, py: Python<'_>) -> PyResult<()> {
        self.check_updated()?;
        for w in self.inner.reference_deviations() {
            warn::<ReferenceDeviationWarning>(py, &w)?;
        }
        print_lines(py, &self.inner.summarize_lines())
    }

    #[doc = "The tables as a list of strings, without printing them. Raises
``RuntimeError`` before any sequence is added."]
    fn summary_lines(&self) -> PyResult<Vec<String>> {
        self.check_updated()?;
        Ok(self.inner.summarize_lines())
    }

    #[doc = "The combined run, TrackEval's ``COMBINED_SEQ``, as a flat dict:
``hotcoco_version``, ``provenance``, and ``reference_deviations``, as every
family's ``results()`` carries them, then the 44 values under TrackEval's
field names, in ``METRIC_NAMES`` order.

``HOTA``, ``DetA``, ``AssA``, ``DetRe``, ``DetPr``, ``AssRe``, ``AssPr``,
``LocA``, and ``OWTA`` are the mean over HOTA's 19 localization thresholds,
as TrackEval's summary takes them; ``curves()`` has the values at each
threshold. Then ``HOTA(0)``, ``LocA(0)``, and ``HOTALocA(0)``, the CLEAR
fields (``MOTA``, ``MOTP``, ... ``IDSW``, ``Frag``), the Identity fields
(``IDF1``, ``IDR``, ``IDP``, ``IDTP``, ``IDFN``, ``IDFP``), and the Count
fields (``Dets``, ``GT_Dets``, ``IDs``, ``GT_IDs``). Scores are fractions
and counts are ints. Raises ``RuntimeError`` before any sequence is added."]
    fn results<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metrics = summary_dict(py, &self.inner.results().map_err(to_pyerr)?)?;
        let d = PyDict::new(py);
        d.set_item("hotcoco_version", env!("CARGO_PKG_VERSION"))?;
        d.set_item("provenance", provenance_str(self.inner.provenance())?)?;
        d.set_item("reference_deviations", self.inner.reference_deviations())?;
        d.update(metrics.as_mapping())?;
        Ok(d)
    }

    #[doc = "Each sequence's result by name, in name order: the 44 values of
``results()`` plus ``Frames``, the sequence's frame count. A sequence with no
ground truth or no predictions left after clean-up has TrackEval's
early-return row: the CLEAR scores are 0 but ``MLR`` is 1 and ``CLR_Frames``
is 0, and ``LocA`` is 1. Raises ``RuntimeError`` before any sequence is added."]
    fn sequence_results<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.check_updated()?;
        let d = PyDict::new(py);
        for (name, r) in self.inner.sequence_results() {
            d.set_item(name, summary_dict(py, r)?)?;
        }
        Ok(d)
    }

    #[doc = "HOTA's per-threshold values, for the combined run or for the sequence
named: ``alphas``, the 19 localization thresholds from 0.05 to 0.95 as
TrackEval builds them, and one value per threshold for ``HOTA``, ``DetA``,
``AssA``, ``DetRe``, ``DetPr``, ``AssRe``, ``AssPr``, ``LocA``, ``OWTA``, and
the counts ``HOTA_TP``, ``HOTA_FN``, and ``HOTA_FP``, as floats. Raises
``RuntimeError`` before any sequence is added, and ``KeyError`` for a
sequence that was not."]
    #[pyo3(signature = (sequence=None))]
    fn curves<'py>(&self, py: Python<'py>, sequence: Option<&str>) -> PyResult<Bound<'py, PyDict>> {
        let curves = match sequence {
            None => self.inner.curves().map_err(to_pyerr)?,
            Some(name) => {
                self.check_updated()?;
                self.inner
                    .sequence_results()
                    .get(name)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "no sequence {name:?} was evaluated"
                        ))
                    })?
                    .curves()
            }
        };
        let d = PyDict::new(py);
        for (name, values) in curves {
            d.set_item(name, values)?;
        }
        Ok(d)
    }

    #[doc = "The run as the ``EvalReport`` dict every hotcoco family produces:
``task`` (``\"tracking\"``), ``provenance``, ``metrics`` (``results()``, every
value a float), ``per_class`` (empty), ``per_group`` (each sequence's
``METRIC_NAMES`` values under its name), ``curves`` (``curves()`` of the
combined run) and ``params`` (``benchmark``, ``clear_threshold``,
``identity_threshold``, ``n_sequences``, ``num_alphas``). Raises
``RuntimeError`` before any sequence is added."]
    fn report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.inner.report().map_err(to_pyerr)?)
    }

    #[doc = "Ways this run departs from what TrackEval would compute. Empty means the
numbers are comparable to the reference: each benchmark is TrackEval's
clean-up for it, and the two thresholds are TrackEval options. The one entry:
with no benchmark, a ground-truth row whose ``conf`` is 0 is evaluated, and
TrackEval drops such a row even with its clean-up off."]
    fn reference_deviations(&self) -> Vec<String> {
        self.inner.reference_deviations()
    }

    #[doc = "``'parity_verified'`` or ``'extension'`` — see ``reference_deviations()``."]
    fn provenance(&self) -> PyResult<String> {
        provenance_str(self.inner.provenance())
    }

    /// Pickle and `copy.deepcopy`, the way a DDP loop gathers shards: the
    /// whole state is the configuration and each sequence's result,
    /// serialized as JSON.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let state = slf.borrow().__getstate__(py)?;
        Ok((slf.get_type(), (), state).into_pyobject(py)?.into_any())
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = serde_json::to_vec(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner = serde_json::from_slice(state)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }
}

/// MOTChallenge rows as `(values, ncols)`, read as `f64_matrix_arg` reads a
/// matrix, but with a bool refused: a row is a record, and a bool is no
/// frame, id, or number in any record. The first bool cell of a list, or of
/// a bool or object array, is a `ValueError` naming it; a numeric array
/// holds none.
fn track_rows(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<(Vec<f64>, usize)> {
    let (values, _, ncols) = f64_matrix_arg(obj, name)?;
    let numeric = obj
        .cast::<numpy::PyUntypedArray>()
        .is_ok_and(|a| matches!(a.dtype().kind(), b'i' | b'u' | b'f'));
    if !numeric {
        for (i, row) in obj.try_iter()?.enumerate() {
            for (j, cell) in row?.try_iter()?.enumerate() {
                let cell = cell?;
                if is_bool(&cell) {
                    let err = pyo3::exceptions::PyValueError::new_err(format!(
                        "expected a number, got {}",
                        cell.repr()?
                    ));
                    return Err(with_context(
                        obj.py(),
                        format_args!("{name}[{i}][{j}]"),
                        err,
                    ));
                }
            }
        }
    }
    Ok((values, ncols))
}

pub fn register(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "tracking")?;
    m.add_class::<PyTrackingEval>()?;
    m.add("METRIC_NAMES", METRIC_NAMES.to_vec())?;
    Ok(m)
}
