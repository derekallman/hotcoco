//! Python bindings for `hotcoco::panoptic` — PQ, SQ, RQ.
//!
//! `PanopticEval` is the driver object, shaped like `COCOeval`; `pq_compute`
//! is the one-call form with panopticapi's signature and return value, for
//! code that imports `from panopticapi.evaluation import pq_compute`.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use hotcoco_core::panoptic::{METRIC_NAMES, PanopticDataset, PanopticEval};

use crate::convert::{from_py_json, py_to_dataset, type_name};
use crate::{PyCOCO, print_lines, provenance_str, serde_to_py, to_pyerr, warn_user};

/// Whether a dict is COCO panoptic JSON — its annotations carry
/// `segments_info` — rather than a detection-style dataset.
fn is_panoptic_shaped(dict: &Bound<'_, PyDict>) -> PyResult<bool> {
    let Some(anns) = dict.get_item("annotations")? else {
        return Ok(false);
    };
    let Ok(anns) = anns.cast::<PyList>() else {
        return Ok(false);
    };
    if anns.is_empty() {
        return Ok(false);
    }
    let first = anns.get_item(0)?;
    let Ok(first) = first.cast::<PyDict>() else {
        return Ok(false);
    };
    first.contains("segments_info")
}

/// Build one side's dataset from whatever Python handed over.
fn dataset_arg(
    obj: &Bound<'_, PyAny>,
    folder: Option<PathBuf>,
    name: &str,
) -> PyResult<PanopticDataset> {
    let masks_carry_no_folder = || {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{name}_folder applies to COCO panoptic JSON; a detection-style dataset carries its masks"
        ))
    };
    let dataset = if let Ok(coco) = obj.extract::<PyRef<'_, PyCOCO>>() {
        if folder.is_some() {
            return Err(masks_carry_no_folder());
        }
        PanopticDataset::from_dataset(&coco.inner.dataset)
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        if is_panoptic_shaped(dict)? {
            // A dict has no path to derive the folder from; without one its
            // segments must carry masks, and the run says so if they do not.
            from_py_json(dict, name)?
        } else {
            if folder.is_some() {
                return Err(masks_carry_no_folder());
            }
            PanopticDataset::from_dataset(&py_to_dataset(dict)?)
        }
    } else if let Ok(path) = obj.extract::<PathBuf>() {
        PanopticDataset::from_file(&path).map_err(to_pyerr)?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a panoptic JSON path (str or os.PathLike), a COCO dataset, or a dict, got {}",
            type_name(obj)
        )));
    };
    Ok(match folder {
        Some(folder) => dataset.with_folder(folder),
        None => dataset,
    })
}

#[doc = "Panoptic quality evaluation: PQ, SQ, and RQ against panopticapi.

Parameters
----------
gt : str, os.PathLike, COCO, or dict
    Ground truth. A path to a COCO panoptic JSON whose PNG files sit in
    ``gt_folder`` (default: the path without ``.json``, panopticapi's
    convention); or a ``COCO`` dataset whose annotations carry masks, one
    annotation per segment, which needs no PNG files; or a dict of either
    shape.
pred : same as ``gt``
    Predictions. Its ``categories`` are ignored; the ground truth's are
    scored, as in panopticapi.
gt_folder, pred_folder : str or os.PathLike, optional
    Where the PNG files are, for the JSON form.

Example
-------
::

    from hotcoco import panoptic

    ev = panoptic.PanopticEval(\"panoptic_val2017.json\", \"predictions.json\")
    ev.run()                 # prints the panopticapi table
    ev.results()[\"All\"][\"pq\"]
    ev.report()              # the EvalReport dict every family shares

Without PNG files, from detection-style datasets with RLE or polygon masks::

    gt = COCO(\"instances.json\")
    ev = panoptic.PanopticEval(gt, gt.load_res(\"segments.json\"))
"]
#[pyclass(name = "PanopticEval", module = "hotcoco.panoptic")]
pub(crate) struct PyPanopticEval {
    inner: PanopticEval,
}

#[pymethods]
impl PyPanopticEval {
    #[new]
    #[pyo3(signature = (gt, pred, *, gt_folder=None, pred_folder=None))]
    fn new(
        gt: &Bound<'_, PyAny>,
        pred: &Bound<'_, PyAny>,
        gt_folder: Option<PathBuf>,
        pred_folder: Option<PathBuf>,
    ) -> PyResult<Self> {
        let gt = dataset_arg(gt, gt_folder, "gt")?;
        let pred = dataset_arg(pred, pred_folder, "pred")?;
        Ok(PyPanopticEval {
            inner: PanopticEval::new(gt, pred),
        })
    }

    #[doc = "Evaluate every ground-truth image against its prediction.

Images run in parallel without the GIL. Raises ``RuntimeError`` on the
inputs panopticapi rejects: an image with no prediction, a predicted
segment in the JSON but not the PNG or the other way around, an unknown
prediction category, or label maps of different sizes."]
    fn evaluate(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| inner.run()).map_err(to_pyerr)
    }

    #[doc = "Print the summary table — PQ, SQ, RQ, N for All, Things, Stuff — in
percent, as panopticapi prints it. Comparability warnings are emitted first
through ``warnings.warn``."]
    fn summarize(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.inner.result().is_none() {
            warn_user(py, "hotcoco: summarize() called before evaluate().")?;
            return Ok(());
        }
        for w in self.inner.reference_deviations() {
            warn_user(py, &w)?;
        }
        print_lines(py, &self.inner.summarize_lines())
    }

    #[doc = "The summary table as a list of strings, without printing it. Empty
before ``evaluate()``."]
    fn summary_lines(&self) -> Vec<String> {
        self.inner.summarize_lines()
    }

    #[doc = "``evaluate()`` then ``summarize()``."]
    fn run(&mut self, py: Python<'_>) -> PyResult<()> {
        self.evaluate(py)?;
        self.summarize(py)
    }

    #[getter]
    #[doc = "The nine headline values, fractions in [0, 1]: PQ, SQ, RQ over all
categories, then over things, then over stuff — ``panoptic.METRIC_NAMES``
gives the order. ``-1.0`` where a split had no category to average. Empty before
``evaluate()``."]
    fn stats(&self) -> Vec<f64> {
        self.inner
            .result()
            .map(|r| r.stats().to_vec())
            .unwrap_or_default()
    }

    #[doc = "Results as a dict in panopticapi's shape.

``All``, ``Things`` and ``Stuff`` each map to ``{\"pq\", \"sq\", \"rq\", \"n\"}``
with the scores as fractions and ``n`` the categories averaged;
``per_class`` maps each ground-truth category id to its ``pq``, ``sq``, ``rq``
and, beyond panopticapi, the ``tp``, ``fp``, ``fn`` and summed ``iou`` they
came from. A category with no segment on either side reports ``-1.0`` for the
three scores, where panopticapi prints ``0.0`` and skips it in the mean —
the counts beside it are all zero either way. Also carries ``provenance``,
``reference_deviations`` and ``hotcoco_version``."]
    fn results(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let results = self.inner.results().map_err(to_pyerr)?;
        let dict = serde_to_py(py, &results)?;
        let dict = dict.bind(py).cast::<PyDict>()?;
        // JSON object keys are strings; panopticapi keys `per_class` by the
        // integer category id, and so does this.
        let per_class = PyDict::new(py);
        for (id, scores) in &results.per_class {
            per_class.set_item(id, serde_to_py(py, scores)?)?;
        }
        dict.set_item("per_class", per_class)?;
        Ok(dict.clone().into_any().unbind())
    }

    #[doc = "The run as the ``EvalReport`` dict every hotcoco family produces:
``task``, ``provenance``, ``metrics`` (the nine headline values), ``per_class``
(PQ/SQ/RQ by category name, evaluable categories only), ``per_group``
(``all``, ``things``, ``stuff`` with their ``n``), ``curves`` (empty) and
``params``."]
    fn report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let report = self.inner.report().map_err(to_pyerr)?;
        serde_to_py(py, &report)
    }

    #[doc = "Ways this run departs from what panopticapi would compute. Empty means
the numbers are comparable to the reference. Readable before ``evaluate()``."]
    fn reference_deviations(&self) -> Vec<String> {
        self.inner.reference_deviations()
    }

    #[doc = "``'parity_verified'`` or ``'extension'`` — see ``reference_deviations()``.
Readable before ``evaluate()``."]
    fn provenance(&self) -> PyResult<String> {
        provenance_str(self.inner.provenance())
    }
}

#[pyfunction]
#[pyo3(
    signature = (gt_json_file, pred_json_file, gt_folder=None, pred_folder=None),
    text_signature = "(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None)"
)]
#[doc = "panopticapi's ``pq_compute``: evaluate two COCO panoptic JSON files, print
the summary table, and return the results dict.

Same arguments and the same return shape (``All``/``Things``/``Stuff`` and
``per_class``) as ``panopticapi.evaluation.pq_compute``, with scores as
fractions. Folders default to each JSON path without its ``.json``. See
``PanopticEval.results()`` for the two additions to the dict and the one
difference in it."]
fn pq_compute(
    py: Python<'_>,
    gt_json_file: &Bound<'_, PyAny>,
    pred_json_file: &Bound<'_, PyAny>,
    gt_folder: Option<PathBuf>,
    pred_folder: Option<PathBuf>,
) -> PyResult<Py<PyAny>> {
    let mut ev = PyPanopticEval::new(gt_json_file, pred_json_file, gt_folder, pred_folder)?;
    ev.run(py)?;
    ev.results(py)
}

pub fn register(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "panoptic")?;
    m.add_class::<PyPanopticEval>()?;
    m.add_function(wrap_pyfunction!(pq_compute, &m)?)?;
    m.add("METRIC_NAMES", METRIC_NAMES.to_vec())?;
    Ok(m)
}
