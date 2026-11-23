//! `hotcoco.Report`: what every evaluator's `compute()` returns.
//!
//! The Python face of [`EvalReport`]. Only an evaluator and `Report.load`
//! make one, so its provenance is the producer's, never a caller's claim; a
//! loaded report renders as recorded. Its containers are plain dicts and
//! arrays, built once: `report.metrics` is the same dict on every read, so it
//! works with `json.dumps` and pandas, and `to_dict()` and `save()` write
//! what those containers hold.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use hotcoco_core::EvalReport;

use crate::convert::{block_to_py, from_py_json};
use crate::{ReferenceDeviationWarning, provenance_str, to_pyerr, warn};

#[doc = "The result of an evaluation, the same shape for every family.

Made by an evaluator's ``compute()`` or by ``Report.load()``; it has no
public constructor, so ``provenance`` is always the producer's.

- ``task``: ``'detection'``, ``'panoptic'``, ``'semantic'``, or ``'tracking'``.
- ``metrics``: headline metric to value, ``None`` where nothing was computed,
  such as an area range with no ground truth.
- ``per_class`` and ``per_group``: name to metric to value. Groups are
  tracking's sequences, LVIS's frequency groups, panoptic's splits.
- ``curves``: name to a ``float64`` array; a point not computed is ``NaN``.
- ``params``: what produced the report.
- ``provenance``: ``'parity_verified'``, ``'extension'``, or ``'user_composed'``.
- ``deviations``: why the run is not comparable to the reference; empty when
  it is.
- ``hotcoco_version``: the version that produced it.

``print(report)`` prints the family's table. The containers are plain and
built once, so an edit to one is what ``to_dict()`` and ``save()`` write."]
#[pyclass(name = "Report", module = "hotcoco", frozen)]
pub(crate) struct PyReport {
    task: String,
    provenance: String,
    hotcoco_version: String,
    summary: Vec<String>,
    deviations: Py<PyList>,
    metrics: Py<PyDict>,
    per_class: Py<PyDict>,
    per_group: Py<PyDict>,
    curves: Py<PyDict>,
    params: Py<PyAny>,
}

impl PyReport {
    /// The Python containers for `report`.
    fn from_report(py: Python<'_>, report: &EvalReport) -> PyResult<Self> {
        let metrics = PyDict::new(py);
        for (name, value) in &report.metrics {
            metrics.set_item(name, *value)?;
        }
        let table = |rows: &BTreeMap<String, BTreeMap<String, f64>>| -> PyResult<Py<PyDict>> {
            let outer = PyDict::new(py);
            for (key, row) in rows {
                let inner = PyDict::new(py);
                for (metric, value) in row {
                    inner.set_item(metric, *value)?;
                }
                outer.set_item(key, inner)?;
            }
            Ok(outer.unbind())
        };
        let curves = PyDict::new(py);
        for (name, values) in &report.curves {
            let points: Vec<f64> = values
                .iter()
                .map(|&v| hotcoco_core::metrics::computed(v).unwrap_or(f64::NAN))
                .collect();
            curves.set_item(name, numpy::PyArray1::from_vec(py, points))?;
        }
        Ok(Self {
            task: report.task.clone(),
            provenance: provenance_str(report.provenance)?,
            hotcoco_version: report.hotcoco_version.clone(),
            summary: report.summary.clone(),
            deviations: PyList::new(py, &report.deviations)?.unbind(),
            metrics: metrics.unbind(),
            per_class: table(&report.per_class)?,
            per_group: table(&report.per_group)?,
            curves: curves.unbind(),
            params: block_to_py(py, &report.params)?.unbind(),
        })
    }

    /// What an evaluator's `compute()` returns: each deviation is emitted as a
    /// `ReferenceDeviationWarning` first, as `summarize()` emits them.
    pub(crate) fn computed(py: Python<'_>, report: &EvalReport) -> PyResult<Self> {
        for deviation in &report.deviations {
            warn::<ReferenceDeviationWarning>(py, deviation)?;
        }
        Self::from_report(py, report)
    }

    /// The containers as the core report, through the JSON `to_dict()` gives,
    /// so an edit that cannot be a report is refused here.
    fn to_core(&self, py: Python<'_>) -> PyResult<EvalReport> {
        from_py_json(self.to_dict(py)?.bind(py), "Report")
    }
}

#[pymethods]
impl PyReport {
    #[getter]
    fn task(&self) -> &str {
        &self.task
    }
    #[getter]
    fn provenance(&self) -> &str {
        &self.provenance
    }
    #[getter]
    fn hotcoco_version(&self) -> &str {
        &self.hotcoco_version
    }
    #[getter]
    fn deviations(&self, py: Python<'_>) -> Py<PyList> {
        self.deviations.clone_ref(py)
    }
    #[getter]
    fn metrics(&self, py: Python<'_>) -> Py<PyDict> {
        self.metrics.clone_ref(py)
    }
    #[getter]
    fn per_class(&self, py: Python<'_>) -> Py<PyDict> {
        self.per_class.clone_ref(py)
    }
    #[getter]
    fn per_group(&self, py: Python<'_>) -> Py<PyDict> {
        self.per_group.clone_ref(py)
    }
    #[getter]
    fn curves(&self, py: Python<'_>) -> Py<PyDict> {
        self.curves.clone_ref(py)
    }
    #[getter]
    fn params(&self, py: Python<'_>) -> Py<PyAny> {
        self.params.clone_ref(py)
    }

    /// The family's table, the lines ``summarize()`` prints.
    fn __str__(&self) -> String {
        self.summary.join("\n")
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "<Report task={} provenance={} metrics={}>",
            self.task,
            self.provenance,
            self.metrics.bind(py).len()
        )
    }

    #[doc = "The report as plain, JSON-ready containers: a copy, with ``None`` for
a metric or curve point that was not computed, and the printed table under
``summary``."]
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let deepcopy = py.import("copy")?.getattr("deepcopy")?;
        let curves = PyDict::new(py);
        for (name, values) in self.curves.bind(py) {
            let points: Vec<Option<f64>> = values
                .extract::<Vec<f64>>()?
                .into_iter()
                .map(|v| (!v.is_nan()).then_some(v))
                .collect();
            curves.set_item(name, points)?;
        }
        let d = PyDict::new(py);
        d.set_item("task", &self.task)?;
        d.set_item("provenance", &self.provenance)?;
        d.set_item("deviations", deepcopy.call1((self.deviations.bind(py),))?)?;
        d.set_item("metrics", deepcopy.call1((self.metrics.bind(py),))?)?;
        d.set_item("per_class", deepcopy.call1((self.per_class.bind(py),))?)?;
        d.set_item("per_group", deepcopy.call1((self.per_group.bind(py),))?)?;
        d.set_item("curves", curves)?;
        d.set_item("params", deepcopy.call1((self.params.bind(py),))?)?;
        d.set_item("summary", &self.summary)?;
        d.set_item("hotcoco_version", &self.hotcoco_version)?;
        Ok(d.unbind())
    }

    #[doc = "Write the report to ``path`` as JSON, the file ``Report.load()`` reads.
Raises ``ValueError`` if an edited container no longer makes a report, and
``OSError`` naming the path if it cannot be written."]
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let report = self.to_core(py)?;
        py.detach(|| report.save(&path)).map_err(to_pyerr)
    }

    #[doc = "Read a report ``save()`` wrote. Its provenance is the one recorded.
Raises ``FileNotFoundError`` or another ``OSError`` naming the path, and
``ValueError`` for a file that is not a report."]
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let report = py.detach(|| EvalReport::load(&path)).map_err(to_pyerr)?;
        Self::from_report(py, &report)
    }

    /// Unpickling: a report from the JSON `__reduce__` wrote.
    #[staticmethod]
    fn _from_json(py: Python<'_>, text: &str) -> PyResult<Self> {
        let report: EvalReport =
            serde_json::from_str(text).map_err(|e| to_pyerr(hotcoco_core::Error::Json(e)))?;
        Self::from_report(py, &report)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let text = serde_json::to_string(&slf.get().to_core(py)?)
            .map_err(|e| to_pyerr(hotcoco_core::Error::Json(e)))?;
        let rebuild = slf.get_type().getattr("_from_json")?;
        Ok((rebuild, (text,)).into_pyobject(py)?.into_any())
    }
}
