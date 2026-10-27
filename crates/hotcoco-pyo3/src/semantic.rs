//! Python bindings for `hotcoco::semantic` — mIoU, fwIoU, mAcc, aAcc.
//!
//! `SemanticEval` is the driver object. Unlike `COCOeval` and `PanopticEval`
//! it is fed incrementally: a training loop calls `update(gt, pred)` per
//! batch, or `update_folders` reads saved PNG files, and every report method
//! reads the counts so far. Its whole state is one confusion matrix, so it
//! pickles and `merge`s, which is what a DDP validation loop needs.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use hotcoco_core::semantic::{METRIC_NAMES, SemanticEval};

use crate::convert::{confusion_counts_to_py, is_bool, label_pair};
use crate::{print_lines, provenance_str, serde_to_py, to_pyerr, warn_user};

#[doc = "Semantic segmentation evaluation: mIoU, fwIoU, mAcc, and aAcc against
mmsegmentation's protocol.

Parameters
----------
classes : int or sequence of str
    The number of classes, labeled ``0..classes - 1``, or their names in
    class-id order.
ignore_index : int or None, default 255
    The ground-truth label of an unlabeled pixel; such pixels are left out on
    both sides. ``None`` counts every pixel.
reduce_zero_label : bool, default False
    mmseg's rule for datasets whose files put \"other\" at ``0`` (ADE20K):
    ground-truth ``0`` becomes ``255`` and every other ground-truth label
    shifts down by one. Needs ``ignore_index=255``; the constructor raises
    ``ValueError`` otherwise.

Example
-------
::

    from hotcoco import semantic

    ev = semantic.SemanticEval(19)                 # Cityscapes train ids
    for gt, pred in loader:                         # (H, W) or (N, H, W) integer arrays
        ev.update(gt, pred.argmax(1).numpy())
    ev.summarize()                                  # per-class IoU and Acc, then the means
    ev.report()[\"metrics\"][\"mIoU\"]

From saved predictions, PNG files named like the ground truth's::

    ev = semantic.SemanticEval(150, reduce_zero_label=True)   # ADE20K
    ev.update_folders(\"ADEChallengeData2016/annotations/validation\", \"preds/\")

Every label must be below ``classes``, on either side, except a ground-truth
pixel equal to ``ignore_index``; one that is not raises ``ValueError`` naming
the pixel, and nothing from that pair is counted."]
#[pyclass(name = "SemanticEval", module = "hotcoco.semantic")]
pub(crate) struct PySemanticEval {
    inner: SemanticEval,
}

/// The evaluator for `classes`: a count names them the way core does, a
/// sequence names them itself.
fn evaluator_for(classes: &Bound<'_, PyAny>) -> PyResult<SemanticEval> {
    // `True` counts as 1 to `extract`, and a flag passed by mistake would
    // then be a one-class evaluator that rejects every label from 1 up.
    if !is_bool(classes)
        && let Ok(n) = classes.extract::<usize>()
    {
        return Ok(SemanticEval::new(n));
    }
    if let Ok(names) = classes.extract::<Vec<String>>()
        && !classes.is_instance_of::<pyo3::types::PyString>()
    {
        return Ok(SemanticEval::with_class_names(names));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "classes must be the number of classes or a sequence of class names",
    ))
}

#[pymethods]
impl PySemanticEval {
    #[new]
    #[pyo3(signature = (classes, *, ignore_index=Some(255), reduce_zero_label=false))]
    fn new(
        classes: &Bound<'_, PyAny>,
        ignore_index: Option<u64>,
        reduce_zero_label: bool,
    ) -> PyResult<Self> {
        let inner = evaluator_for(classes)?
            .ignore_index(ignore_index)
            .reduce_zero_label(reduce_zero_label);
        inner.check_options().map_err(to_pyerr)?;
        Ok(PySemanticEval { inner })
    }

    #[doc = "Count one ground-truth label map against one prediction.

Both are integer numpy arrays of the same shape — ``(H, W)``, or ``(N, H, W)``
for a batch, up to 32 dimensions — in any integer or bool dtype. A
C-contiguous, aligned ``uint8``, ``uint16``, ``uint64``, or ``int64`` array
is read in place; any other array is one numpy copy, ``uint64`` for an
unsigned dtype and ``int64`` otherwise, contiguous and aligned.
The counting itself releases the GIL. Raises
``ValueError`` on a shape mismatch or a label out of range; nothing from the
pair is counted then."]
    fn update(
        &mut self,
        py: Python<'_>,
        gt: &Bound<'_, PyAny>,
        pred: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let (g, p) = label_pair(gt, pred)?;
        let (gt, pred) = (g.map(), p.map());
        let inner = &mut self.inner;
        py.detach(|| inner.update_maps(&gt, &pred))
            .map_err(to_pyerr)
    }

    #[doc = "Count one pair of PNG label maps: 8- or 16-bit grayscale, or a palette
image whose index is the class, as Cityscapes, ADE20K, and VOC store them.
Raises ``OSError`` (``FileNotFoundError`` included) for a file that cannot be
read, and ``ValueError`` for one that is not a label map, two maps of
different sizes, or a label out of range."]
    fn update_png(&mut self, py: Python<'_>, gt: PathBuf, pred: PathBuf) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| inner.update_png(&gt, &pred)).map_err(to_pyerr)
    }

    #[doc = "Count every PNG in ``gt_folder`` against the file of the same name in
``pred_folder``, in parallel without the GIL. Returns how many pairs were
counted. Raises ``OSError`` for a folder that cannot be listed,
``ValueError`` if a ground-truth file has no prediction of the same name, or
whatever ``update_png`` would raise on the first pair it rejects; nothing is
counted then."]
    fn update_folders(
        &mut self,
        py: Python<'_>,
        gt_folder: PathBuf,
        pred_folder: PathBuf,
    ) -> PyResult<usize> {
        let inner = &mut self.inner;
        py.detach(|| inner.update_folders(&gt_folder, &pred_folder))
            .map_err(to_pyerr)
    }

    #[doc = "Add another evaluator's counts into this one — the shards of a
distributed run. Raises ``ValueError`` if it was built with different classes
or options."]
    fn merge(&mut self, other: &PySemanticEval) -> PyResult<()> {
        self.inner.merge(&other.inner).map_err(to_pyerr)
    }

    #[doc = "The confusion matrix behind every number, as a ``uint64`` array of
shape ``(num_classes, num_classes)``: row the ground-truth class, column the
predicted class, after ignoring."]
    fn confusion(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let m = self.inner.confusion();
        confusion_counts_to_py(py, m.counts().to_vec(), m.num_classes())
    }

    #[getter]
    #[doc = "The class names, in class-id order."]
    fn class_names(&self) -> Vec<String> {
        self.inner.class_names().to_vec()
    }

    #[getter]
    fn num_classes(&self) -> usize {
        self.inner.num_classes()
    }

    #[getter]
    #[doc = "Updates counted so far: one per ``update`` call, so an ``(N, H, W)``
batch is one, and one per PNG pair."]
    fn n_images(&self) -> usize {
        self.inner.n_images()
    }

    #[getter]
    #[doc = "The four headline values as fractions, in ``semantic.METRIC_NAMES``
order: mIoU, fwIoU, mAcc, aAcc. ``-1.0`` where nothing was counted."]
    fn stats(&self) -> Vec<f64> {
        self.inner.stats().to_vec()
    }

    #[doc = "Print the table: one row per class with IoU and Acc in percent, then
the four means. Comparability warnings are emitted first through
``warnings.warn``."]
    fn summarize(&self, py: Python<'_>) -> PyResult<()> {
        for w in self.inner.reference_deviations() {
            warn_user(py, &w)?;
        }
        print_lines(py, &self.inner.summarize_lines())
    }

    #[doc = "The table as a list of strings, without printing it."]
    fn summary_lines(&self) -> Vec<String> {
        self.inner.summarize_lines()
    }

    #[doc = "Results as a dict: ``mIoU``, ``fwIoU``, ``mAcc``, ``aAcc`` as fractions;
``per_class``, a list in class-id order of ``{\"name\", \"IoU\", \"Acc\", \"tp\",
\"gt_area\", \"pred_area\"}`` with ``-1.0`` for a score that is not computed;
``n_images``, ``n_pixels``, ``provenance``, ``reference_deviations`` and
``hotcoco_version``."]
    fn results(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.inner.results())
    }

    #[doc = "The run as the ``EvalReport`` dict every hotcoco family produces:
``task``, ``provenance``, ``metrics`` (the four means), ``per_class`` (``IoU``
and ``Acc`` by class name, computed values only), ``per_group`` (empty),
``curves`` (empty) and ``params``."]
    fn report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.inner.report())
    }

    #[doc = "Ways this run departs from what mmsegmentation would compute. Empty
means the numbers are comparable to the reference — every option here is a
reference option with the reference's meaning."]
    fn reference_deviations(&self) -> Vec<String> {
        self.inner.reference_deviations()
    }

    #[doc = "``'parity_verified'`` or ``'extension'`` — see ``reference_deviations()``."]
    fn provenance(&self) -> PyResult<String> {
        provenance_str(self.inner.provenance())
    }

    /// Pickle and `copy.deepcopy`, the way a DDP loop gathers shards: the
    /// whole state is the configuration plus one matrix, serialized as JSON.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let state = slf.borrow().__getstate__(py)?;
        let n = slf.borrow().inner.num_classes();
        Ok((slf.get_type(), (n,), state).into_pyobject(py)?.into_any())
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

pub fn register(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "semantic")?;
    m.add_class::<PySemanticEval>()?;
    m.add("METRIC_NAMES", METRIC_NAMES.to_vec())?;
    Ok(m)
}
