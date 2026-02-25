use std::path::Path;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

mod convert;
mod mask;

use convert::{
    annotation_to_py, category_to_py, dataset_stats_to_py, image_to_py, py_to_annotation, rle_to_py,
};

// ---------------------------------------------------------------------------
// COCO
// ---------------------------------------------------------------------------

#[pyclass(name = "COCO")]
struct PyCOCO {
    inner: hotcoco_core::COCO,
}

impl Clone for PyCOCO {
    fn clone(&self) -> Self {
        PyCOCO {
            inner: hotcoco_core::COCO::from_dataset(self.inner.dataset.clone()),
        }
    }
}

#[pymethods]
impl PyCOCO {
    #[new]
    #[pyo3(signature = (annotation_file=None))]
    fn new(annotation_file: Option<&str>) -> PyResult<Self> {
        match annotation_file {
            Some(path) => {
                let inner = hotcoco_core::COCO::new(Path::new(path))
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
                Ok(PyCOCO { inner })
            }
            None => {
                let inner = hotcoco_core::COCO::from_dataset(hotcoco_core::Dataset {
                    info: None,
                    images: vec![],
                    annotations: vec![],
                    categories: vec![],
                    licenses: vec![],
                });
                Ok(PyCOCO { inner })
            }
        }
    }

    #[pyo3(signature = (img_ids=vec![], cat_ids=vec![], area_rng=None, iscrowd=None))]
    fn get_ann_ids(
        &self,
        img_ids: Vec<u64>,
        cat_ids: Vec<u64>,
        area_rng: Option<[f64; 2]>,
        iscrowd: Option<bool>,
    ) -> Vec<u64> {
        self.inner
            .get_ann_ids(&img_ids, &cat_ids, area_rng, iscrowd)
    }

    #[pyo3(signature = (cat_nms=vec![], sup_nms=vec![], cat_ids=vec![]))]
    fn get_cat_ids(
        &self,
        cat_nms: Vec<String>,
        sup_nms: Vec<String>,
        cat_ids: Vec<u64>,
    ) -> Vec<u64> {
        let cat_nms_ref: Vec<&str> = cat_nms.iter().map(|s| s.as_str()).collect();
        let sup_nms_ref: Vec<&str> = sup_nms.iter().map(|s| s.as_str()).collect();
        self.inner.get_cat_ids(&cat_nms_ref, &sup_nms_ref, &cat_ids)
    }

    #[pyo3(signature = (img_ids=vec![], cat_ids=vec![]))]
    fn get_img_ids(&self, img_ids: Vec<u64>, cat_ids: Vec<u64>) -> Vec<u64> {
        self.inner.get_img_ids(&img_ids, &cat_ids)
    }

    fn load_anns(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        let anns = self.inner.load_anns(&ids);
        let list = PyList::new(
            py,
            anns.iter()
                .map(|a| annotation_to_py(py, a))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(list.into_any().unbind())
    }

    fn load_cats(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        let cats = self.inner.load_cats(&ids);
        let list = PyList::new(
            py,
            cats.iter()
                .map(|c| category_to_py(py, c))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(list.into_any().unbind())
    }

    fn load_imgs(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        let imgs = self.inner.load_imgs(&ids);
        let list = PyList::new(
            py,
            imgs.iter()
                .map(|i| image_to_py(py, i))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(list.into_any().unbind())
    }

    fn load_res(&self, res_file: &str) -> PyResult<PyCOCO> {
        let inner = self
            .inner
            .load_res(Path::new(res_file))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(PyCOCO { inner })
    }

    fn ann_to_rle(&self, py: Python<'_>, ann: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let annotation = py_to_annotation(ann)?;
        match self.inner.ann_to_rle(&annotation) {
            Some(rle) => rle_to_py(py, &rle),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "Could not convert annotation to RLE (image not found?)",
            )),
        }
    }

    fn ann_to_mask<'py>(
        &self,
        py: Python<'py>,
        ann: &Bound<'py, PyDict>,
    ) -> PyResult<Py<PyArray2<u8>>> {
        let annotation = py_to_annotation(ann)?;
        let rle = self.inner.ann_to_rle(&annotation).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Could not convert annotation to RLE (image not found?)",
            )
        })?;
        let col_major = hotcoco_core::mask::decode(&rle);
        let h = rle.h as usize;
        let w = rle.w as usize;
        let arr = unsafe { PyArray2::new(py, [h, w], false) };
        unsafe {
            let ptr: *mut u8 = arr.as_raw_array_mut().as_mut_ptr();
            for y in 0..h {
                for x in 0..w {
                    *ptr.add(y * w + x) = col_major[y + h * x];
                }
            }
        }
        Ok(arr.unbind())
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let s = self.inner.stats();
        dataset_stats_to_py(py, &s)
    }

    /// Filter the dataset by category, image, and/or annotation area.
    ///
    /// Returns a new `COCO` with matching annotations. Images with no matching
    /// annotations are dropped unless `drop_empty_images=False`.
    #[pyo3(signature = (cat_ids=None, img_ids=None, area_rng=None, drop_empty_images=true))]
    fn filter(
        &self,
        cat_ids: Option<Vec<u64>>,
        img_ids: Option<Vec<u64>>,
        area_rng: Option<[f64; 2]>,
        drop_empty_images: bool,
    ) -> PyCOCO {
        let result = self.inner.filter(
            cat_ids.as_deref(),
            img_ids.as_deref(),
            area_rng,
            drop_empty_images,
        );
        PyCOCO {
            inner: hotcoco_core::COCO::from_dataset(result),
        }
    }

    /// Merge a list of `COCO` datasets into one.
    ///
    /// All datasets must share the same category taxonomy (same names and supercategories).
    /// Image and annotation IDs are remapped to be globally unique.
    /// Raises `ValueError` if the taxonomies differ.
    #[classmethod]
    fn merge(_cls: &Bound<'_, PyType>, datasets: Vec<PyRef<'_, PyCOCO>>) -> PyResult<PyCOCO> {
        let ds_refs: Vec<&hotcoco_core::Dataset> =
            datasets.iter().map(|p| &p.inner.dataset).collect();
        let result =
            hotcoco_core::COCO::merge(&ds_refs).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyCOCO {
            inner: hotcoco_core::COCO::from_dataset(result),
        })
    }

    /// Split the dataset into train/val (or train/val/test) subsets.
    ///
    /// Returns a 2-tuple `(train, val)` or 3-tuple `(train, val, test)`.
    /// The shuffle is deterministic for the same `seed`. All splits share the full category list.
    #[pyo3(signature = (val_frac=0.2, test_frac=None, seed=42))]
    fn split(
        &self,
        py: Python<'_>,
        val_frac: f64,
        test_frac: Option<f64>,
        seed: u64,
    ) -> PyResult<PyObject> {
        let (train_ds, val_ds, test_ds) = self.inner.split(val_frac, test_frac, seed);
        let train_py = Py::new(
            py,
            PyCOCO {
                inner: hotcoco_core::COCO::from_dataset(train_ds),
            },
        )?;
        let val_py = Py::new(
            py,
            PyCOCO {
                inner: hotcoco_core::COCO::from_dataset(val_ds),
            },
        )?;
        if let Some(test_ds) = test_ds {
            let test_py = Py::new(
                py,
                PyCOCO {
                    inner: hotcoco_core::COCO::from_dataset(test_ds),
                },
            )?;
            Ok(PyTuple::new(py, [train_py, val_py, test_py])?
                .into_any()
                .unbind())
        } else {
            Ok(PyTuple::new(py, [train_py, val_py])?.into_any().unbind())
        }
    }

    /// Sample a random subset of images with their annotations.
    ///
    /// Provide `n` for an exact count or `frac` for a fraction of total images.
    /// The sample is deterministic for the same `seed`.
    #[pyo3(signature = (n=None, frac=None, seed=42))]
    fn sample(&self, n: Option<usize>, frac: Option<f64>, seed: u64) -> PyCOCO {
        let result = self.inner.sample(n, frac, seed);
        PyCOCO {
            inner: hotcoco_core::COCO::from_dataset(result),
        }
    }

    /// Serialize the dataset to a COCO-format JSON file.
    fn save(&self, path: &str) -> PyResult<()> {
        let file = std::fs::File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.inner.dataset).map_err(
            |e: serde_json::Error| pyo3::exceptions::PyValueError::new_err(e.to_string()),
        )?;
        Ok(())
    }

    // camelCase aliases for pycocotools compatibility
    #[pyo3(name = "getAnnIds")]
    #[pyo3(signature = (img_ids=vec![], cat_ids=vec![], area_rng=None, iscrowd=None))]
    fn get_ann_ids_camel(
        &self,
        img_ids: Vec<u64>,
        cat_ids: Vec<u64>,
        area_rng: Option<[f64; 2]>,
        iscrowd: Option<bool>,
    ) -> Vec<u64> {
        self.get_ann_ids(img_ids, cat_ids, area_rng, iscrowd)
    }

    #[pyo3(name = "getCatIds")]
    #[pyo3(signature = (cat_nms=vec![], sup_nms=vec![], cat_ids=vec![]))]
    fn get_cat_ids_camel(
        &self,
        cat_nms: Vec<String>,
        sup_nms: Vec<String>,
        cat_ids: Vec<u64>,
    ) -> Vec<u64> {
        self.get_cat_ids(cat_nms, sup_nms, cat_ids)
    }

    #[pyo3(name = "getImgIds")]
    #[pyo3(signature = (img_ids=vec![], cat_ids=vec![]))]
    fn get_img_ids_camel(&self, img_ids: Vec<u64>, cat_ids: Vec<u64>) -> Vec<u64> {
        self.get_img_ids(img_ids, cat_ids)
    }

    #[pyo3(name = "loadAnns")]
    fn load_anns_camel(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        self.load_anns(py, ids)
    }

    #[pyo3(name = "loadCats")]
    fn load_cats_camel(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        self.load_cats(py, ids)
    }

    #[pyo3(name = "loadImgs")]
    fn load_imgs_camel(&self, py: Python<'_>, ids: Vec<u64>) -> PyResult<PyObject> {
        self.load_imgs(py, ids)
    }

    #[pyo3(name = "loadRes")]
    fn load_res_camel(&self, res_file: &str) -> PyResult<PyCOCO> {
        self.load_res(res_file)
    }

    #[pyo3(name = "annToRLE")]
    fn ann_to_rle_camel(&self, py: Python<'_>, ann: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        self.ann_to_rle(py, ann)
    }

    #[pyo3(name = "annToMask")]
    fn ann_to_mask_camel<'py>(
        &self,
        py: Python<'py>,
        ann: &Bound<'py, PyDict>,
    ) -> PyResult<Py<PyArray2<u8>>> {
        self.ann_to_mask(py, ann)
    }

    #[getter]
    fn dataset(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ds = &self.inner.dataset;
        let dict = PyDict::new(py);

        let images = PyList::new(
            py,
            ds.images
                .iter()
                .map(|i| image_to_py(py, i))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let annotations = PyList::new(
            py,
            ds.annotations
                .iter()
                .map(|a| annotation_to_py(py, a))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let categories = PyList::new(
            py,
            ds.categories
                .iter()
                .map(|c| category_to_py(py, c))
                .collect::<PyResult<Vec<_>>>()?,
        )?;

        dict.set_item("images", images)?;
        dict.set_item("annotations", annotations)?;
        dict.set_item("categories", categories)?;

        Ok(dict.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

#[pyclass(name = "Params")]
#[derive(Clone)]
struct PyParams {
    inner: hotcoco_core::Params,
}

#[pymethods]
impl PyParams {
    #[new]
    #[pyo3(signature = (iou_type="bbox"))]
    fn new(iou_type: &str) -> PyResult<Self> {
        let iou = parse_iou_type(iou_type)?;
        Ok(PyParams {
            inner: hotcoco_core::Params::new(iou),
        })
    }

    #[getter]
    fn iou_type(&self) -> &str {
        match self.inner.iou_type {
            hotcoco_core::IouType::Bbox => "bbox",
            hotcoco_core::IouType::Segm => "segm",
            hotcoco_core::IouType::Keypoints => "keypoints",
        }
    }

    #[setter]
    fn set_iou_type(&mut self, val: &str) -> PyResult<()> {
        self.inner.iou_type = parse_iou_type(val)?;
        Ok(())
    }

    #[getter]
    fn img_ids(&self) -> Vec<u64> {
        self.inner.img_ids.clone()
    }

    #[setter]
    fn set_img_ids(&mut self, val: Vec<u64>) {
        self.inner.img_ids = val;
    }

    #[getter]
    fn cat_ids(&self) -> Vec<u64> {
        self.inner.cat_ids.clone()
    }

    #[setter]
    fn set_cat_ids(&mut self, val: Vec<u64>) {
        self.inner.cat_ids = val;
    }

    #[getter]
    fn iou_thrs(&self) -> Vec<f64> {
        self.inner.iou_thrs.clone()
    }

    #[setter]
    fn set_iou_thrs(&mut self, val: Vec<f64>) {
        self.inner.iou_thrs = val;
    }

    #[getter]
    fn rec_thrs(&self) -> Vec<f64> {
        self.inner.rec_thrs.clone()
    }

    #[setter]
    fn set_rec_thrs(&mut self, val: Vec<f64>) {
        self.inner.rec_thrs = val;
    }

    #[getter]
    fn max_dets(&self) -> Vec<usize> {
        self.inner.max_dets.clone()
    }

    #[setter]
    fn set_max_dets(&mut self, val: Vec<usize>) {
        self.inner.max_dets = val;
    }

    #[getter]
    fn area_rng(&self) -> Vec<[f64; 2]> {
        self.inner.area_rng.clone()
    }

    #[setter]
    fn set_area_rng(&mut self, val: Vec<[f64; 2]>) {
        self.inner.area_rng = val;
    }

    #[getter]
    fn area_rng_lbl(&self) -> Vec<String> {
        self.inner.area_rng_lbl.clone()
    }

    #[setter]
    fn set_area_rng_lbl(&mut self, val: Vec<String>) {
        self.inner.area_rng_lbl = val;
    }

    #[getter]
    fn use_cats(&self) -> bool {
        self.inner.use_cats
    }

    #[setter]
    fn set_use_cats(&mut self, val: bool) {
        self.inner.use_cats = val;
    }

    #[getter]
    fn kpt_oks_sigmas(&self) -> Vec<f64> {
        self.inner.kpt_oks_sigmas.clone()
    }

    #[setter]
    fn set_kpt_oks_sigmas(&mut self, val: Vec<f64>) {
        self.inner.kpt_oks_sigmas = val;
    }

    // camelCase aliases for pycocotools compatibility
    #[getter(iouType)]
    fn iou_type_camel(&self) -> &str {
        self.iou_type()
    }

    #[setter(iouType)]
    fn set_iou_type_camel(&mut self, val: &str) -> PyResult<()> {
        self.set_iou_type(val)
    }

    #[getter(imgIds)]
    fn img_ids_camel(&self) -> Vec<u64> {
        self.img_ids()
    }

    #[setter(imgIds)]
    fn set_img_ids_camel(&mut self, val: Vec<u64>) {
        self.set_img_ids(val);
    }

    #[getter(catIds)]
    fn cat_ids_camel(&self) -> Vec<u64> {
        self.cat_ids()
    }

    #[setter(catIds)]
    fn set_cat_ids_camel(&mut self, val: Vec<u64>) {
        self.set_cat_ids(val);
    }

    #[getter(iouThrs)]
    fn iou_thrs_camel(&self) -> Vec<f64> {
        self.iou_thrs()
    }

    #[setter(iouThrs)]
    fn set_iou_thrs_camel(&mut self, val: Vec<f64>) {
        self.set_iou_thrs(val);
    }

    #[getter(recThrs)]
    fn rec_thrs_camel(&self) -> Vec<f64> {
        self.rec_thrs()
    }

    #[setter(recThrs)]
    fn set_rec_thrs_camel(&mut self, val: Vec<f64>) {
        self.set_rec_thrs(val);
    }

    #[getter(maxDets)]
    fn max_dets_camel(&self) -> Vec<usize> {
        self.max_dets()
    }

    #[setter(maxDets)]
    fn set_max_dets_camel(&mut self, val: Vec<usize>) {
        self.set_max_dets(val);
    }

    #[getter(areaRng)]
    fn area_rng_camel(&self) -> Vec<[f64; 2]> {
        self.area_rng()
    }

    #[setter(areaRng)]
    fn set_area_rng_camel(&mut self, val: Vec<[f64; 2]>) {
        self.set_area_rng(val);
    }

    #[getter(areaRngLbl)]
    fn area_rng_lbl_camel(&self) -> Vec<String> {
        self.area_rng_lbl()
    }

    #[setter(areaRngLbl)]
    fn set_area_rng_lbl_camel(&mut self, val: Vec<String>) {
        self.set_area_rng_lbl(val);
    }

    #[getter(useCats)]
    fn use_cats_camel(&self) -> bool {
        self.use_cats()
    }

    #[setter(useCats)]
    fn set_use_cats_camel(&mut self, val: bool) {
        self.set_use_cats(val);
    }
}

fn parse_iou_type(s: &str) -> PyResult<hotcoco_core::IouType> {
    match s {
        "bbox" => Ok(hotcoco_core::IouType::Bbox),
        "segm" => Ok(hotcoco_core::IouType::Segm),
        "keypoints" => Ok(hotcoco_core::IouType::Keypoints),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown iou_type: '{}'. Expected 'bbox', 'segm', or 'keypoints'",
            s
        ))),
    }
}

// ---------------------------------------------------------------------------
// COCOeval
// ---------------------------------------------------------------------------

#[doc = "COCO evaluation engine.

Computes AP and AR metrics for bbox, segmentation, and keypoint predictions.
Also supports LVIS federated evaluation via ``lvis_style=True``.

Standard COCO workflow::

    ev = COCOeval(coco_gt, coco_dt, \"bbox\")
    ev.evaluate()    # per-image IoU matching
    ev.accumulate()  # aggregate into precision/recall curves
    ev.summarize()   # print + store the 12 summary metrics in ev.stats

LVIS workflow::

    ev = COCOeval(coco_gt, coco_dt, \"segm\", lvis_style=True)
    ev.run()                    # evaluate + accumulate + summarize in one call
    results = ev.get_results()  # dict with 13 metrics: AP, APr, APc, APf, AR@300, ...
"]
#[pyclass(name = "COCOeval")]
struct PyCOCOeval {
    inner: hotcoco_core::COCOeval,
}

#[pymethods]
impl PyCOCOeval {
    #[new]
    #[pyo3(signature = (coco_gt, coco_dt, iou_type, lvis_style=false))]
    fn new(coco_gt: &PyCOCO, coco_dt: &PyCOCO, iou_type: &str, lvis_style: bool) -> PyResult<Self> {
        let iou = parse_iou_type(iou_type)?;
        let inner = if lvis_style {
            hotcoco_core::COCOeval::new_lvis(coco_gt.clone().inner, coco_dt.clone().inner, iou)
        } else {
            hotcoco_core::COCOeval::new(coco_gt.clone().inner, coco_dt.clone().inner, iou)
        };
        Ok(PyCOCOeval { inner })
    }

    fn evaluate(&mut self) {
        self.inner.evaluate();
    }

    fn accumulate(&mut self) {
        self.inner.accumulate();
    }

    fn summarize(&mut self) {
        self.inner.summarize();
    }

    #[doc = "Run the full evaluation pipeline: evaluate → accumulate → summarize.

Equivalent to calling the three methods in sequence. Primarily used with
LVIS pipelines (Detectron2, MMDetection) that expect a single ``run()`` call."]
    fn run(&mut self) {
        self.inner.run();
    }

    #[doc = "Return summary metrics as a dict.

Must be called after ``summarize()`` (or ``run()``). Returns an empty dict
if ``summarize`` has not been run.

LVIS keys: ``AP``, ``AP50``, ``AP75``, ``APs``, ``APm``, ``APl``,
``APr``, ``APc``, ``APf``, ``AR@300``, ``ARs@300``, ``ARm@300``, ``ARl@300``.

Standard COCO bbox/segm keys: ``AP``, ``AP50``, ``AP75``, ``APs``, ``APm``,
``APl``, ``AR1``, ``AR10``, ``AR100``, ``ARs``, ``ARm``, ``ARl``.

Keypoint keys: ``AP``, ``AP50``, ``AP75``, ``APm``, ``APl``,
``AR``, ``AR50``, ``AR75``, ``ARm``, ``ARl``."]
    fn get_results(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (k, v) in self.inner.get_results() {
            dict.set_item(k, v)?;
        }
        Ok(dict.into_any().unbind())
    }

    #[doc = "Print a formatted results table to stdout.

For LVIS, matches the lvis-api ``print_results()`` style. Must be called after
``summarize()`` (or ``run()``)."]
    fn print_results(&self) {
        self.inner.print_results();
    }

    #[getter]
    fn params(&self) -> PyParams {
        PyParams {
            inner: self.inner.params.clone(),
        }
    }

    #[setter]
    fn set_params(&mut self, params: &PyParams) {
        self.inner.params = params.inner.clone();
    }

    #[getter]
    fn stats(&self) -> Option<Vec<f64>> {
        self.inner.stats.clone()
    }

    #[getter]
    fn eval_imgs(&self, py: Python<'_>) -> PyResult<PyObject> {
        eval_imgs_to_py(py, &self.inner.eval_imgs)
    }

    #[getter(evalImgs)]
    fn eval_imgs_camel(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.eval_imgs(py)
    }

    #[getter(eval)]
    fn get_eval(&self, py: Python<'_>) -> PyResult<PyObject> {
        accumulated_eval_to_py(py, &self.inner.eval)
    }

    #[doc = "Compute a per-category confusion matrix across all images.

Unlike ``evaluate()``, this method compares **all** detections in an image against
**all** ground truth boxes regardless of category, enabling cross-category confusion
analysis (e.g. the model keeps predicting ``dog`` on ``cat`` ground truth).

This method is standalone — no ``evaluate()`` call is needed first.

Returns a dict with:

- ``matrix``: ``np.ndarray`` of shape ``(K+1, K+1)``, dtype ``int64``.  Rows = GT
  category, cols = predicted category.  Index ``K`` is the background row/column
  (unmatched GTs = false negatives end up in the background column; unmatched DTs =
  false positives end up in the background row).
- ``normalized``: ``np.ndarray`` of shape ``(K+1, K+1)``, dtype ``float64``.
  Each row is divided by its row sum (zero rows stay zero).
- ``cat_ids``: ``list[int]`` — category IDs for rows/cols ``0..K-1``.
- ``num_cats``: ``int`` — number of categories (``K``).
- ``iou_thr``: ``float`` — IoU threshold used for matching.

Parameters
----------
iou_thr : float, optional
    IoU threshold for a DT↔GT match.  Default ``0.5``.
max_det : int or None, optional
    Max detections per image (by score, highest first).  ``None`` uses the last
    value of ``params.max_dets``.
min_score : float or None, optional
    Discard detections below this confidence before the ``max_det`` truncation.
    ``None`` keeps all detections.

Example
-------
::

    ev = COCOeval(coco_gt, coco_dt, \"bbox\")
    cm = ev.confusion_matrix(iou_thr=0.5, max_det=100)
    print(cm['matrix'].shape)   # (K+1, K+1)
    print(cm['cat_ids'])        # list of category IDs
"]
    #[pyo3(signature = (iou_thr=0.5, max_det=None, min_score=None))]
    fn confusion_matrix(
        &self,
        py: Python<'_>,
        iou_thr: f64,
        max_det: Option<usize>,
        min_score: Option<f64>,
    ) -> PyResult<PyObject> {
        let cm = self.inner.confusion_matrix(iou_thr, max_det, min_score);
        let k = cm.num_cats + 1;

        // matrix: Vec<u64> → numpy int64, reshaped to (k, k)
        let matrix_i64: Vec<i64> = cm.matrix.iter().map(|&v| v as i64).collect();
        let matrix_arr = PyArray1::<i64>::from_vec(py, matrix_i64);
        let matrix_arr = matrix_arr.call_method1("reshape", ((k, k),))?.unbind();

        // normalized: Vec<f64> → numpy float64, reshaped to (k, k)
        let norm_arr = PyArray1::<f64>::from_vec(py, cm.normalized());
        let norm_arr = norm_arr.call_method1("reshape", ((k, k),))?.unbind();

        let dict = PyDict::new(py);
        dict.set_item("matrix", matrix_arr)?;
        dict.set_item("normalized", norm_arr)?;
        dict.set_item("cat_ids", cm.cat_ids)?;
        dict.set_item("num_cats", cm.num_cats)?;
        dict.set_item("iou_thr", cm.iou_thr)?;

        Ok(dict.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// EvalImg / AccumulatedEval → Python converters
// ---------------------------------------------------------------------------

fn eval_imgs_to_py(
    py: Python<'_>,
    eval_imgs: &[Option<hotcoco_core::EvalImg>],
) -> PyResult<PyObject> {
    let list = PyList::new(
        py,
        eval_imgs
            .iter()
            .map(|opt| match opt {
                None => Ok(py.None()),
                Some(e) => eval_img_to_py(py, e),
            })
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    Ok(list.into_any().unbind())
}

fn eval_img_to_py(py: Python<'_>, e: &hotcoco_core::EvalImg) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("image_id", e.image_id)?;
    dict.set_item("category_id", e.category_id)?;
    dict.set_item("aRng", e.area_rng.to_vec())?;
    dict.set_item("maxDet", e.max_det)?;
    dict.set_item("dtIds", e.dt_ids.clone())?;
    dict.set_item("gtIds", e.gt_ids.clone())?;
    // dt_matches: Vec<Vec<u64>> [T x D] → list of lists
    dict.set_item("dtMatches", &e.dt_matches)?;
    // gt_matches: Vec<Vec<u64>> [T x G] → list of lists
    dict.set_item("gtMatches", &e.gt_matches)?;
    // dt_matched / gt_matched: Vec<Vec<bool>> [T x D/G] → list of lists
    dict.set_item("dtMatched", &e.dt_matched)?;
    dict.set_item("gtMatched", &e.gt_matched)?;
    dict.set_item("dtScores", e.dt_scores.clone())?;
    // dt_ignore: Vec<Vec<bool>> [T x D] → list of lists
    dict.set_item("dtIgnore", &e.dt_ignore)?;
    dict.set_item("gtIgnore", e.gt_ignore.clone())?;
    Ok(dict.into_any().unbind())
}

fn accumulated_eval_to_py(
    py: Python<'_>,
    eval: &Option<hotcoco_core::AccumulatedEval>,
) -> PyResult<PyObject> {
    match eval {
        None => Ok(py.None()),
        Some(e) => {
            let dict = PyDict::new(py);
            let counts = vec![e.t, e.r, e.k, e.a, e.m];
            dict.set_item("counts", counts)?;

            // precision: flat Vec<f64> → numpy array, then reshape to (T, R, K, A, M)
            let precision = PyArray1::from_vec(py, e.precision.clone());
            let precision = precision
                .call_method1("reshape", ((e.t, e.r, e.k, e.a, e.m),))?
                .unbind();
            dict.set_item("precision", precision)?;

            // recall: flat Vec<f64> → numpy array, then reshape to (T, K, A, M)
            let recall = PyArray1::from_vec(py, e.recall.clone());
            let recall = recall
                .call_method1("reshape", ((e.t, e.k, e.a, e.m),))?
                .unbind();
            dict.set_item("recall", recall)?;

            // scores: flat Vec<f64> → numpy array, then reshape to (T, R, K, A, M)
            let scores = PyArray1::from_vec(py, e.scores.clone());
            let scores = scores
                .call_method1("reshape", ((e.t, e.r, e.k, e.a, e.m),))?
                .unbind();
            dict.set_item("scores", scores)?;

            Ok(dict.into_any().unbind())
        }
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Patch `sys.modules` so that `from pycocotools.coco import COCO` etc.
/// transparently use hotcoco.
#[pyfunction]
fn init_as_pycocotools(py: Python<'_>) -> PyResult<()> {
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    let hotcoco = py.import("hotcoco")?;
    let mask_mod = hotcoco.getattr("mask")?;
    modules.set_item("pycocotools", &hotcoco)?;
    modules.set_item("pycocotools.coco", &hotcoco)?;
    modules.set_item("pycocotools.cocoeval", &hotcoco)?;
    modules.set_item("pycocotools.mask", &mask_mod)?;
    Ok(())
}

/// Patch `sys.modules` so that `from lvis import LVIS, LVISEval, LVISResults` etc.
/// transparently use hotcoco.
///
/// After calling this, existing Detectron2 / MMDetection LVIS pipelines work
/// without any other code changes:
///
/// ```python
/// from hotcoco import init_as_lvis
/// init_as_lvis()
///
/// from lvis import LVIS, LVISEval, LVISResults
/// lvis_results = LVISResults(lvis_gt, predictions, max_dets=300)
/// lvis_eval = LVISEval(lvis_gt, lvis_results, "bbox")
/// lvis_eval.run()
/// ```
#[pyfunction]
fn init_as_lvis(py: Python<'_>) -> PyResult<()> {
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    let hotcoco = py.import("hotcoco")?;
    modules.set_item("lvis", &hotcoco)?;
    modules.set_item("lvis.eval", &hotcoco)?;
    modules.set_item("lvis.coco", &hotcoco)?;
    modules.set_item("lvis.results", &hotcoco)?;
    Ok(())
}

#[pymodule]
fn hotcoco(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCOCO>()?;
    m.add_class::<PyCOCOeval>()?;
    m.add_class::<PyParams>()?;
    m.add_function(wrap_pyfunction!(init_as_pycocotools, m)?)?;
    m.add_function(wrap_pyfunction!(init_as_lvis, m)?)?;

    // mask submodule
    let mask_mod = PyModule::new(py, "mask")?;
    mask_mod.add_function(wrap_pyfunction!(mask::encode, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::decode, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::area, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::to_bbox, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::merge, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::iou, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::bbox_iou, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::fr_poly, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::fr_bbox, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::rle_to_string, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::rle_from_string, &mask_mod)?)?;
    m.add_submodule(&mask_mod)?;

    Ok(())
}
