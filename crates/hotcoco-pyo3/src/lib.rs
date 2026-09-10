use std::collections::HashMap;
use std::path::Path;

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use mask::transpose_mask;

use hotcoco_core::Annotation;

mod convert;
mod mask;
mod metrics;
mod primitives;

/// Read `(width, height)` for every image in `dir`, keyed by file stem.
///
/// YOLO, DOTA and Open Images all store coordinates normalized to the image, and
/// none of the three records the pixel size, so denormalizing means measuring the
/// images. Pillow is an optional dependency — its absence is only an error once a
/// caller actually asks for dimensions by passing `images_dir`.
fn read_image_dims(py: Python<'_>, dir: &str) -> PyResult<HashMap<String, (u32, u32)>> {
    let pil_image = py.import("PIL.Image").map_err(|_| {
        pyo3::exceptions::PyImportError::new_err(
            "Pillow is required to read image dimensions. \
             Install it with: pip install Pillow",
        )
    })?;

    let read_dir = std::fs::read_dir(dir).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("cannot read images_dir: {e}"))
    })?;

    let img_exts = hotcoco_core::convert::IMAGE_EXTENSIONS;
    let mut image_dims: HashMap<String, (u32, u32)> = HashMap::new();

    for entry in read_dir.flatten() {
        let path = entry.path();
        let ext_lower = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_lowercase);
        let Some(ext) = ext_lower else { continue };
        if !img_exts.contains(&ext.as_str()) {
            continue;
        }

        let path_str = path.to_string_lossy().into_owned();
        let pil_img = pil_image.getattr("open")?.call1((path_str.as_str(),))?;
        let size: (u32, u32) = pil_img.getattr("size")?.extract()?;
        let _ = pil_img.call_method0("close");
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        image_dims.insert(stem, size);
    }

    Ok(image_dims)
}

/// Convert a hotcoco error to a Python exception with appropriate type mapping.
///
/// The file-wide convention: I/O problems are `IOError`, malformed data is
/// `ValueError`. Every binding that surfaces a core error must come through
/// here (for `ConvertError`, via `e.into()`), not map to `PyRuntimeError` ad
/// hoc — `except OSError` / `except ValueError` in user code should catch what
/// their names promise.
pub(crate) fn to_pyerr(err: hotcoco_core::Error) -> PyErr {
    use hotcoco_core::{ConvertError, Error};
    match err {
        Error::Io(e) | Error::Convert(ConvertError::Io(e)) => {
            pyo3::exceptions::PyIOError::new_err(e.to_string())
        }
        Error::Json(e) => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        Error::JsonParse(e) => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        Error::Convert(e) => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        Error::Other(msg) => pyo3::exceptions::PyRuntimeError::new_err(msg),
    }
}

/// Annotation ids the dataset does not have — a lookup failure, so `KeyError`.
///
/// A sibling of [`to_pyerr`] rather than an arm inside it: the core returns
/// `UnknownAnnIds` as its own type precisely so this one failure maps to the
/// exception Python callers expect from a mapping lookup, and the message has
/// one owner ([`hotcoco_core::UnknownAnnIds`]).
pub(crate) fn unknown_ann_ids_to_pyerr(err: hotcoco_core::UnknownAnnIds) -> PyErr {
    pyo3::exceptions::PyKeyError::new_err(err.to_string())
}

/// Read an annotation id from a Python mapping key.
///
/// Through `i64` rather than straight to `u64`: a negative id is a lookup that
/// cannot succeed, and `KeyError` says that, where a bare `u64` extract reports
/// `OverflowError` and reads like a bug in the caller's arithmetic. A non-integer
/// key still raises the `TypeError` the extract produces.
fn extract_ann_id(key: &Bound<'_, PyAny>) -> PyResult<u64> {
    let id: i64 = key.extract()?;
    u64::try_from(id).map_err(|_| {
        pyo3::exceptions::PyKeyError::new_err(format!("annotation id {id} not in this dataset"))
    })
}

/// Emit a `UserWarning` through Python's `warnings` machinery.
///
/// `eprintln!` writes to fd 2, which bypasses `sys.stderr` — invisible in a
/// Jupyter cell, uncatchable by `warnings.catch_warnings`. Routing through
/// `PyErr::warn` makes filters, `-W` flags, and `pytest.warns` all work.
fn warn_user(py: Python<'_>, msg: &str) -> PyResult<()> {
    let msg = std::ffi::CString::new(msg)
        .unwrap_or_else(|_| c"hotcoco: warning text contained a NUL byte".to_owned());
    PyErr::warn(
        py,
        &py.get_type::<pyo3::exceptions::PyUserWarning>(),
        &msg,
        1,
    )
}

/// Hand Python a serde-serializable value as plain dicts and lists.
///
/// The bindings return plain Python containers rather than wrapped Rust
/// structs, and the cheapest way to get there for a whole report tree is
/// serialize-then-`json.loads`. Used by `healthcheck`, `report` and `results`,
/// so all three agree on the failure type: a serialization failure is
/// `Error::Json`, and `to_pyerr` maps it as it does everywhere else.
fn serde_to_py<T: serde::Serialize + ?Sized>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let json_str =
        serde_json::to_string(value).map_err(|e| to_pyerr(hotcoco_core::Error::Json(e)))?;
    let json_mod = py.import("json")?;
    Ok(json_mod.call_method1("loads", (json_str,))?.unbind())
}

/// The Python spelling of an LVIS frequency bucket.
///
/// Spelled out rather than derived from `Debug`: this string is public API —
/// `metric_defs()[i]["freq_group"]` — and a `#[derive(Debug)]` rename in the
/// core crate must not silently rename a key Python code branches on.
fn freq_group_name(group: hotcoco_core::FreqGroup) -> &'static str {
    match group {
        hotcoco_core::FreqGroup::Rare => "rare",
        hotcoco_core::FreqGroup::Common => "common",
        hotcoco_core::FreqGroup::Frequent => "frequent",
    }
}

use convert::{
    ANNOTATION_KEYS, IdList, NameList, annotation_to_py, category_to_py, confusion_counts_to_py,
    dataset_stats_to_py, f64_array, image_to_py, map_to_dict, py_to_annotation, py_to_dataset,
    rle_to_coco_py, set_scalar_ann_field,
};

// ---------------------------------------------------------------------------
// COCO
// ---------------------------------------------------------------------------

/// COCO dataset — load, index, and query COCO-format annotations.
///
/// `image_dir` is an optional path to the directory containing images for
/// this dataset. It is used by `browse()` and `coco explore` to locate
/// image files. Propagated automatically through `filter`, `split`,
/// `sample`, and `load_res`.
#[pyclass(name = "COCO", subclass, from_py_object)]
struct PyCOCO {
    inner: hotcoco_core::COCO,
    /// Root directory for image files. Used by `browse()` and `coco explore`.
    /// Set at construction time or assign directly: ``coco.image_dir = "/data/images"``.
    #[pyo3(get, set)]
    image_dir: Option<String>,
}

/// Constructors, kept out of `#[pymethods]` so they stay Rust-only.
///
/// Every `PyCOCO` in this file comes from one of these three. The distinction
/// they encode is whether the new object inherits `image_dir`: a dataset derived
/// from this one sits in the same image directory, while one built from a
/// foreign format or merged from several sources does not.
impl PyCOCO {
    /// A dataset derived from this one — same images, so same `image_dir`.
    fn derived(&self, inner: hotcoco_core::COCO) -> PyCOCO {
        PyCOCO {
            inner,
            image_dir: self.image_dir.clone(),
        }
    }

    /// Same, starting from a bare [`Dataset`](hotcoco_core::Dataset).
    fn derived_from(&self, dataset: hotcoco_core::Dataset) -> PyCOCO {
        self.derived(hotcoco_core::COCO::from_dataset(dataset))
    }

    /// A dataset with no image directory to inherit: a conversion from a foreign
    /// format, a merge whose inputs came from different directories, or a view
    /// onto an evaluator's own copy.
    fn without_image_dir(dataset: hotcoco_core::Dataset) -> PyCOCO {
        PyCOCO {
            inner: hotcoco_core::COCO::from_dataset(dataset),
            image_dir: None,
        }
    }
}

impl Clone for PyCOCO {
    fn clone(&self) -> Self {
        self.derived_from(self.inner.dataset.clone())
    }
}

#[pymethods]
impl PyCOCO {
    #[new]
    #[pyo3(signature = (annotation_file=None, image_dir=None))]
    fn new(
        annotation_file: Option<&Bound<'_, PyAny>>,
        image_dir: Option<String>,
    ) -> PyResult<Self> {
        let inner = match annotation_file {
            Some(obj) => {
                if let Ok(path) = obj.extract::<String>() {
                    hotcoco_core::COCO::new(Path::new(&path)).map_err(to_pyerr)?
                } else if let Ok(dict) = obj.cast::<PyDict>() {
                    let dataset = py_to_dataset(dict)?;
                    hotcoco_core::COCO::from_dataset(dataset)
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "COCO() argument must be a file path (str) or dataset dict",
                    ));
                }
            }
            None => hotcoco_core::COCO::from_dataset(hotcoco_core::Dataset::default()),
        };
        Ok(PyCOCO { inner, image_dir })
    }

    #[pyo3(signature = (img_ids=IdList::default(), cat_ids=IdList::default(), area_rng=None, iscrowd=None))]
    fn get_ann_ids(
        &self,
        img_ids: IdList,
        cat_ids: IdList,
        area_rng: Option<[f64; 2]>,
        iscrowd: Option<bool>,
    ) -> Vec<u64> {
        self.inner
            .get_ann_ids(&img_ids.0, &cat_ids.0, area_rng, iscrowd)
    }

    #[pyo3(signature = (cat_nms=NameList::default(), sup_nms=NameList::default(), cat_ids=IdList::default()))]
    fn get_cat_ids(&self, cat_nms: NameList, sup_nms: NameList, cat_ids: IdList) -> Vec<u64> {
        let cat_nms_ref: Vec<&str> = cat_nms.0.iter().map(String::as_str).collect();
        let sup_nms_ref: Vec<&str> = sup_nms.0.iter().map(String::as_str).collect();
        self.inner
            .get_cat_ids(&cat_nms_ref, &sup_nms_ref, &cat_ids.0)
    }

    #[pyo3(signature = (img_ids=IdList::default(), cat_ids=IdList::default()))]
    fn get_img_ids(&self, img_ids: IdList, cat_ids: IdList) -> Vec<u64> {
        self.inner.get_img_ids(&img_ids.0, &cat_ids.0)
    }

    fn load_anns(&self, py: Python<'_>, ids: IdList) -> PyResult<Py<PyAny>> {
        let anns = self.inner.load_anns(&ids.0);
        let list = PyList::new(
            py,
            anns.iter()
                .map(|a| annotation_to_py(py, a))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(list.into_any().unbind())
    }

    fn load_cats(&self, py: Python<'_>, ids: IdList) -> PyResult<Py<PyAny>> {
        let cats = self.inner.load_cats(&ids.0);
        let list = PyList::new(
            py,
            cats.iter()
                .map(|c| category_to_py(py, c))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(list.into_any().unbind())
    }

    fn load_imgs(&self, py: Python<'_>, ids: IdList) -> PyResult<Py<PyAny>> {
        let imgs = self.inner.load_imgs(&ids.0);
        let list = PyList::new(
            py,
            imgs.iter()
                .map(|i| image_to_py(py, i))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(list.into_any().unbind())
    }

    /// Load detection results into a new COCO object.
    ///
    /// Accepts three input formats:
    ///
    /// - **str** — path to a JSON file containing a list of detection dicts.
    /// - **list[dict]** — detection dicts already in memory.
    /// - **numpy.ndarray** — float64 array of shape ``(N, 6)`` or ``(N, 7)``,
    ///   with columns ``[image_id, x, y, w, h, score]`` or
    ///   ``[image_id, x, y, w, h, score, category_id]``.
    ///   Matches pycocotools ``loadNumpyAnnotations`` convention.
    ///
    /// Returns a new ``COCO`` object containing the detections, with images and
    /// categories copied from the ground truth. Missing fields (``area``,
    /// ``segmentation``) are computed automatically.
    ///
    /// Also available as ``loadRes()`` (camelCase alias).
    fn load_res(&self, res: &Bound<'_, PyAny>) -> PyResult<PyCOCO> {
        // Case 1: file path (str)
        if let Ok(path) = res.extract::<String>() {
            return self
                .inner
                .load_res(Path::new(&path))
                .map(|inner| self.derived(inner))
                .map_err(to_pyerr);
        }

        // Case 2: list of annotation dicts
        if let Ok(list) = res.cast::<PyList>() {
            let anns = list
                .iter()
                .map(|item| {
                    let dict = item.cast::<PyDict>().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(
                            "load_res: list elements must be dicts",
                        )
                    })?;
                    py_to_annotation(dict)
                })
                .collect::<PyResult<Vec<_>>>()?;
            return self
                .inner
                .load_res_anns(anns)
                .map(|inner| self.derived(inner))
                .map_err(to_pyerr);
        }

        // Case 3: numpy float64 array, shape (N, 6) or (N, 7)
        //   (N, 6): [image_id, x, y, w, h, score]           — category_id defaults to 1
        //   (N, 7): [image_id, x, y, w, h, score, cat_id]   — matches pycocotools loadNumpyAnnotations
        if let Ok(arr) = res.cast::<PyArray2<f64>>() {
            let arr = arr.readonly();
            let arr = arr.as_array();
            let ncols = arr.ncols();
            if ncols != 6 && ncols != 7 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "load_res: numpy array must have 6 or 7 columns \
                     [image_id, x, y, w, h, score[, category_id]], got {ncols}",
                )));
            }
            let anns = arr
                .rows()
                .into_iter()
                .map(|row| Annotation {
                    id: 0,
                    image_id: row[0] as u64,
                    category_id: if ncols == 7 { row[6] as u64 } else { 1 },
                    bbox: Some([row[1], row[2], row[3], row[4]]),
                    score: Some(row[5]),
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            return self
                .inner
                .load_res_anns(anns)
                .map(|inner| self.derived(inner))
                .map_err(to_pyerr);
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "load_res expects a file path (str), list of annotation dicts, \
             or numpy float64 array of shape (N, 6) or (N, 7)",
        ))
    }

    /// Convert an annotation's segmentation to RLE.
    ///
    /// Returns the same shape as ``mask.encode`` and pycocotools:
    /// ``{"size": [h, w], "counts": bytes}`` — so the result feeds straight
    /// into ``mask.decode`` / ``mask.area`` / ``mask.iou``.
    fn ann_to_rle(&self, py: Python<'_>, ann: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let annotation = py_to_annotation(ann)?;
        match self.inner.ann_to_rle(&annotation) {
            Some(rle) => rle_to_coco_py(py, &rle),
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
        let row_major = transpose_mask(&col_major, w, h);
        let flat = numpy::PyArray1::from_vec(py, row_major);
        let arr = flat.reshape([h, w])?;
        Ok(arr.unbind())
    }

    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let s = self.inner.stats();
        dataset_stats_to_py(py, &s)
    }

    /// Validate this dataset for structural errors, quality warnings, and
    /// distribution issues. If ``dt`` is provided, also checks GT/DT compatibility
    /// such as mismatched image or category IDs.
    ///
    /// Returns a dict with ``"errors"``, ``"warnings"``, and ``"summary"`` keys.
    #[pyo3(signature = (dt=None))]
    fn healthcheck(&self, py: Python<'_>, dt: Option<&PyCOCO>) -> PyResult<Py<PyAny>> {
        let report = match dt {
            Some(dt_coco) => self.inner.healthcheck_compatibility(&dt_coco.inner),
            None => self.inner.healthcheck(),
        };
        serde_to_py(py, &report)
    }

    /// Filter the dataset by category, by image, by annotation area, or by any combination.
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
        self.derived_from(result)
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
        let result = hotcoco_core::COCO::merge(&ds_refs).map_err(to_pyerr)?;
        Ok(PyCOCO::without_image_dir(result))
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
    ) -> PyResult<Py<PyAny>> {
        let (train_ds, val_ds, test_ds) = self.inner.split(val_frac, test_frac, seed);
        let train_py = Py::new(py, self.derived_from(train_ds))?;
        let val_py = Py::new(py, self.derived_from(val_ds))?;
        if let Some(test_ds) = test_ds {
            let test_py = Py::new(py, self.derived_from(test_ds))?;
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
        self.derived_from(result)
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

    // camelCase aliases for pycocotools compatibility. The *parameter* names
    // are camelCase too — `getAnnIds(imgIds=…)` is pycocotools' canonical
    // keyword form (Detectron2 spells it that way), and PyO3 exposes Rust
    // parameter names as Python keywords, so these aliases must spell them
    // the way pycocotools does. non_snake_case is a rustc style lint, not a
    // clippy correctness one; the allow is scoped to exactly these aliases.
    #[allow(non_snake_case)]
    #[pyo3(name = "getAnnIds")]
    #[pyo3(signature = (imgIds=IdList::default(), catIds=IdList::default(), areaRng=None, iscrowd=None))]
    fn get_ann_ids_camel(
        &self,
        imgIds: IdList,
        catIds: IdList,
        areaRng: Option<[f64; 2]>,
        iscrowd: Option<bool>,
    ) -> Vec<u64> {
        self.get_ann_ids(imgIds, catIds, areaRng, iscrowd)
    }

    #[allow(non_snake_case)]
    #[pyo3(name = "getCatIds")]
    #[pyo3(signature = (catNms=NameList::default(), supNms=NameList::default(), catIds=IdList::default()))]
    fn get_cat_ids_camel(&self, catNms: NameList, supNms: NameList, catIds: IdList) -> Vec<u64> {
        self.get_cat_ids(catNms, supNms, catIds)
    }

    #[allow(non_snake_case)]
    #[pyo3(name = "getImgIds")]
    #[pyo3(signature = (imgIds=IdList::default(), catIds=IdList::default()))]
    fn get_img_ids_camel(&self, imgIds: IdList, catIds: IdList) -> Vec<u64> {
        self.get_img_ids(imgIds, catIds)
    }

    #[pyo3(name = "loadAnns")]
    fn load_anns_camel(&self, py: Python<'_>, ids: IdList) -> PyResult<Py<PyAny>> {
        self.load_anns(py, ids)
    }

    #[pyo3(name = "loadCats")]
    fn load_cats_camel(&self, py: Python<'_>, ids: IdList) -> PyResult<Py<PyAny>> {
        self.load_cats(py, ids)
    }

    #[pyo3(name = "loadImgs")]
    fn load_imgs_camel(&self, py: Python<'_>, ids: IdList) -> PyResult<Py<PyAny>> {
        self.load_imgs(py, ids)
    }

    #[pyo3(name = "loadRes")]
    fn load_res_camel(&self, res: &Bound<'_, PyAny>) -> PyResult<PyCOCO> {
        self.load_res(res)
    }

    #[pyo3(name = "annToRLE")]
    fn ann_to_rle_camel(&self, py: Python<'_>, ann: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
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

    /// Convert this dataset to YOLO label format.
    ///
    /// Writes one ``.txt`` label file per image into ``output_dir`` (named by
    /// the image filename stem: ``000042.jpg`` → ``000042.txt``), plus a
    /// ``data.yaml`` file listing the categories.
    ///
    /// Each label line: ``class_idx cx cy w h`` with coordinates normalized to
    /// ``[0, 1]`` by image dimensions. Categories are sorted by COCO ID and
    /// assigned 0-indexed class IDs. Crowd annotations and annotations without
    /// a bounding box are skipped. Images with no annotations produce an empty
    /// ``.txt`` file (standard YOLO convention).
    ///
    /// Parameters
    /// ----------
    /// output_dir : str
    ///     Directory to write label files and ``data.yaml``. Created if it does
    ///     not exist.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{"images": int, "annotations": int, "skipped_crowd": int, "skipped_no_bbox": int}``
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If any image has ``width == 0`` or ``height == 0`` (normalization
    ///     requires valid dimensions), or two images share a file stem.
    /// IOError
    ///     If the output directory cannot be written.
    ///
    /// Examples
    /// --------
    /// >>> coco = COCO("instances_val2017.json")
    /// >>> stats = coco.to_yolo("labels/val2017/")
    /// >>> print(stats)
    /// {'images': 5000, 'annotations': 36781, 'skipped_crowd': 12, 'skipped_no_bbox': 0}
    fn to_yolo(&self, py: Python<'_>, output_dir: &str) -> PyResult<Py<PyAny>> {
        let stats = hotcoco_core::convert::coco_to_yolo(&self.inner.dataset, Path::new(output_dir))
            .map_err(|e| to_pyerr(e.into()))?;
        let dict = PyDict::new(py);
        dict.set_item("images", stats.images)?;
        dict.set_item("annotations", stats.annotations)?;
        dict.set_item("skipped_crowd", stats.skipped_crowd)?;
        dict.set_item("skipped_no_bbox", stats.skipped_no_bbox)?;
        Ok(dict.into_any().unbind())
    }

    /// Load a YOLO label directory as a COCO dataset.
    ///
    /// Reads ``data.yaml`` from ``yolo_dir`` for the category list, then parses
    /// every ``.txt`` label file in the directory. Returns a ``COCO`` object with
    /// sequential image and annotation IDs starting at 1.
    ///
    /// Parameters
    /// ----------
    /// yolo_dir : str
    ///     Directory containing ``.txt`` label files and ``data.yaml``.
    /// images_dir : str, optional
    ///     Directory of source images. When provided, Pillow reads each image to
    ///     populate ``width`` and ``height`` on the resulting image records.
    ///     Without this, images are stored with ``width=0, height=0``.
    ///     Requires ``pip install Pillow``.
    ///
    /// Returns
    /// -------
    /// COCO
    ///     A new ``COCO`` object containing the parsed dataset.
    ///
    /// Raises
    /// ------
    /// ImportError
    ///     If ``images_dir`` is provided but Pillow is not installed.
    /// ValueError
    ///     If ``data.yaml`` is missing or a label file cannot be parsed.
    /// IOError
    ///     If the directory cannot be read.
    ///
    /// Examples
    /// --------
    /// >>> # Without image dims (width/height will be 0)
    /// >>> coco = COCO.from_yolo("labels/val2017/")
    ///
    /// >>> # With image dims read via Pillow
    /// >>> coco = COCO.from_yolo("labels/val2017/", images_dir="images/val2017/")
    /// >>> coco.save("reconstructed.json")
    #[classmethod]
    #[pyo3(signature = (yolo_dir, images_dir=None))]
    fn from_yolo(
        cls: &Bound<'_, PyType>,
        yolo_dir: &str,
        images_dir: Option<&str>,
    ) -> PyResult<PyCOCO> {
        let image_dims = match images_dir {
            Some(dir) => read_image_dims(cls.py(), dir)?,
            None => HashMap::new(),
        };

        hotcoco_core::convert::yolo_to_coco(Path::new(yolo_dir), &image_dims)
            .map(PyCOCO::without_image_dir)
            .map_err(|e| to_pyerr(e.into()))
    }

    /// Export the dataset to Pascal VOC annotation format.
    ///
    /// Writes one XML file per image into ``output_dir/Annotations/``, plus a
    /// ``labels.txt`` file listing category names sorted by COCO ID.
    ///
    /// Parameters
    /// ----------
    /// output_dir : str
    ///     Directory to write the VOC annotations into.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{'images': int, 'annotations': int, 'crowd_as_difficult': int, 'skipped_no_bbox': int}``
    ///
    /// Examples
    /// --------
    /// >>> coco = COCO("instances_val2017.json")
    /// >>> stats = coco.to_voc("voc_output/")
    /// >>> print(stats)
    /// {'images': 5000, 'annotations': 36781, 'crowd_as_difficult': 12, 'skipped_no_bbox': 0}
    fn to_voc(&self, py: Python<'_>, output_dir: &str) -> PyResult<Py<PyAny>> {
        let stats = hotcoco_core::convert::coco_to_voc(&self.inner.dataset, Path::new(output_dir))
            .map_err(|e| to_pyerr(e.into()))?;
        let dict = PyDict::new(py);
        dict.set_item("images", stats.images)?;
        dict.set_item("annotations", stats.annotations)?;
        dict.set_item("crowd_as_difficult", stats.crowd_as_difficult)?;
        dict.set_item("skipped_no_bbox", stats.skipped_no_bbox)?;
        Ok(dict.into_any().unbind())
    }

    /// Load a Pascal VOC annotation directory as a COCO dataset.
    ///
    /// Scans for ``*.xml`` files in ``voc_dir/Annotations/`` (falls back to
    /// ``voc_dir/`` directly). Image dimensions are read from each XML's
    /// ``<size>`` element.
    ///
    /// If ``labels.txt`` exists in ``voc_dir``, it determines category ordering;
    /// otherwise categories are sorted alphabetically.
    ///
    /// Parameters
    /// ----------
    /// voc_dir : str
    ///     Directory containing an ``Annotations/`` subdirectory with ``.xml``
    ///     files, or a flat directory of ``.xml`` files.
    ///
    /// Returns
    /// -------
    /// COCO
    ///     A new ``COCO`` object containing the parsed dataset.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If XML files cannot be parsed or required elements are missing.
    /// IOError
    ///     If the directory cannot be read.
    ///
    /// Examples
    /// --------
    /// >>> coco = COCO.from_voc("VOCdevkit/VOC2012/")
    /// >>> print(len(coco.dataset['images']))
    /// 5717
    /// >>> coco.save("voc2012_as_coco.json")
    #[classmethod]
    fn from_voc(cls: &Bound<'_, PyType>, voc_dir: &str) -> PyResult<PyCOCO> {
        let _ = cls;
        hotcoco_core::convert::voc_to_coco(Path::new(voc_dir))
            .map(PyCOCO::without_image_dir)
            .map_err(|e| to_pyerr(e.into()))
    }

    /// Export the dataset to CVAT for Images 1.1 XML format.
    ///
    /// Writes a single XML file at ``output_path`` containing all images and
    /// annotations. Bounding boxes become ``<box>`` elements; polygon
    /// segmentations become ``<polygon>`` elements.
    ///
    /// Parameters
    /// ----------
    /// output_path : str
    ///     Path to the output XML file.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{'images': int, 'boxes': int, 'polygons': int, 'skipped_no_geometry': int,
    ///     'skipped_degenerate': int}``
    ///
    /// Examples
    /// --------
    /// >>> coco = COCO("instances_val2017.json")
    /// >>> stats = coco.to_cvat("annotations.xml")
    /// >>> print(stats)
    /// {'images': 5000, 'boxes': 36781, 'polygons': 0, 'skipped_no_geometry': 0, 'skipped_degenerate': 0}
    fn to_cvat(&self, py: Python<'_>, output_path: &str) -> PyResult<Py<PyAny>> {
        let stats =
            hotcoco_core::convert::coco_to_cvat(&self.inner.dataset, Path::new(output_path))
                .map_err(|e| to_pyerr(e.into()))?;
        let dict = PyDict::new(py);
        dict.set_item("images", stats.images)?;
        dict.set_item("boxes", stats.boxes)?;
        dict.set_item("polygons", stats.polygons)?;
        dict.set_item("skipped_no_geometry", stats.skipped_no_geometry)?;
        dict.set_item("skipped_degenerate", stats.skipped_degenerate)?;
        Ok(dict.into_any().unbind())
    }

    /// Load a CVAT for Images 1.1 XML file as a COCO dataset.
    ///
    /// Reads a single XML file. Category ordering comes from the
    /// ``<meta><task><labels>`` block. Supports ``<box>`` and ``<polygon>``
    /// elements; ``<polyline>``, ``<points>``, and ``<cuboid>`` are skipped.
    ///
    /// Parameters
    /// ----------
    /// cvat_path : str
    ///     Path to the CVAT XML file.
    ///
    /// Returns
    /// -------
    /// COCO
    ///     A new ``COCO`` object containing the parsed dataset.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the XML file cannot be parsed or required attributes are missing.
    /// IOError
    ///     If the file cannot be read.
    ///
    /// Warns
    /// -----
    /// UserWarning
    ///     When shapes were skipped — kinds COCO cannot express (``<polyline>``,
    ///     ``<points>``, ``<cuboid>``) or polygons with fewer than 3 points —
    ///     so a partial import is visible rather than silent.
    ///
    /// Examples
    /// --------
    /// >>> coco = COCO.from_cvat("annotations.xml")
    /// >>> print(len(coco.dataset['images']))
    /// 5000
    /// >>> coco.save("cvat_as_coco.json")
    #[classmethod]
    fn from_cvat(cls: &Bound<'_, PyType>, cvat_path: &str) -> PyResult<PyCOCO> {
        let (dataset, stats) = hotcoco_core::convert::cvat_to_coco(Path::new(cvat_path))
            .map_err(|e| to_pyerr(e.into()))?;
        if stats.skipped_unsupported > 0 || stats.skipped_degenerate > 0 {
            warn_user(
                cls.py(),
                &format!(
                    "hotcoco: CVAT import of {cvat_path} skipped {} unsupported shape(s) \
                     and {} degenerate polygon(s)",
                    stats.skipped_unsupported, stats.skipped_degenerate
                ),
            )?;
        }
        Ok(PyCOCO::without_image_dir(dataset))
    }

    /// Export the dataset to DOTA oriented-bounding-box format.
    ///
    /// Writes one ``.txt`` file per image into ``output_dir``. Each line holds the
    /// four corner points of the rotated box, the category name, and a difficulty
    /// flag (1 for crowd annotations, 0 otherwise).
    ///
    /// Only annotations carrying an ``obb`` field are written — axis-aligned boxes
    /// have no rotation to record and are counted in ``skipped_no_obb``.
    ///
    /// Parameters
    /// ----------
    /// output_dir : str
    ///     Directory to write the DOTA label files into.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{'images': int, 'annotations': int, 'skipped_no_obb': int}``
    ///
    /// Examples
    /// --------
    /// >>> coco = COCO("aerial_annotations.json")
    /// >>> stats = coco.to_dota("labelTxt/")
    /// >>> print(stats)
    /// {'images': 458, 'annotations': 18211, 'skipped_no_obb': 0}
    fn to_dota(&self, py: Python<'_>, output_dir: &str) -> PyResult<Py<PyAny>> {
        let stats = hotcoco_core::convert::coco_to_dota(&self.inner.dataset, Path::new(output_dir))
            .map_err(|e| to_pyerr(e.into()))?;
        let dict = PyDict::new(py);
        dict.set_item("images", stats.images)?;
        dict.set_item("annotations", stats.annotations)?;
        dict.set_item("skipped_no_obb", stats.skipped_no_obb)?;
        Ok(dict.into_any().unbind())
    }

    /// Load a DOTA label directory as a COCO dataset with oriented boxes.
    ///
    /// Parses every ``.txt`` file in ``label_dir`` as
    /// ``x1 y1 x2 y2 x3 y3 x4 y4 category difficulty``. Each annotation gets both
    /// an ``obb`` (the rotated box) and a ``bbox`` (its axis-aligned envelope), so
    /// the result evaluates under either ``iou_type``.
    ///
    /// Parameters
    /// ----------
    /// label_dir : str
    ///     Directory containing DOTA ``.txt`` label files.
    /// images_dir : str, optional
    ///     Directory of source images. DOTA labels are in absolute pixels, so this
    ///     only populates ``width``/``height`` on the image records; box
    ///     coordinates are unaffected. Requires ``pip install Pillow``.
    /// categories : list of str, optional
    ///     Category names in the order they should be numbered. Without this,
    ///     categories are discovered from the label files and sorted, which means
    ///     two splits of the same dataset can disagree on IDs if one split is
    ///     missing a class.
    ///
    /// Returns
    /// -------
    /// COCO
    ///     A new ``COCO`` object containing the parsed dataset.
    ///
    /// Examples
    /// --------
    /// >>> coco = COCO.from_dota("labelTxt/", images_dir="images/")
    /// >>> ev = COCOeval(coco, preds, "obb")
    #[classmethod]
    #[pyo3(signature = (label_dir, images_dir=None, categories=None))]
    fn from_dota(
        cls: &Bound<'_, PyType>,
        label_dir: &str,
        images_dir: Option<&str>,
        categories: Option<Vec<String>>,
    ) -> PyResult<PyCOCO> {
        let image_dims = match images_dir {
            Some(dir) => read_image_dims(cls.py(), dir)?,
            None => HashMap::new(),
        };

        hotcoco_core::convert::dota_to_coco(Path::new(label_dir), categories, &image_dims)
            .map(PyCOCO::without_image_dir)
            .map_err(|e| to_pyerr(e.into()))
    }

    /// Export the dataset to Open Images challenge CSV format.
    ///
    /// Writes ``ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf`` with coordinates
    /// normalized back to ``[0, 1]``. Note the column order — Open Images puts
    /// ``XMax`` before ``YMin``. When any annotation carries a score, a ``Score``
    /// column is inserted so detection files round-trip too.
    ///
    /// ``LabelName`` is written as the COCO category name. A dataset that was
    /// imported with ``class_descriptions`` therefore exports display names
    /// ("Beer") rather than the MIDs it came from ("/m/0cmf2").
    ///
    /// Parameters
    /// ----------
    /// output_csv : str
    ///     Path of the CSV file to write.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{'images': int, 'annotations': int, 'group_of': int, 'skipped_no_bbox': int}``
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an image has zero or unknown dimensions — coordinates cannot be
    ///     normalized back to ``[0, 1]`` without them.
    ///
    /// Examples
    /// --------
    /// >>> coco = COCO("instances_val2017.json")
    /// >>> stats = coco.to_oid("boxes.csv")
    fn to_oid(&self, py: Python<'_>, output_csv: &str) -> PyResult<Py<PyAny>> {
        let stats = hotcoco_core::convert::coco_to_oid(&self.inner.dataset, Path::new(output_csv))
            .map_err(|e| to_pyerr(e.into()))?;
        let dict = PyDict::new(py);
        dict.set_item("images", stats.images)?;
        dict.set_item("annotations", stats.annotations)?;
        dict.set_item("group_of", stats.group_of)?;
        dict.set_item("skipped_no_bbox", stats.skipped_no_bbox)?;
        Ok(dict.into_any().unbind())
    }

    /// Load an Open Images annotation CSV as a COCO dataset.
    ///
    /// Reads the full V6 layout and the challenge subset alike — columns are
    /// resolved by name, not position. ``IsGroupOf`` becomes the ``is_group_of``
    /// annotation field, which Open Images evaluation matches with IoA rather than
    /// IoU.
    ///
    /// Parameters
    /// ----------
    /// csv_path : str
    ///     Path to the annotations CSV (for example
    ///     ``challenge-2019-validation-detection-bbox.csv``).
    /// class_descriptions : str, optional
    ///     Path to ``class-descriptions-boxable.csv``. Resolves ``LabelName`` MIDs
    ///     such as ``/m/0cmf2`` to readable names such as ``Beer``. Without it,
    ///     category names remain MIDs.
    /// images_dir : str, optional
    ///     Directory of source images, used to denormalize coordinates into
    ///     pixels. Requires ``pip install Pillow``.
    ///
    /// Notes
    /// -----
    /// Open Images coordinates are normalized and the CSV does not record image
    /// sizes. Without ``images_dir``, boxes stay in ``[0, 1]`` against a 1x1 image.
    /// IoU and IoA are ratios of areas scaled identically on both axes, so Open
    /// Images AP is unchanged — but absolute areas, and therefore the
    /// small/medium/large ranges, are not meaningful in that mode.
    ///
    /// Returns
    /// -------
    /// COCO
    ///     A new ``COCO`` object containing the parsed dataset.
    ///
    /// Examples
    /// --------
    /// >>> gt = COCO.from_oid(
    /// ...     "challenge-2019-validation-detection-bbox.csv",
    /// ...     class_descriptions="class-descriptions-boxable.csv",
    /// ... )
    /// >>> dt = gt.load_res_oid("predictions.csv")
    /// >>> ev = COCOeval(gt, dt, "bbox", oid_style=True)
    /// >>> ev.run()
    #[classmethod]
    #[pyo3(signature = (csv_path, class_descriptions=None, images_dir=None))]
    fn from_oid(
        cls: &Bound<'_, PyType>,
        csv_path: &str,
        class_descriptions: Option<&str>,
        images_dir: Option<&str>,
    ) -> PyResult<PyCOCO> {
        let image_dims = match images_dir {
            Some(dir) => read_image_dims(cls.py(), dir)?,
            None => HashMap::new(),
        };

        hotcoco_core::convert::oid_to_coco(
            Path::new(csv_path),
            class_descriptions.map(Path::new),
            &image_dims,
        )
        .map(PyCOCO::without_image_dir)
        .map_err(|e| to_pyerr(e.into()))
    }

    /// Load Open Images detections as a result ``COCO``, aligned to this dataset.
    ///
    /// The Open Images equivalent of :meth:`load_res`. Detections must land on the
    /// same image and category IDs as the ground truth, so ``ImageID`` is matched
    /// against image file-name stems and ``LabelName`` against category names —
    /// pass the same ``class_descriptions`` used to load the ground truth.
    ///
    /// A detection referencing an unknown image or category raises rather than
    /// being dropped: silently discarding detections moves recall, which is
    /// invisible once the metrics come out.
    ///
    /// Parameters
    /// ----------
    /// csv_path : str
    ///     Detection CSV with a ``Score`` column.
    /// class_descriptions : str, optional
    ///     Same file passed to :meth:`from_oid`, if any.
    ///
    /// Returns
    /// -------
    /// COCO
    ///     A result object suitable as the ``coco_dt`` argument to ``COCOeval``.
    ///
    /// Examples
    /// --------
    /// >>> gt = COCO.from_oid("boxes.csv")
    /// >>> dt = gt.load_res_oid("predictions.csv")
    #[pyo3(signature = (csv_path, class_descriptions=None))]
    fn load_res_oid(&self, csv_path: &str, class_descriptions: Option<&str>) -> PyResult<PyCOCO> {
        let anns = hotcoco_core::convert::oid_results_to_anns(
            &self.inner.dataset,
            Path::new(csv_path),
            class_descriptions.map(Path::new),
        )
        .map_err(|e| to_pyerr(e.into()))?;

        self.inner
            .load_res_anns(anns)
            .map(|inner| PyCOCO {
                inner,
                image_dir: None,
            })
            .map_err(to_pyerr)
    }

    /// pycocotools builds COCO objects by assignment — `coco = COCO();
    /// coco.dataset = d; coco.createIndex()` — and torchmetrics' pycocotools
    /// backend uses exactly that flow, so `dataset` must be writable for the
    /// drop-in claim to hold. hotcoco indexes eagerly on assignment, which
    /// makes the follow-up `createIndex()` a no-op.
    #[setter]
    fn set_dataset(&mut self, dataset: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner = hotcoco_core::COCO::from_dataset(py_to_dataset(dataset)?);
        Ok(())
    }

    /// Re-index the current dataset — pycocotools semantics. Under the
    /// assignment flow the `dataset` setter has already indexed, so this is
    /// a formality kept for the canonical `coco.dataset = d;
    /// coco.createIndex()` sequence.
    fn create_index(&mut self) {
        self.inner.create_index();
    }

    #[pyo3(name = "createIndex")]
    fn create_index_camel(&mut self) {
        self.create_index();
    }

    /// Replace whole annotations, matched by ``id``, and re-index immediately.
    ///
    /// The targeted counterpart to ``coco.dataset = d``: it edits the
    /// annotations you name instead of rebuilding the dataset.
    ///
    /// Each dict **replaces** its annotation rather than merging into it: keys
    /// you leave out come back as their defaults. Pass a full annotation dict,
    /// or use :meth:`set_ann_field` to change one field and keep the rest.
    /// Keys outside the COCO schema are preserved, as everywhere else.
    ///
    /// A ``COCOeval`` copies both datasets when it is constructed, so an
    /// evaluator built before this call keeps evaluating the old annotations.
    /// Mutate first, then construct the evaluator.
    ///
    /// Parameters
    /// ----------
    /// anns : list of dict
    ///     Annotation dicts, each with an ``id`` that is already in the
    ///     dataset.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If a dict has no ``id``, or names an ``id`` this dataset does not
    ///     have. Nothing is written in that case — an unknown id is a mistake
    ///     worth surfacing, not an edit to skip quietly.
    /// TypeError
    ///     If ``anns`` is not a list, or an element is not a dict.
    /// ValueError
    ///     If a dict is missing a required field or holds a value that does not
    ///     fit it — the same errors the ``dataset`` setter raises.
    ///
    /// Examples
    /// --------
    /// >>> anns = coco.dataset["annotations"]
    /// >>> for ann in anns:
    /// ...     ann["area"] = ann["bbox"][2] * ann["bbox"][3]
    /// >>> coco.update_anns(anns)
    fn update_anns(&mut self, anns: &Bound<'_, PyList>) -> PyResult<()> {
        let mut updated = Vec::with_capacity(anns.len());
        for item in anns {
            let dict = item.cast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("update_anns: list elements must be dicts")
            })?;
            // Before the conversion, not after: `py_to_annotation` defaults a
            // missing id to 0, which would silently overwrite whichever
            // annotation carries that id, and it rejects a dict missing
            // `image_id` first — so a check placed afterwards never sees the
            // annotation whose only problem is the absent id.
            if dict.get_item("id")?.is_none() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    "every annotation passed to update_anns() needs an 'id'",
                ));
            }
            updated.push(py_to_annotation(dict)?);
        }
        self.inner
            .update_anns(updated)
            .map_err(unknown_ann_ids_to_pyerr)
    }

    /// Set one field on the named annotations, and re-index immediately.
    ///
    /// The narrow form of :meth:`update_anns`: every other field of each
    /// annotation is carried over untouched, so a partial edit cannot drop the
    /// rest of the record. Switching each annotation's ``area`` between its box
    /// area and its mask area between IoU types is the case this exists for —
    /// an edit made through the ``dataset`` copy would not land at all.
    ///
    /// A field outside the COCO schema is a custom key. Setting one that the
    /// annotations already carry works as it does for any other field; adding a
    /// new one needs ``create=True``, so that a misspelled schema field —
    /// ``"Area"``, ``"iscrowed"`` — raises instead of quietly landing beside the
    /// field you meant to change.
    ///
    /// Parameters
    /// ----------
    /// field : str
    ///     Annotation key to set, for example ``"area"`` or ``"iscrowd"``.
    /// values : dict
    ///     Annotation id to new value.
    /// create : bool, keyword-only, default False
    ///     Allow ``field`` to be a custom key the annotations do not have yet.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If an annotation id is not in this dataset, or ``field`` is neither a
    ///     COCO field nor a custom key already on the annotation and ``create``
    ///     is False. Nothing is written in either case.
    /// TypeError
    ///     If a value does not fit the field — ``{1: "big"}`` for ``"area"``.
    /// ValueError
    ///     If ``field`` is ``"id"``.
    ///
    /// Examples
    /// --------
    /// >>> mask_areas = {ann["id"]: mask.area(coco.ann_to_rle(ann))
    /// ...               for ann in coco.dataset["annotations"]}
    /// >>> coco.set_ann_field("area", mask_areas)
    #[pyo3(signature = (field, values, *, create=false))]
    fn set_ann_field(
        &mut self,
        py: Python<'_>,
        field: &str,
        values: &Bound<'_, PyDict>,
        create: bool,
    ) -> PyResult<()> {
        if field == "id" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "set_ann_field() cannot change 'id' — re-key annotations by assigning \
                 coco.dataset instead",
            ));
        }
        let known = ANNOTATION_KEYS.contains(&field);
        let mut updated = Vec::with_capacity(values.len());
        for (key, value) in values.iter() {
            let ann_id = extract_ann_id(&key)?;
            let ann = self.inner.get_ann(ann_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "annotation id {ann_id} not in this dataset"
                ))
            })?;
            if !known && !create && !ann.extra.contains_key(field) {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "'{field}' is not an annotation field, and annotation {ann_id} does not \
                     carry it as a custom key — check the spelling, or pass create=True to \
                     add it"
                )));
            }
            // A scalar field is set on a clone. Everything else — a shaped
            // field, a custom key, a value that does not extract — rebuilds the
            // annotation through the dict converters, so shapes and type errors
            // keep coming from the one conversion path rather than from a
            // second field-name matcher here.
            if let Some(edited) = set_scalar_ann_field(ann, field, &value) {
                updated.push(edited);
                continue;
            }
            let obj = annotation_to_py(py, ann)?.into_bound(py);
            let dict = obj
                .cast::<PyDict>()
                .expect("annotation_to_py builds a dict");
            dict.set_item(field, value)?;
            updated.push(py_to_annotation(dict)?);
        }
        self.inner
            .update_anns(updated)
            .map_err(unknown_ann_ids_to_pyerr)
    }

    /// Warnings collected while loading and indexing this dataset.
    ///
    /// Each entry was also printed to stderr at the moment it arose; this
    /// property exists for notebooks and servers where stderr is invisible.
    /// An empty list means the load was clean.
    #[getter]
    fn load_warnings(&self) -> Vec<String> {
        self.inner.load_warnings().to_vec()
    }

    /// The dataset as plain dicts: ``{"images": [...], "annotations": [...],
    /// "categories": [...]}``.
    ///
    /// **Returns a fresh copy on every access.** Mutating it in place —
    /// ``coco.dataset["annotations"].append(...)`` — changes a temporary and is
    /// a silent no-op. Three ways to make an edit land, cheapest first:
    /// :meth:`set_ann_field` for one field across annotations,
    /// :meth:`update_anns` for whole annotations, and assigning the whole
    /// dataset back (``coco.dataset = d``) when the images or categories
    /// change too. All three keep the indices current.
    #[getter]
    fn dataset(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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

    #[getter]
    fn imgs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for img in &self.inner.dataset.images {
            dict.set_item(img.id, image_to_py(py, img)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    #[getter]
    fn anns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for ann in &self.inner.dataset.annotations {
            dict.set_item(ann.id, annotation_to_py(py, ann)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    #[getter]
    fn cats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for cat in &self.inner.dataset.categories {
            dict.set_item(cat.id, category_to_py(py, cat)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Launch an interactive dataset browser.
    ///
    /// Defined here rather than on a Python subclass so that every ``COCO``
    /// this class hands back — ``load_res``, ``filter``, ``split``, ``sample``,
    /// ``merge`` — carries the full API. A Python-side subclass could not
    /// survive those methods: PyO3 constructs the base class.
    ///
    /// Parameters
    /// ----------
    /// image_dir : str, optional
    ///     Root directory for image files. Overrides ``self.image_dir``.
    /// dt : COCO or str, optional
    ///     Detection results to overlay. Pass a COCO object (from
    ///     ``self.load_res()``) or a path string (auto-loaded).
    /// iou_type : str
    ///     Evaluation type: ``"bbox"``, ``"segm"``, or ``"keypoints"``
    ///     (default ``"bbox"``). Only used when ``dt`` is provided.
    /// iou_thr : float
    ///     Initial IoU threshold for TP/FP classification (default 0.5).
    ///     Sets the starting position of the browser's IoU slider, which
    ///     snaps it to the slider's 0.50-0.95 range in steps of 0.05.
    /// eval : COCOeval, optional
    ///     Pre-computed COCOeval (must have ``evaluate()`` called).
    ///     When provided, ``iou_type`` is ignored.
    /// slices : dict or str, optional
    ///     Image subsets for sliced browsing. Pass a dict mapping slice
    ///     names to image ID lists, or a path to a JSON file.
    /// batch_size : int
    ///     Number of images loaded per batch (default 12).
    /// port : int
    ///     Local server port (default 7860).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``image_dir`` is ``None`` and ``self.image_dir`` is also ``None``.
    /// ImportError
    ///     If browse dependencies are not installed
    ///     (``pip install hotcoco[browse]``).
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(self, image_dir=None, dt=None, iou_type='bbox', iou_thr=0.5, \
                          eval=None, slices=None, batch_size=12, port=7860)"
    )]
    fn browse(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        // The implementation lives in Python (`hotcoco.browse.browse_coco`):
        // it drives a web server, Jupyter display, and matplotlib — Python's
        // half of the project. Arguments pass through verbatim, so the
        // signature above is documentation and the Python function is the
        // arbiter (it raises `TypeError` for anything unexpected).
        let py = slf.py();
        let func = py.import("hotcoco.browse")?.getattr("browse_coco")?;
        let mut all_args: Vec<Py<PyAny>> = vec![slf.as_any().clone().unbind()];
        all_args.extend(args.iter().map(pyo3::Bound::unbind));
        func.call(PyTuple::new(py, all_args)?, kwargs)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

#[doc = "Evaluation parameters controlling IoU thresholds, area ranges, and max detections.

Attribute reads return **copies**: ``p.max_dets.append(200)`` appends to a
temporary list and is a silent no-op. Assign the whole attribute instead —
``p.max_dets = [1, 10, 100, 200]`` — the same way for every list-valued field
and its camelCase alias (``p.maxDets``, ``p.iouThrs``, ...). ``ev.params``
itself is a live object: assigning to its attributes configures the run."]
#[pyclass(name = "Params", from_py_object)]
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
    fn iou_type(&self) -> String {
        self.inner.iou_type.to_string()
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
        self.inner.area_ranges.iter().map(|ar| ar.range).collect()
    }
    #[setter]
    fn set_area_rng(&mut self, val: Vec<[f64; 2]>) {
        // Preserve existing labels when lengths match; otherwise use empty string labels.
        let existing: Vec<String> = self
            .inner
            .area_ranges
            .iter()
            .map(|ar| ar.label.clone())
            .collect();
        self.inner.area_ranges = val
            .into_iter()
            .enumerate()
            .map(|(i, range)| hotcoco_core::AreaRange {
                label: existing.get(i).cloned().unwrap_or_default(),
                range,
            })
            .collect();
    }
    #[getter]
    fn area_rng_lbl(&self) -> Vec<String> {
        self.inner
            .area_ranges
            .iter()
            .map(|ar| ar.label.clone())
            .collect()
    }
    #[setter]
    fn set_area_rng_lbl(&mut self, val: Vec<String>) {
        // Preserve existing ranges when lengths match; otherwise use zero ranges.
        let existing: Vec<[f64; 2]> = self.inner.area_ranges.iter().map(|ar| ar.range).collect();
        self.inner.area_ranges = val
            .into_iter()
            .enumerate()
            .map(|(i, label)| hotcoco_core::AreaRange {
                label,
                range: existing.get(i).copied().unwrap_or([0.0, 0.0]),
            })
            .collect();
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
    fn expand_dt(&self) -> bool {
        self.inner.expand_dt
    }
    #[setter]
    fn set_expand_dt(&mut self, val: bool) {
        self.inner.expand_dt = val;
    }
    #[getter]
    fn kpt_oks_sigmas(&self) -> Vec<f64> {
        self.inner.kpt_oks_sigmas.clone()
    }
    #[setter]
    fn set_kpt_oks_sigmas(&mut self, val: Vec<f64>) {
        self.inner.kpt_oks_sigmas = val;
    }

    // ---- camelCase aliases for pycocotools compatibility ----
    // PyO3 doesn't support macros or multiple #[getter] attrs on one method,
    // so each alias forwards manually.
    #[getter(iouType)]
    fn iou_type_camel(&self) -> String {
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
    s.parse::<hotcoco_core::IouType>()
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

// ---------------------------------------------------------------------------
// Hierarchy
// ---------------------------------------------------------------------------

#[doc = "Category hierarchy for Open Images evaluation.

Supports three construction methods:

- ``Hierarchy.from_parent_map({child_id: parent_id, ...})`` — explicit parent→child mapping.
- ``Hierarchy.from_file(\"hierarchy.json\", label_to_id={...})`` — parse OID hierarchy JSON.
- ``Hierarchy.from_dict(tree_dict, label_to_id={...})`` — from a Python dict (OID format).

Example::

    from hotcoco import COCO, COCOeval, Hierarchy

    h = Hierarchy.from_file(\"bbox_labels_600_hierarchy.json\")
    ev = COCOeval(coco_gt, coco_dt, \"bbox\", oid_style=True, hierarchy=h)
    ev.run()
"]
#[pyclass(name = "Hierarchy", from_py_object)]
#[derive(Clone)]
struct PyHierarchy {
    inner: hotcoco_core::Hierarchy,
}

#[pymethods]
impl PyHierarchy {
    /// Build from a parent map: ``{child_id: parent_id, ...}``
    #[staticmethod]
    fn from_parent_map(parent_map: HashMap<u64, u64>) -> Self {
        Self {
            inner: hotcoco_core::Hierarchy::from_parent_map(parent_map),
        }
    }

    /// Parse an Open Images hierarchy JSON file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the OID hierarchy JSON file (``LabelName``/``Subcategory`` format).
    /// label_to_id : dict, optional
    ///     Maps OID label strings such as ``"/m/dog"`` to category IDs.
    ///     If ``None``, all labels get virtual node IDs.
    #[staticmethod]
    #[pyo3(signature = (path, label_to_id=None))]
    fn from_file(path: &str, label_to_id: Option<HashMap<String, u64>>) -> PyResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let map = label_to_id.unwrap_or_default();
        let inner = hotcoco_core::Hierarchy::from_oid_json(&json, &map)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build from a Python dict representing the OID hierarchy tree.
    ///
    /// Parameters
    /// ----------
    /// tree_dict : dict
    ///     Dict with ``"LabelName"`` and ``"Subcategory"`` keys (the OID format).
    /// label_to_id : dict, optional
    ///     Maps OID label strings to category IDs.
    #[staticmethod]
    #[pyo3(signature = (tree_dict, label_to_id=None))]
    fn from_dict(
        py: Python<'_>,
        tree_dict: &Bound<'_, PyDict>,
        label_to_id: Option<HashMap<String, u64>>,
    ) -> PyResult<Self> {
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (tree_dict,))?.extract()?;
        let map = label_to_id.unwrap_or_default();
        let inner = hotcoco_core::Hierarchy::from_oid_json(&json_str, &map)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get ancestors of a category (inclusive of self).
    fn ancestors(&self, cat_id: u64) -> Vec<u64> {
        self.inner.ancestors(cat_id).to_vec()
    }

    /// Get direct children of a category.
    fn children(&self, cat_id: u64) -> Vec<u64> {
        self.inner.children(cat_id).to_vec()
    }

    /// Get the parent of a category, or ``None`` if root.
    fn parent(&self, cat_id: u64) -> Option<u64> {
        self.inner.parent(cat_id)
    }
}

// ---------------------------------------------------------------------------
// COCOeval
// ---------------------------------------------------------------------------

#[doc = "COCO evaluation engine.

Computes AP and AR metrics for bbox, segmentation, and keypoint predictions.
Also supports LVIS federated evaluation via ``lvis_style=True`` and
Open Images evaluation via ``oid_style=True``.

Standard COCO workflow::

    ev = COCOeval(coco_gt, coco_dt, \"bbox\")
    ev.evaluate()    # per-image IoU matching
    ev.accumulate()  # aggregate into precision/recall curves
    ev.summarize()   # print + store the 12 summary metrics in ev.stats

LVIS workflow::

    ev = COCOeval(coco_gt, coco_dt, \"segm\", lvis_style=True)
    ev.run()                    # evaluate + accumulate + summarize in one call
    results = ev.get_results()  # dict with 13 metrics: AP, APr, APc, APf, AR@300, ...

Open Images workflow::

    ev = COCOeval(coco_gt, coco_dt, \"bbox\", oid_style=True, hierarchy=h)
    ev.run()
    results = ev.results(per_class=True)  # dict with AP + per-class AP
"]
#[pyclass(name = "COCOeval")]
struct PyCOCOeval {
    inner: hotcoco_core::COCOeval,
    /// The `params` object handed to Python, held so every access returns the
    /// *same* object — `E.params.imgIds = [...]` (pycocotools' canonical idiom,
    /// used in its own demo) must configure the run, not mutate a temporary.
    /// `with_params` copies this object's state into `inner` before evaluation.
    params: Py<PyParams>,
    /// Cache for the `.eval` dict built by `get_eval` — pycocotools exposes
    /// `.eval` as an O(1) plain-attribute read, so rebuilding it (cloning the
    /// precision/recall/scores arrays, ~7.8MB each on val2017) on every access
    /// would be needlessly expensive. Invalidated by `evaluate()`,
    /// `accumulate()`, and `run()` — the only methods that change what
    /// `self.inner.accumulated()` returns.
    eval_cache: Option<Py<PyAny>>,
    /// The `params` object in force when `accumulate()` last ran — what
    /// `eval['params']` holds, by reference, the way pycocotools stores
    /// `self.params` into `self.eval`.
    eval_params: Option<Py<PyParams>>,
}

impl PyCOCOeval {
    /// Run `f` against the evaluator with `ev.params` reconciled on both sides:
    /// pull the Python-visible `Params` in, run, push the result back.
    ///
    /// **Every method that reads `params` — drivers, provenance, results, and
    /// the standalone analysis methods alike — must go through here**; a method
    /// that skips the sync silently ignores `ev.params` mutations. The push-back
    /// matters too: `evaluate()` resolves empty `img_ids`/`cat_ids` to the whole
    /// dataset and sorts them, and pycocotools likewise leaves the resolved
    /// lists on `params`, so a caller reading `ev.params.imgIds` afterwards sees
    /// what actually ran.
    fn with_params<R>(
        &mut self,
        py: Python<'_>,
        f: impl FnOnce(&mut hotcoco_core::COCOeval) -> R,
    ) -> R {
        self.inner.params.clone_from(&self.params.borrow(py).inner);
        let out = f(&mut self.inner);
        self.params
            .borrow_mut(py)
            .inner
            .clone_from(&self.inner.params);
        out
    }
}

#[pymethods]
impl PyCOCOeval {
    #[new]
    #[pyo3(signature = (coco_gt=None, coco_dt=None, iou_type=None, lvis_style=false, oid_style=false, hierarchy=None, **kwargs))]
    fn new(
        coco_gt: Option<PyRef<'_, PyCOCO>>,
        coco_dt: Option<PyRef<'_, PyCOCO>>,
        iou_type: Option<String>,
        lvis_style: bool,
        oid_style: bool,
        hierarchy: Option<&PyHierarchy>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        // pycocotools spells the constructor keywords `cocoGt` / `cocoDt` /
        // `iouType`, and consumers pass them that way — torchmetrics' backend
        // calls `COCOeval(gt, dt, iouType=...)`; found by the 1.0
        // third-party-consumer smoke test. Accept either spelling, reject
        // anything else.
        let mut coco_gt = coco_gt;
        let mut coco_dt = coco_dt;
        let mut iou_type = iou_type;
        if let Some(kw) = kwargs {
            for (key, value) in kw.iter() {
                let key: String = key.extract()?;
                match key.as_str() {
                    "cocoGt" if coco_gt.is_none() => coco_gt = Some(value.extract()?),
                    "cocoDt" if coco_dt.is_none() => coco_dt = Some(value.extract()?),
                    "iouType" if iou_type.is_none() => iou_type = Some(value.extract()?),
                    "cocoGt" | "cocoDt" | "iouType" => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                            "COCOeval() got '{key}' and its snake_case form — pass one"
                        )));
                    }
                    _ => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                            "COCOeval() got an unexpected keyword argument '{key}'"
                        )));
                    }
                }
            }
        }
        let (coco_gt, coco_dt) = match (coco_gt, coco_dt) {
            (Some(gt), Some(dt)) => (gt, dt),
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "COCOeval() requires ground truth and detections (coco_gt/cocoGt, coco_dt/cocoDt)",
                ));
            }
        };
        let iou_type = iou_type.ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err(
                "COCOeval() requires iou_type (or pycocotools' iouType)",
            )
        })?;

        if oid_style && lvis_style {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot use both oid_style and lvis_style",
            ));
        }

        let iou = parse_iou_type(&iou_type)?;
        let gt = hotcoco_core::COCO::from_dataset(coco_gt.inner.dataset.clone());
        let dt = hotcoco_core::COCO::from_dataset(coco_dt.inner.dataset.clone());

        let inner = if oid_style {
            if iou != hotcoco_core::IouType::Bbox {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "OID evaluation only supports bbox IoU type",
                ));
            }
            hotcoco_core::COCOeval::new_oid(gt, dt, hierarchy.map(|h| h.inner.clone()))
        } else if lvis_style {
            hotcoco_core::COCOeval::new_lvis(gt, dt, iou)
        } else {
            hotcoco_core::COCOeval::new(gt, dt, iou)
        };
        let params = Python::attach(|py| {
            Py::new(
                py,
                PyParams {
                    inner: inner.params.clone(),
                },
            )
        })?;
        Ok(PyCOCOeval {
            inner,
            params,
            eval_cache: None,
            eval_params: None,
        })
    }

    fn evaluate(&mut self, py: Python<'_>) {
        self.with_params(py, |ev| py.detach(|| ev.evaluate()));
        self.eval_cache = None;
    }

    fn accumulate(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.inner.eval_imgs().is_empty() {
            warn_user(
                py,
                "hotcoco: accumulate() called before evaluate(). \
                 Call evaluate() first or the results will be empty.",
            )?;
        }
        self.with_params(py, |ev| py.detach(|| ev.accumulate()));
        self.eval_cache = None;
        self.eval_params = Some(self.params.clone_ref(py));
        Ok(())
    }

    fn summarize(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.inner.accumulated().is_none() {
            warn_user(
                py,
                "hotcoco: summarize() called before accumulate(). \
                 Call evaluate() then accumulate() first.",
            )?;
        }

        // Re-raise the comparability warnings as real Python warnings.
        // `COCOeval::summarize` writes them with `eprintln!` to fd 2, which
        // bypasses `sys.stderr` — invisible in a Jupyter cell, uncatchable by
        // `warnings.catch_warnings`. Emitting here makes filters, -W flags, and
        // pytest.warns all work. Synced via `with_params` so a post-`evaluate()`
        // params mutation still reaches the comparability check.
        let deviations = self.with_params(py, |ev| ev.reference_deviations());
        for w in &deviations {
            warn_user(py, &format!("hotcoco: {w}"))?;
        }

        // `summarize_lines()` rather than `summarize()`: the latter also prints the
        // same warnings with `eprintln!`, and emitting them on fd 2 *and* as Python
        // warnings would be duplicate output — the fd-2 copy being invisible to
        // notebook users is the whole reason for the block above.
        self.with_params(py, |ev| {
            let _ = ev.summarize_lines();
        });
        Ok(())
    }

    #[doc = "Return summary metric lines as a list of strings without printing.

Computes stats (populating ``ev.stats``) and returns each formatted line.
Use this instead of ``summarize()`` when you need to capture or restyle the output.

>>> lines = ev.summary_lines()
>>> for line in lines:
...     print(line)
"]
    fn summary_lines(&mut self, py: Python<'_>) -> PyResult<Vec<String>> {
        if self.inner.accumulated().is_none() {
            warn_user(
                py,
                "hotcoco: summary_lines() called before accumulate(). \
                 Call evaluate() then accumulate() first.",
            )?;
        }
        Ok(self.with_params(py, hotcoco_core::COCOeval::summarize_lines))
    }

    #[doc = "Run the full evaluation pipeline: evaluate → accumulate → summarize.

Equivalent to calling the three methods in sequence. Primarily used with
LVIS pipelines (Detectron2, MMDetection) that expect a single ``run()`` call."]
    fn run(&mut self, py: Python<'_>) {
        self.with_params(py, |ev| py.detach(|| ev.run()));
        self.eval_cache = None;
        self.eval_params = Some(self.params.clone_ref(py));
    }

    #[getter]
    #[doc = "Category names added by hierarchy expansion (not in the original taxonomy).

Returns an empty list when not in OID mode or before ``evaluate()`` is called.
Use this to distinguish expanded ancestor categories from the model's native classes."]
    fn virtual_cat_names(&self) -> Vec<String> {
        self.inner
            .hierarchy
            .as_ref()
            .map(|h| h.virtual_names.values().cloned().collect())
            .unwrap_or_default()
    }

    #[doc = "Metric key names in canonical display order for this evaluation mode.

Returns the ordered list that drives ``summarize()`` and ``get_results()``.
Standard COCO bbox/segm returns 12 keys, keypoints 10, LVIS 13.

>>> ev.metric_keys()
['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
"]
    fn metric_keys(&mut self, py: Python<'_>) -> Vec<String> {
        self.with_params(py, |ev| {
            ev.metric_keys().into_iter().map(String::from).collect()
        })
    }

    #[doc = "The definition behind every metric key, in the same order as ``metric_keys()``.

Each dict describes one row of the summary table:

- ``name``: ``str`` — the key in ``get_results()``, such as ``'AP50'`` or ``'ARs'``.
- ``ap``: ``bool`` — ``True`` for Average Precision, ``False`` for Average Recall.
- ``iou_thr``: ``float | None`` — a single IoU threshold, or ``None`` when the
  metric averages over the whole sweep.
- ``area``: ``str`` — area-range label, such as ``'all'`` or ``'small'``.
- ``max_det``: ``int`` — detections per image this metric allows.
- ``freq_group``: ``str | None`` — ``'rare'``, ``'common'`` or ``'frequent'`` for
  the LVIS frequency-bucket APs, ``None`` otherwise. When set, the other axes
  are unused: the value is a mean per-category AP over that bucket.

This exists so a renderer never has to parse a metric name. ``'AP50'`` is not a
grammar — ``'AR1'`` is a max-det, ``'APs'`` an area range, ``'APr'`` an LVIS
frequency bucket, and a regex that gets one right gets the next one wrong. Ask
for the axes instead.

>>> next(d for d in ev.metric_defs() if d['name'] == 'AP50')
{'name': 'AP50', 'ap': True, 'iou_thr': 0.5, 'area': 'all', 'max_det': 100, 'freq_group': None}

Returns
-------
list of dict
    One dict per metric key, in canonical display order."]
    fn metric_defs(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let defs = self.with_params(py, |ev| ev.metric_defs());
        let list = PyList::empty(py);
        for d in defs {
            let entry = PyDict::new(py);
            entry.set_item("name", d.name)?;
            entry.set_item("ap", d.ap)?;
            entry.set_item("iou_thr", d.iou_thr)?;
            entry.set_item("area", d.area_lbl)?;
            entry.set_item("max_det", d.max_det)?;
            entry.set_item("freq_group", d.freq_group.map(freq_group_name))?;
            list.append(entry)?;
        }
        Ok(list.into_any().unbind())
    }

    #[doc = "Return summary metrics as a dict.

Must be called after ``summarize()`` (or ``run()``). Returns an empty dict
if ``summarize`` has not been run.

Parameters
----------
prefix : str or None
    If given, each key is prefixed as ``\"{prefix}/{metric}\"``.
per_class : bool
    If True, include per-category AP values keyed as ``\"AP/{cat_name}\"``
    (or ``\"{prefix}/AP/{cat_name}\"`` with a prefix)."]
    #[pyo3(signature = (prefix=None, per_class=false))]
    fn get_results(
        &mut self,
        py: Python<'_>,
        prefix: Option<&str>,
        per_class: bool,
    ) -> PyResult<Py<PyAny>> {
        let results = self.with_params(py, |ev| ev.get_results(prefix, per_class));
        Ok(map_to_dict(py, results)?.into_any().unbind())
    }

    #[doc = "Print a formatted results table to stdout.

For LVIS, matches the lvis-api ``print_results()`` style. Must be called after
``summarize()`` (or ``run()``)."]
    fn print_results(&self) {
        self.inner.print_results();
    }

    #[doc = "Return a full evaluation report as a dict.

Must be called after ``summarize()`` (or ``run()``). This is the shape every
hotcoco metric family reports in, so code that renders a detection report will
render a panoptic or tracking one unchanged.

Returns a dict with:

- ``task``: ``'detection'``
- ``provenance``: ``'parity_verified'`` only when this exact run is comparable to a
  reference implementation — bbox/segm/keypoints against pycocotools, or LVIS
  against lvis-api, **at reference parameters**. Everything else is
  ``'extension'``: oriented boxes (no reference protocol exists), Open Images
  (checked against the TensorFlow reference for group-of handling and AP, but
  missing the challenge's image-level-label rule), and any run with non-default
  ``iou_thrs``, ``rec_thrs``, ``max_dets``, area-range labels or bounds,
  ``use_cats=False``, or custom ``kpt_oks_sigmas``.
  Check this before presenting numbers as comparable to a published leaderboard.
- ``metrics``: summary metrics (AP, AP50, AP75, ...)
- ``per_class``: ``{class_name: {metric: value}}``
- ``per_group``: ``{group_name: {metric: value}}`` — LVIS frequency buckets in
  LVIS mode, empty otherwise
- ``curves``: ``{name: [values]}`` — one aggregate precision-recall curve per IoU
  threshold (``'pr@0.50'`` ...) plus the shared ``'rec_thrs'`` x-axis
- ``params``: the evaluation parameters used

The curves are the aggregate slice a chart draws, averaged over categories at
``area='all'`` and the largest ``max_dets``. For the full per-category arrays use
``.eval['precision']``, which is ~1M floats on COCO.

Examples
--------
>>> ev = COCOeval(gt, dt, 'bbox')
>>> ev.run()
>>> report = ev.report()
>>> report['provenance']
'parity_verified'
>>> plt.plot(report['curves']['rec_thrs'], report['curves']['pr@0.50'])

Returns
-------
dict
    Metrics, breakdowns, curves, provenance, and parameters."]
    fn report(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let report = self.with_params(py, |ev| ev.report()).map_err(to_pyerr)?;
        serde_to_py(py, &report)
    }

    #[doc = "Return this run's provenance without building a full report.

Identical to ``report()['provenance']`` and ``results()['provenance']`` —
``'parity_verified'`` or ``'extension'`` — but reads only the configuration, so
unlike those two it works **before** evaluating. Check it ahead of a long run
rather than discovering afterwards that the numbers cannot be published.

Never re-derive this from ``iou_type`` or the eval mode: parity is a property of
the whole configuration, so a run with custom ``iou_thrs`` is an extension even
in plain COCO bbox mode.

Treat anything other than ``'parity_verified'`` as not leaderboard-comparable.
Use ``reference_deviations()`` for the reasons.

Returns
-------
str
    ``'parity_verified'`` or ``'extension'``."]
    fn provenance(&mut self, py: Python<'_>) -> PyResult<String> {
        // Through `with_params`, so `ev.params.iouThrs = [...]` is visible here.
        // Reading `self.inner` directly returned `parity_verified` for a run that
        // was about to be an extension — the same defect `with_params` was
        // written to close, one call site later.
        let prov = self.with_params(py, |ev| ev.provenance());
        // Serialized rather than matched, so this and ``report()['provenance']``
        // cannot spell the same variant two ways. Falls back to the *non*-verified
        // side: an unrecognized provenance is a reason to caveat, not to certify.
        let value = serde_json::to_value(prov)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(value.as_str().unwrap_or("extension").to_string())
    }

    #[doc = "Whether these numbers can be presented as leaderboard-comparable.

The predicate behind ``provenance()``, exposed so renderers do not re-derive it
with a string compare. Default-deny: only a parity-verified configuration
qualifies, so a provenance variant added later reads as *needs a caveat* until
a renderer is taught what it means.

Returns
-------
bool
    ``True`` only when ``provenance()`` is ``'parity_verified'``."]
    fn is_benchmark_standard(&mut self, py: Python<'_>) -> bool {
        // Through `with_params` for the same reason as `provenance()`.
        self.with_params(py, |ev| ev.provenance().is_benchmark_standard())
    }

    #[doc = "Ways this run's parameters depart from the reference configuration.

Empty when the numbers are directly comparable to the reference implementation's
published output — pycocotools for COCO bbox/segm/keypoints, lvis-api for LVIS —
and non-empty otherwise, with one human-readable sentence per reason.

This is the same predicate that drives ``provenance()`` and the warnings
``summarize()`` writes to stderr, so a report cannot claim parity while the
warnings say otherwise.

Returns
-------
list of str
    One sentence per deviation; empty if the run is reference-comparable.

Examples
--------
>>> ev = COCOeval(gt, dt, 'bbox')
>>> ev.params.iou_thrs = [0.5]
>>> ev.run()
>>> ev.reference_deviations()
['iou_thrs differ from default (0.50:0.05:0.95). ...']"]
    fn reference_deviations(&mut self, py: Python<'_>) -> Vec<String> {
        self.with_params(py, |ev| ev.reference_deviations())
    }

    #[doc = "Return evaluation results as a dict.

Must be called after ``summarize()`` (or ``run()``). Returns a dict with:

- ``hotcoco_version``: hotcoco version string that produced these results.
- ``params``: evaluation parameters (iou_type, iou_thresholds, area_ranges, max_dets)
- ``metrics``: summary metrics (AP, AP50, AP75, and the rest)
- ``per_class``: per-category AP values (only if ``per_class=True``)

Parameters
----------
per_class : bool, optional
    If True, include per-category AP values. Default False.

Returns
-------
dict
    Serializable evaluation results."]
    #[pyo3(signature = (per_class=false))]
    fn results(&mut self, py: Python<'_>, per_class: bool) -> PyResult<Py<PyAny>> {
        let results = self
            .with_params(py, |ev| ev.results(per_class))
            .map_err(to_pyerr)?;
        serde_to_py(py, &results)
    }

    #[doc = "Save evaluation results to a JSON file.

Must be called after ``summarize()`` (or ``run()``).

Parameters
----------
path : str
    Output file path for the JSON results.
per_class : bool, optional
    If True, include per-category AP values. Default False.

Example
-------
::

    ev = COCOeval(coco_gt, coco_dt, \"bbox\")
    ev.run()
    ev.save_results(\"results.json\", per_class=True)
"]
    #[pyo3(signature = (path, per_class=false))]
    fn save_results(&mut self, py: Python<'_>, path: &str, per_class: bool) -> PyResult<()> {
        let results = self
            .with_params(py, |ev| ev.results(per_class))
            .map_err(to_pyerr)?;
        results.save(std::path::Path::new(path)).map_err(to_pyerr)?;
        Ok(())
    }

    #[doc = "Compute F-beta scores. Must be called after ``accumulate()`` (or ``run()``).

Returns three metrics analogous to AP/AP50/AP75, but based on the maximum
achievable F-beta score across confidence thresholds for each category.

Parameters
----------
beta : float, optional
    Trade-off between precision and recall. ``beta=1.0`` (default) gives
    equal weight (F1). ``beta<1`` weights precision more; ``beta>1`` weights
    recall more.

Returns
-------
dict[str, float]
    For ``beta=1.0``: ``{\"F1\": ..., \"F1_50\": ..., \"F1_75\": ...}``.
    For other beta values: ``{\"F<beta>\": ..., \"F<beta>_50\": ..., \"F<beta>_75\": ...}``
    — for example ``F0.5``, ``F0.5_50``, ``F0.5_75``.
    Returns an empty dict if ``accumulate()`` has not been run.

Examples
--------
>>> ev = COCOeval(gt, dt, \"bbox\")
>>> ev.run()
>>> ev.f_scores()
{'F1': 0.523, 'F1_50': 0.712, 'F1_75': 0.581}
>>> ev.f_scores(beta=0.5)   # precision-weighted
>>> ev.f_scores(beta=2.0)   # recall-weighted"]
    #[pyo3(signature = (beta = 1.0))]
    fn f_scores(&mut self, py: Python<'_>, beta: f64) -> PyResult<Py<PyAny>> {
        if self.inner.accumulated().is_none() {
            warn_user(
                py,
                "hotcoco: f_scores() called before accumulate(). \
                 Call evaluate() then accumulate() first. Returning empty dict.",
            )?;
        }
        let scores = self.with_params(py, |ev| ev.f_scores(beta));
        Ok(map_to_dict(py, scores)?.into_any().unbind())
    }

    /// The ground-truth dataset this evaluator was built from.
    ///
    /// **Each access returns a fresh copy** — two reads give two independent
    /// objects, and mutating one never reaches the evaluator. To evaluate
    /// against different ground truth, construct a new ``COCOeval``.
    #[getter]
    fn coco_gt(&self) -> PyCOCO {
        PyCOCO::without_image_dir(self.inner.coco_gt.dataset.clone())
    }

    #[getter(cocoGt)]
    fn coco_gt_camel(&self) -> PyCOCO {
        self.coco_gt()
    }

    /// The detection dataset this evaluator was built from.
    ///
    /// **Each access returns a fresh copy** — see ``coco_gt``.
    #[getter]
    fn coco_dt(&self) -> PyCOCO {
        PyCOCO::without_image_dir(self.inner.coco_dt.dataset.clone())
    }

    #[getter(cocoDt)]
    fn coco_dt_camel(&self) -> PyCOCO {
        self.coco_dt()
    }

    /// The evaluation parameters — the **same live object** on every access.
    ///
    /// Assigning to its attributes (``ev.params.imgIds = [...]``) configures
    /// the run; the evaluator re-reads them on every call. But attribute
    /// *reads* return copies, so ``ev.params.maxDets.append(200)`` mutates a
    /// temporary — assign the whole list instead. See ``Params``.
    #[getter]
    fn params(&self, py: Python<'_>) -> Py<PyParams> {
        self.params.clone_ref(py)
    }

    /// Replace the evaluation parameters.
    ///
    /// **Assignment stores a copy** of ``value`` — mutating the original object
    /// afterwards does not reach the evaluator. Either finish configuring
    /// before assigning, or mutate ``ev.params`` attributes after (that object
    /// is live: the evaluator re-reads it on every call).
    #[setter]
    fn set_params(&mut self, py: Python<'_>, params: &PyParams) -> PyResult<()> {
        self.inner.params = params.inner.clone();
        self.params = Py::new(
            py,
            PyParams {
                inner: params.inner.clone(),
            },
        )?;
        // Deliberately does not invalidate `eval_cache`: `eval['params']` holds
        // the object `accumulate()` ran with (`eval_params`), as pycocotools
        // does, so a reassigned `params` only takes effect on the next
        // evaluate()/accumulate().
        Ok(())
    }

    /// Summary metrics — pycocotools semantics.
    ///
    /// An empty list before ``summarize()`` has run, then a numpy ``float64``
    /// array (12 values for bbox/segm, 10 for keypoints, 13 for LVIS).
    #[getter]
    fn stats(&self, py: Python<'_>) -> Py<PyAny> {
        match self.inner.stats() {
            Some(s) => numpy::PyArray1::from_slice(py, s).into_any().unbind(),
            None => PyList::empty(py).into_any().unbind(),
        }
    }

    #[getter]
    fn eval_imgs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        eval_imgs_to_py(py, self.inner.eval_imgs())
    }

    #[getter(evalImgs)]
    fn eval_imgs_camel(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.eval_imgs(py)
    }

    /// The accumulated results — pycocotools semantics: a plain dict attribute,
    /// so the same object comes back on every access and in-place edits
    /// persist across reads. ``summarize()``, ``stats``, and ``results()``
    /// read the evaluator's own arrays, not this dict, so an edit here does
    /// not change the reported metrics. ``eval['params']`` is the ``params``
    /// object ``accumulate()`` ran with, held by reference as pycocotools
    /// holds ``self.params``. Rebuilt (and any prior in-place edits discarded)
    /// by ``evaluate()``, ``accumulate()``, and ``run()``.
    #[getter(eval)]
    fn get_eval(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(cached) = &self.eval_cache {
            return Ok(cached.clone_ref(py));
        }
        let params = match &self.eval_params {
            Some(p) => p.clone_ref(py),
            // `accumulate()` and `run()` always set `eval_params`; this only
            // covers a future mutation path that forgets to.
            None => Py::new(
                py,
                PyParams {
                    inner: self.inner.params.clone(),
                },
            )?,
        };
        let built = accumulated_eval_to_py(py, self.inner.accumulated(), params)?;
        self.eval_cache = Some(built.clone_ref(py));
        Ok(built)
    }

    #[doc = "Compute a per-category confusion matrix across all images.

Unlike ``evaluate()``, this method compares **all** detections in an image against
**all** ground truth boxes regardless of category, enabling cross-category confusion
analysis — the model keeps predicting ``dog`` on ``cat`` ground truth, say.

This method is standalone — no ``evaluate()`` call is needed first.

Returns a dict with:

- ``matrix``: ``np.ndarray`` of shape ``(K+1, K+1)``, dtype ``uint64``.  Rows = GT
  category, cols = predicted category.  Index ``K`` is the background row/column
  (unmatched GTs = false negatives end up in the background column; unmatched DTs =
  false positives end up in the background row).
- ``normalized``: ``np.ndarray`` of shape ``(K+1, K+1)``, dtype ``float64``.
  Each row is divided by its row sum (zero rows stay zero).
- ``cat_ids``: ``list[int]`` — category IDs for rows/cols ``0..K-1``.
- ``cat_names``: ``list[str]`` — category names for rows/cols ``0..K-1``, same order as ``cat_ids``.
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
        &mut self,
        py: Python<'_>,
        iou_thr: f64,
        max_det: Option<usize>,
        min_score: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let cm = self.with_params(py, |ev| {
            py.detach(|| ev.confusion_matrix(iou_thr, max_det, min_score))
        });
        let k = cm.num_cats + 1;

        // normalized: Vec<f64> → numpy float64, reshaped to (k, k)
        let norm_arr = f64_array(py, cm.normalized(), [k, k])?;

        let matrix_arr = confusion_counts_to_py(py, cm.matrix, k)?;

        let dict = PyDict::new(py);
        dict.set_item("matrix", matrix_arr)?;
        dict.set_item("normalized", norm_arr)?;
        dict.set_item("cat_ids", cm.cat_ids)?;
        dict.set_item("cat_names", cm.cat_names)?;
        dict.set_item("num_cats", cm.num_cats)?;
        dict.set_item("iou_thr", cm.iou_thr)?;

        Ok(dict.into_any().unbind())
    }

    #[doc = "Decompose detection errors into TIDE error types.\n\
\n\
Requires :meth:`evaluate` to have been called first.\n\
\n\
Returns a dict with keys:\n\
\n\
- ``delta_ap``: dict mapping error type → ΔAP (how much AP improves if fixed).\n\
  Keys: ``'Cls'``, ``'Loc'``, ``'Both'``, ``'Dupe'``, ``'Bkg'``, ``'Miss'``, ``'FP'``, ``'FN'``.\n\
  ``'FP'`` and ``'FN'`` are tidecv's special oracles: ``'FP'`` suppresses every\n\
  false positive (perfect precision, recall untouched); ``'FN'`` drops every\n\
  missed ground truth from the denominator (perfect recall, precision\n\
  untouched — a superset of ``'Miss'``).\n\
- ``counts``: dict mapping error type → count across all categories.\n\
  Keys: ``'Cls'``, ``'Loc'``, ``'Both'``, ``'Dupe'``, ``'Bkg'``, ``'Miss'``.\n\
- ``ap_base``: float — baseline AP at ``pos_thr`` (mean over categories with GT).\n\
- ``pos_thr``: float — IoU threshold used for TP/FP classification.\n\
- ``bg_thr``: float — background IoU threshold used for Loc/Both/Bkg discrimination.\n\
\n\
Parameters\n\
----------\n\
pos_thr : float, optional\n\
    IoU threshold for a match.  Default ``0.5``.\n\
bg_thr : float, optional\n\
    Minimum IoU to consider any GT overlap (below = pure background).  Default ``0.1``.\n\
\n\
Example\n\
-------\n\
::\n\
\n\
    ev = COCOeval(coco_gt, coco_dt, \"bbox\")\n\
    ev.evaluate()\n\
    result = ev.tide_errors(pos_thr=0.5, bg_thr=0.1)\n\
    print(result['delta_ap'])\n\
    print(result['counts'])\n\
"]
    #[pyo3(signature = (pos_thr=0.5, bg_thr=0.1))]
    fn tide_errors(&mut self, py: Python<'_>, pos_thr: f64, bg_thr: f64) -> PyResult<Py<PyAny>> {
        let te = self
            .with_params(py, |ev| py.detach(|| ev.tide_errors(pos_thr, bg_thr)))
            .map_err(to_pyerr)?;

        // `TideErrors` uses `BTreeMap`, so iteration is already key-ordered and
        // the dict comes out byte-stable without sorting here.
        let delta_ap = map_to_dict(py, &te.delta_ap)?;
        let counts = map_to_dict(py, &te.counts)?;

        let dict = PyDict::new(py);
        dict.set_item("delta_ap", delta_ap)?;
        dict.set_item("counts", counts)?;
        dict.set_item("ap_base", te.ap_base)?;
        dict.set_item("pos_thr", te.pos_thr)?;
        dict.set_item("bg_thr", te.bg_thr)?;

        Ok(dict.into_any().unbind())
    }

    #[doc = "Compute confidence calibration metrics.\n\
\n\
Measures how well confidence scores predict actual detection accuracy.\n\
Requires :meth:`evaluate` to have been called first.\n\
\n\
Returns a dict with keys:\n\
\n\
- ``ece``: float — Expected Calibration Error (weighted mean of per-bin gaps).\n\
- ``mce``: float — Maximum Calibration Error (worst per-bin gap).\n\
- ``bins``: list of dicts, each with ``bin_lower``, ``bin_upper``,\n\
  ``avg_confidence``, ``avg_accuracy``, ``count``.\n\
- ``per_category``: dict mapping category name → ECE for that category.\n\
- ``iou_threshold``: float — IoU threshold used to define correctness.\n\
- ``n_bins``: int — number of bins.\n\
- ``num_detections``: int — total detections analyzed.\n\
\n\
Parameters\n\
----------\n\
n_bins : int, optional\n\
    Number of equal-width confidence bins. Default ``10``.\n\
iou_threshold : float, optional\n\
    IoU threshold for TP/FP classification. Default ``0.5``.\n\
    Must match one of the thresholds in ``params.iouThrs``.\n\
\n\
Example\n\
-------\n\
::\n\
\n\
    ev = COCOeval(coco_gt, coco_dt, \"bbox\")\n\
    ev.evaluate()\n\
    cal = ev.calibration(n_bins=10, iou_threshold=0.5)\n\
    print(f\"ECE: {cal['ece']:.4f}\")\n\
    print(f\"MCE: {cal['mce']:.4f}\")\n\
"]
    #[pyo3(signature = (n_bins=10, iou_threshold=0.5))]
    fn calibration(
        &mut self,
        py: Python<'_>,
        n_bins: usize,
        iou_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let cal = self
            .with_params(py, |ev| py.detach(|| ev.calibration(n_bins, iou_threshold)))
            .map_err(to_pyerr)?;

        let bins_list = PyList::empty(py);
        for b in &cal.bins {
            bins_list.append(convert::calibration_bin_to_py(py, b)?)?;
        }

        // Map category IDs to names for per_category. Through `COCO::cat_name`,
        // so an id the ground truth does not carry gets the same fallback name
        // here as in `image_diagnostics` and the confusion matrix.
        let per_cat = map_to_dict(
            py,
            cal.per_category
                .iter()
                .map(|(&cat_id, &ece)| (self.inner.coco_gt.cat_name(cat_id), ece)),
        )?;

        let dict = PyDict::new(py);
        dict.set_item("ece", cal.ece)?;
        dict.set_item("mce", cal.mce)?;
        dict.set_item("bins", bins_list)?;
        dict.set_item("per_category", per_cat)?;
        dict.set_item("iou_threshold", cal.iou_threshold)?;
        dict.set_item("n_bins", cal.n_bins)?;
        dict.set_item("num_detections", cal.num_detections)?;

        Ok(dict.into_any().unbind())
    }

    /// Re-accumulate metrics for named image subsets without recomputing IoU.
    ///
    /// ``slices`` is either a dict of ``{name: [img_ids]}`` or a callable that
    /// takes an image dict and returns a slice name (or ``None`` to skip).
    /// Returns a dict with one entry per slice plus ``"_overall"``.
    #[pyo3(signature = (slices))]
    fn slice_by(&mut self, py: Python<'_>, slices: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        use std::collections::HashMap;

        // If slices is callable, group images by return value
        let slice_map: HashMap<String, Vec<u64>> = if slices.is_callable() {
            let gt_images = &self.inner.coco_gt.dataset.images;
            let mut groups: HashMap<String, Vec<u64>> = HashMap::new();
            for img in gt_images {
                // The callable sees the *full* image dict — every standard
                // field plus any custom keys — so slicing on user metadata
                // (`img["weather"]`, `img["camera"]`) works.
                let img_dict = image_to_py(py, img)?;
                let result = slices.call1((img_dict,))?;
                if result.is_none() {
                    continue;
                }
                let name: String = result.extract()?;
                groups.entry(name).or_default().push(img.id);
            }
            groups
        } else {
            let dict = slices.cast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("slices must be a dict or callable")
            })?;
            let mut map = HashMap::new();
            for (k, v) in dict {
                let name: String = k.extract()?;
                let ids: Vec<u64> = v.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "slice values must be sequences of image IDs",
                    )
                })?;
                map.insert(name, ids);
            }
            map
        };

        let results = self
            .with_params(py, |ev| py.detach(|| ev.slice_by(slice_map)))
            .map_err(to_pyerr)?;

        let out = PyDict::new(py);

        let to_dict = |sr: &hotcoco_core::SliceResult, py: Python<'_>| -> PyResult<Py<PyAny>> {
            let d = map_to_dict(py, &sr.metrics)?;
            d.set_item("num_images", sr.num_images)?;
            d.set_item("delta", map_to_dict(py, &sr.delta)?)?;
            Ok(d.into_any().unbind())
        };

        out.set_item("_overall", to_dict(&results.overall, py)?)?;
        for sr in &results.slices {
            out.set_item(&sr.name, to_dict(sr, py)?)?;
        }

        Ok(out.into_any().unbind())
    }

    /// Per-image diagnostics: annotation TP/FP/FN index, per-image F1 and AP scores,
    /// error profiles, and label error candidates.
    ///
    /// Requires ``evaluate()`` to have been called first.
    ///
    /// Parameters
    /// ----------
    /// iou_thr : float, default 0.5
    ///     IoU threshold for TP/FP classification (snapped to nearest in params).
    /// score_thr : float, default 0.5
    ///     Minimum detection confidence for label error candidates.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Keys: ``dt_status``, ``gt_status``, ``dt_match``, ``gt_match``,
    ///     ``img_summary``, ``label_errors``, ``iou_thr``.
    #[pyo3(signature = (iou_thr=0.5, score_thr=0.5))]
    fn image_diagnostics(
        &mut self,
        py: Python<'_>,
        iou_thr: f64,
        score_thr: f64,
    ) -> PyResult<Py<PyAny>> {
        let diag = self
            .with_params(py, |ev| {
                py.detach(|| ev.image_diagnostics(iou_thr, score_thr))
            })
            .map_err(to_pyerr)?;

        // dt_status: {ann_id: "tp" | "fp"}
        let dt_status = PyDict::new(py);
        for (&id, status) in &diag.annotations.dt_status {
            let s = match status {
                hotcoco_core::DtStatus::Tp => "tp",
                hotcoco_core::DtStatus::Fp => "fp",
            };
            dt_status.set_item(id, s)?;
        }

        // gt_status: {ann_id: "matched" | "fn"}
        let gt_status = PyDict::new(py);
        for (&id, status) in &diag.annotations.gt_status {
            let s = match status {
                hotcoco_core::GtStatus::Matched => "matched",
                hotcoco_core::GtStatus::Fn => "fn",
            };
            gt_status.set_item(id, s)?;
        }

        // dt_match / gt_match: {id: id}
        let dt_match = PyDict::new(py);
        for (&dt_id, &gt_id) in &diag.annotations.dt_match {
            dt_match.set_item(dt_id, gt_id)?;
        }
        let gt_match = PyDict::new(py);
        for (&gt_id, &dt_id) in &diag.annotations.gt_match {
            gt_match.set_item(gt_id, dt_id)?;
        }

        // img_summary: {img_id: {tp, fp, fn, f1, ap, error_profile}}
        let img_summary = PyDict::new(py);
        for (&img_id, summary) in &diag.images {
            let d = PyDict::new(py);
            d.set_item("tp", summary.tp)?;
            d.set_item("fp", summary.fp)?;
            d.set_item("fn", summary.fn_count)?;
            d.set_item("f1", summary.f1)?;
            d.set_item("ap", summary.ap)?;
            let profile = match summary.error_profile {
                hotcoco_core::ErrorProfile::Perfect => "perfect",
                hotcoco_core::ErrorProfile::FpHeavy => "fp_heavy",
                hotcoco_core::ErrorProfile::FnHeavy => "fn_heavy",
                hotcoco_core::ErrorProfile::Mixed => "mixed",
            };
            d.set_item("error_profile", profile)?;
            img_summary.set_item(img_id, d)?;
        }

        // label_errors: [{image_id, dt_id, dt_score, dt_category, gt_id, gt_category, iou, type}]
        let label_errors = PyList::empty(py);
        for le in &diag.label_errors {
            let d = PyDict::new(py);
            d.set_item("image_id", le.image_id)?;
            d.set_item("dt_id", le.dt_id)?;
            d.set_item("dt_score", le.dt_score)?;

            // Category names for display, through the one owner of the
            // unknown-id fallback so every surface spells it the same way.
            d.set_item(
                "dt_category",
                self.inner.coco_gt.cat_name(le.dt_category_id),
            )?;
            d.set_item("dt_category_id", le.dt_category_id)?;

            match le.gt_id {
                Some(gt_id) => {
                    d.set_item("gt_id", gt_id)?;
                    let gt_cat_name = le
                        .gt_category_id
                        .map(|cid| self.inner.coco_gt.cat_name(cid));
                    d.set_item("gt_category", gt_cat_name)?;
                    d.set_item("gt_category_id", le.gt_category_id)?;
                }
                None => {
                    d.set_item("gt_id", py.None())?;
                    d.set_item("gt_category", py.None())?;
                    d.set_item("gt_category_id", py.None())?;
                }
            }

            d.set_item("iou", le.iou)?;
            let error_type = match le.error_type {
                hotcoco_core::LabelErrorType::WrongLabel => "wrong_label",
                hotcoco_core::LabelErrorType::MissingAnnotation => "missing_annotation",
            };
            d.set_item("type", error_type)?;
            label_errors.append(d)?;
        }

        let result = PyDict::new(py);
        result.set_item("dt_status", dt_status)?;
        result.set_item("gt_status", gt_status)?;
        result.set_item("dt_match", dt_match)?;
        result.set_item("gt_match", gt_match)?;
        result.set_item("img_summary", img_summary)?;
        result.set_item("label_errors", label_errors)?;
        result.set_item("iou_thr", diag.iou_thr)?;
        result.set_item("score_thr", score_thr)?;

        Ok(result.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// EvalImg / AccumulatedEval → Python converters
// ---------------------------------------------------------------------------

fn eval_imgs_to_py(
    py: Python<'_>,
    eval_imgs: &[Option<hotcoco_core::EvalImg>],
) -> PyResult<Py<PyAny>> {
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

/// Per-threshold rows of a [`hotcoco_core::ThreshMatrix`] as a vec of row
/// slices — `set_item` converts this to a Python list of lists directly, with
/// no owned copy on the Rust side.
fn matrix_rows<T>(m: &hotcoco_core::ThreshMatrix<T>) -> Vec<&[T]> {
    m.iter_rows().collect()
}

fn eval_img_to_py(py: Python<'_>, e: &hotcoco_core::EvalImg) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("image_id", e.image_id)?;
    dict.set_item("category_id", e.category_id)?;
    dict.set_item("aRng", e.area_rng.to_vec())?;
    dict.set_item("maxDet", e.max_det)?;
    dict.set_item("dtIds", e.dt_ids.clone())?;
    dict.set_item("gtIds", e.gt_ids.clone())?;
    // The ThreshMatrix fields are [T x D] / [T x G]; each row becomes one Python
    // list, so the shape Python sees is the same list-of-lists as before the
    // flat-storage change.
    dict.set_item("dtMatches", matrix_rows(&e.dt_matches))?;
    dict.set_item("gtMatches", matrix_rows(&e.gt_matches))?;
    dict.set_item("dtMatched", matrix_rows(&e.dt_matched))?;
    dict.set_item("gtMatched", matrix_rows(&e.gt_matched))?;
    dict.set_item("dtScores", e.dt_scores.clone())?;
    dict.set_item("dtIgnore", matrix_rows(&e.dt_ignore))?;
    dict.set_item("gtIgnore", e.gt_ignore.clone())?;
    dict.set_item("gtInDenominator", e.gt_in_denominator.clone())?;
    Ok(dict.into_any().unbind())
}

fn accumulated_eval_to_py(
    py: Python<'_>,
    eval: Option<&hotcoco_core::AccumulatedEval>,
    params: Py<PyParams>,
) -> PyResult<Py<PyAny>> {
    match eval {
        None => Ok(py.None()),
        Some(e) => {
            let dict = PyDict::new(py);

            // Key order mirrors pycocotools' COCOeval.eval dict for drop-in fidelity:
            // params, counts, date, precision, recall, scores.

            // `params`: the Params object the evaluator ran with, by reference —
            // pycocotools stores `self.params` here, not a copy, so
            // `ev.eval['params'] is ev.params` and attribute edits show through.
            dict.set_item("params", params)?;

            let counts = vec![e.shape.t, e.shape.r, e.shape.k, e.shape.a, e.shape.m];
            dict.set_item("counts", counts)?;

            // `date`: byte-for-byte the pycocotools format
            // (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')). Non-deterministic
            // by construction, so the golden-fixture harness excludes it from comparison.
            let now = py
                .import("datetime")?
                .getattr("datetime")?
                .call_method0("now")?;
            let date_str: String = now
                .call_method1("strftime", ("%Y-%m-%d %H:%M:%S",))?
                .extract()?;
            dict.set_item("date", date_str)?;

            // precision / scores: flat Vec<f64> → numpy (T, R, K, A, M);
            // recall: flat Vec<f64> → numpy (T, K, A, M).
            let trkam = [e.shape.t, e.shape.r, e.shape.k, e.shape.a, e.shape.m];
            let tkam = [e.shape.t, e.shape.k, e.shape.a, e.shape.m];
            dict.set_item("precision", f64_array(py, e.precision.clone(), trkam)?)?;
            dict.set_item("recall", f64_array(py, e.recall.clone(), tkam)?)?;
            dict.set_item("scores", f64_array(py, e.scores.clone(), trkam)?)?;

            Ok(dict.into_any().unbind())
        }
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Patch `sys.modules` so that `from pycocotools.coco import COCO` and friends
/// transparently use hotcoco.
///
/// The submodule names are also set as *attributes* on the hotcoco module:
/// `import pycocotools.coco as pc` binds via `getattr(pycocotools, "coco")`,
/// not `sys.modules["pycocotools.coco"]`, so the sys.modules entries alone
/// cover the `from pycocotools.coco import COCO` form but not the `import
/// … as` form. (`mask` is already a real attribute.)
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
    hotcoco.setattr("coco", &hotcoco)?;
    hotcoco.setattr("cocoeval", &hotcoco)?;
    Ok(())
}

/// Patch `sys.modules` so that `from lvis import LVIS, LVISEval, LVISResults`
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
    // Attribute aliases for the `import lvis.eval as …` binding form — see
    // init_as_pycocotools.
    hotcoco.setattr("coco", &hotcoco)?;
    hotcoco.setattr("eval", &hotcoco)?;
    hotcoco.setattr("results", &hotcoco)?;
    Ok(())
}

/// Compare two model evaluations on the same dataset.
///
/// Both evaluators must have had ``evaluate()`` called. Returns a dict with
/// metric deltas, per-category AP deltas, and optional bootstrap confidence
/// intervals on the summary metric deltas.
///
/// Parameters
/// ----------
/// eval_a : COCOeval
///     First model evaluation (the "baseline").
/// eval_b : COCOeval
///     Second model evaluation (the "improved").
/// n_bootstrap : int
///     Number of bootstrap samples for confidence intervals (0 = disabled).
/// seed : int
///     Random seed for bootstrap reproducibility.
/// confidence : float
///     Confidence level for bootstrap intervals, for example 0.95 for a 95% CI.
///
/// Returns
/// -------
/// dict
///     Keys: ``metrics_a``, ``metrics_b``, ``deltas``, ``ci`` (None if
///     bootstrap disabled), ``per_category``, ``n_bootstrap``, ``num_images``.
#[pyfunction]
#[pyo3(
    signature = (eval_a, eval_b, n_bootstrap=0, seed=42, confidence=0.95),
    text_signature = "(eval_a, eval_b, n_bootstrap=0, seed=42, confidence=0.95)"
)]
fn compare(
    py: Python<'_>,
    eval_a: &PyCOCOeval,
    eval_b: &PyCOCOeval,
    n_bootstrap: usize,
    seed: u64,
    confidence: f64,
) -> PyResult<Py<PyAny>> {
    let opts = hotcoco_core::CompareOpts {
        n_bootstrap,
        seed,
        confidence,
    };
    // `ValueError` for every failure here: not-yet-evaluated inputs and
    // mismatched parameter catalogs (iou_thrs, rec_thrs, max_dets, area
    // ranges) are both bad arguments to *this* call, not runtime faults.
    let result = py
        .detach(|| hotcoco_core::compare(&eval_a.inner, &eval_b.inner, &opts))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let metrics_a = map_to_dict(py, &result.metrics_a)?;
    let metrics_b = map_to_dict(py, &result.metrics_b)?;
    let deltas = map_to_dict(py, &result.deltas)?;

    let ci = match &result.ci {
        Some(ci_map) => {
            let d = PyDict::new(py);
            for (k, ci) in ci_map {
                let ci_dict = PyDict::new(py);
                ci_dict.set_item("lower", ci.lower)?;
                ci_dict.set_item("upper", ci.upper)?;
                ci_dict.set_item("confidence", ci.confidence)?;
                ci_dict.set_item("prob_positive", ci.prob_positive)?;
                ci_dict.set_item("std_err", ci.std_err)?;
                d.set_item(k, ci_dict)?;
            }
            d.into_any().unbind()
        }
        None => py.None(),
    };

    let per_cat_list = PyList::empty(py);
    for cat in &result.per_category {
        let d = PyDict::new(py);
        d.set_item("cat_id", cat.cat_id)?;
        d.set_item("cat_name", &cat.cat_name)?;
        d.set_item("ap_a", cat.ap_a)?;
        d.set_item("ap_b", cat.ap_b)?;
        d.set_item("delta", cat.delta)?;
        per_cat_list.append(d)?;
    }

    let dict = PyDict::new(py);
    dict.set_item("metric_keys", &result.metric_keys)?;
    dict.set_item("metrics_a", metrics_a)?;
    dict.set_item("metrics_b", metrics_b)?;
    dict.set_item("deltas", deltas)?;
    dict.set_item("ci", ci)?;
    dict.set_item("per_category", per_cat_list)?;
    dict.set_item("n_bootstrap", result.n_bootstrap)?;
    dict.set_item("num_images", result.num_images)?;

    Ok(dict.into_any().unbind())
}

#[pymodule]
fn hotcoco(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCOCO>()?;
    m.add_class::<PyCOCOeval>()?;
    m.add_class::<PyParams>()?;
    m.add_class::<PyHierarchy>()?;
    m.add_function(wrap_pyfunction!(init_as_pycocotools, m)?)?;
    m.add_function(wrap_pyfunction!(init_as_lvis, m)?)?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;

    // mask submodule
    let mask_mod = PyModule::new(py, "mask")?;
    mask_mod.add_function(wrap_pyfunction!(mask::encode, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::decode, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::area, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::to_bbox, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::to_bbox_camel, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::merge, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::iou, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::bbox_iou, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::fr_poly, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::fr_poly_camel, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::fr_bbox, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::fr_bbox_camel, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::rle_to_string, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::rle_from_string, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::fr_py_objects, &mask_mod)?)?;
    mask_mod.add_function(wrap_pyfunction!(mask::fr_py_objects_snake, &mask_mod)?)?;
    m.add_submodule(&mask_mod)?;

    // The functional layer — metric functions and matching kernels, callable
    // without a COCOeval. `__init__.py` also registers these in sys.modules so
    // `import hotcoco.metrics` works and not just `from hotcoco import metrics`;
    // add_submodule only sets an attribute.
    let metrics_mod = metrics::register(py)?;
    m.add_submodule(&metrics_mod)?;
    let primitives_mod = primitives::register(py)?;
    m.add_submodule(&primitives_mod)?;

    Ok(())
}
