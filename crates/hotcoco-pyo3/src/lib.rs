use std::collections::HashMap;
use std::path::Path;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use mask::transpose_mask;

use hotcoco_core::Annotation;

mod convert;
mod mask;

use convert::{
    annotation_to_py, category_to_py, dataset_stats_to_py, image_to_py, py_to_annotation,
    py_to_dataset, rle_to_py,
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
#[pyclass(name = "COCO", subclass)]
struct PyCOCO {
    inner: hotcoco_core::COCO,
    /// Root directory for image files. Used by `browse()` and `coco explore`.
    /// Set at construction time or assign directly: ``coco.image_dir = "/data/images"``.
    #[pyo3(get, set)]
    image_dir: Option<String>,
}

impl Clone for PyCOCO {
    fn clone(&self) -> Self {
        PyCOCO {
            inner: hotcoco_core::COCO::from_dataset(self.inner.dataset.clone()),
            image_dir: self.image_dir.clone(),
        }
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
                    hotcoco_core::COCO::new(Path::new(&path))
                        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))?
                } else if let Ok(dict) = obj.downcast::<PyDict>() {
                    let dataset = py_to_dataset(dict)?;
                    hotcoco_core::COCO::from_dataset(dataset)
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "COCO() argument must be a file path (str) or dataset dict",
                    ));
                }
            }
            None => hotcoco_core::COCO::from_dataset(hotcoco_core::Dataset {
                info: None,
                images: vec![],
                annotations: vec![],
                categories: vec![],
                licenses: vec![],
            }),
        };
        Ok(PyCOCO { inner, image_dir })
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
                .map(|inner| PyCOCO {
                    inner,
                    image_dir: self.image_dir.clone(),
                })
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")));
        }

        // Case 2: list of annotation dicts
        if let Ok(list) = res.downcast::<PyList>() {
            let anns = list
                .iter()
                .map(|item| {
                    let dict = item.downcast::<PyDict>().map_err(|_| {
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
                .map(|inner| PyCOCO {
                    inner,
                    image_dir: self.image_dir.clone(),
                })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")));
        }

        // Case 3: numpy float64 array, shape (N, 6) or (N, 7)
        //   (N, 6): [image_id, x, y, w, h, score]           — category_id defaults to 1
        //   (N, 7): [image_id, x, y, w, h, score, cat_id]   — matches pycocotools loadNumpyAnnotations
        if let Ok(arr) = res.downcast::<PyArray2<f64>>() {
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
                    area: None,
                    segmentation: None,
                    iscrowd: false,
                    keypoints: None,
                    num_keypoints: None,
                    is_group_of: None,
                })
                .collect::<Vec<_>>();
            return self
                .inner
                .load_res_anns(anns)
                .map(|inner| PyCOCO {
                    inner,
                    image_dir: self.image_dir.clone(),
                })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "load_res expects a file path (str), list of annotation dicts, \
             or numpy float64 array of shape (N, 6) or (N, 7)",
        ))
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
        let row_major = transpose_mask(&col_major, w, h);
        let flat = numpy::PyArray1::from_vec(py, row_major);
        let arr = flat.reshape([h, w])?;
        Ok(arr.unbind())
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let s = self.inner.stats();
        dataset_stats_to_py(py, &s)
    }

    /// Validate this dataset for structural errors, quality warnings, and
    /// distribution issues. If ``dt`` is provided, also checks GT/DT compatibility
    /// (e.g., mismatched image or category IDs).
    ///
    /// Returns a dict with ``"errors"``, ``"warnings"``, and ``"summary"`` keys.
    #[pyo3(signature = (dt=None))]
    fn healthcheck(&self, py: Python<'_>, dt: Option<&PyCOCO>) -> PyResult<PyObject> {
        let report = match dt {
            Some(dt_coco) => self.inner.healthcheck_compatibility(&dt_coco.inner),
            None => self.inner.healthcheck(),
        };
        let json_str = serde_json::to_string(&report)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json_str,))?;
        Ok(result.into())
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
            image_dir: self.image_dir.clone(),
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
            image_dir: None, // inputs may come from different directories
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
                image_dir: self.image_dir.clone(),
            },
        )?;
        let val_py = Py::new(
            py,
            PyCOCO {
                inner: hotcoco_core::COCO::from_dataset(val_ds),
                image_dir: self.image_dir.clone(),
            },
        )?;
        if let Some(test_ds) = test_ds {
            let test_py = Py::new(
                py,
                PyCOCO {
                    inner: hotcoco_core::COCO::from_dataset(test_ds),
                    image_dir: self.image_dir.clone(),
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
            image_dir: self.image_dir.clone(),
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
    fn load_res_camel(&self, res: &Bound<'_, PyAny>) -> PyResult<PyCOCO> {
        self.load_res(res)
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

    /// Convert this dataset to YOLO label format.
    ///
    /// Writes one ``.txt`` label file per image into ``output_dir`` (named by
    /// the image filename stem, e.g. ``000042.jpg`` → ``000042.txt``), plus a
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
    ///     ``{"images": int, "annotations": int, "skipped_crowd": int, "missing_bbox": int}``
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If any image has ``width == 0`` or ``height == 0`` (normalization
    ///     requires valid dimensions).
    ///
    /// Examples
    /// --------
    /// >>> coco = COCO("instances_val2017.json")
    /// >>> stats = coco.to_yolo("labels/val2017/")
    /// >>> print(stats)
    /// {'images': 5000, 'annotations': 36781, 'skipped_crowd': 12, 'missing_bbox': 0}
    fn to_yolo(&self, py: Python<'_>, output_dir: &str) -> PyResult<PyObject> {
        let stats = hotcoco_core::convert::coco_to_yolo(&self.inner.dataset, Path::new(output_dir))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("images", stats.images)?;
        dict.set_item("annotations", stats.annotations)?;
        dict.set_item("skipped_crowd", stats.skipped_crowd)?;
        dict.set_item("missing_bbox", stats.missing_bbox)?;
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
    /// RuntimeError
    ///     If ``data.yaml`` is missing or a label file cannot be parsed.
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
        use std::collections::HashMap;

        let py = cls.py();
        let mut image_dims: HashMap<String, (u32, u32)> = HashMap::new();

        if let Some(dir) = images_dir {
            let pil_image = py.import("PIL.Image").map_err(|_| {
                pyo3::exceptions::PyImportError::new_err(
                    "Pillow is required to read image dimensions. \
                     Install it with: pip install Pillow",
                )
            })?;

            let read_dir = std::fs::read_dir(dir).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("cannot read images_dir: {e}"))
            })?;

            let img_exts = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"];
            for entry in read_dir.flatten() {
                let path = entry.path();
                let ext_lower = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_lowercase());
                if let Some(ext) = ext_lower {
                    if img_exts.contains(&ext.as_str()) {
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
                }
            }
        }

        hotcoco_core::convert::yolo_to_coco(Path::new(yolo_dir), &image_dims)
            .map(|ds| PyCOCO {
                inner: hotcoco_core::COCO::from_dataset(ds),
                image_dir: None,
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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

    #[getter]
    fn imgs(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for img in &self.inner.dataset.images {
            dict.set_item(img.id, image_to_py(py, img)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    #[getter]
    fn anns(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for ann in &self.inner.dataset.annotations {
            dict.set_item(ann.id, annotation_to_py(py, ann)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    #[getter]
    fn cats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for cat in &self.inner.dataset.categories {
            dict.set_item(cat.id, category_to_py(py, cat)?)?;
        }
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
        self.set_img_ids(val)
    }
    #[getter(catIds)]
    fn cat_ids_camel(&self) -> Vec<u64> {
        self.cat_ids()
    }
    #[setter(catIds)]
    fn set_cat_ids_camel(&mut self, val: Vec<u64>) {
        self.set_cat_ids(val)
    }
    #[getter(iouThrs)]
    fn iou_thrs_camel(&self) -> Vec<f64> {
        self.iou_thrs()
    }
    #[setter(iouThrs)]
    fn set_iou_thrs_camel(&mut self, val: Vec<f64>) {
        self.set_iou_thrs(val)
    }
    #[getter(recThrs)]
    fn rec_thrs_camel(&self) -> Vec<f64> {
        self.rec_thrs()
    }
    #[setter(recThrs)]
    fn set_rec_thrs_camel(&mut self, val: Vec<f64>) {
        self.set_rec_thrs(val)
    }
    #[getter(maxDets)]
    fn max_dets_camel(&self) -> Vec<usize> {
        self.max_dets()
    }
    #[setter(maxDets)]
    fn set_max_dets_camel(&mut self, val: Vec<usize>) {
        self.set_max_dets(val)
    }
    #[getter(areaRng)]
    fn area_rng_camel(&self) -> Vec<[f64; 2]> {
        self.area_rng()
    }
    #[setter(areaRng)]
    fn set_area_rng_camel(&mut self, val: Vec<[f64; 2]>) {
        self.set_area_rng(val)
    }
    #[getter(areaRngLbl)]
    fn area_rng_lbl_camel(&self) -> Vec<String> {
        self.area_rng_lbl()
    }
    #[setter(areaRngLbl)]
    fn set_area_rng_lbl_camel(&mut self, val: Vec<String>) {
        self.set_area_rng_lbl(val)
    }
    #[getter(useCats)]
    fn use_cats_camel(&self) -> bool {
        self.use_cats()
    }
    #[setter(useCats)]
    fn set_use_cats_camel(&mut self, val: bool) {
        self.set_use_cats(val)
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
#[pyclass(name = "Hierarchy")]
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
    ///     Maps OID label strings (e.g. ``"/m/dog"``) to category IDs.
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
}

#[pymethods]
impl PyCOCOeval {
    #[new]
    #[pyo3(signature = (coco_gt, coco_dt, iou_type, lvis_style=false, oid_style=false, hierarchy=None))]
    fn new(
        coco_gt: &PyCOCO,
        coco_dt: &PyCOCO,
        iou_type: &str,
        lvis_style: bool,
        oid_style: bool,
        hierarchy: Option<&PyHierarchy>,
    ) -> PyResult<Self> {
        if oid_style && lvis_style {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot use both oid_style and lvis_style",
            ));
        }

        let iou = parse_iou_type(iou_type)?;
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
        Ok(PyCOCOeval { inner })
    }

    fn evaluate(&mut self) {
        self.inner.evaluate();
    }

    fn accumulate(&mut self) {
        if self.inner.eval_imgs.is_empty() {
            eprintln!(
                "hotcoco: accumulate() called before evaluate(). \
                 Call evaluate() first or the results will be empty."
            );
        }
        self.inner.accumulate();
    }

    fn summarize(&mut self) {
        if self.inner.eval.is_none() {
            eprintln!(
                "hotcoco: summarize() called before accumulate(). \
                 Call evaluate() then accumulate() first."
            );
        }
        self.inner.summarize();
    }

    #[doc = "Run the full evaluation pipeline: evaluate → accumulate → summarize.

Equivalent to calling the three methods in sequence. Primarily used with
LVIS pipelines (Detectron2, MMDetection) that expect a single ``run()`` call."]
    fn run(&mut self) {
        self.inner.run();
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

    #[doc = "Return summary metrics as a dict.

Must be called after ``summarize()`` (or ``run()``). Returns an empty dict
if ``summarize`` has not been run.

LVIS keys: ``AP``, ``AP50``, ``AP75``, ``APs``, ``APm``, ``APl``,
``APr``, ``APc``, ``APf``, ``AR@300``, ``ARs@300``, ``ARm@300``, ``ARl@300``.

Standard COCO bbox/segm keys: ``AP``, ``AP50``, ``AP75``, ``APs``, ``APm``,
``APl``, ``AR1``, ``AR10``, ``AR100``, ``ARs``, ``ARm``, ``ARl``.

Keypoint keys: ``AP``, ``AP50``, ``AP75``, ``APm``, ``APl``,
``AR``, ``AR50``, ``AR75``, ``ARm``, ``ARl``.

Parameters
----------
prefix : str or None
    If given, each key is prefixed as ``\"{prefix}/{metric}\"``.
per_class : bool
    If True, include per-category AP values keyed as ``\"AP/{cat_name}\"``
    (or ``\"{prefix}/AP/{cat_name}\"`` with a prefix)."]
    #[pyo3(signature = (prefix=None, per_class=false))]
    fn get_results(
        &self,
        py: Python<'_>,
        prefix: Option<&str>,
        per_class: bool,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (k, v) in self.inner.get_results(prefix, per_class) {
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

    #[doc = "Return evaluation results as a dict.

Must be called after ``summarize()`` (or ``run()``). Returns a dict with:

- ``hotcoco_version``: hotcoco version string that produced these results.
- ``params``: evaluation parameters (iou_type, iou_thresholds, area_ranges, max_dets)
- ``metrics``: summary metrics (AP, AP50, AP75, etc.)
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
    fn results(&self, py: Python<'_>, per_class: bool) -> PyResult<PyObject> {
        let results = self
            .inner
            .results(per_class)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let json_str = results
            .to_json()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        Ok(dict.unbind())
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
    fn save_results(&self, path: &str, per_class: bool) -> PyResult<()> {
        let results = self
            .inner
            .results(per_class)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        results
            .save(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
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
    For other beta values: ``{\"F<beta>\": ..., \"F<beta>50\": ..., \"F<beta>75\": ...}``.
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
    fn f_scores(&self, py: Python<'_>, beta: f64) -> PyResult<PyObject> {
        if self.inner.eval.is_none() {
            eprintln!(
                "hotcoco: f_scores() called before accumulate(). \
                 Call evaluate() then accumulate() first. Returning empty dict."
            );
        }
        let dict = PyDict::new(py);
        for (k, v) in self.inner.f_scores(beta) {
            dict.set_item(k, v)?;
        }
        Ok(dict.into_any().unbind())
    }

    #[getter]
    fn coco_gt(&self) -> PyCOCO {
        PyCOCO {
            inner: hotcoco_core::COCO::from_dataset(self.inner.coco_gt.dataset.clone()),
            image_dir: None,
        }
    }

    #[getter(cocoGt)]
    fn coco_gt_camel(&self) -> PyCOCO {
        self.coco_gt()
    }

    #[getter]
    fn coco_dt(&self) -> PyCOCO {
        PyCOCO {
            inner: hotcoco_core::COCO::from_dataset(self.inner.coco_dt.dataset.clone()),
            image_dir: None,
        }
    }

    #[getter(cocoDt)]
    fn coco_dt_camel(&self) -> PyCOCO {
        self.coco_dt()
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
    fn tide_errors(&self, py: Python<'_>, pos_thr: f64, bg_thr: f64) -> PyResult<PyObject> {
        let te = self
            .inner
            .tide_errors(pos_thr, bg_thr)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let delta_ap = PyDict::new(py);
        for (k, v) in &te.delta_ap {
            delta_ap.set_item(k, v)?;
        }

        let counts = PyDict::new(py);
        for (k, v) in &te.counts {
            counts.set_item(k, v)?;
        }

        let dict = PyDict::new(py);
        dict.set_item("delta_ap", delta_ap)?;
        dict.set_item("counts", counts)?;
        dict.set_item("ap_base", te.ap_base)?;
        dict.set_item("pos_thr", te.pos_thr)?;
        dict.set_item("bg_thr", te.bg_thr)?;

        Ok(dict.into_any().unbind())
    }

    /// Re-accumulate metrics for named image subsets without recomputing IoU.
    ///
    /// ``slices`` is either a dict of ``{name: [img_ids]}`` or a callable that
    /// takes an image dict and returns a slice name (or ``None`` to skip).
    /// Returns a dict with one entry per slice plus ``"_overall"``.
    #[pyo3(signature = (slices))]
    fn slice_by(&mut self, py: Python<'_>, slices: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        use std::collections::HashMap;

        // If slices is callable, group images by return value
        let slice_map: HashMap<String, Vec<u64>> = if slices.is_callable() {
            let gt_images = &self.inner.coco_gt.dataset.images;
            let mut groups: HashMap<String, Vec<u64>> = HashMap::new();
            for img in gt_images {
                let img_dict = PyDict::new(py);
                img_dict.set_item("id", img.id)?;
                img_dict.set_item("file_name", &img.file_name)?;
                img_dict.set_item("height", img.height)?;
                img_dict.set_item("width", img.width)?;
                let result = slices.call1((img_dict,))?;
                let name: String = result.extract()?;
                groups.entry(name).or_default().push(img.id);
            }
            groups
        } else {
            let dict = slices.downcast::<PyDict>().map_err(|_| {
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
            .inner
            .slice_by(slice_map)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let out = PyDict::new(py);

        let to_dict = |sr: &hotcoco_core::SliceResult, py: Python<'_>| -> PyResult<PyObject> {
            let d = PyDict::new(py);
            for (k, v) in &sr.metrics {
                d.set_item(k, v)?;
            }
            d.set_item("num_images", sr.num_images)?;
            let delta_dict = PyDict::new(py);
            for (k, v) in &sr.delta {
                delta_dict.set_item(k, v)?;
            }
            d.set_item("delta", delta_dict)?;
            Ok(d.into_any().unbind())
        };

        out.set_item("_overall", to_dict(&results.overall, py)?)?;
        for sr in &results.slices {
            out.set_item(&sr.name, to_dict(sr, py)?)?;
        }

        Ok(out.into_any().unbind())
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
            let counts = vec![e.shape.t, e.shape.r, e.shape.k, e.shape.a, e.shape.m];
            dict.set_item("counts", counts)?;

            // precision: flat Vec<f64> → numpy array, then reshape to (T, R, K, A, M)
            let precision = PyArray1::from_vec(py, e.precision.clone());
            let precision = precision
                .call_method1(
                    "reshape",
                    ((e.shape.t, e.shape.r, e.shape.k, e.shape.a, e.shape.m),),
                )?
                .unbind();
            dict.set_item("precision", precision)?;

            // recall: flat Vec<f64> → numpy array, then reshape to (T, K, A, M)
            let recall = PyArray1::from_vec(py, e.recall.clone());
            let recall = recall
                .call_method1("reshape", ((e.shape.t, e.shape.k, e.shape.a, e.shape.m),))?
                .unbind();
            dict.set_item("recall", recall)?;

            // scores: flat Vec<f64> → numpy array, then reshape to (T, R, K, A, M)
            let scores = PyArray1::from_vec(py, e.scores.clone());
            let scores = scores
                .call_method1(
                    "reshape",
                    ((e.shape.t, e.shape.r, e.shape.k, e.shape.a, e.shape.m),),
                )?
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
    m.add_class::<PyHierarchy>()?;
    m.add_function(wrap_pyfunction!(init_as_pycocotools, m)?)?;
    m.add_function(wrap_pyfunction!(init_as_lvis, m)?)?;

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

    Ok(())
}
