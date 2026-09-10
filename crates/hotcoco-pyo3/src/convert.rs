use hotcoco_core::{Annotation, Category, Dataset, DatasetStats, Image, Rle, Segmentation};
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

/// Extract an optional field from a Python dict.
macro_rules! opt {
    ($dict:expr, $key:expr) => {
        $dict.get_item($key)?.map(|v| v.extract()).transpose()?
    };
}

/// Extract a required field from a Python dict, raising `PyValueError` if missing.
macro_rules! req {
    ($dict:expr, $key:expr) => {
        $dict
            .get_item($key)?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(concat!("dict missing '", $key, "'"))
            })?
            .extract()?
    };
}

/// The dict keys each record type owns. Any other key on an incoming dict is a
/// custom key, preserved through the `extra` map (serde-flattened in the core
/// types) so `load → filter → save` keeps user metadata the way pycocotools does.
pub(crate) const ANNOTATION_KEYS: &[&str] = &[
    "id",
    "image_id",
    "category_id",
    "bbox",
    "area",
    "segmentation",
    "iscrowd",
    "keypoints",
    "num_keypoints",
    "obb",
    "score",
    "is_group_of",
];
const IMAGE_KEYS: &[&str] = &[
    "id",
    "file_name",
    "height",
    "width",
    "license",
    "coco_url",
    "flickr_url",
    "date_captured",
    "neg_category_ids",
    "not_exhaustive_category_ids",
];
const CATEGORY_KEYS: &[&str] = &[
    "id",
    "name",
    "supercategory",
    "skeleton",
    "keypoints",
    "frequency",
];

/// Collect every key of `dict` not in `known` into a JSON map.
///
/// Goes through Python's `json.dumps` in one call per record rather than a
/// hand-rolled per-value converter: the values are arbitrary user objects, and
/// `json` already defines exactly which of those a COCO file can hold. A
/// non-serializable value raises the stdlib's own `TypeError`, naming the type.
fn extract_extra(
    dict: &Bound<'_, PyDict>,
    known: &[&str],
) -> PyResult<serde_json::Map<String, serde_json::Value>> {
    let py = dict.py();
    let mut extras: Option<Bound<'_, PyDict>> = None;
    for (k, v) in dict {
        let Ok(key) = k.extract::<String>() else {
            continue; // non-string keys cannot appear in COCO JSON
        };
        if known.contains(&key.as_str()) {
            continue;
        }
        extras
            .get_or_insert_with(|| PyDict::new(py))
            .set_item(key, v)?;
    }
    let Some(extras) = extras else {
        return Ok(serde_json::Map::new());
    };
    let json_str: String = py
        .import("json")?
        .call_method1("dumps", (extras,))?
        .extract()?;
    match serde_json::from_str(&json_str) {
        Ok(serde_json::Value::Object(map)) => Ok(map),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "custom keys did not round-trip through JSON",
        )),
    }
}

/// Merge a record's `extra` map back into its outgoing Python dict.
fn merge_extra(
    dict: &Bound<'_, PyDict>,
    extra: &serde_json::Map<String, serde_json::Value>,
) -> PyResult<()> {
    if extra.is_empty() {
        return Ok(());
    }
    let py = dict.py();
    let extras = crate::serde_to_py(py, extra)?;
    for (k, v) in extras.bind(py).cast::<PyDict>()? {
        dict.set_item(k, v)?;
    }
    Ok(())
}

pub fn annotation_to_py(py: Python<'_>, ann: &Annotation) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("id", ann.id)?;
    dict.set_item("image_id", ann.image_id)?;
    dict.set_item("category_id", ann.category_id)?;
    if let Some(ref bbox) = ann.bbox {
        dict.set_item("bbox", bbox.to_vec())?;
    }
    if let Some(area) = ann.area {
        dict.set_item("area", area)?;
    }
    if let Some(ref seg) = ann.segmentation {
        dict.set_item("segmentation", segmentation_to_py(py, seg)?)?;
    }
    dict.set_item("iscrowd", ann.iscrowd as u8)?;
    if let Some(ref kpts) = ann.keypoints {
        dict.set_item("keypoints", kpts.clone())?;
    }
    if let Some(nk) = ann.num_keypoints {
        dict.set_item("num_keypoints", nk)?;
    }
    if let Some(ref obb) = ann.obb {
        dict.set_item("obb", obb.to_vec())?;
    }
    if let Some(score) = ann.score {
        dict.set_item("score", score)?;
    }
    if let Some(is_group_of) = ann.is_group_of {
        dict.set_item("is_group_of", is_group_of)?;
    }
    merge_extra(&dict, &ann.extra)?;
    Ok(dict.into_any().unbind())
}

pub fn segmentation_to_py(py: Python<'_>, seg: &Segmentation) -> PyResult<Py<PyAny>> {
    match seg {
        Segmentation::Polygon(polys) => {
            let inner_lists: Vec<Bound<'_, PyList>> = polys
                .iter()
                .map(|p| PyList::new(py, p.iter()))
                .collect::<PyResult<_>>()?;
            let list = PyList::new(py, inner_lists)?;
            Ok(list.into_any().unbind())
        }
        Segmentation::CompressedRle { size, counts } => {
            let dict = PyDict::new(py);
            dict.set_item("size", vec![size[0], size[1]])?;
            dict.set_item("counts", counts)?;
            Ok(dict.into_any().unbind())
        }
        Segmentation::UncompressedRle { size, counts } => {
            let dict = PyDict::new(py);
            dict.set_item("size", vec![size[0], size[1]])?;
            dict.set_item("counts", counts.clone())?;
            Ok(dict.into_any().unbind())
        }
    }
}

pub fn py_to_annotation(dict: &Bound<'_, PyDict>) -> PyResult<Annotation> {
    let id: u64 = opt!(dict, "id").unwrap_or(0);
    let image_id: u64 = req!(dict, "image_id");
    let category_id: u64 = opt!(dict, "category_id").unwrap_or(0);
    let bbox: Option<[f64; 4]> = opt!(dict, "bbox");
    let area: Option<f64> = opt!(dict, "area");
    let segmentation: Option<Segmentation> = dict
        .get_item("segmentation")?
        .map(|v| py_to_segmentation(&v))
        .transpose()?;
    let iscrowd: bool = dict
        .get_item("iscrowd")?
        .map(|v| {
            v.extract::<bool>()
                .or_else(|_| v.extract::<u8>().map(|i| i != 0))
        })
        .transpose()?
        .unwrap_or(false);
    let keypoints: Option<Vec<f64>> = opt!(dict, "keypoints");
    let num_keypoints: Option<u32> = opt!(dict, "num_keypoints");
    let obb: Option<[f64; 5]> = opt!(dict, "obb");
    let score: Option<f64> = opt!(dict, "score");
    let is_group_of: Option<bool> = opt!(dict, "is_group_of");
    let extra = extract_extra(dict, ANNOTATION_KEYS)?;

    Ok(Annotation {
        id,
        image_id,
        category_id,
        bbox,
        area,
        segmentation,
        iscrowd,
        keypoints,
        num_keypoints,
        obb,
        score,
        is_group_of,
        extra,
    })
}

/// Set one scalar COCO field on a copy of `ann`, without a dict round-trip.
///
/// The fast path behind [`PyCOCO::set_ann_field`](crate::PyCOCO): only the
/// fields whose Python form is a single number or flag are handled here.
/// Anything else — a shaped field (`bbox`, `segmentation`, `keypoints`, `obb`),
/// a custom key, or a value that does not extract — returns `None`, and the
/// caller falls back to rebuilding the annotation through
/// [`annotation_to_py`] and [`py_to_annotation`].
///
/// Falling back on a failed extraction rather than raising here is what keeps
/// the two paths indistinguishable: a wrong value type is reported by
/// `py_to_annotation`, the way it was before this fast path existed.
pub fn set_scalar_ann_field(
    ann: &Annotation,
    field: &str,
    value: &Bound<'_, PyAny>,
) -> Option<Annotation> {
    let mut out = ann.clone();
    match field {
        "area" => out.area = Some(value.extract().ok()?),
        "score" => out.score = Some(value.extract().ok()?),
        "image_id" => out.image_id = value.extract().ok()?,
        "category_id" => out.category_id = value.extract().ok()?,
        "num_keypoints" => out.num_keypoints = Some(value.extract().ok()?),
        "is_group_of" => out.is_group_of = Some(value.extract().ok()?),
        // Same bool-or-int leniency `py_to_annotation` applies.
        "iscrowd" => {
            out.iscrowd = value
                .extract::<bool>()
                .or_else(|_| value.extract::<u8>().map(|i| i != 0))
                .ok()?;
        }
        _ => return None,
    }
    Some(out)
}

fn py_to_segmentation(obj: &Bound<'_, PyAny>) -> PyResult<Segmentation> {
    // Try as dict (CompressedRle or UncompressedRle)
    if let Ok(dict) = obj.cast::<PyDict>() {
        let size: [u32; 2] = req!(dict, "size");
        let counts_obj = dict
            .get_item("counts")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("dict missing 'counts'"))?;
        // Try str first, then bytes (what `mask.encode` returns, matching
        // pycocotools), then a list of ints (uncompressed RLE). Bytes must be
        // checked before the list: a `bytes` object is a Python sequence of
        // ints, so `extract::<Vec<u32>>()` succeeds on it and silently turns
        // the compressed string's byte values into an uncompressed RLE.
        if let Ok(s) = counts_obj.extract::<String>() {
            return Ok(Segmentation::CompressedRle { size, counts: s });
        }
        if let Ok(b) = counts_obj.cast::<PyBytes>() {
            let s = std::str::from_utf8(b.as_bytes()).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid UTF-8 in RLE counts: {e}"))
            })?;
            return Ok(Segmentation::CompressedRle {
                size,
                counts: s.to_string(),
            });
        }
        let counts: Vec<u32> = counts_obj.extract().map_err(|_| {
            let name = counts_obj
                .get_type()
                .name()
                .map_or_else(|_| "?".to_string(), |n| n.to_string());
            pyo3::exceptions::PyTypeError::new_err(format!(
                "RLE 'counts' must be str, bytes, or a list of ints, got {name}"
            ))
        })?;
        return Ok(Segmentation::UncompressedRle { size, counts });
    }
    // Otherwise it's a polygon (list of lists)
    let polys: Vec<Vec<f64>> = obj.extract()?;
    Ok(Segmentation::Polygon(polys))
}

pub fn image_to_py(py: Python<'_>, img: &Image) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("id", img.id)?;
    dict.set_item("file_name", &img.file_name)?;
    dict.set_item("height", img.height)?;
    dict.set_item("width", img.width)?;
    if let Some(license) = img.license {
        dict.set_item("license", license)?;
    }
    if let Some(ref url) = img.coco_url {
        dict.set_item("coco_url", url)?;
    }
    if let Some(ref url) = img.flickr_url {
        dict.set_item("flickr_url", url)?;
    }
    if let Some(ref dc) = img.date_captured {
        dict.set_item("date_captured", dc)?;
    }
    if !img.neg_category_ids.is_empty() {
        dict.set_item("neg_category_ids", img.neg_category_ids.clone())?;
    }
    if !img.not_exhaustive_category_ids.is_empty() {
        dict.set_item(
            "not_exhaustive_category_ids",
            img.not_exhaustive_category_ids.clone(),
        )?;
    }
    merge_extra(&dict, &img.extra)?;
    Ok(dict.into_any().unbind())
}

pub fn category_to_py(py: Python<'_>, cat: &Category) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("id", cat.id)?;
    dict.set_item("name", &cat.name)?;
    if let Some(ref sc) = cat.supercategory {
        dict.set_item("supercategory", sc)?;
    }
    if let Some(ref sk) = cat.skeleton {
        let skel: Vec<Vec<u32>> = sk.iter().map(|pair| pair.to_vec()).collect();
        dict.set_item("skeleton", skel)?;
    }
    if let Some(ref kpts) = cat.keypoints {
        dict.set_item("keypoints", kpts.clone())?;
    }
    if let Some(ref freq) = cat.frequency {
        dict.set_item("frequency", freq)?;
    }
    merge_extra(&dict, &cat.extra)?;
    Ok(dict.into_any().unbind())
}

pub fn dataset_stats_to_py(py: Python<'_>, stats: &DatasetStats) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("image_count", stats.image_count)?;
    dict.set_item("annotation_count", stats.annotation_count)?;
    dict.set_item("category_count", stats.category_count)?;
    dict.set_item("crowd_count", stats.crowd_count)?;

    let per_cat = PyList::new(
        py,
        stats
            .per_category
            .iter()
            .map(|c| -> PyResult<Py<PyAny>> {
                let d = PyDict::new(py);
                d.set_item("id", c.id)?;
                d.set_item("name", &c.name)?;
                d.set_item("ann_count", c.ann_count)?;
                d.set_item("img_count", c.img_count)?;
                d.set_item("crowd_count", c.crowd_count)?;
                Ok(d.into_any().unbind())
            })
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    dict.set_item("per_category", per_cat)?;

    let summary_to_dict = |s: &hotcoco_core::SummaryStats| -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("min", s.min)?;
        d.set_item("max", s.max)?;
        d.set_item("mean", s.mean)?;
        d.set_item("median", s.median)?;
        Ok(d.into_any().unbind())
    };
    dict.set_item("image_width", summary_to_dict(&stats.image_width)?)?;
    dict.set_item("image_height", summary_to_dict(&stats.image_height)?)?;
    dict.set_item("annotation_area", summary_to_dict(&stats.annotation_area)?)?;

    Ok(dict.into_any().unbind())
}

/// Return an RLE in pycocotools format: `{"size": [h, w], "counts": b"..."}`.
///
/// The `counts` value is a `bytes` object containing the LEB128-compressed
/// string, matching what `pycocotools.mask.encode` returns.
pub fn rle_to_coco_py(py: Python<'_>, rle: &Rle) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("size", vec![rle.h, rle.w])?;
    let compressed = hotcoco_core::mask::rle_to_string(rle);
    let py_bytes = PyBytes::new(py, compressed.as_bytes());
    dict.set_item("counts", py_bytes)?;
    Ok(dict.into_any().unbind())
}

pub fn py_to_rle(dict: &Bound<'_, PyDict>) -> PyResult<Rle> {
    // Support {"h", "w", "counts": [ints]}, {"size": [h,w], "counts": "string"},
    // and {"size": [h,w], "counts": b"bytes"} (pycocotools format)
    if let Some(size_obj) = dict.get_item("size")? {
        let size: [u32; 2] = size_obj.extract()?;
        let counts_obj = dict.get_item("counts")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("RLE dict has 'size' but missing 'counts'")
        })?;
        // Try str first
        if let Ok(s) = counts_obj.extract::<String>() {
            return hotcoco_core::mask::rle_from_string(&s, size[0], size[1])
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
        }
        // Try bytes (pycocotools format)
        if let Ok(b) = counts_obj.cast::<PyBytes>() {
            let s = std::str::from_utf8(b.as_bytes()).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid UTF-8 in RLE counts: {e}"))
            })?;
            return hotcoco_core::mask::rle_from_string(s, size[0], size[1])
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
        }
        // Try list of ints (uncompressed RLE)
        let counts: Vec<u32> = counts_obj.extract()?;
        return Ok(Rle {
            h: size[0],
            w: size[1],
            counts,
        });
    }
    let h: u32 = dict
        .get_item("h")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("RLE dict missing 'h'"))?
        .extract()?;
    let w: u32 = dict
        .get_item("w")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("RLE dict missing 'w'"))?
        .extract()?;
    let counts: Vec<u32> = dict
        .get_item("counts")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("RLE dict missing 'counts'"))?
        .extract()?;
    Ok(Rle { h, w, counts })
}

pub fn py_to_image(dict: &Bound<'_, PyDict>) -> PyResult<Image> {
    let id: u64 = req!(dict, "id");
    let file_name: String = opt!(dict, "file_name").unwrap_or_default();
    // Default rather than require: pycocotools' assignment flow (`coco.dataset
    // = d; coco.createIndex()`) builds images as bare `{"id": …}` — that is
    // what torchmetrics' pycocotools backend passes. Dimensions are only
    // consumed by mask operations, which pycocotools equally cannot perform
    // without them.
    let height: u32 = opt!(dict, "height").unwrap_or_default();
    let width: u32 = opt!(dict, "width").unwrap_or_default();
    let license: Option<u64> = opt!(dict, "license");
    let coco_url: Option<String> = opt!(dict, "coco_url");
    let flickr_url: Option<String> = opt!(dict, "flickr_url");
    let date_captured: Option<String> = opt!(dict, "date_captured");
    let neg_category_ids: Vec<u64> = opt!(dict, "neg_category_ids").unwrap_or_default();
    let not_exhaustive_category_ids: Vec<u64> =
        opt!(dict, "not_exhaustive_category_ids").unwrap_or_default();
    let extra = extract_extra(dict, IMAGE_KEYS)?;

    Ok(Image {
        id,
        file_name,
        height,
        width,
        license,
        coco_url,
        flickr_url,
        date_captured,
        neg_category_ids,
        not_exhaustive_category_ids,
        extra,
    })
}

pub fn py_to_category(dict: &Bound<'_, PyDict>) -> PyResult<Category> {
    let id: u64 = req!(dict, "id");
    let name: String = req!(dict, "name");
    let supercategory: Option<String> = opt!(dict, "supercategory");
    let skeleton: Option<Vec<[u32; 2]>> = opt!(dict, "skeleton");
    let keypoints: Option<Vec<String>> = opt!(dict, "keypoints");
    let frequency: Option<String> = opt!(dict, "frequency");
    let extra = extract_extra(dict, CATEGORY_KEYS)?;

    Ok(Category {
        id,
        name,
        supercategory,
        skeleton,
        keypoints,
        frequency,
        extra,
    })
}

/// Extract a list of dicts from a parent dict, converting each element with `convert_fn`.
/// Returns an empty Vec if the key is absent.
fn extract_dict_list<'py, T>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    convert_fn: impl Fn(&Bound<'py, PyDict>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    match dict.get_item(key)? {
        None => Ok(Vec::new()),
        Some(v) => v
            .cast::<PyList>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be a list of dicts"))
            })?
            .iter()
            .map(|item| {
                let d = item.cast::<PyDict>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "each item in '{key}' must be a dict"
                    ))
                })?;
                convert_fn(d)
            })
            .collect(),
    }
}

pub fn py_to_dataset(dict: &Bound<'_, PyDict>) -> PyResult<Dataset> {
    let images = extract_dict_list(dict, "images", py_to_image)?;
    let annotations = extract_dict_list(dict, "annotations", py_to_annotation)?;
    let categories = extract_dict_list(dict, "categories", py_to_category)?;

    Ok(Dataset {
        info: None,
        images,
        annotations,
        categories,
        licenses: vec![],
    })
}

// ---------------------------------------------------------------------------
// Shared marshaling helpers
// ---------------------------------------------------------------------------

/// Encode one calibration bin as a Python dict.
///
/// Shared by `COCOeval.calibration()` and `metrics.calibration_curve()`. Both
/// expose the same `CalibrationBin`, and when each hand-wrote its own five
/// `set_item` calls, adding a field meant remembering to update two places — so
/// the two Python surfaces could silently start describing the same struct
/// differently.
pub fn calibration_bin_to_py<'py>(
    py: Python<'py>,
    bin: &hotcoco_core::CalibrationBin,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("bin_lower", bin.bin_lower)?;
    d.set_item("bin_upper", bin.bin_upper)?;
    d.set_item("avg_confidence", bin.avg_confidence)?;
    d.set_item("avg_accuracy", bin.avg_accuracy)?;
    d.set_item("count", bin.count)?;
    Ok(d)
}

/// A flat `(side, side)` block of confusion counts as a numpy `uint64` array.
///
/// The one marshaling site for `COCOeval.confusion_matrix()` and
/// `metrics.confusion_matrix()`, so both entry points emit the same dtype —
/// `uint64`, the counts' natural type.
pub fn confusion_counts_to_py(
    py: Python<'_>,
    counts: Vec<u64>,
    side: usize,
) -> PyResult<Py<PyAny>> {
    let arr = PyArray1::from_vec(py, counts);
    Ok(arr.reshape([side, side])?.into_any().unbind())
}

/// A flat `Vec<f64>` as a numpy `float64` array of the given shape.
///
/// The typed counterpart to [`confusion_counts_to_py`] for the sites that
/// reshape a flat float buffer. Generic over the shape rather than the rank, so
/// `[k, k]`, `[t, k, a, m]` and `[t, r, k, a, m]` all use the same helper.
pub fn f64_array<D: numpy::ndarray::IntoDimension>(
    py: Python<'_>,
    values: Vec<f64>,
    dims: D,
) -> PyResult<Py<PyAny>> {
    let arr = PyArray1::from_vec(py, values);
    Ok(arr.reshape(dims)?.into_any().unbind())
}

/// A `Vec<Vec<f64>>` of uniform-length rows, flattened and reshaped to `dims`
/// in one step.
///
/// The single owner of the flatten for `iou`/`bbox_iou`-shaped kernels,
/// which produce a `D`-row, `G`-column `Vec<Vec<f64>>`. Call this rather
/// than flattening at the call site before [`f64_array`].
pub fn f64_matrix<D: numpy::ndarray::IntoDimension>(
    py: Python<'_>,
    rows: &[Vec<f64>],
    dims: D,
) -> PyResult<Py<PyAny>> {
    let mut flat = Vec::with_capacity(rows.iter().map(Vec::len).sum());
    for row in rows {
        flat.extend_from_slice(row);
    }
    f64_array(py, flat, dims)
}

/// A key-value map as a Python dict.
///
/// Eight sites hand-rolled the same three lines — `PyDict::new`, a `for` over a
/// `BTreeMap`, `set_item` — across `get_results`, `f_scores`, `tide_errors`
/// (twice), `calibration`, `slice_by` (twice) and `compare` (which had grown a
/// local closure for its own three uses). Generic over key and value so the
/// `f64` maps and the `u64` count maps share one implementation.
///
/// Iteration order is the caller's: every current caller passes a `BTreeMap`,
/// so the dict comes out key-ordered and byte-stable without sorting here.
pub fn map_to_dict<'py, K, V>(
    py: Python<'py>,
    entries: impl IntoIterator<Item = (K, V)>,
) -> PyResult<Bound<'py, PyDict>>
where
    K: IntoPyObject<'py>,
    V: IntoPyObject<'py>,
{
    let d = PyDict::new(py);
    for (k, v) in entries {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// A 1-D float argument: numpy `float64` in one copy, anything else via the
/// list path.
///
/// The metrics docstrings advertise "lists or numpy arrays", but PyO3's
/// `Vec<f64>` fast path fires only for list/tuple, leaving a numpy array to
/// per-element iteration — one boxed `extract` per element, ~500K calls per
/// argument on a full val2017 run. `PyReadonlyArray1` is a dtype check plus a
/// memcpy; `to_vec` goes through ndarray, so strided views (`scores[::2]`)
/// copy correctly instead of being rejected. Other dtypes (`float32`, object
/// arrays) still work through the fallback, at per-element cost.
pub fn f64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray1<f64>>() {
        return Ok(arr.as_array().to_vec());
    }
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a sequence of floats or a 1-D numpy array"
        ))
    })
}

/// A 1-D bool argument — see [`f64_vec`]. The fast path matters even more
/// here: under abi3, extracting one `numpy.bool_` costs a Python-level
/// `__bool__` call, so a `matched` array was the most expensive argument a
/// caller could pass.
pub fn bool_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<bool>> {
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray1<bool>>() {
        return Ok(arr.as_array().to_vec());
    }
    obj.extract::<Vec<bool>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a sequence of bools or a 1-D numpy bool array"
        ))
    })
}

/// A 2-D float argument — the 2-D member of the [`f64_vec`]/[`bool_vec`]
/// family: numpy `float64` in one copy, a nested sequence (list of lists,
/// tuple of tuples, ...) otherwise, with a named `TypeError` on either
/// failing. Returns a row-major `(flat, nr, nc)` triple, matching
/// `assign::lsap`'s contract even for a Fortran-order or otherwise strided
/// numpy input, since `.as_array().iter()` always yields row-major order.
///
/// A nested sequence gets an additional rectangularity check: mismatched row
/// lengths raise `ValueError`, not `TypeError` — the type is right, the shape
/// isn't. Callers that also need to reject NaN (as `lsap` does) check that
/// locally; whether NaN is meaningful is caller-specific, not a property of
/// extracting a 2-D array.
pub fn f64_matrix_arg(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<(Vec<f64>, usize, usize)> {
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray2<f64>>() {
        let shape = arr.shape();
        let (nr, nc) = (shape[0], shape[1]);
        let flat: Vec<f64> = arr.as_array().iter().copied().collect();
        return Ok((flat, nr, nc));
    }

    let rows: Vec<Vec<f64>> = obj.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a 2-D sequence of floats or a 2-D numpy array"
        ))
    })?;
    let nr = rows.len();
    let nc = rows.first().map_or(0, Vec::len);

    let mut flat = Vec::with_capacity(nr * nc);
    for (i, row) in rows.iter().enumerate() {
        if row.len() != nc {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} must be rectangular; row 0 has {nc} columns but row {i} has {}",
                row.len()
            )));
        }
        flat.extend_from_slice(row);
    }
    Ok((flat, nr, nc))
}

/// An id-list argument read the way pycocotools' `_isArrayLike` reads it: a
/// bare int or any sequence of ints. torchvision's `CocoDetection` calls
/// `coco.getAnnIds(img_id)` with a scalar — found by the 1.0
/// third-party-consumer smoke test. One extractor shared by every query/load
/// method and its camelCase twin, so the two surfaces cannot disagree.
#[derive(Default)]
pub struct IdList(pub Vec<u64>);

impl FromPyObject<'_, '_> for IdList {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        if let Ok(one) = obj.extract::<u64>() {
            return Ok(IdList(vec![one]));
        }
        Ok(IdList(obj.extract()?))
    }
}

/// A name-list argument — see [`IdList`]. The scalar check must come first:
/// a bare `str` is iterable, so the sequence path would split `"person"`
/// into characters. (pycocotools itself has that trap — `_isArrayLike`
/// accepts strings — so accepting the scalar here is strictly kinder than
/// the original.)
#[derive(Default)]
pub struct NameList(pub Vec<String>);

impl FromPyObject<'_, '_> for NameList {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        if let Ok(one) = obj.extract::<String>() {
            return Ok(NameList(vec![one]));
        }
        Ok(NameList(obj.extract()?))
    }
}

/// Reject parallel arrays of differing length.
///
/// Every function taking `(scores, matched)`-style arrays needs this, and a
/// silent truncation would report a metric over a subset the caller never asked
/// for.
pub fn check_parallel(a: usize, b: usize, name_a: &str, name_b: &str) -> PyResult<()> {
    if a != b {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name_a} and {name_b} must have the same length, got {a} and {b}"
        )));
    }
    Ok(())
}
