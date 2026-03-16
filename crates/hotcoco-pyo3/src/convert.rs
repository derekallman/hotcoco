use hotcoco_core::{Annotation, Category, Dataset, DatasetStats, Image, Rle, Segmentation};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

pub fn annotation_to_py(py: Python<'_>, ann: &Annotation) -> PyResult<PyObject> {
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
    if let Some(score) = ann.score {
        dict.set_item("score", score)?;
    }
    if let Some(is_group_of) = ann.is_group_of {
        dict.set_item("is_group_of", is_group_of)?;
    }
    Ok(dict.into_any().unbind())
}

pub fn segmentation_to_py(py: Python<'_>, seg: &Segmentation) -> PyResult<PyObject> {
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
    let id: u64 = dict
        .get_item("id")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0);
    let image_id: u64 = dict
        .get_item("image_id")?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("annotation dict missing 'image_id'")
        })?
        .extract()?;
    let category_id: u64 = dict
        .get_item("category_id")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0);
    let bbox: Option<[f64; 4]> = dict.get_item("bbox")?.map(|v| v.extract()).transpose()?;
    let area: Option<f64> = dict.get_item("area")?.map(|v| v.extract()).transpose()?;
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
    let keypoints: Option<Vec<f64>> = dict
        .get_item("keypoints")?
        .map(|v| v.extract())
        .transpose()?;
    let num_keypoints: Option<u32> = dict
        .get_item("num_keypoints")?
        .map(|v| v.extract())
        .transpose()?;
    let score: Option<f64> = dict.get_item("score")?.map(|v| v.extract()).transpose()?;
    let is_group_of: Option<bool> = dict
        .get_item("is_group_of")?
        .map(|v| v.extract())
        .transpose()?;

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
        score,
        is_group_of,
    })
}

fn py_to_segmentation(obj: &Bound<'_, PyAny>) -> PyResult<Segmentation> {
    // Try as dict (CompressedRle or UncompressedRle)
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let size: [u32; 2] = dict
            .get_item("size")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("RLE dict missing 'size'"))?
            .extract()?;
        let counts_obj = dict
            .get_item("counts")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("RLE dict missing 'counts'"))?;
        if let Ok(s) = counts_obj.extract::<String>() {
            return Ok(Segmentation::CompressedRle { size, counts: s });
        }
        let counts: Vec<u32> = counts_obj.extract()?;
        return Ok(Segmentation::UncompressedRle { size, counts });
    }
    // Otherwise it's a polygon (list of lists)
    let polys: Vec<Vec<f64>> = obj.extract()?;
    Ok(Segmentation::Polygon(polys))
}

pub fn image_to_py(py: Python<'_>, img: &Image) -> PyResult<PyObject> {
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
    Ok(dict.into_any().unbind())
}

pub fn category_to_py(py: Python<'_>, cat: &Category) -> PyResult<PyObject> {
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
    Ok(dict.into_any().unbind())
}

pub fn dataset_stats_to_py(py: Python<'_>, stats: &DatasetStats) -> PyResult<PyObject> {
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
            .map(|c| -> PyResult<PyObject> {
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

    let summary_to_dict = |s: &hotcoco_core::SummaryStats| -> PyResult<PyObject> {
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

pub fn rle_to_py(py: Python<'_>, rle: &Rle) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("h", rle.h)?;
    dict.set_item("w", rle.w)?;
    dict.set_item("counts", rle.counts.clone())?;
    Ok(dict.into_any().unbind())
}

/// Return an RLE in pycocotools format: `{"size": [h, w], "counts": b"..."}`.
///
/// The `counts` value is a `bytes` object containing the LEB128-compressed
/// string, matching what `pycocotools.mask.encode` returns.
pub fn rle_to_coco_py(py: Python<'_>, rle: &Rle) -> PyResult<PyObject> {
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
                .map_err(pyo3::exceptions::PyValueError::new_err);
        }
        // Try bytes (pycocotools format)
        if let Ok(b) = counts_obj.downcast::<PyBytes>() {
            let s = std::str::from_utf8(b.as_bytes()).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid UTF-8 in RLE counts: {e}"))
            })?;
            return hotcoco_core::mask::rle_from_string(s, size[0], size[1])
                .map_err(pyo3::exceptions::PyValueError::new_err);
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
    let id: u64 = dict
        .get_item("id")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("image dict missing 'id'"))?
        .extract()?;
    let file_name: String = dict
        .get_item("file_name")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let height: u32 = dict
        .get_item("height")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("image dict missing 'height'"))?
        .extract()?;
    let width: u32 = dict
        .get_item("width")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("image dict missing 'width'"))?
        .extract()?;
    let license: Option<u64> = dict.get_item("license")?.map(|v| v.extract()).transpose()?;
    let coco_url: Option<String> = dict
        .get_item("coco_url")?
        .map(|v| v.extract())
        .transpose()?;
    let flickr_url: Option<String> = dict
        .get_item("flickr_url")?
        .map(|v| v.extract())
        .transpose()?;
    let date_captured: Option<String> = dict
        .get_item("date_captured")?
        .map(|v| v.extract())
        .transpose()?;
    let neg_category_ids: Vec<u64> = dict
        .get_item("neg_category_ids")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let not_exhaustive_category_ids: Vec<u64> = dict
        .get_item("not_exhaustive_category_ids")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();

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
    })
}

pub fn py_to_category(dict: &Bound<'_, PyDict>) -> PyResult<Category> {
    let id: u64 = dict
        .get_item("id")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("category dict missing 'id'"))?
        .extract()?;
    let name: String = dict
        .get_item("name")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("category dict missing 'name'"))?
        .extract()?;
    let supercategory: Option<String> = dict
        .get_item("supercategory")?
        .map(|v| v.extract())
        .transpose()?;
    let skeleton: Option<Vec<[u32; 2]>> = dict
        .get_item("skeleton")?
        .map(|v| v.extract())
        .transpose()?;
    let keypoints: Option<Vec<String>> = dict
        .get_item("keypoints")?
        .map(|v| v.extract())
        .transpose()?;
    let frequency: Option<String> = dict
        .get_item("frequency")?
        .map(|v| v.extract())
        .transpose()?;

    Ok(Category {
        id,
        name,
        supercategory,
        skeleton,
        keypoints,
        frequency,
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
            .downcast::<PyList>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be a list of dicts"))
            })?
            .iter()
            .map(|item| {
                let d = item.downcast::<PyDict>().map_err(|_| {
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
