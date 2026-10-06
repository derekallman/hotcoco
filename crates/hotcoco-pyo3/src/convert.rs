use hotcoco_core::mask::RleRef;
use hotcoco_core::{Annotation, Category, Dataset, DatasetStats, Extra, Image, Rle, Segmentation};
use numpy::{PyArray1, PyArrayDescrMethods, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyMemoryView, PyString, PyTuple,
};

/// A required field of a Python dict, raising `ValueError("dict missing
/// '<key>'")` when it is absent.
///
/// `$key` is interned (`pyo3::intern!`) rather than passed as a bare `&str`:
/// `get_item` takes anything `IntoPyObject`, and for a plain `&str` that means
/// allocating a fresh `PyString` on every call. Record fields are read in one
/// pass over the dict by [`set_ann_field`], [`set_image_field`], and
/// [`set_category_field`]; their converters call this first only to check
/// the one required key.
macro_rules! req {
    ($dict:expr, $key:literal) => {
        $dict
            .get_item(pyo3::intern!($dict.py(), $key))?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(concat!("dict missing '", $key, "'"))
            })?
    };
}

/// A non-negative integer field that also accepts an integral float.
///
/// Ids come back as `1.0` from a JSON written by pandas or a numpy-backed
/// encoder, and pycocotools accepts them because `1.0 == 1` and they hash
/// alike. A fractional or negative value is still an error, naming the value;
/// a value that does not fit the field's width is an `OverflowError`. The
/// JSON loader applies the same rule through `types::deserialize_uint`.
pub(crate) fn extract_int<T: TryFrom<u64>>(v: &Bound<'_, PyAny>) -> PyResult<T> {
    let n = if let Ok(i) = v.extract::<u64>() {
        i
    } else if let Ok(f) = v.extract::<f64>()
        && f.fract() == 0.0
        && f >= 0.0
        && f <= u64::MAX as f64
    {
        f as u64
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected a non-negative integer, got {}",
            v.repr()?
        )));
    };
    T::try_from(n).map_err(|_| {
        pyo3::exceptions::PyOverflowError::new_err(format!(
            "integer {n} is out of range for this field"
        ))
    })
}

/// A 0/1 flag: `bool`, any integer, or an integral float.
///
/// The one Python-side reader for `iscrowd`, `is_group_of`, `useCats`, and
/// the `iscrowd` argument of the mask functions, so every flag in the API
/// agrees on what counts as one. pycocotools' own default is `useCats = 1`,
/// Open Images spells `IsGroupOf` as `0`/`1`, and a flag column read back
/// from pandas is `0.0`. The JSON loader applies the same rule through
/// `types::deserialize_flag`.
///
/// An exact `bool` or `int` is read first, without the generic path: that is
/// what nearly every flag is, and `extract::<bool>` on an `int` builds an
/// error only to discard it. Anything else, including an `int` outside the
/// `i64` range, takes the generic path, so the result is the same for every
/// input.
pub(crate) fn extract_flag(v: &Bound<'_, PyAny>) -> PyResult<bool> {
    if let Ok(b) = v.cast_exact::<PyBool>() {
        return Ok(b.is_true());
    }
    if v.is_exact_instance_of::<PyInt>()
        && let Ok(i) = v.extract::<i64>()
    {
        return Ok(i != 0);
    }
    if let Ok(b) = v.extract::<bool>() {
        return Ok(b);
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(i != 0);
    }
    if let Ok(f) = v.extract::<f64>()
        && f.fract() == 0.0
    {
        return Ok(f != 0.0);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "expected a bool or 0/1 flag, got {}",
        v.repr()?
    )))
}

/// The Python type name of `obj`, for error messages.
pub(crate) fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map_or_else(|_| "unknown type".to_owned(), |n| n.to_string())
}

/// Compressed RLE `counts`, from either the `str` or the `bytes` spelling;
/// `None` for anything else (an uncompressed list).
///
/// `bytes` is what `mask.encode` and pycocotools emit. It has to be checked
/// before any list branch: Python bytes extract as a sequence of ints, so a
/// caller that tries the list first reads the ASCII codes of the compressed
/// string as run lengths — a silently wrong mask, not an error. `bytearray`
/// and `memoryview` extract the same way, so they are a `TypeError`, as in
/// pycocotools. Every RLE dict reader goes through here so that cannot
/// happen to one and not the other again.
///
/// `bytes` is borrowed in place, and so is a `str` that is ASCII, as every
/// valid `counts` string is; any other `str` gets a UTF-8 copy that CPython
/// keeps on the string object. `bytes` is checked first: it is the common
/// spelling, and a failed `str` cast builds a Python error only to discard it.
fn counts_str<'a>(counts: &'a Bound<'_, PyAny>) -> PyResult<Option<&'a str>> {
    if let Ok(b) = counts.cast::<PyBytes>() {
        let s = std::str::from_utf8(b.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid UTF-8 in RLE counts: {e}"))
        })?;
        return Ok(Some(s));
    }
    if let Ok(s) = counts.cast::<PyString>() {
        // A `str` with no UTF-8 form (a lone surrogate) is left to the list
        // branch, which rejects it.
        return Ok(s.to_str().ok());
    }
    if counts.is_instance_of::<PyByteArray>() || counts.is_instance_of::<PyMemoryView>() {
        return Err(counts_type_error(counts));
    }
    Ok(None)
}

/// The `TypeError` for RLE `counts` that is neither a string nor a run list.
fn counts_type_error(counts: &Bound<'_, PyAny>) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!(
        "RLE 'counts' must be str, bytes, or a list of ints, got {}",
        type_name(counts)
    ))
}

/// Collect every key of `dict` not in `known` into a JSON map, in dict order.
///
/// Plain values, the common case (TorchMetrics adds a float `area_bbox` and an
/// int `area_segm` to every detection; CVAT-style data adds an `attributes`
/// dict), convert in Rust through [`to_json`]. Any value it declines sends the
/// whole record through `json.dumps` ([`extra_from_dict`]), so a value JSON
/// cannot hold raises exactly what it always did, message included.
///
/// `known` says whether a key is a schema key, and may consume it: every
/// caller passes its record type's field setter, such as [`set_ann_field`].
fn extract_extra(
    dict: &Bound<'_, PyDict>,
    mut known: impl FnMut(&str, &Bound<'_, PyAny>) -> PyResult<bool>,
) -> PyResult<Extra> {
    // The custom entries as JSON for as long as `to_json` takes every value,
    // each beside its own key and value objects; from the first value it
    // declines on, a dict of those objects for `json.dumps`.
    let mut converted = Vec::new();
    let mut fallback: Option<Bound<'_, PyDict>> = None;
    for (k, v) in dict {
        let Ok(key) = k.cast::<PyString>() else {
            continue; // non-string keys cannot appear in COCO JSON
        };
        let name = key.to_str()?;
        if known(name, &v)? {
            continue;
        }
        if let Some(fallback) = &fallback {
            fallback.set_item(k, v)?;
            continue;
        }
        match to_json(&v, MAX_JSON_DEPTH) {
            Some(json) => {
                let name = name.to_owned();
                converted.push((k, v, name, json));
            }
            None => {
                // `known` can consume a key, so `dict` cannot be walked a
                // second time, and a lookup by name misses a `str` subclass
                // key with its own `__hash__` or `__eq__`: the entries so far
                // go in as the objects kept beside them.
                let objects = PyDict::new(dict.py());
                for (key, value, _, _) in converted.drain(..) {
                    objects.set_item(key, value)?;
                }
                objects.set_item(k, v)?;
                fallback = Some(objects);
            }
        }
    }
    match fallback {
        Some(objects) => extra_from_dict(&objects),
        None => Ok(converted
            .into_iter()
            .map(|(_, _, name, json)| (name, json))
            .collect()),
    }
}

/// How many levels of nested lists and dicts [`to_json`] follows before it
/// declines.
///
/// Without a limit, a list that contains itself would recurse forever.
/// Declined, it reaches `json.dumps`, which raises its own `Circular
/// reference detected`. 32 levels is far deeper than real custom metadata
/// goes, and far below the 128 that `serde_json` reads back from `json.dumps`
/// output, so every value taken here is one that path takes too.
const MAX_JSON_DEPTH: usize = 32;

/// `value` as the `serde_json::Value` that parsing `json.dumps(value)`
/// yields, when it is `None`, a `bool`, `int`, `float`, or `str`, or an exact
/// `list`, `tuple`, or `dict` of those, nested at most `depth` levels deep;
/// `None` for anything else, which keeps the `json.dumps` path.
///
/// - A subclass of `int`, `float`, or `str` is read as its base value, which
///   is what `json` writes for it: `int.__repr__`, `float.__repr__`, or the
///   string itself, whatever the subclass overrides. So an `IntEnum` member,
///   `numpy.float64`, and `numpy.str_` convert here. Other numpy scalars,
///   such as `numpy.float32` and `numpy.int64`, subclass neither, and `json`
///   rejects them. `bool` cannot be subclassed.
/// - A container is taken only as its exact type: `json` reads a `list` or
///   `tuple` subclass through `__iter__` and a `dict` subclass through
///   `items()`, which a subclass can override.
/// - A `float` stays a float, `1.0` included: its `repr` always has a `.` or
///   an exponent, so `serde_json` never reads it back as an integer, and the
///   core crate's `float_roundtrip` feature makes that read exact. NaN and
///   infinity have no JSON spelling — `serde_json` rejects what `json.dumps`
///   writes for them — so they are not converted here.
/// - An `int` gets the split `serde_json` makes when parsing one: `u64` if
///   non-negative, `i64` if negative. Outside both ranges it parses as a
///   float, so such an int is not converted here.
/// - A `str` with a lone surrogate has no UTF-8 form, so it is not converted
///   here either.
/// - A `tuple` is an array, as `json.dumps` writes it.
/// - A `dict` is converted only when every key is a `str`, which `json`
///   writes as its base string too. `json.dumps` turns an `int`, `float`,
///   `bool`, or `None` key into a string by its own spelling rules, which are
///   not repeated here.
/// - A `dict`'s entries are inserted into a `serde_json::Map`, which orders
///   them exactly as parsing does: sorted by key without `serde_json`'s
///   `preserve_order` feature (this workspace's build), in insertion order —
///   the order `json.dumps` writes — with it.
fn to_json(value: &Bound<'_, PyAny>, depth: usize) -> Option<serde_json::Value> {
    use serde_json::Value;
    if value.is_none() {
        Some(Value::Null)
    } else if let Ok(b) = value.cast_exact::<PyBool>() {
        Some(Value::Bool(b.is_true()))
    } else if let Ok(f) = value.cast::<PyFloat>() {
        // `PyFloat_AsDouble` reads a subclass's value without `__float__`.
        serde_json::Number::from_f64(f.value()).map(Value::Number)
    } else if value.is_instance_of::<PyInt>() {
        // `PyLong_AsLongLong` and `PyLong_AsUnsignedLongLong` read an int
        // subclass's value without `__int__` or `__index__`.
        value
            .extract::<i64>()
            .map(Value::from)
            .or_else(|_| value.extract::<u64>().map(Value::from))
            .ok()
    } else if let Ok(s) = value.cast::<PyString>() {
        s.to_str().ok().map(|s| Value::String(s.to_owned()))
    } else if depth == 0 {
        None
    } else if value.is_exact_instance_of::<PyList>() || value.is_exact_instance_of::<PyTuple>() {
        value
            .try_iter()
            .ok()?
            .map(|item| to_json(&item.ok()?, depth - 1))
            .collect::<Option<_>>()
            .map(Value::Array)
    } else if let Ok(dict) = value.cast_exact::<PyDict>() {
        dict.iter()
            .map(|(k, v)| {
                let key = k.cast::<PyString>().ok()?.to_str().ok()?;
                Some((key.to_owned(), to_json(&v, depth - 1)?))
            })
            .collect::<Option<_>>()
            .map(Value::Object)
    } else {
        None
    }
}

/// The custom keys of a record, as the JSON they will be saved as.
fn extra_from_dict(extras: &Bound<'_, PyDict>) -> PyResult<Extra> {
    let json_str = extras
        .py()
        .import("json")?
        .call_method1("dumps", (extras,))?
        .cast_into::<PyString>()?;
    serde_json::from_str(json_str.to_str()?).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "custom keys did not round-trip through JSON: {e}"
        ))
    })
}

/// `value` as the object `json.loads` returns for the text `serde_json`
/// writes for it, built without the text: the way back from [`to_json`].
///
/// - A number holding an integer is an `int`, and one holding an `f64` is a
///   `float`, `1.0` included: `serde_json` writes an `f64` with a `.` or an
///   exponent, so `json.loads` reads it back as a `float`. The value is the
///   same `f64`, since that text is the shortest that reads back as it and
///   Python's reading is correctly rounded.
/// - An object's entries come in the order of its `serde_json::Map`, which
///   is the order `serde_json` writes them in: sorted by key in this
///   workspace's build, without the `preserve_order` feature.
/// - The recursion needs no limit of its own. Every `Value` in an [`Extra`]
///   was parsed by `serde_json`, which stops at 128 levels, or built by
///   [`to_json`], which stops at [`MAX_JSON_DEPTH`].
fn json_to_py<'py>(py: Python<'py>, value: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    use serde_json::Value;
    let object = match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(f) = n.as_f64().filter(|_| n.is_f64()) {
                PyFloat::new(py, f).into_any()
            } else {
                // Only `serde_json`'s `arbitrary_precision` feature, which
                // this workspace does not enable, holds any other number.
                crate::serde_to_py(py, value)?.into_bound(py)
            }
        }
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(items) => {
            let items = items
                .iter()
                .map(|item| json_to_py(py, item))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            dict.into_any()
        }
    };
    Ok(object)
}

/// Merge a record's `extra` map back into its outgoing Python dict.
///
/// The custom keys go in after the schema keys, in file order. A custom key
/// with a schema key's name, which only the Rust API can create, replaces
/// that key's value where it stands.
fn merge_extra(dict: &Bound<'_, PyDict>, extra: &Extra) -> PyResult<()> {
    for (key, value) in extra {
        dict.set_item(key, json_to_py(dict.py(), value)?)?;
    }
    Ok(())
}

pub fn annotation_to_py(py: Python<'_>, ann: &Annotation) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(pyo3::intern!(py, "id"), ann.id)?;
    dict.set_item(pyo3::intern!(py, "image_id"), ann.image_id)?;
    dict.set_item(pyo3::intern!(py, "category_id"), ann.category_id)?;
    if let Some(ref bbox) = ann.bbox {
        dict.set_item(pyo3::intern!(py, "bbox"), &bbox[..])?;
    }
    if let Some(area) = ann.area {
        dict.set_item(pyo3::intern!(py, "area"), area)?;
    }
    if let Some(ref seg) = ann.segmentation {
        dict.set_item(
            pyo3::intern!(py, "segmentation"),
            segmentation_to_py(py, seg)?,
        )?;
    }
    dict.set_item(pyo3::intern!(py, "iscrowd"), ann.iscrowd as u8)?;
    if let Some(ref kpts) = ann.keypoints {
        dict.set_item(pyo3::intern!(py, "keypoints"), kpts)?;
    }
    if let Some(nk) = ann.num_keypoints {
        dict.set_item(pyo3::intern!(py, "num_keypoints"), nk)?;
    }
    if let Some(ref obb) = ann.obb {
        dict.set_item(pyo3::intern!(py, "obb"), &obb[..])?;
    }
    if let Some(score) = ann.score {
        dict.set_item(pyo3::intern!(py, "score"), score)?;
    }
    if let Some(is_group_of) = ann.is_group_of {
        dict.set_item(pyo3::intern!(py, "is_group_of"), is_group_of)?;
    }
    merge_extra(&dict, &ann.extra)?;
    Ok(dict.into_any().unbind())
}

pub fn segmentation_to_py(py: Python<'_>, seg: &Segmentation) -> PyResult<Py<PyAny>> {
    match seg {
        Segmentation::Polygon(polys) => polygons_to_py(py, polys),
        Segmentation::Rect(bbox) => polygons_to_py(py, &[Segmentation::rect_corners(bbox)]),
        Segmentation::CompressedRle { size, counts } => {
            let dict = PyDict::new(py);
            dict.set_item(pyo3::intern!(py, "size"), &size[..])?;
            dict.set_item(pyo3::intern!(py, "counts"), counts)?;
            Ok(dict.into_any().unbind())
        }
        Segmentation::UncompressedRle { size, counts } => {
            let dict = PyDict::new(py);
            dict.set_item(pyo3::intern!(py, "size"), &size[..])?;
            dict.set_item(pyo3::intern!(py, "counts"), counts)?;
            Ok(dict.into_any().unbind())
        }
        // `Segmentation` is `#[non_exhaustive]`: a format the core adds must
        // be given a Python shape here, and this arm makes forgetting loud.
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "segmentation format has no Python representation yet: {seg:?}"
        ))),
    }
}

fn polygons_to_py<P: AsRef<[f64]>>(py: Python<'_>, polys: &[P]) -> PyResult<Py<PyAny>> {
    let inner_lists: Vec<Bound<'_, PyList>> = polys
        .iter()
        .map(|p| PyList::new(py, p.as_ref()))
        .collect::<PyResult<_>>()?;
    Ok(PyList::new(py, inner_lists)?.into_any().unbind())
}

/// Set the annotation field a schema key names; `false` for any other key,
/// which is a custom key for `extra`.
///
/// The one place a dict key becomes an annotation field, for a whole record
/// ([`py_to_annotation`]) and for a partial edit ([`merge_ann_dict`]) alike,
/// so both read a value by the same rules. A `None` on an optional field
/// reads as absent, as `null` does in a file ([`optional`]).
fn set_ann_field(ann: &mut Annotation, key: &str, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    match key {
        "id" => ann.id = extract_int(value)?,
        "image_id" => ann.image_id = extract_int(value)?,
        "category_id" => ann.category_id = extract_int(value)?,
        "bbox" => ann.bbox = value.extract()?,
        "area" => ann.area = value.extract()?,
        "segmentation" => ann.segmentation = optional(value, py_to_segmentation)?,
        "iscrowd" => ann.iscrowd = extract_flag(value)?,
        "keypoints" => ann.keypoints = optional(value, py_to_keypoints)?,
        "num_keypoints" => ann.num_keypoints = optional(value, extract_int)?,
        "obb" => ann.obb = value.extract::<Option<[f64; 5]>>()?.map(Box::new),
        "score" => ann.score = value.extract()?,
        "is_group_of" => ann.is_group_of = optional(value, extract_flag)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// An optional field read with `read`, or `None` for a Python `None` — what
/// the JSON loader makes of `null`, so a dict and the file it came from load
/// alike. A field PyO3 extracts directly gets the same rule from extracting
/// an `Option`.
fn optional<'py, T>(
    value: &Bound<'py, PyAny>,
    read: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<Option<T>> {
    if value.is_none() {
        Ok(None)
    } else {
        read(value).map(Some)
    }
}

/// Flat `[x, y, v, …]` is the COCO spelling; an `(N, 3)` array is how the
/// same triplets sit in a tensor. Both mean one thing.
fn py_to_keypoints(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    value.extract::<Vec<f64>>().or_else(|flat_err| {
        value
            .extract::<Vec<[f64; 3]>>()
            .map(|rows| rows.into_iter().flatten().collect())
            .map_err(|_| flat_err)
    })
}

/// Apply every key of `dict` to `ann` in one pass: schema keys set their
/// fields, the rest land in `extra`, replacing a custom key of the same
/// name. Every schema-unknown key is allowed to create a new custom key —
/// used for a whole record, which has no prior state to check a key
/// against.
pub(crate) fn merge_ann_dict(ann: &mut Annotation, dict: &Bound<'_, PyDict>) -> PyResult<()> {
    merge_ann_dict_checked(ann, dict, true)
}

/// [`merge_ann_dict`], but a schema-unknown key that is not already a
/// custom key on `ann` is an error unless `create` is true — so a
/// misspelled schema field (`"Area"`, `"iscrowed"`) raises instead of
/// quietly landing beside the field the caller meant to change.
pub(crate) fn merge_ann_dict_checked(
    ann: &mut Annotation,
    dict: &Bound<'_, PyDict>,
    create: bool,
) -> PyResult<()> {
    let mut unknown_field = None;
    let custom = extract_extra(dict, |key, value| {
        let set = set_ann_field(ann, key, value)?;
        if !set && !create && unknown_field.is_none() && !ann.extra.contains_key(key) {
            unknown_field = Some(key.to_string());
        }
        Ok(set)
    })?;
    if let Some(field) = unknown_field {
        return Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "'{field}' is not an annotation field, and annotation {} does not carry it as a \
             custom key — check the spelling, or pass create=True to add it",
            ann.id
        )));
    }
    if ann.extra.is_empty() {
        ann.extra = custom;
    } else if !custom.is_empty() {
        ann.extra = std::mem::take(&mut ann.extra)
            .into_iter()
            .chain(custom)
            .collect();
    }
    Ok(())
}

pub fn py_to_annotation(dict: &Bound<'_, PyDict>) -> PyResult<Annotation> {
    req!(dict, "image_id");
    let mut ann = Annotation::default();
    merge_ann_dict(&mut ann, dict)?;
    Ok(ann)
}

pub fn py_to_segmentation(obj: &Bound<'_, PyAny>) -> PyResult<Segmentation> {
    // Try as dict (CompressedRle or UncompressedRle)
    if let Ok(dict) = obj.cast::<PyDict>() {
        return RleDict::read(dict)?.into_segmentation();
    }
    // Otherwise it's a polygon (list of lists)
    let polys: Vec<Vec<f64>> = obj.extract()?;
    Ok(Segmentation::Polygon(polys))
}

pub fn image_to_py(py: Python<'_>, img: &Image) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(pyo3::intern!(py, "id"), img.id)?;
    dict.set_item(pyo3::intern!(py, "file_name"), &img.file_name)?;
    dict.set_item(pyo3::intern!(py, "height"), img.height)?;
    dict.set_item(pyo3::intern!(py, "width"), img.width)?;
    if let Some(license) = img.license {
        dict.set_item(pyo3::intern!(py, "license"), license)?;
    }
    if let Some(ref url) = img.coco_url {
        dict.set_item(pyo3::intern!(py, "coco_url"), url)?;
    }
    if let Some(ref url) = img.flickr_url {
        dict.set_item(pyo3::intern!(py, "flickr_url"), url)?;
    }
    if let Some(ref dc) = img.date_captured {
        dict.set_item(pyo3::intern!(py, "date_captured"), dc)?;
    }
    if !img.neg_category_ids.is_empty() {
        dict.set_item(pyo3::intern!(py, "neg_category_ids"), &img.neg_category_ids)?;
    }
    if !img.not_exhaustive_category_ids.is_empty() {
        dict.set_item(
            pyo3::intern!(py, "not_exhaustive_category_ids"),
            &img.not_exhaustive_category_ids,
        )?;
    }
    merge_extra(&dict, &img.extra)?;
    Ok(dict.into_any().unbind())
}

pub fn category_to_py(py: Python<'_>, cat: &Category) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(pyo3::intern!(py, "id"), cat.id)?;
    dict.set_item(pyo3::intern!(py, "name"), &cat.name)?;
    if let Some(ref sc) = cat.supercategory {
        dict.set_item(pyo3::intern!(py, "supercategory"), sc)?;
    }
    if let Some(ref sk) = cat.skeleton {
        // Each `[u32; 2]` pair becomes a list, as `Vec<Vec<u32>>` would.
        dict.set_item(pyo3::intern!(py, "skeleton"), sk)?;
    }
    if let Some(ref kpts) = cat.keypoints {
        dict.set_item(pyo3::intern!(py, "keypoints"), kpts)?;
    }
    if let Some(ref freq) = cat.frequency {
        dict.set_item(pyo3::intern!(py, "frequency"), freq)?;
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
    compressed_rle_to_py(py, rle.h, rle.w, &hotcoco_core::mask::rle_to_string(rle))
}

/// [`rle_to_coco_py`] for an RLE already compressed to its `counts` string.
pub(crate) fn compressed_rle_to_py(
    py: Python<'_>,
    h: u32,
    w: u32,
    counts: &str,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(pyo3::intern!(py, "size"), [h, w])?;
    let py_bytes = PyBytes::new(py, counts.as_bytes());
    dict.set_item(pyo3::intern!(py, "counts"), py_bytes)?;
    Ok(dict.into_any().unbind())
}

pub fn py_to_rle(dict: &Bound<'_, PyDict>) -> PyResult<Rle> {
    RleDict::read(dict)?.into_rle()
}

/// A core RLE error as the `ValueError` every RLE reader raises for a string
/// that does not decode or runs that overrun the mask.
pub(crate) fn rle_result<T>(result: Result<T, hotcoco_core::Error>) -> PyResult<T> {
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// An RLE dict, read: the one reader every RLE dict from Python goes through,
/// a record's `segmentation` and every `hotcoco.mask` function alike, so the
/// spellings they take and the errors they raise cannot drift apart.
///
/// Takes the pycocotools spelling `{"size": [h, w], "counts": ...}` and
/// `{"h": h, "w": w, "counts": ...}`, with `counts` as `bytes`, `str`, or a
/// list of ints in either. A compressed `counts` stays in its Python object
/// and [`view`](Self::view) borrows it; a run list is read here.
pub(crate) struct RleDict<'py> {
    h: u32,
    w: u32,
    counts: Counts<'py>,
}

enum Counts<'py> {
    /// `bytes` or `str`, as [`counts_str`] reads it.
    Compressed(Bound<'py, PyAny>),
    Runs(Vec<u32>),
}

impl<'py> RleDict<'py> {
    pub(crate) fn read(dict: &Bound<'py, PyDict>) -> PyResult<Self> {
        let py = dict.py();
        let missing = |key: &str| {
            pyo3::exceptions::PyValueError::new_err(format!("RLE dict missing '{key}'"))
        };
        let (h, w) = if let Some(size) = dict.get_item(pyo3::intern!(py, "size"))? {
            let [h, w]: [u32; 2] = size.extract()?;
            (h, w)
        } else if let Some(h) = dict.get_item(pyo3::intern!(py, "h"))? {
            let w = dict
                .get_item(pyo3::intern!(py, "w"))?
                .ok_or_else(|| missing("w"))?;
            (h.extract()?, w.extract()?)
        } else {
            return Err(missing("size"));
        };
        let counts = dict
            .get_item(pyo3::intern!(py, "counts"))?
            .ok_or_else(|| missing("counts"))?;
        let counts = if is_compressed(&counts)? {
            Counts::Compressed(counts)
        } else {
            Counts::Runs(counts.extract().map_err(|_| counts_type_error(&counts))?)
        };
        Ok(RleDict { h, w, counts })
    }

    /// The RLE as the core's borrowed view, for its area, box, or run list.
    pub(crate) fn view(&self) -> PyResult<RleRef<'_>> {
        let (h, w) = (self.h, self.w);
        Ok(match &self.counts {
            Counts::Compressed(obj) => RleRef::Compressed {
                counts: compressed_counts(obj)?,
                h,
                w,
            },
            Counts::Runs(runs) => RleRef::Runs { counts: runs, h, w },
        })
    }

    /// The RLE's run list, validated as [`view`](Self::view) validates it,
    /// keeping a run list that was read rather than copying it.
    pub(crate) fn into_rle(self) -> PyResult<Rle> {
        let (h, w) = (self.h, self.w);
        match self.counts {
            Counts::Compressed(obj) => rle_result(hotcoco_core::mask::rle_from_string(
                compressed_counts(&obj)?,
                h,
                w,
            )),
            Counts::Runs(counts) => {
                rle_result(hotcoco_core::mask::check_counts(&counts, h, w))?;
                Ok(Rle { h, w, counts })
            }
        }
    }

    /// The RLE as a record's `segmentation`. Not validated: a loaded dataset
    /// keeps an RLE as given until a mask is drawn from it.
    pub(crate) fn into_segmentation(self) -> PyResult<Segmentation> {
        let size = [self.h, self.w];
        Ok(match self.counts {
            Counts::Compressed(obj) => Segmentation::CompressedRle {
                size,
                counts: compressed_counts(&obj)?.to_owned(),
            },
            Counts::Runs(counts) => Segmentation::UncompressedRle { size, counts },
        })
    }
}

/// Whether `counts` is a compressed string, `bytes` or `str`, judged by type
/// alone: [`counts_str`] validates it when the string is read. A byte buffer
/// it rejects is the same `TypeError` here.
fn is_compressed(counts: &Bound<'_, PyAny>) -> PyResult<bool> {
    if counts.is_instance_of::<PyBytes>() || counts.is_instance_of::<PyString>() {
        return Ok(true);
    }
    if counts.is_instance_of::<PyByteArray>() || counts.is_instance_of::<PyMemoryView>() {
        return Err(counts_type_error(counts));
    }
    Ok(false)
}

/// The string of a `counts` that [`RleDict::read`] found compressed.
fn compressed_counts<'a>(obj: &'a Bound<'_, PyAny>) -> PyResult<&'a str> {
    counts_str(obj)?.ok_or_else(|| counts_type_error(obj))
}

/// One RLE dict from a Python object (`size` + `counts`, or `h` + `w` +
/// `counts`).
pub(crate) fn extract_coco_rle(obj: &Bound<'_, PyAny>) -> PyResult<Rle> {
    py_to_rle(obj.cast::<PyDict>()?)
}

/// A list of RLE dicts, or a single dict as a list of one. Shared by
/// `hotcoco.mask` and `hotcoco.primitives`.
pub(crate) fn extract_rle_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Rle>> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        return Ok(vec![py_to_rle(dict)?]);
    }
    let list: Vec<Bound<'_, PyAny>> = obj.extract()?;
    list.iter().map(extract_coco_rle).collect()
}

/// Set the image field a schema key names; `false` for any other key, which
/// is a custom key for `extra`. The one place a dict key becomes an image
/// field, as [`set_ann_field`] is for annotations, with the same rule for
/// `None`.
fn set_image_field(img: &mut Image, key: &str, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    match key {
        "id" => img.id = extract_int(value)?,
        "file_name" => img.file_name = value.extract()?,
        "height" => img.height = extract_int(value)?,
        "width" => img.width = extract_int(value)?,
        "license" => img.license = optional(value, extract_int)?,
        "coco_url" => img.coco_url = value.extract()?,
        "flickr_url" => img.flickr_url = value.extract()?,
        "date_captured" => img.date_captured = value.extract()?,
        "neg_category_ids" => img.neg_category_ids = value.extract()?,
        "not_exhaustive_category_ids" => img.not_exhaustive_category_ids = value.extract()?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// Set the category field a schema key names; `false` for any other key.
/// See [`set_image_field`].
fn set_category_field(cat: &mut Category, key: &str, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    match key {
        "id" => cat.id = extract_int(value)?,
        "name" => cat.name = value.extract()?,
        "supercategory" => cat.supercategory = value.extract()?,
        "skeleton" => cat.skeleton = value.extract()?,
        "keypoints" => cat.keypoints = value.extract()?,
        "frequency" => cat.frequency = value.extract()?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// Read every key of `dict` in one pass: schema keys set their fields, the
/// rest are the image's custom keys.
pub fn py_to_image(dict: &Bound<'_, PyDict>) -> PyResult<Image> {
    // Only `id` is required. pycocotools' assignment flow (`coco.dataset = d;
    // coco.createIndex()`) builds images as bare `{"id": …}` — that is what
    // torchmetrics' pycocotools backend passes. Dimensions are only consumed
    // by mask operations, which pycocotools equally cannot perform without
    // them.
    req!(dict, "id");
    let mut img = Image::default();
    img.extra = extract_extra(dict, |key, value| set_image_field(&mut img, key, value))?;
    Ok(img)
}

/// [`py_to_image`] for a category.
pub fn py_to_category(dict: &Bound<'_, PyDict>) -> PyResult<Category> {
    // Only `id` is required. A missing `name` is tolerated the way pycocotools
    // tolerates it (it stores raw dicts); `COCO::create_index` fills the
    // placeholder.
    req!(dict, "id");
    let mut cat = Category::default();
    cat.extra = extract_extra(dict, |key, value| set_category_field(&mut cat, key, value))?;
    Ok(cat)
}

/// Convert a list of dicts element by element. `what` names the list in the
/// `TypeError` a non-dict element raises.
pub fn dict_list<'py, T>(
    list: &Bound<'py, PyList>,
    what: &str,
    convert_fn: impl Fn(&Bound<'py, PyDict>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    list.iter()
        .map(|item| {
            let d = item.cast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "each item in '{what}' must be a dict"
                ))
            })?;
            convert_fn(d)
        })
        .collect()
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
        Some(v) => {
            let list = v.cast::<PyList>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!("'{key}' must be a list of dicts"))
            })?;
            dict_list(list, key, convert_fn)
        }
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

/// A numpy array of any integer or float dtype as a `float64` array of `D`
/// dimensions: the array itself when it is `float64`, otherwise numpy's
/// `astype` copy of it, one pass in C. The cast is the conversion a Python
/// `float()` of each element makes, so it is exact for every `float32` and
/// `float16`, and for every integer up to 2^53. `None` for anything else —
/// a list, a bool or object array, an array of other dimensions — which the
/// caller reads element by element.
pub(crate) fn numeric_array<'py, D: numpy::ndarray::Dimension>(
    obj: &Bound<'py, PyAny>,
) -> PyResult<Option<numpy::PyReadonlyArray<'py, f64, D>>> {
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray<'py, f64, D>>() {
        return Ok(Some(arr));
    }
    let Ok(arr) = obj.cast::<numpy::PyUntypedArray>() else {
        return Ok(None);
    };
    if D::NDIM.is_some_and(|ndim| ndim != arr.ndim())
        || !matches!(arr.dtype().kind(), b'i' | b'u' | b'f')
    {
        return Ok(None);
    }
    let py = obj.py();
    let wide = obj.call_method1(pyo3::intern!(py, "astype"), (numpy::dtype::<f64>(py),))?;
    Ok(Some(wide.extract()?))
}

/// A 1-D float argument: a numpy array of any integer or float dtype in one
/// pass ([`numeric_array`]), anything else via the list path.
///
/// The metrics docstrings advertise "lists or numpy arrays", but PyO3's
/// `Vec<f64>` fast path fires only for list/tuple, leaving a numpy array to
/// per-element iteration — one boxed `extract` per element, ~500K calls per
/// argument on a full val2017 run. `to_vec` goes through ndarray, so strided
/// views (`scores[::2]`) copy correctly instead of being rejected.
pub fn f64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if let Some(arr) = numeric_array::<numpy::Ix1>(obj)? {
        return Ok(arr.as_array().to_vec());
    }
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a sequence of floats or a 1-D numpy array"
        ))
    })
}

/// A 1-D non-negative integer argument — see [`f64_vec`]. numpy `int64` and
/// `int32` (what detection code usually holds) in one copy, a sequence of ints
/// otherwise. A negative value raises `ValueError`: ids are unsigned.
pub fn u64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u64>> {
    let signed: Vec<i64> = if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray1<i64>>() {
        arr.as_array().to_vec()
    } else if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray1<i32>>() {
        arr.as_array().iter().map(|&v| i64::from(v)).collect()
    } else {
        obj.extract::<Vec<i64>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{name} must be a sequence of ints or a 1-D numpy int array"
            ))
        })?
    };
    signed
        .into_iter()
        .map(|v| {
            u64::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} must not contain negative values, got {v}"
                ))
            })
        })
        .collect()
}

/// A 1-D flag argument — `iscrowd` for `from_arrays` and the IoU functions.
///
/// A numpy bool array is read as is and an `int64`/`int32` array as non-zero
/// means true, each in one copy; anything else (a list of `0`/`1` or bools,
/// a float array read back from pandas) goes through [`extract_flag`] item by
/// item, so every flag spelling the API accepts elsewhere works here too.
/// COCO JSON stores `iscrowd` as `0`/`1`, and pycocotools takes
/// `maskUtils.iou(dt, gt, [a["iscrowd"] for a in anns])` straight through.
pub fn flag_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<bool>> {
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray1<bool>>() {
        return Ok(arr.as_array().to_vec());
    }
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray1<i64>>() {
        return Ok(arr.as_array().iter().map(|&v| v != 0).collect());
    }
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray1<i32>>() {
        return Ok(arr.as_array().iter().map(|&v| v != 0).collect());
    }
    let iter = obj.try_iter().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a sequence of flags or a 1-D numpy array, got {}",
            type_name(obj)
        ))
    })?;
    iter.map(|item| extract_flag(&item?)).collect()
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
/// family: a numpy array of any integer or float dtype in one pass
/// ([`numeric_array`]), a nested sequence (list of lists, tuple of tuples,
/// ...) otherwise, with a named `TypeError` on either failing. Returns a row-major `(flat, nr, nc)` triple, matching
/// `assign::lsap`'s contract even for a Fortran-order or otherwise strided
/// numpy input, since `.as_array().iter()` always yields row-major order.
///
/// A nested sequence gets an additional rectangularity check: mismatched row
/// lengths raise `ValueError`, not `TypeError` — the type is right, the shape
/// isn't. Callers that also need to reject NaN (as `lsap` does) check that
/// locally; whether NaN is meaningful is caller-specific, not a property of
/// extracting a 2-D array.
pub fn f64_matrix_arg(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<(Vec<f64>, usize, usize)> {
    if let Some(arr) = numeric_array::<numpy::Ix2>(obj)? {
        let view = arr.as_array();
        let (nr, nc) = view.dim();
        return Ok((view.iter().copied().collect(), nr, nc));
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

/// An id-list argument read the way pycocotools reads it: a bare int or any
/// iterable of ints. torchvision's `CocoDetection` calls
/// `coco.getAnnIds(img_id)` with a scalar — found by the 1.0
/// third-party-consumer smoke test — and pycocotools' `_isArrayLike` admits a
/// `set` or a `dict.keys()` view as readily as a list. One extractor shared by
/// every query/load method and its camelCase twin, so the two surfaces cannot
/// disagree.
#[derive(Default)]
pub struct IdList(pub Vec<u64>);

impl FromPyObject<'_, '_> for IdList {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        if let Ok(one) = obj.extract::<u64>() {
            return Ok(IdList(vec![one]));
        }
        // A list, tuple, or numpy array takes the bulk path; a set, a
        // `dict.keys()` view, or a generator is walked item by item.
        if let Ok(ids) = obj.extract::<Vec<u64>>() {
            return Ok(IdList(ids));
        }
        let not_ids = || {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "expected an int or an iterable of ints, got {}",
                type_name(&obj)
            ))
        };
        let iter = obj.try_iter().map_err(|_| not_ids())?;
        iter.map(|item| item?.extract::<u64>().map_err(|_| not_ids()))
            .collect::<PyResult<_>>()
            .map(IdList)
    }
}

/// A keyword flag on a pycocotools-compatible method, read by [`extract_flag`]:
/// `getAnnIds(iscrowd=0)` and `mask.merge(rles, 1)` are how pycocotools callers
/// spell them. The snake_case methods take a plain `bool`; only the
/// compatibility surface is this lenient.
pub struct Flag(pub bool);

impl FromPyObject<'_, '_> for Flag {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        extract_flag(&obj).map(Flag)
    }
}

/// pycocotools' `areaRng`: `[lo, hi]`, or an empty sequence for no range
/// filter — `getAnnIds` tests `len(areaRng) == 0`, so `[]` is how callers
/// spell "unset" as often as `None` is.
pub struct AreaRng(pub Option<[f64; 2]>);

impl FromPyObject<'_, '_> for AreaRng {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        if obj.len().is_ok_and(|n| n == 0) {
            return Ok(AreaRng(None));
        }
        Ok(AreaRng(Some(obj.extract()?)))
    }
}

/// An `(N, 4)` box argument, read by [`f64_matrix_arg`]: a numpy array of
/// any integer or float dtype in one pass, a sequence of 4-element rows
/// element by element. An empty input is zero boxes whatever its width.
pub fn boxes_arg(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<[f64; 4]>> {
    let (flat, nrows, ncols) = f64_matrix_arg(obj, name)?;
    if ncols != 4 && nrows != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape (N, 4), got ({nrows}, {ncols})"
        )));
    }
    Ok(flat
        .chunks_exact(4)
        .map(|b| [b[0], b[1], b[2], b[3]])
        .collect())
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
