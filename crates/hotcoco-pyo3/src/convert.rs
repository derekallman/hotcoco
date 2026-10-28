use std::borrow::Cow;

use hotcoco_core::mask::RleRef;
use hotcoco_core::metrics::semantic::LabelMap;
use hotcoco_core::panoptic::{MatchIds, PanopticAnnotation, PanopticDataset, SegmentInfo};
use hotcoco_core::{Annotation, Category, Dataset, DatasetStats, Extra, Image, Rle, Segmentation};
use numpy::{PyArray1, PyArrayDescrMethods, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyMemoryView, PySequence,
    PyString, PyTuple,
};

use crate::to_pyerr;

/// A required field of a Python dict, raising `ValueError("dict missing
/// '<key>'")` when it is absent.
///
/// `$key` is interned (`pyo3::intern!`) rather than passed as a bare `&str`:
/// `get_item` takes anything `IntoPyObject`, and for a plain `&str` that means
/// allocating a fresh `PyString` on every call. Record fields are read in one
/// pass over the dict by [`set_ann_field`], [`set_image_field`], and
/// [`set_category_field`]; their converters call this first only to check
/// the required keys.
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
/// alike. A fractional, negative, or NaN number is a `ValueError` naming the
/// value, and anything that is not a number a `TypeError`; a value that does
/// not fit the field's width is an `OverflowError`. Inside a record all three
/// are `ValueError` ([`field_error`]). The JSON loader applies the same rule
/// through `types::deserialize_uint`.
///
/// A `bool` is a `ValueError` too: `True` is the int 1 to Python, but a JSON
/// `true` is not a number, and a record must load alike from a dict and from
/// the file it is saved to.
pub(crate) fn extract_int<T: TryFrom<u64>>(v: &Bound<'_, PyAny>) -> PyResult<T> {
    // An exact `int` is what nearly every id is, and it is never a `bool`;
    // everything else is read by kind, bools first, since `True` is an index
    // to Python and a `numpy.bool_` still is to numpy 1.x.
    let id = if v.is_exact_instance_of::<PyInt>() {
        v.extract::<u64>().ok()
    } else if v.is_instance_of::<PyFloat>() {
        hotcoco_core::types::id_from_f64(v.extract()?)
    } else if is_bool(v) {
        None
    } else if let Ok(i) = v.extract::<u64>() {
        Some(i)
    } else {
        v.extract::<f64>()
            .ok()
            .and_then(hotcoco_core::types::id_from_f64)
    };
    let Some(n) = id else {
        return Err(not_a(v, "a non-negative integer")?);
    };
    T::try_from(n).map_err(|_| {
        pyo3::exceptions::PyOverflowError::new_err(format!(
            "integer {n} is out of range for this field"
        ))
    })
}

/// Whether `v` is a bool of any spelling: `bool`, `numpy.bool_`, or a 0-d
/// bool array. The numpy type is compared by pointer, once cached.
pub(crate) fn is_bool(v: &Bound<'_, PyAny>) -> bool {
    static NUMPY_BOOL: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
        pyo3::sync::PyOnceLock::new();
    let py = v.py();
    v.is_instance_of::<PyBool>()
        || NUMPY_BOOL
            .get_or_try_init(py, || {
                let ty = py.import("numpy")?.getattr(pyo3::intern!(py, "bool_"))?;
                Ok::<_, PyErr>(ty.cast_into::<pyo3::types::PyType>()?.unbind())
            })
            .is_ok_and(|ty| v.get_type().is(ty))
        || v.cast::<numpy::PyUntypedArray>()
            .is_ok_and(|a| a.dtype().kind() == b'b')
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
    Err(not_a(v, "a bool or 0/1 flag")?)
}

/// A number inside a record — a box edge, an area, a score, a keypoint — as
/// a float: `int`, `float`, or a numpy number, never a bool of any spelling.
/// `True` is the number 1 to Python, but a JSON `true` is not a number, and a
/// record must load alike from a dict and from the file it is saved to, the
/// rule [`extract_int`] applies to an id.
pub(crate) fn extract_float(v: &Bound<'_, PyAny>) -> PyResult<f64> {
    // A `float` subclass is a `numpy.float64`, the common case for a record
    // built from arrays; `bool` is not one, so the fast path cannot take it.
    if let Ok(f) = v.cast::<PyFloat>() {
        return Ok(f.value());
    }
    if v.is_exact_instance_of::<PyInt>() {
        return v.extract();
    }
    if is_bool(v) {
        return Err(not_a(v, "a number")?);
    }
    v.extract()
}

/// `v` checked to be a sequence of numbers to read by index — a list, a
/// tuple, or a numpy array, which is not a `Sequence` to Python but indexes
/// like one — and not any iterable: a set has no order and a dict iterates
/// its keys, and a JSON file can hold neither. A `str` is refused, as PyO3
/// refuses it for a `Vec`: its items are strings, never numbers.
fn number_sequence<'a, 'py>(v: &'a Bound<'py, PyAny>) -> PyResult<&'a Bound<'py, PyAny>> {
    let indexable = !v.is_instance_of::<PyString>()
        && (v.cast::<PySequence>().is_ok() || v.cast::<numpy::PyUntypedArray>().is_ok());
    if indexable {
        Ok(v)
    } else {
        Err(not_a(v, "a sequence of numbers")?)
    }
}

/// Every number of a sequence, each read by [`extract_float`].
fn float_vec(v: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let seq = number_sequence(v)?;
    let mut out = Vec::with_capacity(seq.len()?);
    for item in seq.try_iter()? {
        out.push(extract_float(&item?)?);
    }
    Ok(out)
}

/// Exactly `N` numbers from a sequence, each read by [`extract_float`].
/// Reads the items in place, as PyO3's own array extraction does: a box is
/// read for every annotation of a dict dataset.
fn float_array<const N: usize>(v: &Bound<'_, PyAny>) -> PyResult<[f64; N]> {
    let seq = number_sequence(v)?;
    let len = seq.len()?;
    if len != N {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected a sequence of length {N} (got {len})"
        )));
    }
    let mut out = [0.0; N];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = extract_float(&seq.get_item(i)?)?;
    }
    Ok(out)
}

/// The error for `v`, which is not `expected`: a `ValueError` when `v` is a
/// number (an `int` too large for a float is one), a `TypeError` otherwise.
fn not_a(v: &Bound<'_, PyAny>, expected: &str) -> PyResult<PyErr> {
    let message = format!("expected {expected}, got {}", v.repr()?);
    Ok(
        if v.is_instance_of::<PyInt>() || v.extract::<f64>().is_ok() {
            pyo3::exceptions::PyValueError::new_err(message)
        } else {
            pyo3::exceptions::PyTypeError::new_err(message)
        },
    )
}

/// An error raised while reading the field `key` of a record, as the
/// `ValueError` with the field's name in front of its message.
///
/// A record is data wherever it comes from: a dict, the JSON file it was
/// saved to, and an array row must fail alike for the same value, and a JSON
/// file cannot tell a wrong type from a bad value. So inside a record a
/// `TypeError` or `OverflowError` is a `ValueError` too. Any other exception,
/// a subclass of `ValueError` such as `UnicodeEncodeError` included, passes
/// through unchanged.
pub(crate) fn field_error(py: Python<'_>, key: impl std::fmt::Display, err: PyErr) -> PyErr {
    if is_plain_value_error(py, &err)
        || err.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
        || err.is_instance_of::<pyo3::exceptions::PyOverflowError>(py)
    {
        pyo3::exceptions::PyValueError::new_err(format!("{key}: {}", err.value(py)))
    } else {
        err
    }
}

/// Whether `err` is exactly a `ValueError`, which can be raised again with
/// more context; a subclass keeps its own type and arguments.
fn is_plain_value_error(py: Python<'_>, err: &PyErr) -> bool {
    err.get_type(py)
        .is(py.get_type::<pyo3::exceptions::PyValueError>())
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
    if !is_compressed(counts)? {
        return Ok(None);
    }
    if let Ok(b) = counts.cast::<PyBytes>() {
        let s = std::str::from_utf8(b.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid UTF-8 in RLE counts: {e}"))
        })?;
        return Ok(Some(s));
    }
    // A `str` with no UTF-8 form (a lone surrogate) is left to the list
    // branch, which rejects it.
    Ok(counts.cast::<PyString>()?.to_str().ok())
}

/// Whether `counts` is a compressed string, `bytes` or `str`, judged by type
/// alone, so a reader can classify it without scanning it. The one list of
/// the types a `counts` may be: a byte buffer (`bytearray`, `memoryview`) is
/// the `TypeError`, and anything else is left to the run-list branch.
fn is_compressed(counts: &Bound<'_, PyAny>) -> PyResult<bool> {
    if counts.is_instance_of::<PyBytes>() || counts.is_instance_of::<PyString>() {
        return Ok(true);
    }
    if counts.is_instance_of::<PyByteArray>() || counts.is_instance_of::<PyMemoryView>() {
        return Err(counts_type_error(counts));
    }
    Ok(false)
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
        if known(name, &v).map_err(|e| field_error(v.py(), format_args!("field '{name}'"), e))? {
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
    from_py_json(extras, "custom keys")
}

/// Deserialize a Python object through `json.dumps`: the way in for a value
/// whose Rust type already knows its JSON shape. `what` names it in the
/// error a non-JSON value raises.
pub(crate) fn from_py_json<T: serde::de::DeserializeOwned>(
    obj: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<T> {
    let json_str = obj
        .py()
        .import("json")?
        .call_method1("dumps", (obj,))?
        .cast_into::<PyString>()?;
    serde_json::from_str(json_str.to_str()?).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{what} did not round-trip through JSON: {e}"
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
        "bbox" => ann.bbox = optional(value, float_array::<4>)?,
        "area" => ann.area = optional(value, extract_float)?,
        "segmentation" => ann.segmentation = optional(value, py_to_segmentation)?,
        "iscrowd" => ann.iscrowd = extract_flag(value)?,
        "keypoints" => ann.keypoints = optional(value, py_to_keypoints)?,
        "num_keypoints" => ann.num_keypoints = optional(value, extract_int)?,
        "obb" => ann.obb = optional(value, float_array::<5>)?.map(Box::new),
        "score" => ann.score = optional(value, extract_float)?,
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
    float_vec(value).or_else(|flat_err| {
        let seq = number_sequence(value)?;
        let mut out = Vec::with_capacity(3 * seq.len()?);
        for row in seq.try_iter()? {
            out.extend(float_array::<3>(&row?).map_err(|_| flat_err.clone_ref(value.py()))?);
        }
        Ok(out)
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

/// [`py_to_annotation`] for a record of a dataset or a results list, which
/// needs a `category_id` too, as the JSON file it could be saved to does. A
/// lone annotation handed to `ann_to_rle` needs only what pycocotools reads.
pub fn py_to_record(dict: &Bound<'_, PyDict>) -> PyResult<Annotation> {
    req!(dict, "image_id");
    req!(dict, "category_id");
    let mut ann = Annotation::default();
    merge_ann_dict(&mut ann, dict)?;
    Ok(ann)
}

pub fn py_to_segmentation(obj: &Bound<'_, PyAny>) -> PyResult<Segmentation> {
    // Try as dict (CompressedRle or UncompressedRle)
    if let Ok(dict) = obj.cast::<PyDict>() {
        return RleDict::read(dict)?.into_segmentation();
    }
    // Otherwise it is a polygon: a list of rings, each a flat list of
    // numbers read by the record rule, so a bool is refused here too.
    let polys = number_sequence(obj)?
        .try_iter()?
        .map(|ring| float_vec(&ring?))
        .collect::<PyResult<Vec<_>>>()?;
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
    if let Some(isthing) = cat.isthing {
        dict.set_item(pyo3::intern!(py, "isthing"), isthing)?;
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
            Counts::Compressed(obj) => {
                hotcoco_core::mask::rle_from_string(compressed_counts(&obj)?, h, w)
                    .map_err(to_pyerr)
            }
            Counts::Runs(counts) => {
                hotcoco_core::mask::check_counts(&counts, h, w).map_err(to_pyerr)?;
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
        "neg_category_ids" => img.neg_category_ids = extract_id_list(value)?,
        "not_exhaustive_category_ids" => img.not_exhaustive_category_ids = extract_id_list(value)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// A list of category ids, each read by the one id rule ([`extract_int`]),
/// as the JSON loader reads these fields.
fn extract_id_list(value: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let items: Vec<Bound<'_, PyAny>> = value.extract()?;
    items.iter().map(extract_int).collect()
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
        // `1`/`0` in COCO panoptic files, a bool from a Python caller; `None`
        // is no `isthing`, as `null` is in a file.
        "isthing" => cat.isthing = optional(value, extract_flag)?,
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

/// Set the segment field a `segments_info` key names. `true` for every key:
/// a segment keeps no extra, so any other key is dropped, as the JSON loader
/// drops it. See [`set_image_field`].
fn set_segment_field(seg: &mut SegmentInfo, key: &str, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    match key {
        "id" => seg.id = extract_int(value)?,
        "category_id" => seg.category_id = extract_int(value)?,
        "area" => seg.area = optional(value, extract_int)?,
        "bbox" => seg.bbox = optional(value, float_array::<4>)?,
        "iscrowd" => seg.iscrowd = extract_flag(value)?,
        "segmentation" => seg.segmentation = optional(value, py_to_segmentation)?,
        _ => {}
    }
    Ok(true)
}

/// One `segments_info` entry: `id` and `category_id` required, the rest
/// optional, every field read by the record rules the dataset readers use.
/// The one reader of a segment from Python, for a panoptic dict and for
/// `primitives.match_segments` alike.
pub(crate) fn py_to_segment(dict: &Bound<'_, PyDict>) -> PyResult<SegmentInfo> {
    req!(dict, "id");
    req!(dict, "category_id");
    let mut seg = SegmentInfo {
        id: 0,
        category_id: 0,
        area: None,
        bbox: None,
        iscrowd: false,
        segmentation: None,
    };
    // Every key is consumed, so no extra is built.
    extract_extra(dict, |key, value| set_segment_field(&mut seg, key, value))?;
    Ok(seg)
}

/// A COCO panoptic dataset from a dict, read field by field as a detection
/// dataset is ([`py_to_dataset`]): a record fails here as it fails in the
/// JSON file it could be saved to.
pub(crate) fn py_to_panoptic_dataset(dict: &Bound<'_, PyDict>) -> PyResult<PanopticDataset> {
    let annotation = |d: &Bound<'_, PyDict>| -> PyResult<PanopticAnnotation> {
        req!(d, "image_id");
        req!(d, "segments_info");
        let mut ann = PanopticAnnotation {
            image_id: 0,
            file_name: None,
            segments_info: extract_dict_list(d, "segments_info", py_to_segment)?,
        };
        // The scalar fields of one panoptic annotation — one image;
        // `segments_info` is read beside them, and any other key is dropped,
        // as the JSON loader drops it.
        extract_extra(d, |key, value| {
            match key {
                "image_id" => ann.image_id = extract_int(value)?,
                "file_name" => ann.file_name = optional(value, |v| v.extract())?,
                _ => {}
            }
            Ok(true)
        })?;
        Ok(ann)
    };
    Ok(PanopticDataset {
        images: extract_dict_list(dict, "images", py_to_image)?,
        annotations: extract_dict_list(dict, "annotations", annotation)?,
        categories: extract_dict_list(dict, "categories", py_to_category)?,
        ..PanopticDataset::default()
    })
}

/// Convert a list of dicts element by element. `what` names the list, and a
/// `ValueError` from an element (a non-dict, or a bad field) says which:
/// `annotations[3]: field 'bbox': ...`.
pub fn dict_list<'py, T>(
    list: &Bound<'py, PyList>,
    what: &str,
    convert_fn: impl Fn(&Bound<'py, PyDict>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    let py = list.py();
    list.iter()
        .enumerate()
        .map(|(i, item)| {
            let d = item.cast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{what}[{i}] must be a dict, got {}",
                    type_name(&item)
                ))
            })?;
            convert_fn(d).map_err(|e| with_context(py, format_args!("{what}[{i}]"), e))
        })
        .collect()
}

/// A `ValueError` or `TypeError` with `context` in front of its message,
/// such as the item of a list it came from: `annotations[3]: field 'bbox':
/// ...`. Any other exception, a subclass of either included, passes through
/// unchanged.
pub(crate) fn with_context(py: Python<'_>, context: impl std::fmt::Display, err: PyErr) -> PyErr {
    let message = || format!("{context}: {}", err.value(py));
    if is_plain_value_error(py, &err) {
        pyo3::exceptions::PyValueError::new_err(message())
    } else if err
        .get_type(py)
        .is(py.get_type::<pyo3::exceptions::PyTypeError>())
    {
        pyo3::exceptions::PyTypeError::new_err(message())
    } else {
        err
    }
}

/// Entry `i` of a `segmentation=` list given beside an array or columns, a
/// bad entry named by its position.
pub(crate) fn segmentation_at(seg: &Bound<'_, PyAny>, i: usize) -> PyResult<Segmentation> {
    py_to_segmentation(seg).map_err(|e| field_error(seg.py(), format_args!("segmentation[{i}]"), e))
}

/// A list of records as a `list`: the list itself, or a tuple's items — the
/// array `json.dump` writes for either, so a dict loads as its file does.
/// `None` for anything else.
pub(crate) fn record_list<'py>(obj: &Bound<'py, PyAny>) -> Option<Bound<'py, PyList>> {
    if let Ok(list) = obj.cast::<PyList>() {
        return Some(list.clone());
    }
    obj.cast::<PyTuple>().ok().map(PyTupleMethods::to_list)
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
            let list = record_list(&v).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("'{key}' must be a list of dicts"))
            })?;
            dict_list(&list, key, convert_fn)
        }
    }
}

pub fn py_to_dataset(dict: &Bound<'_, PyDict>) -> PyResult<Dataset> {
    let images = extract_dict_list(dict, "images", py_to_image)?;
    let annotations = extract_dict_list(dict, "annotations", py_to_record)?;
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

/// One image's segment matches into `dict`: the four keys
/// `primitives.match_segments` returns and every `PanopticEval.eval_imgs`
/// record carries after its `image_id`. One writer, so the two cannot drift.
pub(crate) fn set_segment_matches(dict: &Bound<'_, PyDict>, m: &MatchIds) -> PyResult<()> {
    dict.set_item("matched", &m.matched)?;
    dict.set_item("missed", &m.missed)?;
    dict.set_item("false_positives", &m.false_positives)?;
    dict.set_item("ignored", &m.ignored)?;
    Ok(())
}

/// A label map — a numpy integer or bool array of any shape — held as a
/// read-only borrow of its buffer, so the counting reads the pixels where
/// they are rather than from a copy; at memory bandwidth a copy of an
/// `int64` map costs about what counting it does.
///
/// `uint8` (a PNG-backed ground truth), `uint16` (a wide label set),
/// `uint64`, `int32` (panopticapi's `rgb2id`), and `int64` (`argmax`) are
/// read in place; any other unsigned dtype is a `uint64` copy and any other
/// dtype — `int16`, bool — an `int64` one ([`contiguous_array`]), so a
/// negative label survives to be reported rather than wrapping, and an
/// unsigned one of 2^63 or more is never reported as negative.
pub(crate) enum LabelArray<'py> {
    U8(Contiguous<'py, u8, numpy::IxDyn>),
    U16(Contiguous<'py, u16, numpy::IxDyn>),
    U64(Contiguous<'py, u64, numpy::IxDyn>),
    I32(Contiguous<'py, i32, numpy::IxDyn>),
    I64(Contiguous<'py, i64, numpy::IxDyn>),
}

impl LabelArray<'_> {
    pub(crate) fn shape(&self) -> &[usize] {
        match self {
            LabelArray::U8(a) => a.shape(),
            LabelArray::U16(a) => a.shape(),
            LabelArray::U64(a) => a.shape(),
            LabelArray::I32(a) => a.shape(),
            LabelArray::I64(a) => a.shape(),
        }
    }

    /// The pixels as core's map: the buffer itself, row-major.
    pub(crate) fn map(&self) -> LabelMap<'_> {
        match self {
            LabelArray::U8(a) => LabelMap::U8(Cow::Borrowed(a.as_slice())),
            LabelArray::U16(a) => LabelMap::U16(Cow::Borrowed(a.as_slice())),
            LabelArray::U64(a) => LabelMap::U64(Cow::Borrowed(a.as_slice())),
            LabelArray::I32(a) => LabelMap::I32(Cow::Borrowed(a.as_slice())),
            LabelArray::I64(a) => LabelMap::I64(Cow::Borrowed(a.as_slice())),
        }
    }
}

/// `obj` as a [`LabelArray`]. Shape and pixel order are the caller's to
/// compare: a pair of maps need only agree with each other.
fn label_array<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<LabelArray<'py>> {
    let Ok(arr) = obj.cast::<numpy::PyUntypedArray>() else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a numpy integer array (a label map), got {}",
            type_name(obj)
        )));
    };
    // The narrow widths are read at their own width only; any other unsigned
    // dtype (`uint32`, a swapped byte order) takes the lossless `uint64`
    // copy, and everything else the `int64` one.
    if let Some(a) = contiguous_array::<u8, numpy::IxDyn>(obj, name, b"")? {
        return Ok(LabelArray::U8(a));
    }
    if let Some(a) = contiguous_array::<u16, numpy::IxDyn>(obj, name, b"")? {
        return Ok(LabelArray::U16(a));
    }
    if let Some(a) = contiguous_array::<u64, numpy::IxDyn>(obj, name, b"u")? {
        return Ok(LabelArray::U64(a));
    }
    if let Some(a) = contiguous_array::<i32, numpy::IxDyn>(obj, name, b"")? {
        return Ok(LabelArray::I32(a));
    }
    match contiguous_array::<i64, numpy::IxDyn>(obj, name, b"iub")? {
        Some(a) => Ok(LabelArray::I64(a)),
        None => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} has dtype {}; a label map holds integer class ids",
            arr.dtype()
        ))),
    }
}

/// A 2-D count matrix — `metrics.mean_iou`'s argument — as a row-major
/// `(flat, nr, nc)` triple: a numpy integer array of any dtype in one pass,
/// a nested sequence of ints otherwise. The integer twin of
/// [`f64_matrix_arg`]; a negative count is a `ValueError`, counts are
/// unsigned. A `uint64` array — what `pixel_confusion` and
/// `SemanticEval.confusion()` hand back — needs no sign check, but a cell of
/// 2⁶³ or more is refused like a negative one: no image has that many pixels,
/// and it is what an underflowed `uint64` subtraction of two matrices yields.
pub fn u64_matrix_arg(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<(Vec<u64>, usize, usize)> {
    fn unsigned(name: &str, values: impl Iterator<Item = i64>) -> PyResult<Vec<u64>> {
        values
            .map(|v| {
                u64::try_from(v).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{name} must not contain negative values, got {v}"
                    ))
                })
            })
            .collect()
    }
    if let Some(arr) = numpy_array::<u64, numpy::Ix2>(obj, name, b"u", Layout::Any)? {
        let view = arr.as_array();
        let (nr, nc) = view.dim();
        if let Some(&v) = view.iter().find(|&&v| v > i64::MAX as u64) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} holds {v}, which is not a pixel count (an underflowed uint64 difference?)"
            )));
        }
        return Ok((view.iter().copied().collect(), nr, nc));
    }
    if let Some(arr) = numpy_array::<i64, numpy::Ix2>(obj, name, b"i", Layout::Any)? {
        let view = arr.as_array();
        let (nr, nc) = view.dim();
        return Ok((unsigned(name, view.iter().copied())?, nr, nc));
    }
    let (signed, nr, nc) = nested_matrix::<i64>(obj, name, "ints or a 2-D numpy integer array")?;
    Ok((unsigned(name, signed.into_iter())?, nr, nc))
}

/// Both sides of one label-map pair, checked to be the same shape.
pub(crate) fn label_pair<'py>(
    gt: &Bound<'py, PyAny>,
    pred: &Bound<'py, PyAny>,
) -> PyResult<(LabelArray<'py>, LabelArray<'py>)> {
    let (g, p) = (label_array(gt, "gt")?, label_array(pred, "pred")?);
    if g.shape() != p.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gt and pred must have the same shape, got {:?} and {:?}",
            g.shape(),
            p.shape()
        )));
    }
    Ok((g, p))
}

/// [`label_pair`] for the free functions, where a nested list is welcome as
/// it is everywhere in `metrics` and `primitives`: `numpy.asarray` first,
/// which hands an array back unchanged. `SemanticEval.update` takes arrays
/// only, and calls [`label_pair`] itself.
pub(crate) fn label_pair_or_lists<'py>(
    gt: &Bound<'py, PyAny>,
    pred: &Bound<'py, PyAny>,
) -> PyResult<(LabelArray<'py>, LabelArray<'py>)> {
    let asarray = gt.py().import("numpy")?.getattr("asarray")?;
    label_pair(&asarray.call1((gt,))?, &asarray.call1((pred,))?)
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

/// A flat `Vec<T>` as a numpy array of `T`'s dtype in the given shape —
/// `float64` for a float buffer, `int64` or `bool` for match records.
///
/// The general counterpart to [`confusion_counts_to_py`] for the sites that
/// reshape a flat buffer. Generic over the shape rather than the rank, so
/// `[k, k]`, `[t, k, a, m]` and `[t, r, k, a, m]` all use the same helper.
pub fn shaped_array<T: numpy::Element, D: numpy::ndarray::IntoDimension>(
    py: Python<'_>,
    values: Vec<T>,
    dims: D,
) -> PyResult<Py<PyAny>> {
    let arr = PyArray1::from_vec(py, values);
    Ok(arr.reshape(dims)?.into_any().unbind())
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

/// The dtype kinds a `float64` reader casts from: any integer or float. The
/// cast is the conversion a Python `float()` of each element makes, so it is
/// exact for every `float32` and `float16`, and for every integer up to
/// 2^53. A bool or object array is left to the caller's element-wise path.
pub(crate) const NUMERIC_KINDS: &[u8] = b"iuf";

/// A numpy array of any integer or float dtype ([`NUMERIC_KINDS`]) as a
/// `float64` array of `D` dimensions, read through an ndarray view
/// ([`numpy_array`]). `None` for anything else — a list, a bool or object
/// array, an array of other dimensions — which the caller reads element by
/// element.
pub(crate) fn numeric_array<'py, D: numpy::ndarray::Dimension>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<numpy::PyReadonlyArray<'py, f64, D>>> {
    numpy_array::<f64, D>(obj, name, NUMERIC_KINDS, Layout::Any)
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
    if let Some(arr) = numeric_array::<numpy::Ix1>(obj, name)? {
        return Ok(arr.as_array().to_vec());
    }
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a sequence of floats or a 1-D numpy array"
        ))
    })
}

/// What a numpy buffer must satisfy to be read where it is.
#[derive(Clone, Copy)]
enum Layout {
    /// Aligned, any strides: read through an ndarray view, which walks them.
    Any,
    /// Aligned and C-contiguous: one row-major slice, as [`Contiguous`]
    /// hands it out.
    Contiguous,
}

impl Layout {
    fn holds(self, arr: &Bound<'_, numpy::PyUntypedArray>) -> bool {
        // numpy flags every empty array aligned, whatever its data pointer;
        // an empty view of a `frombuffer` at an odd offset is not one to
        // ndarray, and its copy is free.
        !arr.is_empty()
            && arr.is_aligned()
            && (matches!(self, Layout::Any) || arr.is_c_contiguous())
    }
}

/// A numpy array read with [`Layout::Contiguous`]: its buffer is one
/// row-major slice. ndarray's layout test ignores axes of length one, as
/// numpy's `C_CONTIGUOUS` flag does, so the two agree on every shape.
pub(crate) struct Contiguous<'py, T: numpy::Element, D: numpy::ndarray::Dimension>(
    numpy::PyReadonlyArray<'py, T, D>,
);

impl<T: numpy::Element, D: numpy::ndarray::Dimension> Contiguous<'_, T, D> {
    pub(crate) fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    /// The buffer, row-major. ndarray's `to_slice`, not numpy's `as_slice`,
    /// which also hands back a Fortran-order buffer in its column-major
    /// order.
    pub(crate) fn as_slice(&self) -> &[T] {
        self.0
            .as_array()
            .to_slice()
            .expect("read with Layout::Contiguous, so one row-major slice")
    }
}

/// [`numpy_array`] with [`Layout::Contiguous`]: `obj` as a `T` array whose
/// buffer is one row-major slice, for a reader that borrows it as one.
pub(crate) fn contiguous_array<'py, T: numpy::Element, D: numpy::ndarray::Dimension>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    kinds: &[u8],
) -> PyResult<Option<Contiguous<'py, T, D>>> {
    Ok(numpy_array::<T, D>(obj, name, kinds, Layout::Contiguous)?.map(Contiguous))
}

/// The one reader of a numpy argument: `obj` as a `T` array of `D`
/// dimensions in `layout`. The array itself when it already is one; one
/// numpy copy of it (`astype` in C order, which casts and lays out in the
/// same pass, in C and without the GIL) when its dtype is `T` or one of the
/// `kinds` the caller casts from; `None` for anything else — a list, an array
/// of another rank or dtype — which the caller reads some other way or
/// refuses by name.
///
/// Every check a fast path needs is made here, so no reader can borrow a
/// buffer it must not. Each of these reached review in the 1.3 cycle from a
/// reader that decided for itself: a Fortran-order map paired with a C-order
/// one, a `frombuffer` view at an odd offset that is no `&[u16]`, a `uint64`
/// wrapped negative by an `int64` cast. `tests/test_array_spellings.py` holds
/// every dtype in every layout to the answer a fresh copy gives.
fn numpy_array<'py, T: numpy::Element, D: numpy::ndarray::Dimension>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    kinds: &[u8],
    layout: Layout,
) -> PyResult<Option<numpy::PyReadonlyArray<'py, T, D>>> {
    let Ok(arr) = obj.cast::<numpy::PyUntypedArray>() else {
        return Ok(None);
    };
    if D::NDIM.is_some_and(|ndim| ndim != arr.ndim()) {
        return Ok(None);
    }
    let py = obj.py();
    let want = numpy::dtype::<T>(py);
    let same = arr.dtype().is_equiv_to(&want);
    if !same && !kinds.contains(&arr.dtype().kind()) {
        return Ok(None);
    }
    // rust-numpy views at most 32 dimensions and panics past that; numpy 2
    // allows 64.
    if arr.ndim() > 32 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} has {} dimensions; at most 32 are supported",
            arr.ndim()
        )));
    }
    if same && layout.holds(arr) {
        return Ok(Some(obj.extract()?));
    }
    let made = obj.call_method1(pyo3::intern!(py, "astype"), (want, pyo3::intern!(py, "C")))?;
    Ok(Some(made.extract()?))
}

/// A 1-D non-negative integer argument — see [`f64_vec`]. A numpy array of
/// any unsigned dtype is read as `uint64`, of any signed dtype as `int64`, and
/// of any float dtype as `float64`, each in one copy; a list or tuple value by
/// value ([`extract_int`]). Ids follow the rule a record's do: a whole float
/// such as `1.0` is the id 1, and a negative or fractional number, or a bool,
/// is a `ValueError` naming its position. A string, anything that is not a
/// sequence (a set has no order to pair with the other columns), or an item
/// that is not a number is a `TypeError`.
pub fn u64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u64>> {
    let bad = |i: usize, v: &dyn std::fmt::Display| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{name}[{i}]: expected a non-negative integer, got {v}"
        ))
    };
    if let Some(arr) = numpy_array::<u64, numpy::Ix1>(obj, name, b"u", Layout::Any)? {
        return Ok(arr.as_array().to_vec());
    }
    if let Some(arr) = numpy_array::<i64, numpy::Ix1>(obj, name, b"i", Layout::Any)? {
        return (arr.as_array().iter().enumerate())
            .map(|(i, &v)| u64::try_from(v).map_err(|_| bad(i, &v)))
            .collect();
    }
    if let Some(arr) = numpy_array::<f64, numpy::Ix1>(obj, name, b"f", Layout::Any)? {
        return (arr.as_array().iter().enumerate())
            .map(|(i, &v)| hotcoco_core::types::id_from_f64(v).ok_or_else(|| bad(i, &v)))
            .collect();
    }
    let not_ids = || {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a sequence of ints or a 1-D numpy int array"
        ))
    };
    if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyBytes>() {
        return Err(not_ids());
    }
    // Any other array, a bool array included, is read value by value, so a
    // bool is refused the way a record's is: by position, as a `ValueError`.
    let items: Vec<Bound<'_, PyAny>> = obj.extract().map_err(|_| not_ids())?;
    (items.iter().enumerate())
        .map(|(i, item)| {
            extract_int(item).map_err(|e| with_context(obj.py(), format_args!("{name}[{i}]"), e))
        })
        .collect()
}

/// A 1-D bool numpy array, cast from `kinds` by numpy's rule (non-zero is
/// true), as a `Vec`; `None` for anything else — see [`numpy_array`].
fn bool_array(obj: &Bound<'_, PyAny>, name: &str, kinds: &[u8]) -> PyResult<Option<Vec<bool>>> {
    Ok(
        numpy_array::<bool, numpy::Ix1>(obj, name, kinds, Layout::Any)?
            .map(|arr| arr.as_array().to_vec()),
    )
}

/// A 1-D flag argument — `iscrowd` for `from_arrays` and the IoU functions.
///
/// A numpy bool array is read as is and an array of any integer dtype as
/// non-zero means true; anything else (a list of `0`/`1` or bools, a float
/// array read back from pandas) goes through [`extract_flag`] item by item,
/// so every flag spelling the API accepts elsewhere works here too. COCO
/// JSON stores `iscrowd` as `0`/`1`, and pycocotools takes
/// `maskUtils.iou(dt, gt, [a["iscrowd"] for a in anns])` straight through.
pub fn flag_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<bool>> {
    if let Some(flags) = bool_array(obj, name, b"iu")? {
        return Ok(flags);
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
    if let Some(bools) = bool_array(obj, name, b"")? {
        return Ok(bools);
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
    if let Some(arr) = numeric_array::<numpy::Ix2>(obj, name)? {
        let view = arr.as_array();
        let (nr, nc) = view.dim();
        return Ok((view.iter().copied().collect(), nr, nc));
    }
    nested_matrix(obj, name, "floats or a 2-D numpy array")
}

/// A nested sequence (list of lists, tuple of tuples, ...) as a row-major
/// `(flat, nr, nc)` triple: the sequence path of [`f64_matrix_arg`] and
/// [`u64_matrix_arg`]. Mismatched row lengths raise `ValueError`, not
/// `TypeError` — the type is right, the shape is not.
fn nested_matrix<T: Clone + for<'a, 'py> FromPyObject<'a, 'py>>(
    obj: &Bound<'_, PyAny>,
    name: &str,
    expected: &str,
) -> PyResult<(Vec<T>, usize, usize)> {
    let rows: Vec<Vec<T>> = obj.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a 2-D sequence of {expected}"
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
