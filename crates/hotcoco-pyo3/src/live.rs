//! The `COCO` wrapper's data and the dict `dataset` hands out, kept in step.
//!
//! pycocotools code edits `coco.dataset` in place — appends an annotation,
//! changes a field, assigns a whole dict — and expects the next call to see it.
//! The dict is plain Python, so nothing observes the edit. Instead the wrapper
//! remembers whether Python has had the dict since the data was last built
//! from it, and reads the dict back before any method reads the data. The data
//! is private to this module and [`LiveDataset::get`] is its one reader, so a
//! binding method cannot skip the read-back and compile; `tests/test_live_dataset.py`
//! calls every public method after an edit to prove it.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyMapping};

use crate::convert::{annotation_to_py, category_to_py, image_to_py, py_to_dataset};
use crate::{DatasetWarning, warn};

pub(crate) struct LiveDataset {
    inner: Arc<hotcoco_core::COCO>,
    /// The dict `dataset` hands out — the same object on every access, as
    /// pycocotools' is, and the one assigned to `dataset` if one was. `None`
    /// until the first access.
    dict: Option<Py<PyDict>>,
    /// Whether `dict` has been handed to Python since `inner` was last built
    /// from it, and so may carry an edit `inner` has not seen.
    handed_out: bool,
    /// `imgToAnns` and `catToImgs`, each kept from its first read until
    /// `inner` next changes — see [`index_map`](Self::index_map).
    index_maps: [Option<Py<PyAny>>; 2],
}

/// Which of pycocotools' two index maps; the slot in `index_maps`.
#[derive(Clone, Copy)]
pub(crate) enum IndexMap {
    /// `imgToAnns`: image id to its annotation dicts.
    ImageToAnns,
    /// `catToImgs`: category id to the image id of each of its annotations.
    CategoryToImages,
}

impl LiveDataset {
    pub(crate) fn new(inner: impl Into<Arc<hotcoco_core::COCO>>) -> Self {
        Self {
            inner: inner.into(),
            dict: None,
            handed_out: false,
            index_maps: [None, None],
        }
    }

    /// The data, with an edit to the dict read back first.
    ///
    /// A bad edit raises here as a `ValueError` naming the record and field,
    /// and the dict stays pending, so the next call reads the repaired dict.
    pub(crate) fn get(&mut self, py: Python<'_>) -> PyResult<Arc<hotcoco_core::COCO>> {
        self.sync(py)?;
        Ok(Arc::clone(&self.inner))
    }

    /// One of the index maps, with an edit to the dict read back first: built
    /// by `build` on the first read and the same object on every read after,
    /// as pycocotools' attribute is, until the data changes. Framework loaders
    /// read `coco.imgToAnns[img_id]` once per image, and a map built on every
    /// read made that loop quadratic (139 ms a read on val2017).
    ///
    /// Every path that replaces or edits `inner` drops both maps: `sync`,
    /// `edit`, and `reindex`.
    pub(crate) fn index_map(
        &mut self,
        py: Python<'_>,
        which: IndexMap,
        build: impl FnOnce(&hotcoco_core::COCO) -> PyResult<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.sync(py)?;
        if let Some(map) = &self.index_maps[which as usize] {
            return Ok(map.clone_ref(py));
        }
        let map = build(&self.inner)?;
        self.index_maps[which as usize] = Some(map.clone_ref(py));
        Ok(map)
    }

    /// A Rust-side edit of the data. The dict is read back first, so an edit
    /// made to it is kept, and the annotations `edit` returns the ids of are
    /// written over their dicts afterwards, so a dict the caller holds stays
    /// the dataset. Every Rust-side mutation goes through here for that reason.
    pub(crate) fn edit(
        &mut self,
        py: Python<'_>,
        edit: impl FnOnce(&mut Arc<hotcoco_core::COCO>) -> PyResult<Vec<u64>>,
    ) -> PyResult<()> {
        self.sync(py)?;
        self.index_maps = [None, None];
        let ids = edit(&mut self.inner)?;
        self.mirror(py, &ids)
    }

    /// The dict: built once from the data, then the same object every time.
    /// Handing it out marks it as possibly edited. No read-back here: a dict
    /// already out is the truth, edited or not, so a loop that touches
    /// `coco.dataset` on every step parses nothing.
    pub(crate) fn dict(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = match &self.dict {
            Some(dict) => dict.clone_ref(py),
            None => {
                let dict = dataset_to_py(py, &self.inner.dataset)?;
                self.dict = Some(dict.clone_ref(py));
                dict
            }
        };
        self.handed_out = true;
        Ok(dict)
    }

    /// `coco.dataset = d`: `d` becomes the dict, unparsed. The caller still
    /// holds `d` and may edit it before the next call, as pycocotools'
    /// `coco.dataset = d; ...; coco.createIndex()` flow does, so the next call
    /// parses it once, assignment and edits alike.
    pub(crate) fn assign(&mut self, dict: &Bound<'_, PyDict>) {
        self.dict = Some(dict.clone().unbind());
        self.handed_out = true;
    }

    /// `createIndex()`: read the dict back now, handed out since the last
    /// read or not — the way to see an edit made through a reference kept
    /// from before the last call.
    pub(crate) fn reindex(&mut self, py: Python<'_>) -> PyResult<()> {
        self.handed_out = self.dict.is_some();
        self.index_maps = [None, None];
        self.sync(py)
    }

    /// Rebuild `inner` from the dict if Python has had it since the last
    /// build. The rest of the `COCO` — the loader's warnings, the LVIS cap
    /// marker — carries over, and only a warning the edit introduced is
    /// emitted.
    fn sync(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(dict) = self.dict.as_ref().filter(|_| self.handed_out) else {
            return Ok(());
        };
        let rebuilt = self.inner.reindexed(py_to_dataset(dict.bind(py))?);
        for w in &rebuilt.load_warnings()[self.inner.load_warnings().len()..] {
            warn::<DatasetWarning>(py, w)?;
        }
        self.inner = Arc::new(rebuilt);
        self.handed_out = false;
        self.index_maps = [None, None];
        Ok(())
    }

    /// Write the annotations with `ids` over their dicts, in place, so a
    /// reference the caller holds to one stays current.
    ///
    /// The dict becomes the record, key for key: a field the edit cleared
    /// leaves the dict too, or the next read-back would parse it back in.
    /// Each dict is found by its `id`, at the record's own position when the
    /// list is still in `inner`'s order. A list reordered or shortened through
    /// a reference kept from before the last call is not: `sync` does not see
    /// that edit, the position then names another annotation's dict, and an
    /// annotation no longer listed has no dict to write.
    fn mirror(&self, py: Python<'_>, ids: &[u64]) -> PyResult<()> {
        let Some(dict) = &self.dict else {
            return Ok(());
        };
        let Some(anns) = dict.bind(py).get_item(pyo3::intern!(py, "annotations"))? else {
            return Ok(());
        };
        let Ok(anns) = anns.cast::<PyList>() else {
            return Ok(());
        };
        fn holds<'py>(item: &Bound<'py, PyAny>, id: u64) -> Option<Bound<'py, PyDict>> {
            let item = item.cast::<PyDict>().ok()?;
            let found = item.get_item("id").ok()??.extract::<u64>().ok()?;
            (found == id).then(|| item.clone())
        }
        for &id in ids {
            let Some(i) = self.inner.ann_position(id) else {
                continue;
            };
            let in_place = anns.get_item(i).ok().and_then(|item| holds(&item, id));
            let Some(target) = in_place.or_else(|| anns.iter().find_map(|item| holds(&item, id)))
            else {
                continue;
            };
            let record = annotation_to_py(py, &self.inner.dataset.annotations[i])?;
            target.clear();
            target.update(record.bind(py).cast::<PyMapping>()?)?;
        }
        Ok(())
    }
}

/// The dataset as the dict `COCO.dataset` returns: `images`, `annotations`,
/// and `categories`, each a list of plain dicts.
pub(crate) fn dataset_to_py(py: Python<'_>, ds: &hotcoco_core::Dataset) -> PyResult<Py<PyDict>> {
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
    // As pycocotools' dict has them when the file does. Left out when the
    // dataset has none, so a dataset built from records stays three keys.
    if let Some(info) = &ds.info {
        dict.set_item("info", crate::convert::block_to_py(py, info)?)?;
    }
    if !ds.licenses.is_empty() {
        dict.set_item("licenses", crate::convert::block_to_py(py, &ds.licenses)?)?;
    }
    Ok(dict.unbind())
}
