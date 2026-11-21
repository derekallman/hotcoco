//! COCO dataset loading and querying API.
//!
//! Faithful port of `pycocotools/coco.py`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use rayon::prelude::*;

use crate::primitives::sim::ForkSafe;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::ann_index::{AnnIndex, Duplicates, IdIndex};
use crate::error::Error;
use crate::mask;
use crate::primitives::greedy::descending_score_key;
use crate::primitives::sim::worth_parallel;
use crate::types::{Annotation, Category, Dataset, Image, Rle, Segmentation, bbox_within_image};

/// The COCO dataset API for loading, querying, and indexing annotations.
#[derive(Clone)]
pub struct COCO {
    /// The raw dataset. Public and mutable for pycocotools-style direct
    /// manipulation — but the query indices do **not** track it: after
    /// mutating `dataset` in place, call [`create_index`](Self::create_index)
    /// or every `get_*`/`load_*` method answers from the stale index.
    pub dataset: Dataset,
    /// Non-fatal problems noticed while loading/indexing; see
    /// [`load_warnings`](Self::load_warnings).
    warnings: Vec<String>,
    /// ann_id -> index into dataset.annotations
    anns: IdIndex,
    /// img_id -> index into dataset.images
    imgs: FxHashMap<u64, usize>,
    /// cat_id -> index into dataset.categories
    cats: FxHashMap<u64, usize>,
    /// cat_id -> [img_id, ...] (unique)
    /// `pub(crate)` so `quality::stats` can read it — `COCO::stats` lives there,
    /// since dataset statistics are introspection output rather than schema.
    pub(crate) cat_to_imgs: FxHashMap<u64, Vec<u64>>,
    /// Annotation ids by image and by (image, category), in JSON array
    /// order — see [`AnnIndex`] for why that order is contract.
    index: AnnIndex,
    /// Set by [`cap_detections_per_image`](Self::cap_detections_per_image):
    /// this results set is an lvis-api `LVISResults`, already capped (or
    /// explicitly left uncapped), so LVIS evaluation takes it as is instead of
    /// applying its default cap again.
    per_image_capped: bool,
}

/// Detections LVIS keeps per image, across categories: lvis-api's
/// `LVISResults` default, which `LVISEval` applies to every raw results list
/// whatever its own `params.max_dets`. The only owner of the number — both the
/// LVIS default `max_dets` and the per-image cap read it.
pub(crate) const LVIS_MAX_DETS_PER_IMAGE: usize = 300;

/// What kind of results a detection file holds, decided from its first
/// annotation.
///
/// pycocotools' `loadRes` infers this the same way and in the same order — a
/// bbox wins outright, then a segmentation, then keypoints — and derives the
/// missing geometry to match. The kind is a property of the *file*, not of each
/// annotation: one detection deciding for all of them is the pycocotools
/// behavior, faithfully kept.
///
/// `Obb` is a hotcoco extension with no pycocotools counterpart, so it sits last
/// and cannot displace any of the three.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ResultKind {
    Bbox,
    Segm,
    Keypoints,
    Obb,
}

impl ResultKind {
    /// `None` when the annotation carries no geometry at all, in which case
    /// nothing is derived and the results pass through untouched.
    fn of(first: &Annotation) -> Option<Self> {
        if first.bbox.is_some() {
            Some(ResultKind::Bbox)
        } else if first.segmentation.is_some() {
            // Segmentation outranks keypoints even when both are present —
            // pycocotools' `elif 'segmentation' in anns[0]` order.
            Some(ResultKind::Segm)
        } else if first.keypoints.is_some() {
            Some(ResultKind::Keypoints)
        } else if first.obb.is_some() {
            Some(ResultKind::Obb)
        } else {
            None
        }
    }
}

/// The fields of one result `load_res` checks before keeping it, read from a
/// record or from a row of a detection array, so the two are checked by one
/// function: [`COCO::check_results`].
struct ResultFields<'a> {
    id: u64,
    image_id: u64,
    category_id: u64,
    score: Option<f64>,
    bbox: Option<&'a [f64; 4]>,
}

/// A detection row's category: its seventh column, or 1 for a six-column
/// row, as pycocotools' `loadNumpyAnnotations` assigns.
fn detection_category(row: &[f64], ncols: usize) -> u64 {
    if ncols == 7 { row[6] as u64 } else { 1 }
}

/// A fraction-of-the-images argument of [`COCO::split`] or [`COCO::sample`]:
/// in `[0, 1]`, which NaN is not.
fn fraction(name: &str, value: f64) -> crate::error::Result<()> {
    if (0.0..=1.0).contains(&value) {
        return Ok(());
    }
    Err(Error::InvalidInput(format!(
        "{name} must be between 0 and 1, got {value}"
    )))
}

/// Ground-truth annotations as columns, for [`COCO::from_arrays`]: parallel,
/// one entry per annotation, each optional column defaulted when `None`.
#[derive(Debug, Clone, Default)]
pub struct AnnotationColumns {
    /// The image each annotation is on.
    pub image_ids: Vec<u64>,
    /// The category of each annotation.
    pub category_ids: Vec<u64>,
    /// COCO `[x, y, w, h]` boxes.
    pub boxes: Vec<[f64; 4]>,
    /// Annotation ids. Default: `1` to `n`.
    pub ids: Option<Vec<u64>>,
    /// Areas. Default: each box's `w * h`. Required with `segmentation`.
    pub area: Option<Vec<f64>>,
    /// Crowd flags. Default: all false.
    pub iscrowd: Option<Vec<bool>>,
    /// An RLE or polygon segmentation per annotation.
    pub segmentation: Option<Vec<Segmentation>>,
}

/// How many rows of `ncols` columns `rows` holds.
///
/// # Panics
///
/// If `ncols` is not 6 or 7, or `rows` is not a whole number of rows.
fn detection_row_count(rows: &[f64], ncols: usize) -> usize {
    assert!(
        ncols == 6 || ncols == 7,
        "detections_from_rows: 6 or 7 columns, got {ncols}"
    );
    assert_eq!(rows.len() % ncols, 0, "detections_from_rows: a partial row");
    rows.len() / ncols
}

/// The area `load_res` gives a box result: its width times its height.
pub(crate) fn box_area(bbox: &[f64; 4]) -> f64 {
    bbox[2] * bbox[3]
}

impl<'a> ResultFields<'a> {
    fn of(ann: &'a Annotation) -> Self {
        ResultFields {
            id: ann.id,
            image_id: ann.image_id,
            category_id: ann.category_id,
            score: ann.score,
            bbox: ann.bbox.as_ref(),
        }
    }
}

/// How a missing `area` is derived: COCO's instance area for ground truth, or
/// `load_res`'s precedence for results.
#[derive(Clone, Copy, Debug)]
pub(crate) enum AreaRule {
    /// The mask's pixel count when it has any, then the rotated box's `w × h`,
    /// then the box's, then the extent of the labeled keypoints.
    Instance,
    /// The order `load_res` derives a result's area in ([`ResultKind::of`]):
    /// the box's `w × h`, then the mask's pixel count, then the extent of every
    /// keypoint, as pycocotools' `loadRes` takes it, then the rotated box's.
    Result,
}

/// The area-range predicate behind [`COCO::get_ann_ids`] and [`COCO::filter`].
/// pycocotools is strict in `getAnnIds` (`lo < area < hi`) and inclusive in
/// evaluation (`area < lo or area > hi` ignores), so `get_ann_ids` passes
/// `Bounds::Open` and `filter`, which prepares data for evaluation,
/// `Bounds::Closed`.
///
/// An annotation without an `area` value never matches an explicit range.
/// pycocotools' `getAnnIds` reads `ann['area']` unconditionally and raises
/// `KeyError` on a missing key; a filter cannot raise, so exclusion is the
/// closest faithful behavior (it never fabricates an area of 0.0, which used
/// to make area-less annotations match any range starting at 0). An evaluator
/// derives the area instead ([`COCO::fill_missing_areas`]), because it has to
/// place every annotation in an area range.
fn area_in_range(ann: &Annotation, rng: [f64; 2], bounds: Bounds) -> bool {
    ann.area.is_some_and(|a| match bounds {
        Bounds::Open => a > rng[0] && a < rng[1],
        Bounds::Closed => a >= rng[0] && a <= rng[1],
    })
}

/// Whether an area range includes its end points; see [`area_in_range`].
#[derive(Clone, Copy)]
enum Bounds {
    Open,
    Closed,
}

/// How many annotations a `COCO`, or evaluated pairs a `COCOeval`, must hold
/// before dropping it frees them on the thread pool rather than the caller's
/// thread.
pub(crate) const BACKGROUND_DROP: usize = 100_000;

/// A large annotation list is freed off the caller's thread. Freeing 1.5M
/// annotations walks every record for the parts it may own — about 25 ms on
/// an M1 — and a Python caller would hold the GIL throughout; evaluation
/// frameworks drop their result sets at the end of every `compute()`.
impl Drop for COCO {
    fn drop(&mut self) {
        if self.dataset.annotations.len() >= BACKGROUND_DROP {
            let annotations = std::mem::take(&mut self.dataset.annotations);
            crate::primitives::sim::drop_in_background(annotations);
        }
    }
}

/// How `bbox` sits in a `img_w × img_h` image under each reading of its four
/// numbers. `Some(true)`: it runs past the image as `[x, y, w, h]` and fits it
/// as `[x1, y1, x2, y2]`, the signature of a corner-pair box in a results file.
/// `Some(false)`: any other box. `None`: the image size is unknown, so there is
/// nothing to measure against.
fn reads_as_corner_pair(bbox: &[f64; 4], img_w: u32, img_h: u32) -> Option<bool> {
    if img_w == 0 || img_h == 0 {
        return None;
    }
    let [x1, y1, x2, y2] = *bbox;
    let as_xyxy = [x1, y1, x2 - x1, y2 - y1];
    Some(!bbox_within_image(bbox, img_w, img_h) && bbox_within_image(&as_xyxy, img_w, img_h))
}

/// One `load_res_anns` validation pass's result: the first NaN score, unknown
/// image, and unknown category, then the corner-pair census as
/// `(boxes measured, boxes reading as corner pairs)`.
type Checked = ([Option<usize>; 3], (usize, usize));

/// Why `load_res` rejects a NaN score, from a file and a list alike.
/// The error for annotation `i` carrying a NaN score; `context` is the caller's
/// prefix (`"load_res(): "`), empty for the evaluator.
fn nan_score_error(context: &str, i: usize, id: u64, image_id: u64) -> Error {
    Error::InvalidInput(format!(
        "{context}annotation {i} (id {id}, image_id {image_id}) has a NaN score. {NAN_SCORE_WHY}"
    ))
}

const NAN_SCORE_WHY: &str = "Scores order the detection ranking, and NaN makes that order \
     undefined — every metric downstream would be meaningless. Filter or repair these \
     detections before evaluating.";

impl COCO {
    /// Load a COCO annotation JSON file and build indices.
    ///
    /// Non-fatal problems (non-finite floats normalized to `null`, duplicate
    /// annotation ids) are retained on the returned object — see
    /// [`load_warnings`](Self::load_warnings).
    pub fn new(annotation_file: &Path) -> crate::error::Result<Self> {
        let (dataset, non_finite) = crate::json::read_dataset(annotation_file)?;
        let mut coco = Self::from_dataset(dataset);
        let n_fixed = non_finite.rewritten;
        if n_fixed > 0 {
            coco.warn(format!(
                "normalized {n_fixed} non-finite float value(s) (NaN/Infinity) to null while loading {}",
                annotation_file.display()
            ));
        }
        Ok(coco)
    }

    /// Warnings collected while loading and indexing this dataset.
    ///
    /// The library records them and prints nothing: the Python bindings emit
    /// each one as a `DatasetWarning`, and `hotcoco-cli` prints them after a
    /// load. Empty for a clean load.
    pub fn load_warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Record a non-fatal problem for [`load_warnings`](Self::load_warnings).
    fn warn(&mut self, msg: String) {
        self.warnings.push(msg);
    }

    /// Build a COCO object from an already-loaded Dataset.
    pub fn from_dataset(dataset: Dataset) -> Self {
        let mut coco = COCO {
            dataset,
            warnings: Vec::new(),
            anns: IdIndex::default(),
            imgs: FxHashMap::default(),
            cats: FxHashMap::default(),
            cat_to_imgs: FxHashMap::default(),
            index: AnnIndex::default(),
            per_image_capped: false,
        };
        coco.create_index();
        coco
    }

    /// Build a ground-truth dataset from columns, one entry per annotation,
    /// with no record built until the end: the Rust side of Python's
    /// `COCO.from_arrays`. The result equals [`from_dataset`](Self::from_dataset)
    /// over the same annotations, built one by one.
    ///
    /// Each annotation gets the id `ids[i]`, or `i + 1` without `ids`; the
    /// area `area[i]`, or its box's `w * h` without `area`; and `iscrowd[i]`,
    /// or false without `iscrowd`.
    ///
    /// # Errors
    ///
    /// `InvalidInput` for a column whose length differs from `image_ids`', and
    /// for `segmentation` without `area`, because the box area would not
    /// match the mask.
    pub fn from_arrays(
        images: Vec<Image>,
        categories: Vec<Category>,
        columns: AnnotationColumns,
    ) -> crate::error::Result<COCO> {
        let AnnotationColumns {
            image_ids,
            category_ids,
            boxes,
            ids,
            area,
            iscrowd,
            segmentation,
        } = columns;
        let n = image_ids.len();
        let same_length = |name: &str, len: usize| {
            if len == n {
                Ok(())
            } else {
                Err(Error::InvalidInput(format!(
                    "from_arrays: {name} has {len} entries but image_ids has {n}"
                )))
            }
        };
        same_length("category_ids", category_ids.len())?;
        same_length("boxes", boxes.len())?;
        for (name, len) in [
            ("ids", ids.as_ref().map(Vec::len)),
            ("area", area.as_ref().map(Vec::len)),
            ("iscrowd", iscrowd.as_ref().map(Vec::len)),
            ("segmentation", segmentation.as_ref().map(Vec::len)),
        ] {
            if let Some(len) = len {
                same_length(name, len)?;
            }
        }
        if segmentation.is_some() && area.is_none() {
            return Err(Error::InvalidInput(
                "from_arrays: area is required with segmentation, because the box \
                 area would not match the mask"
                    .into(),
            ));
        }
        let mut segmentation = segmentation.map(Vec::into_iter);
        let annotations = (0..n)
            .map(|i| Annotation {
                id: ids.as_ref().map_or(i as u64 + 1, |ids| ids[i]),
                image_id: image_ids[i],
                category_id: category_ids[i],
                bbox: Some(boxes[i]),
                area: Some(area.as_ref().map_or(boxes[i][2] * boxes[i][3], |a| a[i])),
                iscrowd: iscrowd.as_ref().is_some_and(|c| c[i]),
                segmentation: segmentation.as_mut().and_then(Iterator::next),
                ..Default::default()
            })
            .collect();
        Ok(COCO::from_dataset(Dataset {
            images,
            annotations,
            categories,
            ..Default::default()
        }))
    }

    /// This `COCO` over a replaced dataset, re-indexed, with the rest of its
    /// state kept: the per-image cap marker of
    /// [`cap_detections_per_image`](Self::cap_detections_per_image) and the
    /// warnings the loader recorded, with the warnings the new index raises
    /// that were not recorded before appended after them. The Python
    /// `dataset` setter and `createIndex()` build through this, so a results
    /// set that was capped stays capped and a warning is recorded once.
    #[must_use]
    pub fn reindexed(&self, dataset: Dataset) -> COCO {
        Self::reindexed_from(&self.warnings, self.per_image_capped, dataset)
    }

    /// [`reindexed`](Self::reindexed) from the state it keeps.
    fn reindexed_from(warnings: &[String], per_image_capped: bool, dataset: Dataset) -> COCO {
        let mut rebuilt = COCO::from_dataset(dataset);
        let fresh: Vec<String> = rebuilt
            .warnings
            .drain(..)
            .filter(|w| !warnings.contains(w))
            .collect();
        rebuilt.warnings = warnings.to_vec();
        rebuilt.warnings.extend(fresh);
        rebuilt.per_image_capped = per_image_capped;
        rebuilt
    }

    /// Where the annotation with `id` sits in `dataset.annotations`, or `None`.
    /// With duplicate ids, the last occurrence, as [`get_ann`](Self::get_ann).
    pub fn ann_position(&self, id: u64) -> Option<usize> {
        self.anns.position(id)
    }

    /// Rebuild the internal query indices from `dataset`.
    ///
    /// Call this after mutating [`dataset`](Self::dataset) directly (the
    /// pycocotools `createIndex()` idiom) — the indices are snapshots, not
    /// views, and every `get_*`/`load_*` method answers from them.
    ///
    /// Duplicate annotation ids are indexed the way pycocotools does —
    /// last-write-wins in the id lookup, while per-image lists keep every
    /// occurrence — and reported via [`load_warnings`](Self::load_warnings).
    pub fn create_index(&mut self) {
        let (anns, dups) = IdIndex::build(&self.dataset.annotations);
        self.anns = anns;
        if let Some(Duplicates { count, first }) = dups {
            // pycocotools parity: the id lookup keeps the last annotation with
            // a given id, while imgToAnns keeps every occurrence — both are
            // preserved here, and the condition is surfaced instead of silent.
            self.warn(format!(
                "{count} duplicate annotation id(s) found (first: {first}). Lookups by id \
                 see only the last occurrence; per-image annotation lists keep every \
                 occurrence, so duplicates are double-counted there (pycocotools behaves \
                 the same way). Deduplicate ids to make this dataset unambiguous."
            ));
        }
        self.index_annotations();

        self.imgs = FxHashMap::with_capacity_and_hasher(self.dataset.images.len(), FxBuildHasher);
        for (i, img) in self.dataset.images.iter().enumerate() {
            self.imgs.insert(img.id, i);
        }

        let unnamed = self.fill_placeholder_cat_names();
        if unnamed > 0 {
            self.warn(format!(
                "{unnamed} category record(s) without a name; using cat_<id> as the display name."
            ));
        }
        self.cats =
            FxHashMap::with_capacity_and_hasher(self.dataset.categories.len(), FxBuildHasher);
        for (i, cat) in self.dataset.categories.iter().enumerate() {
            self.cats.insert(cat.id, i);
        }
    }

    /// The groupings that key on an annotation's `image_id` and
    /// `category_id`: rebuilt by [`create_index`](Self::create_index) and by
    /// an [`update_anns`](Self::update_anns) that moves one.
    fn index_annotations(&mut self) {
        self.index = AnnIndex::build(&self.dataset.annotations);
        // One push per distinct (image, category) pair, so each image appears
        // once in a category's list.
        self.cat_to_imgs =
            FxHashMap::with_capacity_and_hasher(self.dataset.categories.len(), FxBuildHasher);
        for (img_id, cat_id) in self.index.pairs() {
            self.cat_to_imgs.entry(cat_id).or_default().push(img_id);
        }
    }

    /// Replace annotations by id, re-indexing only when a replacement moves
    /// one.
    ///
    /// The targeted counterpart to replacing the whole
    /// [`dataset`](Self::dataset): each annotation in `anns` overwrites the one
    /// that carries the same `id`. Ids do not move, so the id lookup survives
    /// untouched; the per-image and per-category groupings are rebuilt only if
    /// a replacement changes an `image_id` or a `category_id`, which is what
    /// they key on. Editing `area` across a whole dataset — the multi-IoU-type
    /// case — therefore costs one pass over `anns`, not one over the dataset.
    ///
    /// Every id is checked before anything is written: the ids that are not
    /// in the dataset come back as [`Error::UnknownAnnIds`] and the dataset is
    /// left untouched, so a partial update never happens.
    ///
    /// In a dataset with duplicate annotation ids, the id lookup holds the
    /// *last* occurrence (pycocotools parity — see
    /// [`create_index`](Self::create_index)), so that is the one replaced.
    pub fn update_anns(&mut self, anns: Vec<Annotation>) -> crate::error::Result<()> {
        let targets = self.ann_positions(anns.iter().map(|ann| ann.id))?;

        let mut moved = false;
        for (i, ann) in targets.into_iter().zip(anns) {
            let old = &self.dataset.annotations[i];
            moved |= old.image_id != ann.image_id || old.category_id != ann.category_id;
            self.dataset.annotations[i] = ann;
        }
        if moved {
            self.index_annotations();
        }
        Ok(())
    }

    /// Set the `area` of annotations by id, in place: `areas[i]` goes to the
    /// annotation with id `ids[i]`.
    ///
    /// [`update_anns`](Self::update_anns) for one field, without building a
    /// replacement annotation, and so without copying an annotation's
    /// segmentation to change one number. An area moves no annotation, so
    /// nothing is re-indexed. Every id is checked before anything is written,
    /// as in `update_anns`; the ids that are not in the dataset come back as
    /// [`Error::UnknownAnnIds`]. Errors when `ids` and `areas` differ in
    /// length.
    pub fn update_ann_areas(&mut self, ids: &[u64], areas: &[f64]) -> crate::error::Result<()> {
        if ids.len() != areas.len() {
            return Err(Error::InvalidInput(format!(
                "update_ann_areas: {} areas for {} ids",
                areas.len(),
                ids.len()
            )));
        }
        let targets = self.ann_positions(ids.iter().copied())?;
        for (i, &area) in targets.into_iter().zip(areas) {
            self.dataset.annotations[i].area = Some(area);
        }
        Ok(())
    }

    /// Every id in `ids` is an annotation of this dataset, or the ones that are
    /// not as [`Error::UnknownAnnIds`]: the check
    /// [`update_ann_areas`](Self::update_ann_areas) makes before it writes,
    /// for a caller that wants it before paying for a copy to write into.
    pub fn check_ann_ids(&self, ids: &[u64]) -> crate::error::Result<()> {
        self.ann_positions(ids.iter().copied()).map(drop)
    }

    /// The position of each id's annotation, or every id the dataset does not
    /// have as [`Error::UnknownAnnIds`]: the check an edit by id makes before
    /// it writes anything.
    fn ann_positions(&self, ids: impl Iterator<Item = u64>) -> crate::error::Result<Vec<usize>> {
        let mut targets = Vec::with_capacity(ids.size_hint().0);
        let mut missing = Vec::new();
        for id in ids {
            match self.anns.position(id) {
                Some(i) => targets.push(i),
                None => missing.push(id),
            }
        }
        if missing.is_empty() {
            Ok(targets)
        } else {
            Err(Error::UnknownAnnIds(missing))
        }
    }

    /// Get annotation IDs matching the given filters.
    ///
    /// All filter parameters are optional (pass empty slices / None to skip).
    /// Ids come back in pycocotools' order: dataset order, image by image in
    /// `img_ids` order when that is given. `area_rng` excludes its end points,
    /// as pycocotools' `getAnnIds` does; [`COCO::filter`] includes them, as
    /// evaluation does.
    pub fn get_ann_ids(
        &self,
        img_ids: &[u64],
        cat_ids: &[u64],
        area_rng: Option<[f64; 2]>,
        is_crowd: Option<bool>,
    ) -> Vec<u64> {
        let filter = |ann: &&Annotation| -> bool {
            if !cat_ids.is_empty() && !cat_ids.contains(&ann.category_id) {
                return false;
            }
            if let Some(rng) = area_rng {
                if !area_in_range(ann, rng, Bounds::Open) {
                    return false;
                }
            }
            if let Some(crowd) = is_crowd {
                if ann.iscrowd != crowd {
                    return false;
                }
            }
            true
        };

        if !img_ids.is_empty() {
            img_ids
                .iter()
                .flat_map(|&id| self.index.for_img(id))
                .filter_map(|&id| self.get_ann(id))
                .filter(filter)
                .map(|ann| ann.id)
                .collect()
        } else {
            self.dataset
                .annotations
                .iter()
                .filter(filter)
                .map(|ann| ann.id)
                .collect()
        }
    }

    /// Get category IDs matching the given filters.
    pub fn get_cat_ids(&self, cat_nms: &[&str], sup_nms: &[&str], cat_ids: &[u64]) -> Vec<u64> {
        let mut result: Vec<u64> = self
            .dataset
            .categories
            .iter()
            .filter(|cat| {
                if !cat_nms.is_empty() && !cat_nms.contains(&cat.name.as_str()) {
                    return false;
                }
                if !sup_nms.is_empty() {
                    match &cat.supercategory {
                        Some(sc) if sup_nms.contains(&sc.as_str()) => {}
                        _ => return false,
                    }
                }
                if !cat_ids.is_empty() && !cat_ids.contains(&cat.id) {
                    return false;
                }
                true
            })
            .map(|cat| cat.id)
            .collect();
        result.sort_unstable();
        result
    }

    /// Get image IDs matching the given filters.
    pub fn get_img_ids(&self, img_ids: &[u64], cat_ids: &[u64]) -> Vec<u64> {
        let mut ids: Vec<u64> = if !img_ids.is_empty() {
            img_ids.to_vec()
        } else {
            self.dataset.images.iter().map(|img| img.id).collect()
        };

        if !cat_ids.is_empty() {
            let mut valid: Vec<u64> = cat_ids
                .iter()
                .flat_map(|cid| self.cat_to_imgs.get(cid).map_or(&[][..], Vec::as_slice))
                .copied()
                .collect();
            valid.sort_unstable();
            valid.dedup();
            ids.retain(|id| valid.binary_search(id).is_ok());
        }

        ids.sort_unstable();
        ids
    }

    /// Load annotations by IDs.
    pub fn load_anns(&self, ids: &[u64]) -> Vec<&Annotation> {
        ids.iter().filter_map(|&id| self.get_ann(id)).collect()
    }

    /// Load categories by IDs.
    pub fn load_cats(&self, ids: &[u64]) -> Vec<&Category> {
        ids.iter()
            .filter_map(|id| self.cats.get(id).map(|&i| &self.dataset.categories[i]))
            .collect()
    }

    /// Load images by IDs.
    pub fn load_imgs(&self, ids: &[u64]) -> Vec<&Image> {
        ids.iter()
            .filter_map(|id| self.imgs.get(id).map(|&i| &self.dataset.images[i]))
            .collect()
    }

    /// Get a single annotation by ID.
    pub fn get_ann(&self, id: u64) -> Option<&Annotation> {
        self.anns.position(id).map(|i| &self.dataset.annotations[i])
    }

    /// Get a single image by ID.
    pub fn get_img(&self, id: u64) -> Option<&Image> {
        self.imgs.get(&id).map(|&i| &self.dataset.images[i])
    }

    /// Get a single category by ID.
    pub fn get_cat(&self, id: u64) -> Option<&Category> {
        self.cats.get(&id).map(|&i| &self.dataset.categories[i])
    }

    /// The display name for a category id, falling back to `cat_{id}`.
    ///
    /// The one owner of the unnamed-category fallback. Three surfaces invented
    /// their own and disagreed: the confusion matrix rendered `cat_7`, the
    /// model-comparison table rendered `7`, and the Python layer rendered its
    /// own third spelling — so the same missing category record produced three
    /// different labels depending on which report a user was reading. Anything
    /// that puts a category name in front of a user goes through here, the
    /// Python bindings included.
    ///
    /// `cat_{id}` rather than the bare id because a name column holding `7` next
    /// to `person` reads as a category *named* seven; the prefix says it is a
    /// stand-in.
    pub fn cat_name(&self, id: u64) -> String {
        self.get_cat(id)
            .map_or_else(|| Self::placeholder_cat_name(id), |c| c.name.clone())
    }

    /// Give every category loaded without a `name` its
    /// [`placeholder_cat_name`](Self::placeholder_cat_name); returns how many.
    ///
    /// Part of every index rebuild, so an indexed dataset never carries an
    /// empty name. Idempotent: zero on any rebuild after the first.
    fn fill_placeholder_cat_names(&mut self) -> usize {
        let mut unnamed = 0;
        for cat in &mut self.dataset.categories {
            if cat.name.is_empty() {
                cat.name = Self::placeholder_cat_name(cat.id);
                unnamed += 1;
            }
        }
        unnamed
    }

    /// The stand-in name for a category that has none: `cat_{id}`.
    ///
    /// Used both for an id the dataset does not know and for a category
    /// record loaded without a `name` (pycocotools tolerates the omission, and
    /// TorchMetrics emits bare `{"id": i}` records). One owner, so the two
    /// cases read the same in a report.
    pub fn placeholder_cat_name(id: u64) -> String {
        crate::types::placeholder_cat_name(id)
    }

    /// Get annotation IDs for a specific (image, category) pair.
    ///
    /// One hash probe and a binary search — much faster than `get_ann_ids` with filtering.
    pub fn get_ann_ids_for_img_cat(&self, img_id: u64, cat_id: u64) -> &[u64] {
        self.index.for_img_cat(img_id, cat_id)
    }

    /// Get annotation IDs for a specific image.
    pub fn get_ann_ids_for_img(&self, img_id: u64) -> &[u64] {
        self.index.for_img(img_id)
    }

    /// Returns (img_id, cat_id) pairs that have at least one annotation:
    /// images in first-seen order, categories ascending within an image.
    pub fn nonempty_img_cat_pairs(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.index.pairs()
    }

    /// The ids by image and by (image, category).
    pub(crate) fn ann_index(&self) -> &AnnIndex {
        &self.index
    }

    /// Annotation id to position in `dataset.annotations`.
    pub(crate) fn id_index(&self) -> &IdIndex {
        &self.anns
    }

    /// How many (image, category) pairs hold an annotation.
    pub(crate) fn nonempty_pair_count(&self) -> usize {
        self.index.pair_count()
    }

    /// The categories one image has annotations in, ascending.
    pub(crate) fn cat_ids_of_img(&self, img_id: u64) -> impl Iterator<Item = u64> + '_ {
        self.index.cats_of(img_id)
    }

    /// Returns image IDs that have at least one annotation (any category), in
    /// first-seen order.
    pub fn nonempty_img_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.index.img_ids()
    }

    /// Load detection/result annotations into a new COCO object.
    ///
    /// The result file can be a JSON array of annotation dicts, or a JSON object
    /// with an `annotations` field. The result COCO object shares the images
    /// and categories from self.
    pub fn load_res(&self, res_file: &Path) -> crate::error::Result<COCO> {
        let (anns, non_finite) = crate::json::read_results(res_file)?;
        if let Some(line) = non_finite.nan_score_line {
            return Err(Error::InvalidInput(format!(
                "load_res(): {} has a NaN score at line {line}. {NAN_SCORE_WHY}",
                res_file.display()
            )));
        }
        let mut res = self.load_res_anns(anns)?;
        let n_fixed = non_finite.rewritten;
        if n_fixed > 0 {
            res.warn(format!(
                "normalized {n_fixed} non-finite float value(s) (NaN/Infinity) to null while loading {}",
                res_file.display()
            ));
        }
        Ok(res)
    }

    /// Detections from rows of `[image_id, x, y, w, h, score]` or
    /// `[image_id, x, y, w, h, score, category_id]`, back to back in `rows`:
    /// the array [`load_res_anns`](Self::load_res_anns) takes in Python, the
    /// pycocotools `loadNumpyAnnotations` convention. Six-column rows get
    /// category 1, as in pycocotools. Rows are converted in parallel past the
    /// shared fan-out threshold.
    ///
    /// # Errors
    ///
    /// An `image_id` or `category_id` that is NaN, infinite, negative, or not
    /// integral, naming the first such row: cast to an integer, NaN and
    /// negatives would saturate to 0, which is a real id in some datasets, and
    /// `1.5` would silently become id 1. pycocotools' `int()` raises on NaN.
    /// Also `ncols` other than 6 or 7, or `rows` that is not a whole number
    /// of rows.
    pub fn detections_from_rows(
        rows: &[f64],
        ncols: usize,
    ) -> crate::error::Result<Vec<Annotation>> {
        Self::check_detection_rows(rows, ncols)?;
        let parallel = worth_parallel(rows.len() / ncols);
        let detection = |row: &[f64]| Annotation {
            image_id: row[0] as u64,
            category_id: detection_category(row, ncols),
            bbox: Some([row[1], row[2], row[3], row[4]]),
            score: Some(row[5]),
            ..Default::default()
        };
        Ok(if parallel {
            rows.par_chunks_exact(ncols)
                .fork_safe()
                .map(detection)
                .collect()
        } else {
            rows.chunks_exact(ncols).map(detection).collect()
        })
    }

    /// [`detections_from_rows`](Self::detections_from_rows)' check of the
    /// rows' shape and ids, which [`DetectionColumns::check_rows`] runs too.
    fn check_detection_rows(rows: &[f64], ncols: usize) -> crate::error::Result<()> {
        if ncols != 6 && ncols != 7 {
            return Err(Error::InvalidInput(format!(
                "detections must have 6 or 7 columns \
                 [image_id, x, y, w, h, score[, category_id]], got {ncols}"
            )));
        }
        if rows.len() % ncols != 0 {
            return Err(Error::InvalidInput(format!(
                "detections: {} values is not a whole number of {ncols}-column rows",
                rows.len()
            )));
        }
        let n = rows.len() / ncols;
        let valid = |v: f64| crate::types::id_from_f64(v).is_some();
        // Ids are checked in a read of the rows first, so the records can be
        // written straight into the result: collecting a fallible map in
        // parallel would gather chunks and copy them all again.
        let first_bad = |(i, row): (usize, &[f64])| {
            if !valid(row[0]) {
                Some((i, "image_id", row[0]))
            } else if ncols == 7 && !valid(row[6]) {
                Some((i, "category_id", row[6]))
            } else {
                None
            }
        };
        let parallel = worth_parallel(n);
        // Ids are almost always fine: an unordered check of every row first,
        // and the ordered search for the first bad one only when there is one.
        if parallel
            && rows
                .par_chunks_exact(ncols)
                .fork_safe()
                .enumerate()
                .all(|row| first_bad(row).is_none())
        {
            return Ok(());
        }
        let bad = if parallel {
            rows.par_chunks_exact(ncols)
                .fork_safe()
                .enumerate()
                .find_map_first(first_bad)
        } else {
            rows.chunks_exact(ncols).enumerate().find_map(first_bad)
        };
        match bad {
            Some((row, column, value)) => Err(Error::InvalidInput(format!(
                "row {row} has {column} {value}; ids must be finite, non-negative integers"
            ))),
            None => Ok(()),
        }
    }

    /// Load detection results from an already-parsed list of annotations.
    ///
    /// This is the in-memory equivalent of [`load_res`](Self::load_res). It applies
    /// the same area, segmentation, and bbox fixups and returns a new `COCO` object
    /// sharing the images and categories from `self`.
    ///
    /// Prefer this over `load_res` when results are already in memory — it avoids
    /// a round-trip through the filesystem. The Python binding uses this internally
    /// when `load_res` is called with a list of dicts or a numpy array.
    pub fn load_res_anns(&self, mut anns: Vec<Annotation>) -> crate::error::Result<COCO> {
        // One kind for the whole file, from the first annotation, as pycocotools
        // does — then fill in whatever geometry that kind implies.
        let kind = anns.first().and_then(ResultKind::of);
        let census = kind == Some(ResultKind::Bbox);
        let warnings = self.check_results(anns.len(), census, |i| ResultFields::of(&anns[i]))?;

        // Per annotation and, for masks, an RLE decode each: the one expensive
        // step of loading results. A decode costs enough to fan out at any
        // batch size; the other kinds derive a few numbers per annotation, so
        // they fan out only past the shared threshold.
        let fill = |(i, ann): (usize, &mut Annotation)| {
            // Ids are 1-indexed and assigned unconditionally, as pycocotools does.
            ann.id = (i + 1) as u64;
            if let Some(kind) = kind {
                // Detection results are never crowd regions, whatever the input
                // file claimed.
                ann.iscrowd = false;
                match kind {
                    ResultKind::Bbox => Self::derive_from_bbox(ann),
                    ResultKind::Segm => self.derive_from_segmentation(ann),
                    ResultKind::Keypoints => Self::derive_from_keypoints(ann),
                    ResultKind::Obb => Self::derive_from_obb(ann),
                }
            }
        };
        if matches!(kind, Some(ResultKind::Segm)) || worth_parallel(anns.len()) {
            anns.par_iter_mut().fork_safe().enumerate().for_each(fill);
        } else {
            anns.iter_mut().enumerate().for_each(fill);
        }

        let dataset = Dataset {
            info: self.dataset.info.clone(),
            images: self.dataset.images.clone(),
            annotations: anns,
            categories: self.dataset.categories.clone(),
            licenses: self.dataset.licenses.clone(),
        };

        let mut res = COCO::from_dataset(dataset);
        for w in warnings {
            res.warn(w);
        }
        Ok(res)
    }

    /// What `load_res` checks of `n` results before keeping them, the `i`th
    /// read through `at`, and the warnings it keeps. Both
    /// [`load_res_anns`](Self::load_res_anns) and
    /// [`DetectionColumns::from_checked_rows`] check through this, so a results set
    /// loaded either way raises and warns alike.
    ///
    /// One read-only pass, probing the index maps this `COCO` already built
    /// (`self.imgs`/`self.cats`) rather than building a `HashSet` of GT ids per
    /// call. It fans out past the shared threshold; the first offender of each
    /// kind is the one reported either way.
    ///
    /// A NaN score is rejected rather than warned about: it corrupts the
    /// whole run, not one annotation. NaN has no place in a score order: a
    /// sort with `partial_cmp(..).unwrap_or(Equal)` is not transitive once
    /// it is present, and the bit keys `accumulate()` sorts on put it
    /// wherever its bits fall, so AP becomes a function of the sort
    /// implementation. (`healthcheck` reports the same condition as an
    /// error and points here.) An image_id or category_id not in the GT is
    /// a common mistake that makes the detections silently score low, so
    /// the first of each is a warning.
    /// With `census`, beside the first offenders, a census of boxes that read
    /// as corner pairs: a corner-pair box evaluates to plausible, wrong numbers
    /// with no other symptom, so the loader is the one place to catch it. One
    /// box past the edge is a dataset quirk; a majority is a format.
    fn check_results<'a>(
        &self,
        n: usize,
        census: bool,
        at: impl Fn(usize) -> ResultFields<'a> + Sync,
    ) -> crate::error::Result<Vec<String>> {
        let has_cats = !self.cats.is_empty();
        // Results usually come image by image, so the image is looked up
        // again only when it changes: `memo` holds the last one.
        let check = |memo: &mut Option<(u64, Option<usize>)>, i: usize| {
            let ann = at(i);
            let img = match *memo {
                Some((id, img)) if id == ann.image_id => img,
                _ => {
                    let img = self.imgs.get(&ann.image_id).copied();
                    *memo = Some((ann.image_id, img));
                    img
                }
            };
            let firsts = [
                ann.score.is_some_and(f64::is_nan).then_some(i),
                img.is_none().then_some(i),
                (has_cats && !self.cats.contains_key(&ann.category_id)).then_some(i),
            ];
            let reading = match (census, ann.bbox, img) {
                (true, Some(b), Some(img)) => {
                    let img = &self.dataset.images[img];
                    reads_as_corner_pair(b, img.width, img.height)
                }
                _ => None,
            };
            (firsts, reading.map_or((0, 0), |r| (1, usize::from(r))))
        };
        let combine = |(a, (n, m)): Checked, (b, (n2, m2)): Checked| {
            let firsts = std::array::from_fn(|j| match (a[j], b[j]) {
                (Some(x), Some(y)) => Some(x.min(y)),
                (x, y) => x.or(y),
            });
            (firsts, (n + n2, m + m2))
        };
        let none: Checked = ([None; 3], (0, 0));
        let ([first_nan, first_img, first_cat], (n, m)) = if worth_parallel(n) {
            (0..n)
                .into_par_iter()
                .fork_safe()
                .fold(
                    || (none, None),
                    |(acc, mut memo), i| {
                        let checked = check(&mut memo, i);
                        (combine(acc, checked), memo)
                    },
                )
                .map(|(acc, _)| acc)
                .reduce(|| none, combine)
        } else {
            let mut memo = None;
            (0..n).fold(none, |acc, i| combine(acc, check(&mut memo, i)))
        };
        if let Some(i) = first_nan {
            let ann = at(i);
            return Err(nan_score_error("load_res(): ", i, ann.id, ann.image_id));
        }
        let mut warnings = Vec::new();
        if 2 * m > n {
            warnings.push(format!(
                "load_res() warning — {m} of {n} detection boxes run past their image \
                 when read as [x, y, width, height] and fit it when read as \
                 [x1, y1, x2, y2]. COCO boxes are [x, y, width, height]; corner-pair \
                 boxes evaluate to plausible but wrong numbers. If yours are corner \
                 pairs, convert them before evaluating: \
                 https://derekallman.github.io/hotcoco/getting-started/troubleshooting/#bbox-format-x1-y1-x2-y2-vs-x-y-w-h"
            ));
        }
        if let Some(i) = first_img {
            warnings.push(format!(
                "load_res() warning — found annotation with image_id {} not in the \
                 GT dataset. These DTs will never match. Check your results file matches the \
                 correct GT split.",
                at(i).image_id
            ));
        }
        if let Some(i) = first_cat {
            warnings.push(format!(
                "load_res() warning — found annotation with category_id {} not \
                 in the GT dataset. These DTs will never match.",
                at(i).category_id
            ));
        }
        Ok(warnings)
    }

    /// Fill in `area` on every annotation that has none, the way a ground
    /// truth's is derived: the mask's pixel count, then the rotated box's
    /// `w × h`, then the box's, then the extent of the labeled keypoints.
    /// Annotations with an `area` are left as they
    /// are, so a COCO file's authored areas survive; one with nothing to derive
    /// an area from stays without one.
    ///
    /// `area` places an annotation in an area range, and the matcher would read
    /// a missing one as 0, putting every such object in `small`; pycocotools
    /// raises `KeyError` instead. [`COCOeval`](crate::COCOeval) and
    /// [`StreamingEval`](crate::StreamingEval) fill their own copies of the
    /// datasets they evaluate, the ground truth by this rule and the detections
    /// by `load_res`'s, so targets that never carried the field
    /// (torchvision-style dicts) evaluate as if they had.
    pub fn fill_missing_areas(&mut self) {
        let derived = self.missing_areas(AreaRule::Instance);
        self.set_areas_at(derived);
    }

    /// The `area` each annotation without one would get by `rule`, as
    /// `(index, area)` pairs, computed without writing anything, so a caller
    /// holding a shared `COCO` copies it only when there is something to fill.
    /// Masks are drawn in parallel past the shared fan-out threshold.
    pub(crate) fn missing_areas(&self, rule: AreaRule) -> Vec<(usize, f64)> {
        let anns = &self.dataset.annotations;
        // Usually none: every evaluator runs this over both datasets.
        let missing: Vec<usize> = if worth_parallel(anns.len()) {
            anns.par_iter()
                .fork_safe()
                .positions(|ann| ann.area.is_none())
                .collect()
        } else {
            (0..anns.len())
                .filter(|&i| anns[i].area.is_none())
                .collect()
        };
        let derive = |&i: &usize| {
            let ann = &anns[i];
            let from_box = || ann.bbox.as_ref().map(box_area);
            let from_obb = || ann.obb.as_deref().map(|obb| obb[2] * obb[3]);
            // A mask of no pixels, such as `"segmentation": []` beside a box,
            // says nothing about the object's size.
            let from_mask = || {
                ann.segmentation
                    .as_ref()
                    .and_then(|_| self.mask_area(ann))
                    .filter(|&area| area > 0)
                    .map(|area| area as f64)
            };
            let from_keypoints =
                |labeled_only| Self::keypoint_extent(ann, labeled_only).map(|bb| bb[2] * bb[3]);
            let area = match rule {
                AreaRule::Instance => from_mask()
                    .or_else(from_obb)
                    .or_else(from_box)
                    .or_else(|| from_keypoints(true)),
                AreaRule::Result => from_box()
                    .or_else(from_mask)
                    .or_else(|| from_keypoints(false))
                    .or_else(from_obb),
            }?;
            Some((i, area))
        };
        if worth_parallel(missing.len()) {
            missing.par_iter().fork_safe().filter_map(derive).collect()
        } else {
            missing.iter().filter_map(derive).collect()
        }
    }

    /// Write the pairs [`missing_areas`](Self::missing_areas) returned.
    pub(crate) fn set_areas_at(&mut self, derived: Vec<(usize, f64)>) {
        for (i, area) in derived {
            self.dataset.annotations[i].area = Some(area);
        }
    }

    /// Area from the box, and a rectangular segmentation when none was given.
    fn derive_from_bbox(ann: &mut Annotation) {
        let Some(bbox) = ann.bbox else {
            return;
        };
        ann.area = Some(box_area(&bbox));
        if ann.segmentation.is_none() {
            ann.segmentation = Some(Segmentation::Rect(bbox));
        }
    }

    /// Area and box from the mask.
    ///
    /// Only `CompressedRle` is handled, matching pycocotools' `loadRes`: polygon
    /// and uncompressed-RLE results are not expected in a detection output file.
    /// Image lookups go through the GT `COCO` (`self`), since detection results
    /// share its images and therefore its dimensions.
    fn derive_from_segmentation(&self, ann: &mut Annotation) {
        let Some(rle @ mask::RleRef::Compressed { .. }) =
            ann.segmentation.as_ref().and_then(Segmentation::rle_ref)
        else {
            return;
        };
        // `ann_to_rle`'s gate: no mask for an image with no record.
        if self.raster_dims(ann).is_none() {
            return;
        }
        // A given box is kept, as pycocotools keeps it; only the area comes
        // from the mask.
        if ann.bbox.is_some() {
            if let Ok(area) = rle.area() {
                ann.area = Some(area as f64);
            }
        } else if let Ok((area, bbox)) = rle.area_and_bbox() {
            ann.area = Some(area as f64);
            ann.bbox = Some(bbox);
        }
    }

    /// Area and box from the extent of the keypoints.
    ///
    /// An annotation with no keypoint values is left untouched (`area`/`bbox`
    /// stay `None`): folding an empty extent would produce a
    /// `[inf, inf, -inf, -inf]` bbox with infinite area that flows into
    /// evaluation unflagged. (pycocotools errors outright on an empty
    /// keypoints array here.)
    fn derive_from_keypoints(ann: &mut Annotation) {
        if let Some(bbox) = Self::keypoint_extent(ann, false) {
            ann.area = Some(bbox[2] * bbox[3]);
            ann.bbox = Some(bbox);
        }
    }

    /// The `[x, y, w, h]` extent of an annotation's keypoints, or of the
    /// labeled ones (visibility above 0) with `labeled_only`; `None` with no
    /// such point. A result's extent takes every point, as pycocotools'
    /// `loadRes` does; a ground truth's unlabeled points sit at `(0, 0)` and
    /// would stretch it to the image origin.
    fn keypoint_extent(ann: &Annotation, labeled_only: bool) -> Option<[f64; 4]> {
        let kpts = ann.keypoints.as_ref()?;
        // Keypoints are flat (x, y, visibility) triples; a trailing pair
        // without its visibility still counts as a point.
        let points = kpts
            .chunks(3)
            .filter(|p| p.len() >= 2)
            .filter(|p| !labeled_only || p.get(2).is_some_and(|&v| v > 0.0));
        let (mut x0, mut y0) = (f64::INFINITY, f64::INFINITY);
        let (mut x1, mut y1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut any = false;
        for p in points {
            (x0, x1) = (x0.min(p[0]), x1.max(p[0]));
            (y0, y1) = (y0.min(p[1]), y1.max(p[1]));
            any = true;
        }
        any.then_some([x0, y0, x1 - x0, y1 - y0])
    }

    /// Area from the rotated box, and its axis-aligned envelope as the bbox.
    fn derive_from_obb(ann: &mut Annotation) {
        let Some(obb) = ann.obb.as_deref() else {
            return;
        };
        ann.area = Some(obb[2] * obb[3]);
        ann.bbox = Some(crate::geometry::obb_to_aabb(obb));
    }

    /// Whether this annotation's mask is rasterized onto the image's canvas.
    ///
    /// Polygons, rectangles, and bbox-only records take their raster size from
    /// the image; either RLE form carries its own `size`.
    fn needs_image_dims(ann: &Annotation) -> bool {
        match &ann.segmentation {
            Some(Segmentation::Polygon(_) | Segmentation::Rect(_)) => true,
            Some(Segmentation::CompressedRle { .. } | Segmentation::UncompressedRle { .. }) => {
                false
            }
            None => ann.bbox.is_some(),
        }
    }

    /// Check that no annotation has a NaN score.
    ///
    /// [`load_res`](Self::load_res) refuses one, with this error: scores order
    /// the detection ranking, and NaN makes that order undefined, so AP becomes
    /// a function of the sort. A results set built another way — a dataset dict
    /// assigned to a `COCO`, [`from_dataset`](Self::from_dataset) — skips the
    /// loader, so [`COCOeval::check_inputs`](crate::COCOeval::check_inputs)
    /// runs this on the detections first.
    ///
    /// # Errors
    ///
    /// `Error::InvalidInput` naming the first such annotation.
    pub fn check_scores(&self) -> crate::error::Result<()> {
        let anns = &self.dataset.annotations;
        // Every `evaluate()` runs this over the whole results set, which at
        // validation scale is a walk over a million records: in parallel past
        // the shared threshold, still naming the first offender.
        let is_nan = |ann: &Annotation| ann.score.is_some_and(f64::is_nan);
        let first = if worth_parallel(anns.len()) {
            anns.par_iter().fork_safe().position_first(is_nan)
        } else {
            anns.iter().position(is_nan)
        };
        match first {
            Some(i) => Err(nan_score_error("", i, anns[i].id, anns[i].image_id)),
            None => Ok(()),
        }
    }

    /// Check that every mask an evaluation of `img_ids` and `cat_ids` would
    /// rasterize has a canvas to land on.
    ///
    /// `height` and `width` are optional on an image record — box evaluation
    /// never reads them — and default to 0. A polygon drawn onto a 0×0 canvas is
    /// an empty mask, so a segm evaluation over such a dataset reports AP 0 with
    /// nothing to say it went wrong; pycocotools raises `KeyError` instead. This
    /// is the check the segm entry points run first, so the failure is an error
    /// naming the images, not a plausible number.
    ///
    /// Only annotations on `img_ids` in `cat_ids` are checked — the ones the
    /// evaluation covers, as pycocotools rasterizes only what `_prepare` loads.
    /// `cat_ids = None` means every category, as when `use_cats` is off. An
    /// annotation outside that scope, including one on an image with no image
    /// record, is never rasterized and never an error.
    /// [`COCOeval::check_inputs`](crate::COCOeval::check_inputs) passes the
    /// evaluation's resolved ids.
    ///
    /// # Errors
    ///
    /// An error when an annotation in scope whose mask comes from
    /// the image size (a polygon, a rectangle, or a bbox-only record — see
    /// [`ann_to_rle`](Self::ann_to_rle)) belongs to an image without `height`
    /// and `width`, or an id in `img_ids` with no image record.
    pub fn check_mask_dims(
        &self,
        img_ids: &[u64],
        cat_ids: Option<&[u64]>,
    ) -> crate::error::Result<()> {
        let imgs: rustc_hash::FxHashSet<u64> = img_ids.iter().copied().collect();
        let cats: Option<rustc_hash::FxHashSet<u64>> =
            cat_ids.map(|ids| ids.iter().copied().collect());
        let mut bad = std::collections::BTreeSet::new();
        for ann in &self.dataset.annotations {
            let in_scope = imgs.contains(&ann.image_id)
                && cats.as_ref().is_none_or(|c| c.contains(&ann.category_id));
            if !in_scope || !Self::needs_image_dims(ann) {
                continue;
            }
            if self.raster_dims(ann).is_none() {
                bad.insert(ann.image_id);
            }
        }
        if bad.is_empty() {
            return Ok(());
        }
        let shown: Vec<String> = bad.iter().take(10).map(u64::to_string).collect();
        let more = if bad.len() > 10 {
            format!(", and {} more", bad.len() - 10)
        } else {
            String::new()
        };
        Err(Error::InvalidInput(format!(
            "segmentation evaluation needs `height` and `width` on every image whose \
             annotations are polygons or boxes; {} image(s) lack them (ids {}{more}). \
             Add the fields to the image records, or pass masks as RLE, which carries \
             its own size.",
            bad.len(),
            shown.join(", "),
        )))
    }

    /// Keep each image's `max_det` highest-scoring annotations, across every
    /// category — lvis-api's per-image detection cap, as `LVISResults` applies it.
    ///
    /// lvis-api applies this once, when `LVISResults` loads the detections
    /// (`limit_dets_per_image`, lvis-api 0.5.3 `lvis/results.py`): each image's
    /// results are sorted by score and the top `max_dets` (300 by default)
    /// kept before `LVISEval` sees them, so detections on categories the
    /// federated rule later drops still spend the budget. `None` keeps every
    /// detection, lvis-api's `max_dets=-1`.
    ///
    /// The returned set is marked as capped, and LVIS evaluation takes a
    /// marked set as is — lvis-api's `LVISEval` likewise uses an
    /// `LVISResults` unchanged and caps only results it loads itself. An
    /// unmarked set, such as a plain [`load_res`](Self::load_res) result, gets
    /// the default 300 cap from [`COCOeval::new_lvis`](crate::COCOeval::new_lvis).
    /// So call this to evaluate under a different cap, or none.
    ///
    /// Ties keep dataset order — results-file order for a set loaded by
    /// [`load_res`](Self::load_res) — as Python's stable `sorted` does in
    /// lvis-api. A missing score sorts as 0.
    ///
    /// Takes `self` so that the common case, where no image holds more than
    /// `max_det` annotations, copies nothing; otherwise the result is a new,
    /// re-indexed `COCO` with the same images and categories.
    #[must_use]
    pub fn cap_detections_per_image(self, max_det: Option<usize>) -> COCO {
        let mut capped = match max_det.and_then(|n| self.capped_copy(n)) {
            Some(capped) => capped,
            None => self,
        };
        capped.per_image_capped = true;
        capped
    }

    /// [`cap_detections_per_image`](Self::cap_detections_per_image) for a set
    /// held behind an `Arc` that others share — the Python binding's case,
    /// where taking `self` by value would mean copying first.
    ///
    /// Returns `self` itself, not a copy, when the cap drops nothing and the
    /// capped mark would change nothing: no image holds more than `max_det`
    /// annotations nor more than LVIS's own 300, so LVIS evaluation's default
    /// cap is a no-op on it too. Otherwise a capped copy, or a marked one when
    /// the set must be shielded from that default cap (`None`, or a `max_det`
    /// above 300, on a set with an image over 300).
    #[must_use]
    pub fn cap_detections_per_image_shared(
        self: std::sync::Arc<Self>,
        max_det: Option<usize>,
    ) -> std::sync::Arc<COCO> {
        if let Some(mut capped) = max_det.and_then(|n| self.capped_copy(n)) {
            capped.per_image_capped = true;
            return std::sync::Arc::new(capped);
        }
        if self.per_image_capped || self.images_over(LVIS_MAX_DETS_PER_IMAGE).is_empty() {
            return self;
        }
        let mut marked = std::sync::Arc::unwrap_or_clone(self);
        marked.per_image_capped = true;
        std::sync::Arc::new(marked)
    }

    /// Whether [`cap_detections_per_image`](Self::cap_detections_per_image)
    /// produced this set, so LVIS evaluation must not cap it again.
    pub fn is_per_image_capped(&self) -> bool {
        self.per_image_capped
    }

    /// The images holding more than `max_det` annotations, with their counts.
    /// Counted from the annotations, not the index: `dataset` is public, and a
    /// caller that trimmed it without `create_index` must still get a cap that
    /// agrees with what it holds.
    fn images_over(&self, max_det: usize) -> FxHashMap<u64, usize> {
        let mut per_img: FxHashMap<u64, usize> = FxHashMap::default();
        for ann in &self.dataset.annotations {
            *per_img.entry(ann.image_id).or_default() += 1;
        }
        per_img.retain(|_, &mut n| n > max_det);
        per_img
    }

    /// The per-image cap's working half: `None` when no image holds more than
    /// `max_det` annotations, so a caller holding only a reference copies
    /// nothing in the common case. The copy is not marked as capped.
    pub(crate) fn capped_copy(&self, max_det: usize) -> Option<COCO> {
        let per_img = self.images_over(max_det);
        if per_img.is_empty() {
            return None;
        }
        let anns = &self.dataset.annotations;

        // (score, position) for each over-cap image's annotations, in dataset
        // order, so the stable sort below breaks ties by position.
        let mut over: FxHashMap<u64, Vec<(f64, usize)>> =
            per_img.into_keys().map(|id| (id, Vec::new())).collect();
        for (pos, ann) in anns.iter().enumerate() {
            if let Some(dets) = over.get_mut(&ann.image_id) {
                dets.push((ann.score.unwrap_or(0.0), pos));
            }
        }
        let mut keep = vec![true; anns.len()];
        for dets in over.values_mut() {
            dets.sort_by_key(|&(score, _)| descending_score_key(score));
            for &(_, pos) in &dets[max_det..] {
                keep[pos] = false;
            }
        }
        Some(COCO::from_dataset(Dataset {
            info: self.dataset.info.clone(),
            images: self.dataset.images.clone(),
            annotations: anns
                .iter()
                .zip(&keep)
                .filter(|&(_, &k)| k)
                .map(|(ann, _)| ann.clone())
                .collect(),
            categories: self.dataset.categories.clone(),
            licenses: self.dataset.licenses.clone(),
        }))
    }

    /// Convert an annotation's segmentation to RLE.
    ///
    /// `None` when the annotation's image is unknown, its RLE counts overrun the
    /// mask, or its mask would be rasterized from an image with no `height` and
    /// `width` — a 0×0 canvas holds no mask, and [`check_mask_dims`](Self::check_mask_dims)
    /// is how a caller turns that into an error up front.
    pub fn ann_to_rle(&self, ann: &Annotation) -> Option<Rle> {
        let (h, w) = self.raster_dims(ann)?;
        match &ann.segmentation {
            // Either RLE spelling is validated on the way: a string that does
            // not decode, or runs past `h * w`, is no mask.
            Some(seg) => seg.to_rle(h, w).ok(),
            None => {
                // For bbox-only annotations, convert bbox to RLE
                ann.bbox
                    .as_ref()
                    .and_then(|bb| mask::fr_bbox(bb, h, w).ok())
            }
        }
    }

    /// The `(height, width)` [`ann_to_rle`](Self::ann_to_rle) rasterizes
    /// `ann` at, or `None` when it returns `None` before looking at the mask:
    /// the image is unknown, or the mask is drawn on the image's canvas and
    /// the image has no size. [`check_mask_dims`](Self::check_mask_dims)
    /// reports the second case up front.
    fn raster_dims(&self, ann: &Annotation) -> Option<(u32, u32)> {
        let img = self.get_img(ann.image_id)?;
        if (img.height == 0 || img.width == 0) && Self::needs_image_dims(ann) {
            return None;
        }
        Some((img.height, img.width))
    }

    /// The pixel count of `ann`'s mask:
    /// `ann_to_rle(ann).map(|rle| mask::area(&rle))`, with an RLE summed
    /// where it is — a compressed one straight off its `counts` string, a run
    /// list without the copy `ann_to_rle` makes of it.
    ///
    /// An RLE gets `ann_to_rle`'s gate and validation: [`RleRef::area`]
    /// rejects exactly what [`RleRef::to_rle`] does, so a malformed RLE is
    /// `None` here too.
    ///
    /// [`RleRef::area`]: mask::RleRef::area
    /// [`RleRef::to_rle`]: mask::RleRef::to_rle
    fn mask_area(&self, ann: &Annotation) -> Option<u64> {
        match ann.segmentation.as_ref().and_then(Segmentation::rle_ref) {
            // `ann_to_rle`'s gate: no mask for an image with no record.
            Some(rle) => {
                self.raster_dims(ann)?;
                rle.area().ok()
            }
            None => self.ann_to_rle(ann).map(|rle| mask::area(&rle)),
        }
    }

    /// Convert an annotation to a binary mask.
    pub fn ann_to_mask(&self, ann: &Annotation) -> Option<Vec<u8>> {
        self.ann_to_rle(ann).map(|rle| mask::decode(&rle))
    }

    /// Filter the dataset, returning a new `Dataset` with matching images, annotations, and categories.
    ///
    /// Annotations are kept when they match **all** provided criteria. If `drop_empty_images` is
    /// `true`, images with no matching annotations are removed; otherwise all images are kept
    /// (intersected with `img_ids` if provided).
    pub fn filter(
        &self,
        cat_ids: Option<&[u64]>,
        img_ids: Option<&[u64]>,
        area_rng: Option<[f64; 2]>,
        drop_empty_images: bool,
    ) -> Dataset {
        let cat_set: Option<std::collections::HashSet<u64>> =
            cat_ids.map(|ids| ids.iter().copied().collect());
        let img_set: Option<std::collections::HashSet<u64>> =
            img_ids.map(|ids| ids.iter().copied().collect());

        let filtered_anns: Vec<Annotation> = self
            .dataset
            .annotations
            .iter()
            .filter(|ann| {
                if let Some(ref cids) = cat_set {
                    if !cids.contains(&ann.category_id) {
                        return false;
                    }
                }
                if let Some(ref iids) = img_set {
                    if !iids.contains(&ann.image_id) {
                        return false;
                    }
                }
                if let Some(rng) = area_rng {
                    if !area_in_range(ann, rng, Bounds::Closed) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        let img_ids_with_anns: std::collections::HashSet<u64> =
            filtered_anns.iter().map(|a| a.image_id).collect();

        let filtered_images: Vec<Image> = self
            .dataset
            .images
            .iter()
            .filter(|img| {
                if drop_empty_images {
                    img_ids_with_anns.contains(&img.id)
                } else if let Some(ref iids) = img_set {
                    iids.contains(&img.id)
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        let cat_ids_used: std::collections::HashSet<u64> =
            filtered_anns.iter().map(|a| a.category_id).collect();
        let filtered_cats: Vec<Category> = self
            .dataset
            .categories
            .iter()
            .filter(|cat| cat_ids_used.contains(&cat.id))
            .cloned()
            .collect();

        Dataset {
            info: self.dataset.info.clone(),
            images: filtered_images,
            annotations: filtered_anns,
            categories: filtered_cats,
            licenses: self.dataset.licenses.clone(),
        }
    }

    /// Merge multiple datasets into one.
    ///
    /// All datasets must share the same category taxonomy (same names + supercategories).
    /// Image and annotation IDs are remapped to ensure global uniqueness.
    pub fn merge(datasets: &[&Dataset]) -> crate::error::Result<Dataset> {
        if datasets.is_empty() {
            return Ok(Dataset {
                info: None,
                images: vec![],
                annotations: vec![],
                categories: vec![],
                licenses: vec![],
            });
        }

        let canonical_cats = &datasets[0].categories;
        let canonical_key_to_id: HashMap<(String, Option<String>), u64> = canonical_cats
            .iter()
            .map(|c| ((c.name.clone(), c.supercategory.clone()), c.id))
            .collect();

        // Build per-dataset category ID remaps (dataset[0] is identity)
        let mut cat_remaps: Vec<HashMap<u64, u64>> = Vec::new();
        let identity: HashMap<u64, u64> = canonical_cats.iter().map(|c| (c.id, c.id)).collect();
        cat_remaps.push(identity);

        for ds in datasets.iter().skip(1) {
            if ds.categories.len() != canonical_cats.len() {
                return Err(Error::InvalidInput(format!(
                    "Cannot merge: datasets have different numbers of categories ({} vs {})",
                    canonical_cats.len(),
                    ds.categories.len()
                )));
            }
            let mut remap = HashMap::new();
            for cat in &ds.categories {
                let key = (cat.name.clone(), cat.supercategory.clone());
                match canonical_key_to_id.get(&key) {
                    Some(&canonical_id) => {
                        remap.insert(cat.id, canonical_id);
                    }
                    None => {
                        return Err(Error::InvalidInput(format!(
                            "Cannot merge: category '{}' not found in first dataset",
                            cat.name
                        )));
                    }
                }
            }
            cat_remaps.push(remap);
        }

        let mut all_images: Vec<Image> = Vec::new();
        let mut all_anns: Vec<Annotation> = Vec::new();
        let mut current_max_img_id: u64 = 0;
        let mut current_max_ann_id: u64 = 0;

        // All id arithmetic is checked: ids come from untrusted JSON, and a
        // wrapped offset in release would produce silent id collisions in the
        // merged dataset.
        let id_overflow = |what: &str, id: u64, offset: u64| {
            Error::InvalidInput(format!(
                "Cannot merge: {what} id {id} + offset {offset} overflows u64. \
                 Renumber the input ids to something smaller first."
            ))
        };
        for (i, ds) in datasets.iter().enumerate() {
            let img_offset = current_max_img_id;
            let ann_offset = current_max_ann_id;
            let cat_remap = &cat_remaps[i];

            let mut max_img_id = 0u64;
            for img in &ds.images {
                let mut new_img = img.clone();
                new_img.id = img
                    .id
                    .checked_add(img_offset)
                    .ok_or_else(|| id_overflow("image", img.id, img_offset))?;
                all_images.push(new_img);
                max_img_id = max_img_id.max(img.id);
            }

            let mut max_ann_id = 0u64;
            for ann in &ds.annotations {
                let mut new_ann = ann.clone();
                new_ann.id = ann
                    .id
                    .checked_add(ann_offset)
                    .ok_or_else(|| id_overflow("annotation", ann.id, ann_offset))?;
                new_ann.image_id = ann
                    .image_id
                    .checked_add(img_offset)
                    .ok_or_else(|| id_overflow("annotation image", ann.image_id, img_offset))?;
                new_ann.category_id = *cat_remap.get(&ann.category_id).unwrap_or(&ann.category_id);
                all_anns.push(new_ann);
                max_ann_id = max_ann_id.max(ann.id);
            }

            current_max_img_id = max_img_id
                .checked_add(img_offset)
                .ok_or_else(|| id_overflow("image", max_img_id, img_offset))?;
            current_max_ann_id = max_ann_id
                .checked_add(ann_offset)
                .ok_or_else(|| id_overflow("annotation", max_ann_id, ann_offset))?;
        }

        Ok(Dataset {
            info: datasets[0].info.clone(),
            images: all_images,
            annotations: all_anns,
            categories: canonical_cats.clone(),
            licenses: datasets[0].licenses.clone(),
        })
    }

    /// Create a dataset subset containing only the given image IDs and their annotations.
    fn subset_by_img_ids(&self, ids: &[u64]) -> Dataset {
        let id_set: std::collections::HashSet<u64> = ids.iter().copied().collect();
        let images: Vec<Image> = self
            .dataset
            .images
            .iter()
            .filter(|img| id_set.contains(&img.id))
            .cloned()
            .collect();
        let annotations: Vec<Annotation> = self
            .dataset
            .annotations
            .iter()
            .filter(|ann| id_set.contains(&ann.image_id))
            .cloned()
            .collect();
        Dataset {
            info: self.dataset.info.clone(),
            images,
            annotations,
            categories: self.dataset.categories.clone(),
            licenses: self.dataset.licenses.clone(),
        }
    }

    /// Split the dataset into train/val (and optionally test) subsets.
    ///
    /// Images are shuffled deterministically using `seed`, then partitioned.
    /// All splits share the full category list.
    ///
    /// # Errors
    ///
    /// A fraction outside `[0, 1]` (NaN included), or `val_frac + test_frac`
    /// over 1. Neither is clamped: clamping makes `val_frac = -1` an empty
    /// validation set and `0.9 + 0.9` a test set of a tenth.
    pub fn split(
        &self,
        val_frac: f64,
        test_frac: Option<f64>,
        seed: u64,
    ) -> crate::error::Result<(Dataset, Dataset, Option<Dataset>)> {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;

        fraction("val_frac", val_frac)?;
        if let Some(test_frac) = test_frac {
            fraction("test_frac", test_frac)?;
            if val_frac + test_frac > 1.0 {
                return Err(Error::InvalidInput(format!(
                    "val_frac + test_frac must be at most 1, got {val_frac} + {test_frac}"
                )));
            }
        }

        let mut img_ids: Vec<u64> = self.dataset.images.iter().map(|img| img.id).collect();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        img_ids.shuffle(&mut rng);

        let n = img_ids.len();
        let n_val = ((n as f64 * val_frac).round() as usize).min(n);
        let n_test = test_frac.map_or(0, |f| {
            ((n as f64 * f).round() as usize).min(n.saturating_sub(n_val))
        });
        let n_train = n.saturating_sub(n_val + n_test);

        let train_ids = &img_ids[..n_train];
        let val_ids = &img_ids[n_train..n_train + n_val];
        let test_ids = if test_frac.is_some() {
            Some(&img_ids[n_train + n_val..])
        } else {
            None
        };

        let train = self.subset_by_img_ids(train_ids);
        let val = self.subset_by_img_ids(val_ids);
        let test = test_ids.map(|ids| self.subset_by_img_ids(ids));

        Ok((train, val, test))
    }

    /// Sample a random subset of images (with their annotations).
    ///
    /// Provide either `n` (exact count) or `frac` (fraction of images).
    /// The sample is deterministic given the same `seed`.
    ///
    /// # Errors
    ///
    /// A `frac` outside `[0, 1]`, NaN included.
    pub fn sample(
        &self,
        n: Option<usize>,
        frac: Option<f64>,
        seed: u64,
    ) -> crate::error::Result<Dataset> {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;

        if let Some(frac) = frac {
            fraction("frac", frac)?;
        }

        let total = self.dataset.images.len();
        let count = match (n, frac) {
            (Some(n), _) => n.min(total),
            // Truncated, past the float error in the product: `0.29 * 100` is
            // 28.999…, which is 29 images, not 28.
            (None, Some(f)) => ((total as f64 * f + 1e-9) as usize).min(total),
            (None, None) => total,
        };

        let mut img_ids: Vec<u64> = self.dataset.images.iter().map(|img| img.id).collect();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        img_ids.shuffle(&mut rng);

        Ok(self.subset_by_img_ids(&img_ids[..count]))
    }

    /// Run a health check on this dataset.
    pub fn healthcheck(&self) -> crate::quality::HealthReport {
        crate::quality::healthcheck(&self.dataset)
    }

    /// Run a health check including GT/DT compatibility.
    pub fn healthcheck_compatibility(&self, dt: &COCO) -> crate::quality::HealthReport {
        crate::quality::healthcheck_compatibility(&self.dataset, &dt.dataset)
    }
}

/// Box detections held as rows: what [`COCO::load_res_anns`] makes of a
/// detection array, without building its record per detection.
///
/// A results `COCO` holds a 200-byte [`Annotation`] per detection, and at the
/// million-detection scale of a detector's validation pass, building and
/// indexing those records costs more than evaluating them. This is the same
/// results set kept as a 48-byte table row plus the index the records would get:
/// [`check_rows`](Self::check_rows) then
/// [`from_checked_rows`](Self::from_checked_rows) run `load_res`'s checks,
/// with the same errors and warnings, and [`coco`](Self::coco) is the results
/// `COCO` itself, built on the first call. An evaluation reads the table
/// directly — see `COCOeval::new_columns` — and anything that needs a record
/// asks for the `COCO`.
///
/// Plumbing between this crate and its Python bindings, not an API to build
/// on: its shape follows what the evaluator reads.
#[doc(hidden)]
pub struct DetectionColumns {
    /// The ground truth `load_res` was called on: the images and categories
    /// [`coco`](Self::coco) copies, as `load_res_anns` does.
    gt: Arc<COCO>,
    /// What an evaluator reads of each detection, in row order.
    pub(crate) table: crate::table::DetectionTable,
    /// Row `i` is id `i + 1`, as `load_res_anns` numbers its results.
    pub(crate) ids: IdIndex,
    /// The index [`COCO::create_index`] builds over the records, ids
    /// `1..=len`, built from the rows by the same code.
    pub(crate) index: AnnIndex,
    warnings: Vec<String>,
    /// The results `COCO`, once something has asked for it: built once, and
    /// shared by every reader — an evaluator and the Python wrapper alike.
    coco: OnceLock<Arc<COCO>>,
}

impl DetectionColumns {
    /// `load_res`'s first check of `rows`, `ncols` (6 or 7) to a row: their
    /// ids, with the error [`COCO::detections_from_rows`] returns.
    /// [`from_checked_rows`](Self::from_checked_rows) makes the rest. Apart,
    /// so a caller can report this one apart, as the Python binding names the
    /// call it came from.
    ///
    /// # Errors
    ///
    /// `ncols` not 6 or 7, a partial row, or an `image_id` or `category_id`
    /// that is not a non-negative integer.
    pub fn check_rows(rows: &[f64], ncols: usize) -> crate::error::Result<()> {
        COCO::check_detection_rows(rows, ncols)
    }

    /// The detections of rows [`check_rows`](Self::check_rows) accepted, as
    /// `gt.load_res_anns(COCO::detections_from_rows(rows, ncols)?)` would load
    /// them: the rest of `load_res`'s checks, then the columns.
    ///
    /// # Errors
    ///
    /// A NaN score.
    ///
    /// # Panics
    ///
    /// `ncols` not 6 or 7, or a partial row.
    pub fn from_checked_rows(
        gt: Arc<COCO>,
        rows: &[f64],
        ncols: usize,
    ) -> crate::error::Result<Self> {
        let n = detection_row_count(rows, ncols);
        // The records `detections_from_rows` would write, as the check reads
        // them: id 0 until `load_res_anns` numbers them, and a box on every
        // one, which makes the results set's kind `Bbox` and turns the
        // corner-pair census on.
        let warnings = gt.check_results(n, n > 0, |i| {
            let row = &rows[i * ncols..(i + 1) * ncols];
            ResultFields {
                id: 0,
                image_id: row[0] as u64,
                category_id: detection_category(row, ncols),
                score: Some(row[5]),
                bbox: <&[f64; 4]>::try_from(&row[1..5]).ok(),
            }
        })?;
        // The image and category of each row go to the index, read from the
        // rows as `detections_from_rows` reads them; the rest is kept.
        let index = AnnIndex::from_keys(n, |i| {
            let row = &rows[i * ncols..(i + 1) * ncols];
            (row[0] as u64, detection_category(row, ncols), i as u64 + 1)
        });
        let fields = |row: &[f64]| ([row[1], row[2], row[3], row[4]], row[5]);
        let (boxes, scores): (Vec<[f64; 4]>, Vec<f64>) = if worth_parallel(n) {
            rows.par_chunks_exact(ncols).fork_safe().map(fields).unzip()
        } else {
            rows.chunks_exact(ncols).map(fields).unzip()
        };
        Ok(DetectionColumns {
            gt,
            table: crate::table::DetectionTable::from_boxes(boxes, scores),
            ids: IdIndex::Dense { len: n },
            index,
            warnings,
            coco: OnceLock::new(),
        })
    }

    /// How many detections.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.table.len() == 0
    }

    /// The warnings `load_res` records for these detections, as
    /// [`COCO::load_warnings`] lists them on the results `COCO`.
    pub fn load_warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The ground truth these detections were loaded against. Its images and
    /// categories are the results `COCO`'s, which `load_res_anns` copies
    /// from it, so a read of those alone need not build the results.
    pub fn ground_truth(&self) -> &Arc<COCO> {
        &self.gt
    }

    /// [`coco`](Self::coco)`.`[`reindexed`](COCO::reindexed)`(dataset)`,
    /// without building the results `COCO` it would replace: a dict assigned
    /// over the columns is the results set from then on.
    #[must_use]
    pub fn reindexed(&self, dataset: Dataset) -> COCO {
        // `load_res_anns` never caps a results set.
        COCO::reindexed_from(&self.warnings, false, dataset)
    }

    /// The results `COCO` these detections stand for: exactly what
    /// `load_res_anns` builds from the same rows. Built on the first call.
    pub fn coco(&self) -> &Arc<COCO> {
        self.coco.get_or_init(|| Arc::new(self.build_coco()))
    }

    fn build_coco(&self) -> COCO {
        // Each detection's image and category, from the (image, category)
        // group of the index that holds its id: every id is in exactly one.
        let n = self.table.len();
        let (mut image_ids, mut category_ids) = (vec![0u64; n], vec![0u64; n]);
        for (img_id, cat_id) in self.index.pairs() {
            for &id in self.index.for_img_cat(img_id, cat_id) {
                let i = (id - 1) as usize;
                image_ids[i] = img_id;
                category_ids[i] = cat_id;
            }
        }
        let anns: Vec<Annotation> = (0..n)
            .map(|i| Annotation {
                image_id: image_ids[i],
                category_id: category_ids[i],
                bbox: self.table.bbox(i),
                score: Some(self.table.score(i)),
                ..Default::default()
            })
            .collect();
        self.gt
            .load_res_anns(anns)
            .expect("from_checked_rows ran load_res's checks on these rows")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_test_dataset() -> Dataset {
        Dataset {
            info: None,
            images: vec![
                Image {
                    id: 1,
                    file_name: "img1.jpg".into(),
                    height: 100,
                    width: 100,
                    ..Default::default()
                },
                Image {
                    id: 2,
                    file_name: "img2.jpg".into(),
                    height: 200,
                    width: 200,
                    ..Default::default()
                },
            ],
            annotations: vec![
                Annotation {
                    id: 1,
                    image_id: 1,
                    category_id: 1,
                    bbox: Some([10.0, 10.0, 20.0, 20.0]),
                    area: Some(400.0),
                    ..Default::default()
                },
                Annotation {
                    id: 2,
                    image_id: 1,
                    category_id: 2,
                    bbox: Some([30.0, 30.0, 10.0, 10.0]),
                    area: Some(100.0),
                    ..Default::default()
                },
                Annotation {
                    id: 3,
                    image_id: 2,
                    category_id: 1,
                    bbox: Some([0.0, 0.0, 50.0, 50.0]),
                    area: Some(2500.0),
                    iscrowd: true,
                    ..Default::default()
                },
            ],
            categories: vec![
                Category {
                    id: 1,
                    name: "cat".into(),
                    supercategory: Some("animal".into()),
                    ..Default::default()
                },
                Category {
                    id: 2,
                    name: "dog".into(),
                    supercategory: Some("animal".into()),
                    ..Default::default()
                },
            ],
            licenses: vec![],
        }
    }

    #[test]
    fn test_create_index() {
        let coco = COCO::from_dataset(make_test_dataset());
        assert!((1..=3).all(|id| coco.get_ann(id).is_some()));
        assert!(coco.get_ann(4).is_none());
        assert_eq!(coco.imgs.len(), 2);
        assert_eq!(coco.cats.len(), 2);
    }

    /// A box result's segmentation is the box itself; it must rasterize to
    /// the same RLE as the four-corner polygon pycocotools' `loadRes` builds.
    #[test]
    fn a_box_result_masks_like_its_corner_polygon() {
        let gt = COCO::from_dataset(make_test_dataset());
        let bbox = [10.5, 20.25, 30.0, 40.125];
        let res = gt
            .load_res_anns(vec![Annotation {
                image_id: 1,
                category_id: 1,
                bbox: Some(bbox),
                score: Some(0.9),
                ..Default::default()
            }])
            .unwrap();
        let ann = res.get_ann(1).unwrap();
        assert!(matches!(ann.segmentation, Some(Segmentation::Rect(b)) if b == bbox));
        assert_eq!(ann.area, Some(bbox[2] * bbox[3]));

        let polygon = Annotation {
            segmentation: Some(Segmentation::Polygon(vec![
                Segmentation::rect_corners(&bbox).to_vec(),
            ])),
            ..ann.clone()
        };
        assert_eq!(
            res.ann_to_rle(ann).unwrap(),
            res.ann_to_rle(&polygon).unwrap()
        );
    }

    /// `cat_to_imgs` must be derived from the distinct `(img, cat)` pairs, not
    /// pushed once per annotation — img1/cat1 has three annotations (ids given
    /// out of JSON order) and must collapse to one `cat_to_imgs` entry, while
    /// the pair index for that pair must keep every id, in dataset order.
    #[test]
    fn test_cat_to_imgs_derived_from_pair_keys() {
        let dataset = Dataset {
            info: None,
            images: vec![
                Image {
                    id: 1,
                    file_name: "img1.jpg".into(),
                    height: 100,
                    width: 100,
                    ..Default::default()
                },
                Image {
                    id: 2,
                    file_name: "img2.jpg".into(),
                    height: 100,
                    width: 100,
                    ..Default::default()
                },
            ],
            annotations: vec![
                // img1/cat1, three annotations, ids given in descending order —
                // dataset (JSON array) order is 30, 20, 10, not ascending.
                Annotation {
                    id: 30,
                    image_id: 1,
                    category_id: 1,
                    ..Default::default()
                },
                Annotation {
                    id: 20,
                    image_id: 1,
                    category_id: 1,
                    ..Default::default()
                },
                Annotation {
                    id: 10,
                    image_id: 1,
                    category_id: 1,
                    ..Default::default()
                },
                // img1/cat2, one annotation
                Annotation {
                    id: 40,
                    image_id: 1,
                    category_id: 2,
                    ..Default::default()
                },
                // img2/cat1, one annotation
                Annotation {
                    id: 50,
                    image_id: 2,
                    category_id: 1,
                    ..Default::default()
                },
            ],
            categories: vec![
                Category {
                    id: 1,
                    name: "cat".into(),
                    ..Default::default()
                },
                Category {
                    id: 2,
                    name: "dog".into(),
                    ..Default::default()
                },
            ],
            licenses: vec![],
        };
        let coco = COCO::from_dataset(dataset);

        // (1) cat_to_imgs: exact membership, three img1/cat1 annotations
        // collapse to one entry, not three.
        assert_eq!(coco.cat_to_imgs.get(&1), Some(&vec![1, 2]));
        assert_eq!(coco.cat_to_imgs.get(&2), Some(&vec![1]));

        // (2) the pair index keeps every id, in dataset (JSON array) order —
        // not sorted ascending, not deduplicated by anything upstream.
        assert_eq!(
            coco.get_ann_ids_for_img_cat(1, 1),
            &[30, 20, 10],
            "must preserve dataset order, the greedy tie-break's visibility contract"
        );

        // (3) the one externally observable consumer of cat_to_imgs's length.
        let stats = coco.stats();
        let cat1 = stats
            .per_category
            .iter()
            .find(|c| c.id == 1)
            .expect("category 1 present");
        assert_eq!(
            cat1.img_count, 2,
            "cat 1 appears on img1 and img2, once each"
        );
    }

    /// `COCOeval`'s constructor now copies an already-indexed `COCO` (`.clone()`)
    /// instead of cloning the dataset and rebuilding the index from scratch —
    /// safe only because every `PyCOCO` write path keeps `inner`'s index in
    /// lockstep with `inner.dataset` (see `lib.rs`'s ctor comment). This pins
    /// that a clone is a faithful stand-in for a fresh `create_index()` pass:
    /// same six index maps, same warnings, and independent afterward.
    #[test]
    fn test_clone_is_a_faithful_reindex() {
        let mut dataset = make_test_dataset();
        // Reuse an existing id so `create_index` records a duplicate-id
        // warning, giving `warnings` something to compare.
        dataset.annotations.push(Annotation {
            id: 3,
            image_id: 2,
            category_id: 1,
            bbox: Some([5.0, 5.0, 5.0, 5.0]),
            area: Some(25.0),
            ..Default::default()
        });

        let a = COCO::from_dataset(dataset.clone());
        let mut b = a.clone();
        let c = COCO::from_dataset(dataset);

        assert_eq!(b.anns, c.anns);
        assert_eq!(b.imgs, c.imgs);
        assert_eq!(b.cats, c.cats);
        assert_eq!(b.cat_to_imgs, c.cat_to_imgs);
        assert_eq!(b.index, c.index);
        assert!(
            !c.warnings.is_empty(),
            "fixture must trigger a duplicate-id warning"
        );
        assert_eq!(b.warnings, c.warnings);

        // The evaluator's copy must be a snapshot, not a shared view: mutating
        // it and re-indexing must leave the original untouched.
        b.dataset.annotations.truncate(1);
        b.create_index();
        assert_eq!(a.anns, c.anns);
        assert_eq!(a.index, c.index);
    }

    #[test]
    fn test_get_ann_ids_by_img() {
        let coco = COCO::from_dataset(make_test_dataset());
        let ids = coco.get_ann_ids(&[1], &[], None, None);
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_get_ann_ids_by_cat() {
        let coco = COCO::from_dataset(make_test_dataset());
        let ids = coco.get_ann_ids(&[], &[1], None, None);
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn test_get_ann_ids_by_crowd() {
        let coco = COCO::from_dataset(make_test_dataset());
        let ids = coco.get_ann_ids(&[], &[], None, Some(true));
        assert_eq!(ids, vec![3]);
    }

    #[test]
    fn test_get_cat_ids() {
        let coco = COCO::from_dataset(make_test_dataset());
        let ids = coco.get_cat_ids(&["cat"], &[], &[]);
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn test_get_img_ids() {
        let coco = COCO::from_dataset(make_test_dataset());
        let ids = coco.get_img_ids(&[], &[1]);
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_get_img_ids_by_cat2() {
        let coco = COCO::from_dataset(make_test_dataset());
        let ids = coco.get_img_ids(&[], &[2]);
        assert_eq!(ids, vec![1]);
    }
}
