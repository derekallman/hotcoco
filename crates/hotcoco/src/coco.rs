//! COCO dataset loading and querying API.
//!
//! Faithful port of `pycocotools/coco.py`.

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::ann_index::{AnnIndex, Duplicates, IdIndex};
use crate::error::Error;
use crate::mask;
use crate::metrics::counts::descending_score_key;
use crate::primitives::sim::worth_parallel;
use crate::types::{Annotation, Category, Dataset, Image, Rle, Segmentation};

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
#[derive(Clone, Copy)]
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

/// Inclusive area-range predicate shared by [`COCO::get_ann_ids`] and
/// [`COCO::filter`].
///
/// An annotation without an `area` value never matches an explicit range.
/// pycocotools' `getAnnIds` reads `ann['area']` unconditionally and raises
/// `KeyError` on a missing key; a filter cannot raise, so exclusion is the
/// closest faithful behavior (it never fabricates an area of 0.0, which used
/// to make area-less annotations match any range starting at 0). An evaluator
/// derives the area instead ([`COCO::fill_missing_areas`]), because it has to
/// place every annotation in an area range.
fn area_in_range(ann: &Annotation, rng: [f64; 2]) -> bool {
    ann.area.is_some_and(|a| a >= rng[0] && a <= rng[1])
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
            rayon::spawn(move || drop(annotations));
        }
    }
}

impl COCO {
    /// Load a COCO annotation JSON file and build indices.
    ///
    /// Non-fatal problems (non-finite floats normalized to `null`, duplicate
    /// annotation ids) are printed to stderr and retained on the returned
    /// object — see [`load_warnings`](Self::load_warnings).
    pub fn new(annotation_file: &Path) -> crate::error::Result<Self> {
        let (dataset, n_fixed) = crate::json::read_dataset(annotation_file)?;
        let mut coco = Self::from_dataset(dataset);
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
    /// Each entry was also printed to stderr at the moment it arose (so CLI
    /// behavior is unchanged); this accessor exists for callers — the Python
    /// bindings, notebooks, servers — where stderr is invisible. Empty for a
    /// clean load.
    pub fn load_warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Record a non-fatal problem: print it to stderr and retain it for
    /// [`load_warnings`](Self::load_warnings).
    fn warn(&mut self, msg: String) {
        eprintln!("hotcoco: {msg}");
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
            return Err(Error::from(format!(
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
                if !area_in_range(ann, rng) {
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

        let mut result: Vec<u64> = if !img_ids.is_empty() {
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
        };
        result.sort_unstable();
        result
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
        format!("cat_{id}")
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

    /// One image's annotation ids grouped by category, ascending, and in array
    /// order within a category.
    pub(crate) fn ann_ids_for_img_by_cat(&self, img_id: u64) -> &[u64] {
        self.index.for_img_by_cat(img_id)
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
        let (anns, n_fixed) = crate::json::read_results(res_file)?;
        let mut res = self.load_res_anns(anns)?;
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
    ///
    /// # Panics
    ///
    /// If `ncols` is not 6 or 7, or `rows` is not a whole number of rows.
    pub fn detections_from_rows(
        rows: &[f64],
        ncols: usize,
    ) -> crate::error::Result<Vec<Annotation>> {
        assert!(
            ncols == 6 || ncols == 7,
            "detections_from_rows: 6 or 7 columns, got {ncols}"
        );
        assert_eq!(rows.len() % ncols, 0, "detections_from_rows: a partial row");
        let valid = |v: f64| v.is_finite() && v >= 0.0 && v.fract() == 0.0;
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
        let parallel = worth_parallel(rows.len() / ncols);
        let bad = if parallel {
            rows.par_chunks_exact(ncols)
                .enumerate()
                .find_map_first(first_bad)
        } else {
            rows.chunks_exact(ncols).enumerate().find_map(first_bad)
        };
        if let Some((row, column, value)) = bad {
            return Err(Error::Other(format!(
                "row {row} has {column} {value}; ids must be finite, non-negative integers"
            )));
        }
        let detection = |row: &[f64]| Annotation {
            image_id: row[0] as u64,
            category_id: if ncols == 7 { row[6] as u64 } else { 1 },
            bbox: Some([row[1], row[2], row[3], row[4]]),
            score: Some(row[5]),
            ..Default::default()
        };
        Ok(if parallel {
            rows.par_chunks_exact(ncols).map(detection).collect()
        } else {
            rows.chunks_exact(ncols).map(detection).collect()
        })
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
        let has_cats = !self.cats.is_empty();

        // Validate in one read-only pass, probing the index maps this `COCO`
        // already built (`self.imgs`/`self.cats`) rather than building a
        // `HashSet` of GT ids per call, then assign ids and derive geometry in
        // one writing pass. Both fan out past the shared threshold; the
        // first offender of each kind is the one reported either way.
        // `COCO::from_dataset` below walks `anns` again to build the result's
        // own index; that walk is `create_index`'s.
        //
        // A NaN score is rejected rather than warned about: it corrupts the
        // whole run, not one annotation. NaN has no place in a score order: a
        // sort with `partial_cmp(..).unwrap_or(Equal)` is not transitive once
        // it is present, and the bit keys `accumulate()` sorts on put it
        // wherever its bits fall, so AP becomes a function of the sort
        // implementation. (`healthcheck` reports the same condition as an
        // error and points here.) An image_id or category_id not in the GT is
        // a common mistake that makes the detections silently score low, so
        // the first of each is a warning.
        let check = |i: usize, ann: &Annotation| {
            [
                ann.score.is_some_and(f64::is_nan).then_some(i),
                (!self.imgs.contains_key(&ann.image_id)).then_some(i),
                (has_cats && !self.cats.contains_key(&ann.category_id)).then_some(i),
            ]
        };
        let earliest = |a: [Option<usize>; 3], b: [Option<usize>; 3]| {
            std::array::from_fn(|j| match (a[j], b[j]) {
                (Some(x), Some(y)) => Some(x.min(y)),
                (x, y) => x.or(y),
            })
        };
        let [first_nan, first_img, first_cat] = if worth_parallel(anns.len()) {
            anns.par_iter()
                .enumerate()
                .map(|(i, ann)| check(i, ann))
                .reduce(|| [None; 3], earliest)
        } else {
            anns.iter()
                .enumerate()
                .map(|(i, ann)| check(i, ann))
                .fold([None; 3], earliest)
        };
        if let Some(i) = first_nan {
            let ann = &anns[i];
            return Err(format!(
                "load_res(): annotation {} (id {}, image_id {}) has a NaN score. \
                 Scores order the detection ranking, and NaN makes that order \
                 undefined — every metric downstream would be meaningless. Filter \
                 or repair these detections before evaluating.",
                i, ann.id, ann.image_id
            )
            .into());
        }
        let mut warnings = Vec::new();
        if let Some(i) = first_img {
            warnings.push(format!(
                "load_res() warning — found annotation with image_id {} not in the \
                 GT dataset. These DTs will never match. Check your results file matches the \
                 correct GT split.",
                anns[i].image_id
            ));
        }
        if let Some(i) = first_cat {
            warnings.push(format!(
                "load_res() warning — found annotation with category_id {} not \
                 in the GT dataset. These DTs will never match.",
                anns[i].category_id
            ));
        }

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
            anns.par_iter_mut().enumerate().for_each(fill);
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
                .positions(|ann| ann.area.is_none())
                .collect()
        } else {
            (0..anns.len())
                .filter(|&i| anns[i].area.is_none())
                .collect()
        };
        let derive = |&i: &usize| {
            let ann = &anns[i];
            let from_box = || ann.bbox.map(|bb| bb[2] * bb[3]);
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
            missing.par_iter().filter_map(derive).collect()
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
        ann.area = Some(bbox[2] * bbox[3]);
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
        Err(crate::error::Error::Other(format!(
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
    pub(crate) fn is_per_image_capped(&self) -> bool {
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
            Some(Segmentation::Polygon(polys)) => mask::fr_polys(polys, h, w).ok(),
            Some(Segmentation::Rect(bbox)) => mask::fr_bbox(bbox, h, w).ok(),
            // Either RLE spelling, validated the same way: a string that does
            // not decode, or runs past `h * w`, is no mask.
            Some(seg) => seg.rle_ref()?.to_rle().ok(),
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
                    if !area_in_range(ann, rng) {
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
                return Err(format!(
                    "Cannot merge: datasets have different numbers of categories ({} vs {})",
                    canonical_cats.len(),
                    ds.categories.len()
                )
                .into());
            }
            let mut remap = HashMap::new();
            for cat in &ds.categories {
                let key = (cat.name.clone(), cat.supercategory.clone());
                match canonical_key_to_id.get(&key) {
                    Some(&canonical_id) => {
                        remap.insert(cat.id, canonical_id);
                    }
                    None => {
                        return Err(format!(
                            "Cannot merge: category '{}' not found in first dataset",
                            cat.name
                        )
                        .into());
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
            crate::error::Error::from(format!(
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
    pub fn split(
        &self,
        val_frac: f64,
        test_frac: Option<f64>,
        seed: u64,
    ) -> (Dataset, Dataset, Option<Dataset>) {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;

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

        (train, val, test)
    }

    /// Sample a random subset of images (with their annotations).
    ///
    /// Provide either `n` (exact count) or `frac` (fraction of images).
    /// The sample is deterministic given the same `seed`.
    pub fn sample(&self, n: Option<usize>, frac: Option<f64>, seed: u64) -> Dataset {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;

        let total = self.dataset.images.len();
        let count = match (n, frac) {
            (Some(n), _) => n.min(total),
            (None, Some(f)) => ((total as f64 * f) as usize).min(total),
            (None, None) => total,
        };

        let mut img_ids: Vec<u64> = self.dataset.images.iter().map(|img| img.id).collect();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        img_ids.shuffle(&mut rng);

        self.subset_by_img_ids(&img_ids[..count])
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
