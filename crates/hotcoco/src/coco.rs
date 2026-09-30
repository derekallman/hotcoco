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
}

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

/// Inclusive area-range predicate shared by [`COCO::get_ann_ids`] and
/// [`COCO::filter`] — the one owner of the missing-`area` convention.
///
/// An annotation without an `area` value never matches an explicit range.
/// pycocotools' `getAnnIds` reads `ann['area']` unconditionally and raises
/// `KeyError` on a missing key; a filter cannot raise, so exclusion is the
/// closest faithful behavior (it never fabricates an area of 0.0, which used
/// to make area-less annotations match any range starting at 0).
fn area_in_range(ann: &Annotation, rng: [f64; 2]) -> bool {
    ann.area.is_some_and(|a| a >= rng[0] && a <= rng[1])
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
        let mut targets = Vec::with_capacity(anns.len());
        let mut missing = Vec::new();
        for ann in &anns {
            match self.anns.position(ann.id) {
                Some(i) => targets.push(i),
                None => missing.push(ann.id),
            }
        }
        if !missing.is_empty() {
            return Err(Error::UnknownAnnIds(missing));
        }

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

    /// Returns (img_id, cat_id) pairs that have at least one annotation.
    ///
    /// Used by COCOeval to enumerate only non-empty pairs instead of the full
    /// Cartesian product, which is critical for large-scale datasets.
    pub fn nonempty_img_cat_pairs(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.index.pairs()
    }

    /// Returns image IDs that have at least one annotation (any category).
    ///
    /// Used by COCOeval when `use_cats = false` (all categories treated as one).
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
        let mut warnings = Vec::new();
        let mut img_mismatch_warned = false;
        let mut cat_mismatch_warned = false;

        // Validate and assign ids in one pass, probing the index maps this
        // `COCO` already built (`self.imgs`/`self.cats`) rather than building a
        // `HashSet` of GT ids per call. `COCO::from_dataset` below walks `anns`
        // again to build the result's own index; that walk is `create_index`'s.
        for (i, ann) in anns.iter_mut().enumerate() {
            // A NaN score is rejected rather than warned about: it corrupts the
            // whole run, not one annotation. Every ranking path sorts with
            // `partial_cmp(..).unwrap_or(Equal)`, which is not transitive once NaN
            // is present — the sort silently produces an arbitrary order, so AP
            // becomes a function of the sort implementation. (`healthcheck`
            // reports the same condition as an error and points here.)
            if ann.score.is_some_and(f64::is_nan) {
                return Err(format!(
                    "load_res(): annotation {} (id {}, image_id {}) has a NaN score. \
                     Scores order the detection ranking, and NaN makes that order \
                     undefined — every metric downstream would be meaningless. Filter \
                     or repair these detections before evaluating.",
                    i, ann.id, ann.image_id
                )
                .into());
            }

            // Warn on the first annotation whose image_id or category_id isn't in
            // the GT — a common mistake that causes DTs to silently produce
            // misleadingly low metrics.
            if !img_mismatch_warned && !self.imgs.contains_key(&ann.image_id) {
                warnings.push(format!(
                    "load_res() warning — found annotation with image_id {} not in the \
                     GT dataset. These DTs will never match. Check your results file matches the \
                     correct GT split.",
                    ann.image_id
                ));
                img_mismatch_warned = true;
            }
            if has_cats && !cat_mismatch_warned && !self.cats.contains_key(&ann.category_id) {
                warnings.push(format!(
                    "load_res() warning — found annotation with category_id {} not \
                     in the GT dataset. These DTs will never match.",
                    ann.category_id
                ));
                cat_mismatch_warned = true;
            }

            // Assign IDs to result annotations (1-indexed, unconditional like pycocotools)
            ann.id = (i + 1) as u64;
        }

        if let Some(kind) = kind {
            // Per annotation and, for masks, an RLE decode each: the one
            // expensive step of loading results, so it runs in parallel.
            anns.par_iter_mut().for_each(|ann| {
                // Detection results are never crowd regions, whatever the input
                // file claimed.
                ann.iscrowd = false;
                match kind {
                    ResultKind::Bbox => Self::derive_from_bbox(ann),
                    ResultKind::Segm => self.derive_from_segmentation(ann),
                    ResultKind::Keypoints => Self::derive_from_keypoints(ann),
                    ResultKind::Obb => Self::derive_from_obb(ann),
                }
            });
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
        if !matches!(ann.segmentation, Some(Segmentation::CompressedRle { .. })) {
            return;
        }
        let Some(rle) = self.ann_to_rle(ann) else {
            return;
        };
        ann.area = Some(mask::area(&rle) as f64);
        if ann.bbox.is_none() {
            ann.bbox = Some(mask::to_bbox(&rle));
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
        let Some(kpts) = ann.keypoints.as_ref() else {
            return;
        };
        if kpts.len() < 2 {
            return;
        }
        // Keypoints are flat (x, y, visibility) triples.
        let extent = |offset: usize| {
            kpts.iter()
                .skip(offset)
                .step_by(3)
                .copied()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), v| {
                    (mn.min(v), mx.max(v))
                })
        };
        let (x0, x1) = extent(0);
        let (y0, y1) = extent(1);
        ann.area = Some((x1 - x0) * (y1 - y0));
        ann.bbox = Some([x0, y0, x1 - x0, y1 - y0]);
    }

    /// Area from the rotated box, and its axis-aligned envelope as the bbox.
    fn derive_from_obb(ann: &mut Annotation) {
        let Some(obb) = ann.obb.as_deref() else {
            return;
        };
        ann.area = Some(obb[2] * obb[3]);
        ann.bbox = Some(crate::geometry::obb_to_aabb(obb));
    }

    /// Convert an annotation's segmentation to RLE.
    pub fn ann_to_rle(&self, ann: &Annotation) -> Option<Rle> {
        let img = self.get_img(ann.image_id)?;
        let h = img.height;
        let w = img.width;

        match &ann.segmentation {
            Some(Segmentation::Polygon(polys)) => mask::fr_polys(polys, h, w).ok(),
            Some(Segmentation::Rect(bbox)) => {
                mask::fr_poly(&Segmentation::rect_corners(bbox), h, w).ok()
            }
            Some(Segmentation::CompressedRle { size, counts }) => {
                mask::rle_from_string(counts, size[0], size[1]).ok()
            }
            Some(Segmentation::UncompressedRle { size, counts }) => {
                // Same untrusted boundary as the compressed form, same
                // validation: counts must fit the image (`rle_from_string`
                // checks this for compressed input).
                let total: u64 = counts.iter().map(|&c| c as u64).sum();
                if total > size[0] as u64 * size[1] as u64 {
                    return None;
                }
                Some(Rle {
                    h: size[0],
                    w: size[1],
                    counts: counts.clone(),
                })
            }
            None => {
                // For bbox-only annotations, convert bbox to RLE
                ann.bbox
                    .as_ref()
                    .and_then(|bb| mask::fr_bbox(bb, h, w).ok())
            }
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
