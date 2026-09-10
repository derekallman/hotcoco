//! COCO dataset loading and querying API.
//!
//! Faithful port of `pycocotools/coco.py`.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::error::UnknownAnnIds;
use crate::mask;
use crate::types::{Annotation, Category, Dataset, Image, Rle, Segmentation};

/// The COCO dataset API for loading, querying, and indexing annotations.
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
    anns: HashMap<u64, usize>,
    /// img_id -> index into dataset.images
    imgs: HashMap<u64, usize>,
    /// cat_id -> index into dataset.categories
    cats: HashMap<u64, usize>,
    /// img_id -> [ann_id, ...]
    img_to_anns: HashMap<u64, Vec<u64>>,
    /// cat_id -> [img_id, ...] (unique)
    /// `pub(crate)` so `quality::stats` can read it — `COCO::stats` lives there,
    /// since dataset statistics are introspection output rather than schema.
    pub(crate) cat_to_imgs: HashMap<u64, Vec<u64>>,
    /// (img_id, cat_id) -> [ann_id, ...] in JSON array order.
    ///
    /// Deliberately *not* sorted by id: pycocotools builds `_gts` by iterating
    /// `dataset['annotations']` once, so array order is what feeds the matcher,
    /// and the greedy tie-break (`>=`, later GT wins on equal IoU) makes that
    /// order observable through `evalImgs`. Official COCO files are id-ordered
    /// anyway; converted or merged files are where the two orders differ.
    img_cat_to_anns: HashMap<(u64, u64), Vec<u64>>,
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

/// Normalize non-finite JSON float tokens (`NaN`, `Infinity`, `-Infinity`) to
/// `null`, matching the leniency of Python's `json` module.
///
/// Python emits these bare tokens by default and reads them back, so files
/// produced by pycocotools / numpy pipelines frequently contain them, even
/// though they are not valid JSON. serde_json (correctly) rejects them. To load
/// such files, each non-finite token is rewritten to `null` — which serde also
/// uses when *serializing* a non-finite `f64` — but only when the token appears
/// outside a JSON string, so string values that merely contain the substring
/// `"NaN"`/`"Infinity"` — a file name, say — are left untouched. On `Option<f64>`
/// fields (`area`, `score`) the `null` deserializes to `None`.
///
/// Returns the input unchanged and borrowed (no allocation) when it contains no
/// such tokens, so the common case pays only a single linear scan. The second
/// element is the number of tokens rewritten.
fn sanitize_non_finite(input: &[u8]) -> (Cow<'_, [u8]>, usize) {
    // Prefilter: if the tokens never occur as substrings *anywhere* — even
    // inside strings, where they would not count — the scan below cannot
    // rewrite anything. Two SIMD substring searches cost ~1ms on a 19 MB
    // file; the byte-at-a-time state machine they skip cost ~24ms, paid on
    // every load of a clean file, which is nearly every load. ("-Infinity"
    // contains "Infinity", so two needles cover all three tokens.)
    if memchr::memmem::find(input, b"NaN").is_none()
        && memchr::memmem::find(input, b"Infinity").is_none()
    {
        return (Cow::Borrowed(input), 0);
    }

    let n = input.len();
    let mut out: Option<Vec<u8>> = None;
    let mut count = 0usize;
    let mut in_string = false;
    let mut i = 0;

    while i < n {
        let b = input[i];

        if in_string {
            if b == b'\\' {
                // Copy the backslash and the escaped byte verbatim so an
                // escaped quote (`\"`) does not toggle the string state.
                if let Some(o) = out.as_mut() {
                    o.push(b);
                    if i + 1 < n {
                        o.push(input[i + 1]);
                    }
                }
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            if let Some(o) = out.as_mut() {
                o.push(b);
            }
            i += 1;
            continue;
        }

        if b == b'"' {
            in_string = true;
            if let Some(o) = out.as_mut() {
                o.push(b);
            }
            i += 1;
            continue;
        }

        // Outside a string, the only bare identifier-like tokens are
        // true/false/null and the non-finite floats we rewrite here. Gate the
        // substring comparisons on the first byte so the common case (digits,
        // punctuation, whitespace) skips them entirely.
        let token_len = match b {
            b'N' if input[i..].starts_with(b"NaN") => Some(3),
            b'I' if input[i..].starts_with(b"Infinity") => Some(8),
            b'-' if input[i..].starts_with(b"-Infinity") => Some(9),
            _ => None,
        };

        if let Some(len) = token_len {
            let o = out.get_or_insert_with(|| {
                let mut v = Vec::with_capacity(n);
                v.extend_from_slice(&input[..i]);
                v
            });
            o.extend_from_slice(b"null");
            count += 1;
            i += len;
            continue;
        }

        if let Some(o) = out.as_mut() {
            o.push(b);
        }
        i += 1;
    }

    match out {
        Some(v) => (Cow::Owned(v), count),
        None => (Cow::Borrowed(input), count),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod sanitize_tests {
    use super::sanitize_non_finite;

    fn run(s: &str) -> (String, usize) {
        let (bytes, n) = sanitize_non_finite(s.as_bytes());
        (String::from_utf8(bytes.into_owned()).unwrap(), n)
    }

    #[test]
    fn clean_input_is_borrowed_unchanged() {
        let input = br#"{"a": [1.0, -2.5], "b": null}"#;
        let (bytes, n) = sanitize_non_finite(input);
        assert_eq!(n, 0);
        assert!(matches!(bytes, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn rewrites_the_non_finite_family() {
        let (out, n) = run(r#"{"a": NaN, "b": Infinity, "c": -Infinity, "d": -3.5}"#);
        assert_eq!(n, 3);
        // -3.5 (a real negative number) must be preserved, not mangled.
        assert_eq!(out, r#"{"a": null, "b": null, "c": null, "d": -3.5}"#);
    }

    #[test]
    fn leaves_non_finite_substrings_inside_strings_alone() {
        // Strings containing the tokens — including an escaped quote — untouched.
        let (out, n) = run(r#"{"name": "NaN and \"Infinity\"", "v": NaN}"#);
        assert_eq!(n, 1);
        assert_eq!(out, r#"{"name": "NaN and \"Infinity\"", "v": null}"#);
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
        let raw = std::fs::read(annotation_file)?;
        let (mut bytes, n_fixed) = Self::sanitize_owned(raw);
        let dataset: Dataset = simd_json::serde::from_slice(&mut bytes)?;
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

    /// [`sanitize_non_finite`] over an owned buffer: hands the original buffer
    /// back untouched when the input is clean, so the common case pays no copy.
    /// Owned because `simd_json` parses in place and needs `&mut` bytes.
    fn sanitize_owned(raw: Vec<u8>) -> (Vec<u8>, usize) {
        match sanitize_non_finite(&raw) {
            (Cow::Owned(fixed), n) => (fixed, n),
            (Cow::Borrowed(_), n) => (raw, n),
        }
    }

    /// Build a COCO object from an already-loaded Dataset.
    pub fn from_dataset(dataset: Dataset) -> Self {
        let mut coco = COCO {
            dataset,
            warnings: Vec::new(),
            anns: HashMap::new(),
            imgs: HashMap::new(),
            cats: HashMap::new(),
            img_to_anns: HashMap::new(),
            cat_to_imgs: HashMap::new(),
            img_cat_to_anns: HashMap::new(),
        };
        coco.create_index();
        coco
    }

    /// Rebuild the internal query indices from `dataset`, and report duplicate
    /// annotation ids.
    ///
    /// Call this after mutating [`dataset`](Self::dataset) directly (the
    /// pycocotools `createIndex()` idiom) — the indices are snapshots, not
    /// views, and every `get_*`/`load_*` method answers from them.
    ///
    /// Duplicate annotation ids are indexed the way pycocotools does —
    /// last-write-wins in the id lookup, while per-image lists keep every
    /// occurrence — and reported via [`load_warnings`](Self::load_warnings).
    pub fn create_index(&mut self) {
        if let Some((count, first)) = self.rebuild_index() {
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
    }

    /// The index rebuild alone, returning `(duplicate count, first duplicate
    /// id)` instead of reporting it.
    ///
    /// Split out because reporting is a *load-time* event: it prints to stderr
    /// and appends to [`load_warnings`](Self::load_warnings), which is right
    /// once per dataset and wrong once per edit. [`update_anns`](Self::update_anns)
    /// re-indexes as often as a caller edits, so it rebuilds through here and
    /// leaves the warning to the load and to explicit `create_index()` calls.
    fn rebuild_index(&mut self) -> Option<(usize, u64)> {
        let n_anns = self.dataset.annotations.len();
        let n_imgs = self.dataset.images.len();
        let n_cats = self.dataset.categories.len();

        self.anns.clear();
        self.anns.reserve(n_anns);
        self.imgs.clear();
        self.imgs.reserve(n_imgs);
        self.cats.clear();
        self.cats.reserve(n_cats);
        self.img_to_anns.clear();
        self.img_to_anns.reserve(n_imgs);
        self.cat_to_imgs.clear();
        self.cat_to_imgs.reserve(n_cats);
        self.img_cat_to_anns.clear();
        self.img_cat_to_anns.reserve(n_anns);

        // Single pass over annotations: build all annotation-derived indices at once
        let mut dup_ann_ids = 0usize;
        let mut first_dup: Option<u64> = None;
        for (i, ann) in self.dataset.annotations.iter().enumerate() {
            if self.anns.insert(ann.id, i).is_some() {
                dup_ann_ids += 1;
                first_dup.get_or_insert(ann.id);
            }
            self.img_to_anns
                .entry(ann.image_id)
                .or_default()
                .push(ann.id);
            self.img_cat_to_anns
                .entry((ann.image_id, ann.category_id))
                .or_default()
                .push(ann.id);
            self.cat_to_imgs
                .entry(ann.category_id)
                .or_default()
                .push(ann.image_id);
        }

        for (i, img) in self.dataset.images.iter().enumerate() {
            self.imgs.insert(img.id, i);
        }

        for (i, cat) in self.dataset.categories.iter().enumerate() {
            self.cats.insert(cat.id, i);
        }

        // Deduplicate cat_to_imgs (multiple annotations per image produce duplicates)
        for ids in self.cat_to_imgs.values_mut() {
            ids.sort_unstable();
            ids.dedup();
        }
        // img_cat_to_anns stays in JSON array order — see the field doc.

        first_dup.map(|id| (dup_ann_ids, id))
    }

    /// Replace annotations by id, re-indexing only when the replacement moves
    /// one.
    ///
    /// The targeted counterpart to replacing the whole
    /// [`dataset`](Self::dataset): each annotation in `anns` overwrites the one
    /// that carries the same `id`. Ids do not move, so the id lookup survives
    /// untouched; the per-image and per-category indices are rebuilt only if a
    /// replacement changes an `image_id` or a `category_id`, which is what they
    /// key on. Editing `area` across a whole dataset — the multi-IoU-type case —
    /// therefore costs one pass over `anns`, not one over the dataset.
    ///
    /// Every id is checked before anything is written: an id that is not in the
    /// dataset returns [`UnknownAnnIds`] and leaves the dataset untouched, so a
    /// partial update never happens. Silently skipping unknown ids would
    /// reproduce the no-op that this method exists to remove.
    ///
    /// In a dataset with duplicate annotation ids, the id lookup holds the
    /// *last* occurrence (pycocotools parity — see
    /// [`create_index`](Self::create_index)), so that is the one replaced.
    pub fn update_anns(&mut self, anns: Vec<Annotation>) -> std::result::Result<(), UnknownAnnIds> {
        let mut targets = Vec::with_capacity(anns.len());
        let mut missing = Vec::new();
        for ann in &anns {
            match self.anns.get(&ann.id) {
                Some(&i) => targets.push(i),
                None => missing.push(ann.id),
            }
        }
        if !missing.is_empty() {
            return Err(UnknownAnnIds(missing));
        }

        let mut moved = false;
        for (i, ann) in targets.into_iter().zip(anns) {
            let old = &self.dataset.annotations[i];
            moved |= old.image_id != ann.image_id || old.category_id != ann.category_id;
            self.dataset.annotations[i] = ann;
        }
        if moved {
            self.rebuild_index();
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
                // Borrowed, not cloned: the index already holds one `Vec` per
                // image, and cloning it per key allocated and dropped the whole
                // list again just to walk it.
                .flat_map(|id| self.img_to_anns.get(id).map_or(&[][..], Vec::as_slice))
                .filter_map(|id| self.anns.get(id).map(|&i| &self.dataset.annotations[i]))
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
        ids.iter()
            .filter_map(|id| self.anns.get(id).map(|&i| &self.dataset.annotations[i]))
            .collect()
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
        self.anns.get(&id).map(|&i| &self.dataset.annotations[i])
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
            .map_or_else(|| format!("cat_{id}"), |c| c.name.clone())
    }

    /// Get annotation IDs for a specific (image, category) pair.
    ///
    /// Single HashMap lookup — much faster than `get_ann_ids` with filtering.
    pub fn get_ann_ids_for_img_cat(&self, img_id: u64, cat_id: u64) -> &[u64] {
        self.img_cat_to_anns
            .get(&(img_id, cat_id))
            .map_or(&[], std::vec::Vec::as_slice)
    }

    /// Get annotation IDs for a specific image.
    pub fn get_ann_ids_for_img(&self, img_id: u64) -> &[u64] {
        self.img_to_anns
            .get(&img_id)
            .map_or(&[], std::vec::Vec::as_slice)
    }

    /// Returns (img_id, cat_id) pairs that have at least one annotation.
    ///
    /// Used by COCOeval to enumerate only non-empty pairs instead of the full
    /// Cartesian product, which is critical for large-scale datasets.
    pub fn nonempty_img_cat_pairs(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.img_cat_to_anns.keys().copied()
    }

    /// Returns image IDs that have at least one annotation (any category).
    ///
    /// Used by COCOeval when `use_cats = false` (all categories treated as one).
    pub fn nonempty_img_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.img_to_anns.keys().copied()
    }

    /// Load detection/result annotations into a new COCO object.
    ///
    /// The result file can be a JSON array of annotation dicts, or a JSON object
    /// with an `annotations` field. The result COCO object shares the images
    /// and categories from self.
    pub fn load_res(&self, res_file: &Path) -> crate::error::Result<COCO> {
        let raw = std::fs::read(res_file)?;
        let (mut bytes, n_fixed) = Self::sanitize_owned(raw);

        // The shape is decided by the first non-whitespace byte rather than by
        // try-parse-then-fallback: simd-json parses in place (it unescapes
        // strings into the buffer as it goes), so a failed first attempt would
        // leave the buffer unusable for a second one.
        let is_array = bytes
            .iter()
            .find(|b| !b.is_ascii_whitespace())
            .is_some_and(|&b| b == b'[');
        let anns: Vec<Annotation> = if is_array {
            simd_json::serde::from_slice(&mut bytes)?
        } else {
            let ds: Dataset = simd_json::serde::from_slice(&mut bytes)?;
            ds.annotations
        };

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
    pub fn load_res_anns(&self, anns: Vec<Annotation>) -> crate::error::Result<COCO> {
        let warnings = self.validate_results(&anns)?;

        let mut dataset = Dataset {
            info: self.dataset.info.clone(),
            images: self.dataset.images.clone(),
            annotations: anns,
            categories: self.dataset.categories.clone(),
            licenses: self.dataset.licenses.clone(),
        };

        // One kind for the whole file, from the first annotation, as pycocotools
        // does — then fill in whatever geometry that kind implies.
        if let Some(kind) = dataset.annotations.first().and_then(ResultKind::of) {
            for ann in &mut dataset.annotations {
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
        }

        // Assign IDs to result annotations (1-indexed, unconditional like pycocotools)
        for (i, ann) in dataset.annotations.iter_mut().enumerate() {
            ann.id = (i + 1) as u64;
        }

        let mut res = COCO::from_dataset(dataset);
        for w in warnings {
            res.warn(w);
        }
        Ok(res)
    }

    /// Reject results that would make the run meaningless, and warn about ones
    /// that merely make it wrong.
    ///
    /// The split is deliberate: a mismatched id yields misleadingly low metrics
    /// that the user can still investigate, while a NaN score has no correct
    /// interpretation at all.
    ///
    /// Returns the warnings so the caller can attach them to the result `COCO`
    /// (they are emitted to stderr there, via [`warn`](Self::warn)).
    fn validate_results(&self, anns: &[Annotation]) -> crate::error::Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Warn on the first annotation whose image_id or category_id isn't in the GT —
        // a common mistake that causes DTs to silently produce misleadingly low metrics.
        let gt_img_ids: HashSet<u64> = self.dataset.images.iter().map(|i| i.id).collect();
        if let Some(ann) = anns.iter().find(|a| !gt_img_ids.contains(&a.image_id)) {
            warnings.push(format!(
                "load_res() warning — found annotation with image_id {} not in the \
                 GT dataset. These DTs will never match. Check your results file matches the \
                 correct GT split.",
                ann.image_id
            ));
        }

        if !self.dataset.categories.is_empty() {
            let gt_cat_ids: HashSet<u64> = self.dataset.categories.iter().map(|c| c.id).collect();
            if let Some(ann) = anns.iter().find(|a| !gt_cat_ids.contains(&a.category_id)) {
                warnings.push(format!(
                    "load_res() warning — found annotation with category_id {} not \
                     in the GT dataset. These DTs will never match.",
                    ann.category_id
                ));
            }
        }

        // A NaN score is rejected rather than warned about: it corrupts the whole
        // run, not one annotation. Every ranking path sorts with
        // `partial_cmp(..).unwrap_or(Equal)`, which is not transitive once NaN is
        // present — the sort silently produces an arbitrary order, so AP becomes
        // a function of the sort implementation. (`healthcheck` reports the same
        // condition as an error and points here.)
        if let Some((i, ann)) = anns
            .iter()
            .enumerate()
            .find(|(_, a)| a.score.is_some_and(f64::is_nan))
        {
            return Err(format!(
                "load_res(): annotation {} (id {}, image_id {}) has a NaN score. \
                 Scores order the detection ranking, and NaN makes that order \
                 undefined — every metric downstream would be meaningless. Filter \
                 or repair these detections before evaluating.",
                i, ann.id, ann.image_id
            )
            .into());
        }

        Ok(warnings)
    }

    /// Area from the box, and a rectangular segmentation when none was given.
    fn derive_from_bbox(ann: &mut Annotation) {
        let Some(bbox) = ann.bbox else {
            return;
        };
        ann.area = Some(bbox[2] * bbox[3]);
        if ann.segmentation.is_none() {
            let (x1, y1, bw, bh) = (bbox[0], bbox[1], bbox[2], bbox[3]);
            let (x2, y2) = (x1 + bw, y1 + bh);
            ann.segmentation = Some(Segmentation::Polygon(vec![vec![
                x1, y1, x1, y2, x2, y2, x2, y1,
            ]]));
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
        let Some(obb) = ann.obb else {
            return;
        };
        ann.area = Some(obb[2] * obb[3]);
        ann.bbox = Some(crate::geometry::obb_to_aabb(&obb));
    }

    /// Convert an annotation's segmentation to RLE.
    pub fn ann_to_rle(&self, ann: &Annotation) -> Option<Rle> {
        let img = self.get_img(ann.image_id)?;
        let h = img.height;
        let w = img.width;

        match &ann.segmentation {
            Some(Segmentation::Polygon(polys)) => mask::fr_polys(polys, h, w).ok(),
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
        assert_eq!(coco.anns.len(), 3);
        assert_eq!(coco.imgs.len(), 2);
        assert_eq!(coco.cats.len(), 2);
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
