//! COCO dataset loading and querying API.
//!
//! Faithful port of `pycocotools/coco.py`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::mask;
use crate::types::{
    Annotation, Category, CategoryStats, Dataset, DatasetStats, Image, Rle, Segmentation,
    SummaryStats,
};

/// The COCO dataset API for loading, querying, and indexing annotations.
pub struct COCO {
    pub dataset: Dataset,
    /// ann_id -> index into dataset.annotations
    anns: HashMap<u64, usize>,
    /// img_id -> index into dataset.images
    imgs: HashMap<u64, usize>,
    /// cat_id -> index into dataset.categories
    cats: HashMap<u64, usize>,
    /// img_id -> [ann_id, ...]
    img_to_anns: HashMap<u64, Vec<u64>>,
    /// cat_id -> [img_id, ...] (unique)
    cat_to_imgs: HashMap<u64, Vec<u64>>,
    /// (img_id, cat_id) -> [ann_id, ...] (sorted)
    img_cat_to_anns: HashMap<(u64, u64), Vec<u64>>,
}

impl COCO {
    /// Load a COCO annotation JSON file and build indices.
    pub fn new(annotation_file: &Path) -> crate::error::Result<Self> {
        let file = std::fs::File::open(annotation_file)?;
        let reader = std::io::BufReader::new(file);
        let dataset: Dataset = serde_json::from_reader(reader)?;
        Ok(Self::from_dataset(dataset))
    }

    /// Build a COCO object from an already-loaded Dataset.
    pub fn from_dataset(dataset: Dataset) -> Self {
        let mut coco = COCO {
            dataset,
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

    /// Build internal index structures from the dataset.
    fn create_index(&mut self) {
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

        // Single pass over annotations: build all annotation-derived indices at once
        for (i, ann) in self.dataset.annotations.iter().enumerate() {
            self.anns.insert(ann.id, i);
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
        // Sort img_cat_to_anns values
        for ids in self.img_cat_to_anns.values_mut() {
            ids.sort_unstable();
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
                let a = ann.area.unwrap_or(0.0);
                if a < rng[0] || a > rng[1] {
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
                .flat_map(|id| self.img_to_anns.get(id).cloned().unwrap_or_default())
                .filter_map(|id| self.anns.get(&id).map(|&i| &self.dataset.annotations[i]))
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
                .flat_map(|cid| self.cat_to_imgs.get(cid).cloned().unwrap_or_default())
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
        // Read once into memory so we can retry parsing without re-opening.
        let bytes = std::fs::read(res_file)?;

        // Try to parse as array first, then as Dataset.
        let anns: Vec<Annotation> = match serde_json::from_slice::<Vec<Annotation>>(&bytes) {
            Ok(a) => a,
            Err(_) => {
                let ds: Dataset = serde_json::from_slice(&bytes)?;
                ds.annotations
            }
        };

        self.load_res_anns(anns)
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
        // Warn on the first annotation whose image_id or category_id isn't in the GT —
        // a common mistake that causes DTs to silently produce misleadingly low metrics.
        let gt_img_ids: HashSet<u64> = self.dataset.images.iter().map(|i| i.id).collect();
        if let Some(ann) = anns.iter().find(|a| !gt_img_ids.contains(&a.image_id)) {
            eprintln!(
                "hotcoco: load_res() warning — found annotation with image_id {} not in the \
                 GT dataset. These DTs will never match. Check your results file matches the \
                 correct GT split.",
                ann.image_id
            );
        }

        if !self.dataset.categories.is_empty() {
            let gt_cat_ids: HashSet<u64> = self.dataset.categories.iter().map(|c| c.id).collect();
            if let Some(ann) = anns.iter().find(|a| !gt_cat_ids.contains(&a.category_id)) {
                eprintln!(
                    "hotcoco: load_res() warning — found annotation with category_id {} not \
                     in the GT dataset. These DTs will never match.",
                    ann.category_id
                );
            }
        }

        let mut dataset = Dataset {
            info: self.dataset.info.clone(),
            images: self.dataset.images.clone(),
            annotations: anns,
            categories: self.dataset.categories.clone(),
            licenses: self.dataset.licenses.clone(),
        };

        // Determine result type from first annotation (matching pycocotools loadRes logic)
        // Priority: bbox (if present and non-empty) > segmentation > keypoints
        if let Some(first) = dataset.annotations.first() {
            let has_bbox = first.bbox.is_some();
            let has_seg = first.segmentation.is_some();
            let has_kpts = first.keypoints.is_some();
            let has_obb = first.obb.is_some();

            if has_bbox {
                // bbox results: area = bbox w*h, create segmentation from bbox if missing
                for ann in &mut dataset.annotations {
                    if let Some(ref bbox) = ann.bbox {
                        ann.area = Some(bbox[2] * bbox[3]);
                        if ann.segmentation.is_none() {
                            let (x1, y1, bw, bh) = (bbox[0], bbox[1], bbox[2], bbox[3]);
                            let (x2, y2) = (x1 + bw, y1 + bh);
                            ann.segmentation = Some(crate::types::Segmentation::Polygon(vec![
                                vec![x1, y1, x1, y2, x2, y2, x2, y1],
                            ]));
                        }
                    }
                    ann.iscrowd = false;
                }
            } else if has_seg && !has_kpts {
                // segmentation results: area from mask RLE.
                // Only CompressedRle is handled here, matching pycocotools' loadRes behavior.
                // Polygon and UncompressedRle results are not expected in detection output files.
                // Use the GT COCO (self) for image lookups — detection results
                // share the same images, so self.get_img finds the right dims.
                for ann in &mut dataset.annotations {
                    if let Some(crate::types::Segmentation::CompressedRle { .. }) = ann.segmentation
                    {
                        if let Some(rle) = self.ann_to_rle(ann) {
                            ann.area = Some(mask::area(&rle) as f64);
                            if ann.bbox.is_none() {
                                let bb = mask::to_bbox(&rle);
                                ann.bbox = Some(bb);
                            }
                        }
                    }
                    ann.iscrowd = false;
                }
            } else if has_kpts {
                // keypoints results: area and bbox from keypoint extent
                for ann in &mut dataset.annotations {
                    if let Some(ref kpts) = ann.keypoints {
                        let (x0, x1) = kpts
                            .iter()
                            .step_by(3)
                            .copied()
                            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), v| {
                                (mn.min(v), mx.max(v))
                            });
                        let (y0, y1) = kpts
                            .iter()
                            .skip(1)
                            .step_by(3)
                            .copied()
                            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), v| {
                                (mn.min(v), mx.max(v))
                            });
                        ann.area = Some((x1 - x0) * (y1 - y0));
                        ann.bbox = Some([x0, y0, x1 - x0, y1 - y0]);
                    }
                    ann.iscrowd = false;
                }
            } else if has_obb {
                // OBB results: area = w*h, bbox = axis-aligned bounding box of rotated rect
                for ann in &mut dataset.annotations {
                    if let Some(ref obb) = ann.obb {
                        ann.area = Some(obb[2] * obb[3]);
                        ann.bbox = Some(crate::geometry::obb_to_aabb(obb));
                    }
                    ann.iscrowd = false;
                }
            }
        }

        // Assign IDs to result annotations (1-indexed, unconditional like pycocotools)
        for (i, ann) in dataset.annotations.iter_mut().enumerate() {
            ann.id = (i + 1) as u64;
        }

        Ok(COCO::from_dataset(dataset))
    }

    /// Convert an annotation's segmentation to RLE.
    pub fn ann_to_rle(&self, ann: &Annotation) -> Option<Rle> {
        let img = self.get_img(ann.image_id)?;
        let h = img.height;
        let w = img.width;

        match &ann.segmentation {
            Some(Segmentation::Polygon(polys)) => Some(mask::fr_polys(polys, h, w)),
            Some(Segmentation::CompressedRle { size, counts }) => {
                mask::rle_from_string(counts, size[0], size[1]).ok()
            }
            Some(Segmentation::UncompressedRle { size, counts }) => Some(Rle {
                h: size[0],
                w: size[1],
                counts: counts.clone(),
            }),
            None => {
                // For bbox-only annotations, convert bbox to RLE
                ann.bbox.as_ref().map(|bb| mask::fr_bbox(bb, h, w))
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
                    let a = ann.area.unwrap_or(0.0);
                    if a < rng[0] || a > rng[1] {
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

        for (i, ds) in datasets.iter().enumerate() {
            let img_offset = current_max_img_id;
            let ann_offset = current_max_ann_id;
            let cat_remap = &cat_remaps[i];

            let mut max_img_id = 0u64;
            for img in &ds.images {
                let mut new_img = img.clone();
                new_img.id = img.id + img_offset;
                all_images.push(new_img);
                max_img_id = max_img_id.max(img.id);
            }

            let mut max_ann_id = 0u64;
            for ann in &ds.annotations {
                let mut new_ann = ann.clone();
                new_ann.id = ann.id + ann_offset;
                new_ann.image_id = ann.image_id + img_offset;
                new_ann.category_id = *cat_remap.get(&ann.category_id).unwrap_or(&ann.category_id);
                all_anns.push(new_ann);
                max_ann_id = max_ann_id.max(ann.id);
            }

            current_max_img_id = max_img_id + img_offset;
            current_max_ann_id = max_ann_id + ann_offset;
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

    /// Compute dataset health-check statistics.
    pub fn stats(&self) -> DatasetStats {
        let mut cat_ann_counts: HashMap<u64, usize> = HashMap::new();
        let mut cat_crowd_counts: HashMap<u64, usize> = HashMap::new();
        let mut areas: Vec<f64> = Vec::new();
        let mut crowd_count = 0usize;

        for ann in &self.dataset.annotations {
            *cat_ann_counts.entry(ann.category_id).or_default() += 1;
            if ann.iscrowd {
                crowd_count += 1;
                *cat_crowd_counts.entry(ann.category_id).or_default() += 1;
            }
            if let Some(area) = ann.area {
                areas.push(area);
            }
        }

        let (widths, heights): (Vec<f64>, Vec<f64>) = self
            .dataset
            .images
            .iter()
            .map(|img| (img.width as f64, img.height as f64))
            .unzip();

        let mut per_category: Vec<CategoryStats> = self
            .dataset
            .categories
            .iter()
            .map(|cat| CategoryStats {
                id: cat.id,
                name: cat.name.clone(),
                ann_count: cat_ann_counts.get(&cat.id).copied().unwrap_or(0),
                img_count: self.cat_to_imgs.get(&cat.id).map_or(0, std::vec::Vec::len),
                crowd_count: cat_crowd_counts.get(&cat.id).copied().unwrap_or(0),
            })
            .collect();
        per_category.sort_by(|a, b| b.ann_count.cmp(&a.ann_count));

        DatasetStats {
            image_count: self.dataset.images.len(),
            annotation_count: self.dataset.annotations.len(),
            category_count: self.dataset.categories.len(),
            crowd_count,
            per_category,
            image_width: summary_stats(widths),
            image_height: summary_stats(heights),
            annotation_area: summary_stats(areas),
        }
    }

    /// Run a health check on this dataset.
    pub fn healthcheck(&self) -> crate::healthcheck::HealthReport {
        crate::healthcheck::healthcheck(&self.dataset)
    }

    /// Run a health check including GT/DT compatibility.
    pub fn healthcheck_compatibility(&self, dt: &COCO) -> crate::healthcheck::HealthReport {
        crate::healthcheck::healthcheck_compatibility(&self.dataset, &dt.dataset)
    }
}

fn summary_stats(mut values: Vec<f64>) -> SummaryStats {
    if values.is_empty() {
        return SummaryStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
        };
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sorted = values;
    let min = sorted[0];
    let max = *sorted.last().expect("non-empty after early return");
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    };
    SummaryStats {
        min,
        max,
        mean,
        median,
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
                    license: None,
                    coco_url: None,
                    flickr_url: None,
                    date_captured: None,
                    neg_category_ids: vec![],
                    not_exhaustive_category_ids: vec![],
                },
                Image {
                    id: 2,
                    file_name: "img2.jpg".into(),
                    height: 200,
                    width: 200,
                    license: None,
                    coco_url: None,
                    flickr_url: None,
                    date_captured: None,
                    neg_category_ids: vec![],
                    not_exhaustive_category_ids: vec![],
                },
            ],
            annotations: vec![
                Annotation {
                    id: 1,
                    image_id: 1,
                    category_id: 1,
                    bbox: Some([10.0, 10.0, 20.0, 20.0]),
                    area: Some(400.0),
                    segmentation: None,
                    iscrowd: false,
                    keypoints: None,
                    num_keypoints: None,
                    obb: None,
                    score: None,
                    is_group_of: None,
                },
                Annotation {
                    id: 2,
                    image_id: 1,
                    category_id: 2,
                    bbox: Some([30.0, 30.0, 10.0, 10.0]),
                    area: Some(100.0),
                    segmentation: None,
                    iscrowd: false,
                    keypoints: None,
                    num_keypoints: None,
                    obb: None,
                    score: None,
                    is_group_of: None,
                },
                Annotation {
                    id: 3,
                    image_id: 2,
                    category_id: 1,
                    bbox: Some([0.0, 0.0, 50.0, 50.0]),
                    area: Some(2500.0),
                    segmentation: None,
                    iscrowd: true,
                    keypoints: None,
                    num_keypoints: None,
                    obb: None,
                    score: None,
                    is_group_of: None,
                },
            ],
            categories: vec![
                Category {
                    id: 1,
                    name: "cat".into(),
                    supercategory: Some("animal".into()),
                    skeleton: None,
                    keypoints: None,
                    frequency: None,
                },
                Category {
                    id: 2,
                    name: "dog".into(),
                    supercategory: Some("animal".into()),
                    skeleton: None,
                    keypoints: None,
                    frequency: None,
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
