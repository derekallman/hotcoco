//! COCO dataset loading and querying API.
//!
//! Faithful port of `pycocotools/coco.py`.

use std::collections::HashMap;
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
    pub fn new(annotation_file: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(annotation_file)?;
        let reader = std::io::BufReader::new(file);
        let dataset: Dataset = serde_json::from_reader(reader)?;
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
        Ok(coco)
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
        self.anns.clear();
        self.imgs.clear();
        self.cats.clear();
        self.img_to_anns.clear();
        self.cat_to_imgs.clear();
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
        let anns: Box<dyn Iterator<Item = &Annotation>> = if !img_ids.is_empty() {
            let ann_ids: Vec<u64> = img_ids
                .iter()
                .flat_map(|id| self.img_to_anns.get(id).cloned().unwrap_or_default())
                .collect();
            Box::new(
                ann_ids
                    .into_iter()
                    .filter_map(|id| self.anns.get(&id).map(|&i| &self.dataset.annotations[i])),
            )
        } else {
            Box::new(self.dataset.annotations.iter())
        };

        let mut result: Vec<u64> = anns
            .filter(|ann| {
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
            })
            .map(|ann| ann.id)
            .collect();
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
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get annotation IDs for a specific image.
    pub fn get_ann_ids_for_img(&self, img_id: u64) -> &[u64] {
        self.img_to_anns
            .get(&img_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Load detection/result annotations into a new COCO object.
    ///
    /// The result file can be a JSON array of annotation dicts, or a JSON object
    /// with an `annotations` field. The result COCO object shares the images
    /// and categories from self.
    pub fn load_res(&self, res_file: &Path) -> Result<COCO, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(res_file)?;
        let reader = std::io::BufReader::new(file);

        // Try to parse as array first, then as Dataset
        let anns: Vec<Annotation> = match serde_json::from_reader::<_, Vec<Annotation>>(reader) {
            Ok(a) => a,
            Err(_) => {
                let file = std::fs::File::open(res_file)?;
                let reader = std::io::BufReader::new(file);
                let ds: Dataset = serde_json::from_reader(reader)?;
                ds.annotations
            }
        };

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
                // segmentation results: area from mask RLE
                // Build a temporary COCO to use ann_to_rle
                let temp = COCO::from_dataset(Dataset {
                    info: dataset.info.clone(),
                    images: dataset.images.clone(),
                    annotations: dataset.annotations.clone(),
                    categories: dataset.categories.clone(),
                    licenses: dataset.licenses.clone(),
                });
                for ann in &mut dataset.annotations {
                    if let Some(crate::types::Segmentation::CompressedRle { .. }) = ann.segmentation
                    {
                        if let Some(rle) = temp.ann_to_rle(ann) {
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
                        let xs: Vec<f64> = kpts.iter().step_by(3).copied().collect();
                        let ys: Vec<f64> = kpts.iter().skip(1).step_by(3).copied().collect();
                        let x0 = xs.iter().copied().fold(f64::INFINITY, f64::min);
                        let x1 = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let y0 = ys.iter().copied().fold(f64::INFINITY, f64::min);
                        let y1 = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        ann.area = Some((x1 - x0) * (y1 - y0));
                        ann.bbox = Some([x0, y0, x1 - x0, y1 - y0]);
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

        let widths: Vec<f64> = self
            .dataset
            .images
            .iter()
            .map(|img| img.width as f64)
            .collect();
        let heights: Vec<f64> = self
            .dataset
            .images
            .iter()
            .map(|img| img.height as f64)
            .collect();

        let mut per_category: Vec<CategoryStats> = self
            .dataset
            .categories
            .iter()
            .map(|cat| CategoryStats {
                id: cat.id,
                name: cat.name.clone(),
                ann_count: cat_ann_counts.get(&cat.id).copied().unwrap_or(0),
                img_count: self.cat_to_imgs.get(&cat.id).map(|v| v.len()).unwrap_or(0),
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
            image_width: summary_stats(&widths),
            image_height: summary_stats(&heights),
            annotation_area: summary_stats(&areas),
        }
    }
}

fn summary_stats(values: &[f64]) -> SummaryStats {
    if values.is_empty() {
        return SummaryStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
        };
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = sorted[0];
    let max = *sorted.last().unwrap();
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    SummaryStats {
        min,
        max,
        mean,
        median,
    }
}

#[cfg(test)]
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
                    score: None,
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
                    score: None,
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
                    score: None,
                },
            ],
            categories: vec![
                Category {
                    id: 1,
                    name: "cat".into(),
                    supercategory: Some("animal".into()),
                    skeleton: None,
                    keypoints: None,
                },
                Category {
                    id: 2,
                    name: "dog".into(),
                    supercategory: Some("animal".into()),
                    skeleton: None,
                    keypoints: None,
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
