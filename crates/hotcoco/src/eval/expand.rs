//! GT/DT expansion for Open Images hierarchy-aware evaluation.
//!
//! Expands annotations up the category hierarchy so that a "Dog" detection
//! also counts as an "Animal" detection (if Animal is an ancestor of Dog).

use std::collections::HashSet;

use crate::types::{Annotation, Category, Dataset};
use crate::{COCO, Hierarchy};

/// Expand ground-truth annotations up the category hierarchy.
///
/// For each annotation, creates additional copies at every ancestor category
/// (excluding self). Deduplicates by `(image_id, bbox_bits, category_id)` so
/// pre-expanded inputs are idempotent. Adds virtual categories for any
/// hierarchy-only node IDs not already present in the dataset.
pub fn expand_gt(coco: &COCO, hierarchy: &Hierarchy) -> COCO {
    expand_annotations(coco, hierarchy)
}

/// Expand detection annotations up the category hierarchy.
///
/// Same logic as [`expand_gt`] — both GT and DT expansion use the same
/// ancestor-propagation strategy.
pub fn expand_dt(coco: &COCO, hierarchy: &Hierarchy) -> COCO {
    expand_annotations(coco, hierarchy)
}

fn expand_annotations(coco: &COCO, hierarchy: &Hierarchy) -> COCO {
    let mut seen: HashSet<(u64, [u64; 4], u64)> = HashSet::new();
    let mut expanded_anns: Vec<Annotation> = Vec::new();
    let mut next_id = coco
        .dataset
        .annotations
        .iter()
        .map(|a| a.id)
        .max()
        .unwrap_or(0)
        + 1;

    // Record existing annotations in `seen` and copy them to output
    for ann in &coco.dataset.annotations {
        let bbox_bits = bbox_to_bits(ann.bbox.unwrap_or([0.0; 4]));
        seen.insert((ann.image_id, bbox_bits, ann.category_id));
        expanded_anns.push(ann.clone());
    }

    // For each original annotation, add ancestor copies
    for ann in &coco.dataset.annotations {
        let ancestors = hierarchy.ancestors(ann.category_id);
        let bbox_bits = bbox_to_bits(ann.bbox.unwrap_or([0.0; 4]));

        for &ancestor_id in ancestors {
            if ancestor_id == ann.category_id {
                continue; // skip self
            }
            let key = (ann.image_id, bbox_bits, ancestor_id);
            if seen.contains(&key) {
                continue; // dedup
            }
            seen.insert(key);

            let mut new_ann = ann.clone();
            new_ann.id = next_id;
            new_ann.category_id = ancestor_id;
            next_id += 1;
            expanded_anns.push(new_ann);
        }
    }

    // Collect existing category IDs
    let existing_cat_ids: HashSet<u64> = coco.dataset.categories.iter().map(|c| c.id).collect();

    // Add virtual categories for hierarchy-only nodes
    let mut categories = coco.dataset.categories.clone();
    for &id in &hierarchy.all_ids() {
        if !existing_cat_ids.contains(&id) {
            categories.push(Category {
                id,
                name: hierarchy.name_of(id).unwrap_or("_unknown").to_string(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            });
        }
    }

    let dataset = Dataset {
        info: coco.dataset.info.clone(),
        images: coco.dataset.images.clone(),
        annotations: expanded_anns,
        categories,
        licenses: coco.dataset.licenses.clone(),
    };

    COCO::from_dataset(dataset)
}

fn bbox_to_bits(bbox: [f64; 4]) -> [u64; 4] {
    [
        bbox[0].to_bits(),
        bbox[1].to_bits(),
        bbox[2].to_bits(),
        bbox[3].to_bits(),
    ]
}
