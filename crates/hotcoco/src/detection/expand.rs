//! GT/DT expansion for Open Images hierarchy-aware evaluation.
//!
//! Expands annotations up the category hierarchy so that a "Dog" detection
//! also counts as an "Animal" detection (if Animal is an ancestor of Dog).

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::types::{Annotation, Category, Dataset, Segmentation};
use crate::{COCO, Hierarchy};

/// Expand a dataset's annotations up the category hierarchy.
///
/// For each annotation, creates additional copies at every ancestor category
/// (excluding self). Adds virtual categories for any hierarchy-only node IDs
/// not already present in the dataset.
///
/// Each annotation expands on its own, as the Open Images protocol expands
/// each box: a cat detection at 0.3 and a dog detection at 0.9 on the same box
/// become two animal detections, at 0.3 and 0.9. A copy is skipped only when
/// an annotation identical to it — same image, category, score, and geometry
/// (box, oriented box, and segmentation) — already exists, which makes
/// expanding an expanded dataset a no-op: `evaluate()` expands the handles it
/// replaced, so a second `evaluate()` must not expand further. The cost is
/// that two children with the same score *and* the same geometry yield one
/// ancestor copy; nothing in an annotation tells a re-expanded copy from such a
/// twin.
///
/// One function for both sides: Open Images expands ground truth always and
/// detections when `params.expand_dt` is set, with the identical
/// ancestor-propagation strategy.
pub fn expand_annotations(coco: &COCO, hierarchy: &Hierarchy) -> COCO {
    let anns = &coco.dataset.annotations;
    // One digest per annotation, shared by every ancestor copy: only the
    // category varies between them.
    let digests: Vec<u64> = anns.iter().map(identity_digest).collect();
    let mut seen: HashSet<CopyKey> = HashSet::with_capacity(anns.len());
    let mut expanded_anns: Vec<Annotation> = Vec::with_capacity(anns.len());
    let mut next_id = anns.iter().map(|a| a.id).max().unwrap_or(0) + 1;

    // Record existing annotations in `seen` and copy them to output
    for (ann, &digest) in anns.iter().zip(&digests) {
        seen.insert((ann.image_id, ann.category_id, digest));
        expanded_anns.push(ann.clone());
    }

    // For each original annotation, add ancestor copies
    for (ann, &digest) in anns.iter().zip(&digests) {
        let ancestors = hierarchy.ancestors(ann.category_id);

        for &ancestor_id in ancestors {
            if ancestor_id == ann.category_id {
                continue; // skip self
            }
            if !seen.insert((ann.image_id, ancestor_id, digest)) {
                continue; // this copy already exists
            }

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
                ..Default::default()
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

/// What makes two annotations the same annotation for expansion: everything
/// but the id, as `(image_id, category_id, identity_digest)`. See
/// [`expand_annotations`] for why the score is part of it.
type CopyKey = (u64, u64, u64);

/// A digest of an annotation's score and geometry — box, oriented box, and
/// segmentation — computed once per annotation rather than once per ancestor
/// copy, so the set holds neither a second copy of every mask nor a re-hash of
/// it per ancestor. Two distinct annotations would also need the same image and
/// category to collide.
fn identity_digest(ann: &Annotation) -> u64 {
    let mut h = DefaultHasher::new();
    ann.score.map(f64::to_bits).hash(&mut h);
    ann.bbox.map(|b| b.map(f64::to_bits)).hash(&mut h);
    ann.obb.as_deref().map(|o| o.map(f64::to_bits)).hash(&mut h);
    match &ann.segmentation {
        None => 0u8.hash(&mut h),
        Some(seg) => {
            1u8.hash(&mut h);
            segmentation_digest(seg, &mut h);
        }
    }
    h.finish()
}

fn segmentation_digest(seg: &Segmentation, h: &mut DefaultHasher) {
    let floats = |h: &mut DefaultHasher, xs: &[f64]| {
        xs.len().hash(h);
        for x in xs {
            x.to_bits().hash(h);
        }
    };
    match seg {
        Segmentation::Polygon(polys) => {
            0u8.hash(h);
            polys.len().hash(h);
            for poly in polys {
                floats(h, poly);
            }
        }
        Segmentation::Rect(bbox) => {
            1u8.hash(h);
            floats(h, bbox);
        }
        Segmentation::CompressedRle { size, counts } => {
            2u8.hash(h);
            size.hash(h);
            counts.hash(h);
        }
        Segmentation::UncompressedRle { size, counts } => {
            3u8.hash(h);
            size.hash(h);
            counts.hash(h);
        }
    }
}
