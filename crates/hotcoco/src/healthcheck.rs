//! Dataset healthcheck: 4-layer validation for COCO annotation files.
//!
//! Layers run in order, each catching progressively subtler issues:
//!
//! 1. **Structural** — duplicate IDs, orphaned references (errors)
//! 2. **Quality** — degenerate/zero-area bboxes, out-of-bounds, extreme aspect ratios,
//!    near-duplicates (warnings)
//! 3. **Distribution** — category imbalance, low/zero-instance categories (warnings)
//! 4. **Compatibility** — GT/DT image/category mismatches (requires detections)

use std::collections::{HashMap, HashSet};

use crate::types::Dataset;
use serde::Serialize;

/// Severity layer for a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Layer {
    Structural,
    Quality,
    Distribution,
    Compatibility,
}

/// A single issue found during healthcheck.
#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    /// Machine-readable code (e.g., "duplicate_ann_id", "degenerate_bbox").
    pub code: &'static str,
    /// Human-readable message with actionable guidance.
    pub message: String,
    /// IDs of the entity being checked. The `code` field disambiguates the ID kind
    /// (annotation IDs for annotation checks, image IDs for image checks, etc.).
    pub affected_ids: Vec<u64>,
    /// Which check layer produced this finding.
    pub layer: Layer,
}

/// Distribution summary focused on red flags.
#[derive(Debug, Clone, Serialize)]
pub struct DatasetSummary {
    pub num_images: usize,
    pub num_annotations: usize,
    pub num_categories: usize,
    pub images_without_annotations: usize,
    /// Category instance counts, sorted descending.
    pub category_counts: Vec<(String, usize)>,
    /// max / min among categories with at least 1 instance. 1.0 if all equal.
    pub imbalance_ratio: f64,
}

/// Complete healthcheck report.
#[derive(Debug, Clone, Serialize)]
pub struct HealthReport {
    pub errors: Vec<Finding>,
    pub warnings: Vec<Finding>,
    pub summary: DatasetSummary,
}

/// Run healthcheck layers 1–3 (structural, quality, distribution) on a dataset.
///
/// Returns a [`HealthReport`] with errors, warnings, and a distribution summary.
/// For GT/DT compatibility checks (layer 4), use [`healthcheck_compatibility`] instead.
pub fn healthcheck(dataset: &Dataset) -> HealthReport {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    check_structural(dataset, &mut errors);
    check_quality(dataset, &mut warnings);
    let summary = build_summary(dataset, &mut warnings);

    HealthReport {
        errors,
        warnings,
        summary,
    }
}

/// Run all 4 healthcheck layers, including GT/DT compatibility.
///
/// Runs layers 1–3 on the ground-truth dataset, then checks that detection
/// image IDs and category IDs are subsets of what exists in the ground truth.
pub fn healthcheck_compatibility(gt: &Dataset, dt: &Dataset) -> HealthReport {
    let mut report = healthcheck(gt);
    check_compatibility(gt, dt, &mut report.errors, &mut report.warnings);
    report
}

fn find_duplicate_ids<T, F>(items: &[T], id_fn: F) -> Vec<u64>
where
    F: Fn(&T) -> u64,
{
    let mut seen = HashSet::new();
    items
        .iter()
        .filter_map(|item| {
            let id = id_fn(item);
            if seen.insert(id) { None } else { Some(id) }
        })
        .collect()
}

fn push_if_nonempty(
    findings: &mut Vec<Finding>,
    ids: Vec<u64>,
    code: &'static str,
    message: String,
    layer: Layer,
) {
    if !ids.is_empty() {
        findings.push(Finding {
            code,
            message,
            affected_ids: ids,
            layer,
        });
    }
}

fn check_structural(dataset: &Dataset, errors: &mut Vec<Finding>) {
    let dup_img = find_duplicate_ids(&dataset.images, |i| i.id);
    push_if_nonempty(
        errors,
        dup_img,
        "duplicate_image_id",
        "Duplicate image ID(s) found. Each image must have a unique ID.".into(),
        Layer::Structural,
    );

    let dup_ann = find_duplicate_ids(&dataset.annotations, |a| a.id);
    push_if_nonempty(
        errors,
        dup_ann,
        "duplicate_ann_id",
        "Duplicate annotation ID(s) found. Each annotation must have a unique ID.".into(),
        Layer::Structural,
    );

    let dup_cat = find_duplicate_ids(&dataset.categories, |c| c.id);
    push_if_nonempty(
        errors,
        dup_cat,
        "duplicate_category_id",
        "Duplicate category ID(s) found. Each category must have a unique ID.".into(),
        Layer::Structural,
    );

    let image_ids: HashSet<u64> = dataset.images.iter().map(|img| img.id).collect();
    let orphan_img: Vec<u64> = dataset
        .annotations
        .iter()
        .filter(|ann| !image_ids.contains(&ann.image_id))
        .map(|ann| ann.id)
        .collect();
    push_if_nonempty(
        errors,
        orphan_img,
        "orphan_image_id",
        "Annotation(s) reference image IDs not present in images.".into(),
        Layer::Structural,
    );

    let cat_ids: HashSet<u64> = dataset.categories.iter().map(|c| c.id).collect();
    let orphan_cat: Vec<u64> = dataset
        .annotations
        .iter()
        .filter(|ann| !cat_ids.contains(&ann.category_id))
        .map(|ann| ann.id)
        .collect();
    push_if_nonempty(
        errors,
        orphan_cat,
        "orphan_category_id",
        "Annotation(s) reference category IDs not present in categories.".into(),
        Layer::Structural,
    );

    let missing_geom: Vec<u64> = dataset
        .annotations
        .iter()
        .filter(|ann| ann.bbox.is_none() && ann.segmentation.is_none() && ann.keypoints.is_none())
        .map(|ann| ann.id)
        .collect();
    push_if_nonempty(
        errors,
        missing_geom,
        "missing_geometry",
        "Annotation(s) have no bbox, segmentation, or keypoints.".into(),
        Layer::Structural,
    );

    let zero_dim: Vec<u64> = dataset
        .images
        .iter()
        .filter(|img| img.height == 0 || img.width == 0)
        .map(|img| img.id)
        .collect();
    push_if_nonempty(
        errors,
        zero_dim,
        "zero_dimensions",
        "Image(s) have zero height or width.".into(),
        Layer::Structural,
    );
}

/// Compute IoU between two bboxes in [x, y, w, h] format.
fn bbox_iou_pair(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let ax2 = a[0] + a[2];
    let ay2 = a[1] + a[3];
    let bx2 = b[0] + b[2];
    let by2 = b[1] + b[3];

    let inter_x = (ax2.min(bx2) - a[0].max(b[0])).max(0.0);
    let inter_y = (ay2.min(by2) - a[1].max(b[1])).max(0.0);
    let inter = inter_x * inter_y;

    let area_a = a[2] * a[3];
    let area_b = b[2] * b[3];
    let union = area_a + area_b - inter;

    if union <= 0.0 { 0.0 } else { inter / union }
}

fn check_quality(dataset: &Dataset, warnings: &mut Vec<Finding>) {
    // Build image dimension lookup
    let img_dims: HashMap<u64, (u32, u32)> = dataset
        .images
        .iter()
        .map(|img| (img.id, (img.width, img.height)))
        .collect();

    let mut degenerate_ids = Vec::new();
    let mut zero_area_ids = Vec::new();
    let mut oob_ids = Vec::new();
    let mut extreme_ar_ids = Vec::new();

    // Near-duplicate detection groups, built in the same pass
    struct AnnBbox {
        id: u64,
        bbox: [f64; 4],
    }
    let mut groups: HashMap<(u64, u64), Vec<AnnBbox>> = HashMap::new();

    for ann in &dataset.annotations {
        if let Some(bbox) = &ann.bbox {
            let w = bbox[2];
            let h = bbox[3];

            // Degenerate bbox — skip further checks on this annotation
            if w <= 0.0 || h <= 0.0 {
                degenerate_ids.push(ann.id);
                continue;
            }

            // Zero area
            if let Some(area) = ann.area {
                if area == 0.0 {
                    zero_area_ids.push(ann.id);
                }
            }

            // Bbox out of bounds
            if let Some(&(img_w, img_h)) = img_dims.get(&ann.image_id) {
                let x2 = bbox[0] + w;
                let y2 = bbox[1] + h;
                if x2 > img_w as f64 || y2 > img_h as f64 {
                    oob_ids.push(ann.id);
                }
            }

            // Extreme aspect ratio (>20:1)
            let ar = if w > h { w / h } else { h / w };
            if ar > 20.0 {
                extreme_ar_ids.push(ann.id);
            }

            // Group for near-duplicate detection (non-degenerate only)
            groups
                .entry((ann.image_id, ann.category_id))
                .or_default()
                .push(AnnBbox {
                    id: ann.id,
                    bbox: *bbox,
                });
        }
    }

    let n = degenerate_ids.len();
    push_if_nonempty(
        warnings,
        degenerate_ids,
        "degenerate_bbox",
        format!("{n} annotation(s) have degenerate bboxes (width or height <= 0)."),
        Layer::Quality,
    );

    let n = zero_area_ids.len();
    push_if_nonempty(
        warnings,
        zero_area_ids,
        "zero_area",
        format!("{n} annotation(s) have zero area."),
        Layer::Quality,
    );

    let n = oob_ids.len();
    push_if_nonempty(
        warnings,
        oob_ids,
        "bbox_out_of_bounds",
        format!("{n} annotation(s) have bboxes extending outside the image boundary."),
        Layer::Quality,
    );

    let n = extreme_ar_ids.len();
    push_if_nonempty(
        warnings,
        extreme_ar_ids,
        "extreme_aspect_ratio",
        format!("{n} annotation(s) have extreme aspect ratios (>20:1)."),
        Layer::Quality,
    );

    // Near-duplicate detection: same class, same image, IoU > 0.95
    let mut near_dup_ids = HashSet::new();
    let mut skipped_img_ids = Vec::new();
    for ((img_id, _), anns) in &groups {
        // Skip groups with >100 annotations to avoid O(n^2) blowup
        if anns.len() > 100 {
            skipped_img_ids.push(*img_id);
            continue;
        }
        for i in 0..anns.len() {
            for j in (i + 1)..anns.len() {
                if bbox_iou_pair(&anns[i].bbox, &anns[j].bbox) > 0.95 {
                    near_dup_ids.insert(anns[i].id);
                    near_dup_ids.insert(anns[j].id);
                }
            }
        }
    }

    if !near_dup_ids.is_empty() {
        let mut ids: Vec<u64> = near_dup_ids.into_iter().collect();
        ids.sort_unstable();
        warnings.push(Finding {
            code: "near_duplicate",
            message: format!(
                "{} annotation(s) appear to be near-duplicates (same class, same image, IoU > 0.95).",
                ids.len()
            ),
            affected_ids: ids,
            layer: Layer::Quality,
        });
    }

    skipped_img_ids.sort_unstable();
    skipped_img_ids.dedup();
    let n = skipped_img_ids.len();
    push_if_nonempty(
        warnings,
        skipped_img_ids,
        "near_duplicate_check_skipped",
        format!("{n} image(s) have >100 same-class annotations; near-duplicate check was skipped."),
        Layer::Quality,
    );
}

fn build_summary(dataset: &Dataset, warnings: &mut Vec<Finding>) -> DatasetSummary {
    // Images without annotations
    let annotated_img_ids: HashSet<u64> =
        dataset.annotations.iter().map(|ann| ann.image_id).collect();
    let images_without_annotations = dataset
        .images
        .iter()
        .filter(|img| !annotated_img_ids.contains(&img.id))
        .count();

    // Per-category annotation counts
    let cat_name_map: HashMap<u64, &str> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();

    let mut cat_counts: HashMap<u64, usize> = HashMap::new();
    for cat in &dataset.categories {
        cat_counts.insert(cat.id, 0);
    }
    for ann in &dataset.annotations {
        if let Some(count) = cat_counts.get_mut(&ann.category_id) {
            *count += 1;
        }
    }

    let mut category_counts: Vec<(String, usize)> = cat_counts
        .iter()
        .filter_map(|(&cat_id, &count)| {
            cat_name_map
                .get(&cat_id)
                .map(|name| ((*name).to_string(), count))
        })
        .collect();
    category_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Imbalance ratio
    let nonzero_counts: Vec<usize> = category_counts
        .iter()
        .map(|(_, c)| *c)
        .filter(|&c| c > 0)
        .collect();
    let imbalance_ratio = if nonzero_counts.len() < 2 {
        1.0
    } else {
        let max = *nonzero_counts.iter().max().expect("len >= 2") as f64;
        let min = *nonzero_counts.iter().min().expect("len >= 2") as f64;
        max / min
    };

    // Warn about categories with 0 instances
    let zero_cats: Vec<u64> = cat_counts
        .iter()
        .filter(|&(_, &count)| count == 0)
        .map(|(&id, _)| id)
        .collect();
    if !zero_cats.is_empty() {
        warnings.push(Finding {
            code: "zero_instance_category",
            message: format!(
                "{} category/categories have zero annotation instances.",
                zero_cats.len()
            ),
            affected_ids: zero_cats,
            layer: Layer::Distribution,
        });
    }

    // Warn about categories with <10 instances (but >0)
    let low_cats: Vec<u64> = cat_counts
        .iter()
        .filter(|&(_, &count)| count > 0 && count < 10)
        .map(|(&id, _)| id)
        .collect();
    if !low_cats.is_empty() {
        warnings.push(Finding {
            code: "low_instance_category",
            message: format!(
                "{} category/categories have fewer than 10 annotation instances.",
                low_cats.len()
            ),
            affected_ids: low_cats,
            layer: Layer::Distribution,
        });
    }

    DatasetSummary {
        num_images: dataset.images.len(),
        num_annotations: dataset.annotations.len(),
        num_categories: dataset.categories.len(),
        images_without_annotations,
        category_counts,
        imbalance_ratio,
    }
}

fn check_compatibility(
    gt: &Dataset,
    dt: &Dataset,
    errors: &mut Vec<Finding>,
    warnings: &mut Vec<Finding>,
) {
    let gt_image_ids: HashSet<u64> = gt.images.iter().map(|img| img.id).collect();
    let gt_cat_ids: HashSet<u64> = gt.categories.iter().map(|c| c.id).collect();

    // DT annotations referencing image_ids not in GT
    let orphan_img: Vec<u64> = dt
        .annotations
        .iter()
        .filter(|ann| !gt_image_ids.contains(&ann.image_id))
        .map(|ann| ann.id)
        .collect();
    let n = orphan_img.len();
    push_if_nonempty(
        errors,
        orphan_img,
        "dt_orphan_image_id",
        format!("{n} detection(s) reference image IDs not present in ground truth."),
        Layer::Compatibility,
    );

    // DT annotations referencing category_ids not in GT
    let orphan_cat: Vec<u64> = dt
        .annotations
        .iter()
        .filter(|ann| !gt_cat_ids.contains(&ann.category_id))
        .map(|ann| ann.id)
        .collect();
    let n = orphan_cat.len();
    push_if_nonempty(
        errors,
        orphan_cat,
        "dt_orphan_category_id",
        format!("{n} detection(s) reference category IDs not present in ground truth."),
        Layer::Compatibility,
    );

    // DT annotations missing score
    let missing_score: Vec<u64> = dt
        .annotations
        .iter()
        .filter(|ann| ann.score.is_none())
        .map(|ann| ann.id)
        .collect();
    let n = missing_score.len();
    push_if_nonempty(
        warnings,
        missing_score,
        "dt_missing_score",
        format!("{n} detection(s) are missing a confidence score."),
        Layer::Compatibility,
    );

    // DT annotations with score outside [0, 1]
    let bad_score: Vec<u64> = dt
        .annotations
        .iter()
        .filter(|ann| {
            if let Some(score) = ann.score {
                !(0.0..=1.0).contains(&score)
            } else {
                false
            }
        })
        .map(|ann| ann.id)
        .collect();
    let n = bad_score.len();
    push_if_nonempty(
        warnings,
        bad_score,
        "dt_score_out_of_range",
        format!("{n} detection(s) have scores outside the [0, 1] range."),
        Layer::Compatibility,
    );
}
