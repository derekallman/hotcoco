//! DOTA format conversion (oriented bounding boxes).
//!
//! DOTA uses one `.txt` file per image with lines:
//! `x1 y1 x2 y2 x3 y3 x4 y4 category difficulty`
//! where (x1,y1)...(x4,y4) are the four corner points of the rotated rectangle.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::geometry::{corners_to_obb, obb_to_aabb, obb_to_corners};
use crate::types::{Annotation, Category, Dataset, Image};

use super::{anns_by_image, file_stem, ConvertError};

/// Statistics from a COCO → DOTA conversion.
#[derive(Debug, Default)]
pub struct DotaStats {
    pub images: usize,
    pub annotations: usize,
    pub skipped_no_obb: usize,
}

/// Convert a COCO dataset with OBB annotations to DOTA text format.
///
/// Creates one `.txt` file per image in `output_dir`. Each line contains
/// the 4 corner points, category name, and difficulty (1 for crowd, 0 otherwise).
pub fn coco_to_dota(dataset: &Dataset, output_dir: &Path) -> Result<DotaStats, ConvertError> {
    fs::create_dir_all(output_dir)?;

    let cat_map: HashMap<u64, &str> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();

    let grouped = anns_by_image(dataset);
    let mut stats = DotaStats::default();

    for img in &dataset.images {
        let stem = file_stem(&img.file_name);
        let path = output_dir.join(format!("{stem}.txt"));
        let mut file = fs::File::create(&path)?;

        if let Some(anns) = grouped.get(&img.id) {
            for ann in anns {
                let obb = match &ann.obb {
                    Some(o) => o,
                    None => {
                        stats.skipped_no_obb += 1;
                        continue;
                    }
                };

                let cat_name = cat_map.get(&ann.category_id).copied().unwrap_or("unknown");
                let difficulty = i32::from(ann.iscrowd);
                let corners = obb_to_corners(obb);

                writeln!(
                    file,
                    "{:.1} {:.1} {:.1} {:.1} {:.1} {:.1} {:.1} {:.1} {} {}",
                    corners[0].0,
                    corners[0].1,
                    corners[1].0,
                    corners[1].1,
                    corners[2].0,
                    corners[2].1,
                    corners[3].0,
                    corners[3].1,
                    cat_name,
                    difficulty,
                )?;
                stats.annotations += 1;
            }
        }
        stats.images += 1;
    }

    Ok(stats)
}

/// Convert DOTA text files to a COCO dataset with OBB annotations.
///
/// Reads `.txt` files from `label_dir`. Each line is parsed as
/// `x1 y1 x2 y2 x3 y3 x4 y4 category difficulty`.
///
/// Image dimensions are read from `image_dir` (uses `imagesize` crate if available,
/// otherwise defaults to 0×0). Categories are auto-discovered from the label files
/// unless explicitly provided.
pub fn dota_to_coco(
    label_dir: &Path,
    image_dims: &HashMap<String, (u32, u32)>,
    categories: Option<Vec<String>>,
) -> Result<Dataset, ConvertError> {
    let mut cat_name_to_id: HashMap<String, u64> = HashMap::new();
    let mut cat_list: Vec<Category> = Vec::new();

    // Pre-populate categories if provided
    if let Some(ref cats) = categories {
        for (i, name) in cats.iter().enumerate() {
            let id = (i + 1) as u64;
            cat_name_to_id.insert(name.clone(), id);
            cat_list.push(Category {
                id,
                name: name.clone(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            });
        }
    }

    let mut images: Vec<Image> = Vec::new();
    let mut annotations: Vec<Annotation> = Vec::new();
    let mut img_id: u64 = 1;
    let mut ann_id: u64 = 1;

    // Collect and sort label files for deterministic output
    let mut entries: Vec<_> = fs::read_dir(label_dir)?
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "txt"))
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);

    for entry in entries {
        let path = entry.path();
        let stem = file_stem(
            path.file_name()
                .expect("entry has a file name")
                .to_str()
                .unwrap_or(""),
        );

        let (width, height) = super::lookup_image_dims(image_dims, stem);

        images.push(Image {
            id: img_id,
            file_name: format!("{stem}.png"),
            width,
            height,
            license: None,
            coco_url: None,
            flickr_url: None,
            date_captured: None,
            neg_category_ids: vec![],
            not_exhaustive_category_ids: vec![],
        });

        let file = fs::File::open(&path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 9 {
                continue; // Skip malformed lines
            }

            // Parse 8 corner coordinates
            let coords: Result<Vec<f64>, _> = parts[..8].iter().map(|s| s.parse::<f64>()).collect();
            let coords = match coords {
                Ok(c) => c,
                Err(_) => continue, // Skip unparseable lines
            };

            let cat_name = parts[8].to_string();
            let difficulty: u8 = parts.get(9).and_then(|s| s.parse().ok()).unwrap_or(0);

            // Auto-discover category
            let category_id = if let Some(&id) = cat_name_to_id.get(&cat_name) {
                id
            } else {
                let id = (cat_list.len() + 1) as u64;
                cat_name_to_id.insert(cat_name.clone(), id);
                cat_list.push(Category {
                    id,
                    name: cat_name,
                    supercategory: None,
                    skeleton: None,
                    keypoints: None,
                    frequency: None,
                });
                id
            };

            let obb = corners_to_obb(&coords);

            annotations.push(Annotation {
                id: ann_id,
                image_id: img_id,
                category_id,
                bbox: Some(obb_to_aabb(&obb)),
                area: Some(obb[2] * obb[3]),
                segmentation: None,
                iscrowd: difficulty > 0,
                keypoints: None,
                num_keypoints: None,
                obb: Some(obb),
                score: None,
                is_group_of: None,
            });
            ann_id += 1;
        }
        img_id += 1;
    }

    Ok(Dataset {
        info: None,
        images,
        annotations,
        categories: cat_list,
        licenses: vec![],
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const EPS: f64 = 0.2; // DOTA uses 1 decimal place, so ≤0.1 round-trip error per coord

    #[test]
    fn test_corners_to_obb_axis_aligned() {
        // Rectangle from (0,0) to (4,3) — corners CCW
        let coords = [0.0, 0.0, 4.0, 0.0, 4.0, 3.0, 0.0, 3.0];
        let obb = corners_to_obb(&coords);
        assert!((obb[0] - 2.0).abs() < 1e-9, "cx"); // cx
        assert!((obb[1] - 1.5).abs() < 1e-9, "cy"); // cy
        assert!((obb[2] - 4.0).abs() < 1e-9, "w"); // w
        assert!((obb[3] - 3.0).abs() < 1e-9, "h"); // h
        assert!(obb[4].abs() < 1e-9, "angle should be 0"); // angle
    }

    #[test]
    fn test_dota_round_trip() {
        let dataset = Dataset {
            info: None,
            images: vec![Image {
                id: 1,
                file_name: "test.png".into(),
                width: 800,
                height: 600,
                license: None,
                coco_url: None,
                flickr_url: None,
                date_captured: None,
                neg_category_ids: vec![],
                not_exhaustive_category_ids: vec![],
            }],
            annotations: vec![Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 40.0, 20.0]),
                area: Some(800.0),
                segmentation: None,
                iscrowd: false,
                keypoints: None,
                num_keypoints: None,
                obb: Some([30.0, 20.0, 40.0, 20.0, 0.0]),
                score: None,
                is_group_of: None,
            }],
            categories: vec![Category {
                id: 1,
                name: "vehicle".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            }],
            licenses: vec![],
        };

        let tmp = TempDir::new().unwrap();
        let label_dir = tmp.path().join("labels");

        // Export
        let stats = coco_to_dota(&dataset, &label_dir).unwrap();
        assert_eq!(stats.images, 1);
        assert_eq!(stats.annotations, 1);

        // Import
        let mut dims = HashMap::new();
        dims.insert("test".into(), (800, 600));
        let result = dota_to_coco(&label_dir, &dims, None).unwrap();

        assert_eq!(result.annotations.len(), 1);
        let ann = &result.annotations[0];
        let obb = ann.obb.unwrap();

        // Check round-trip accuracy (limited by .1 decimal formatting)
        assert!((obb[0] - 30.0).abs() < EPS, "cx: {}", obb[0]);
        assert!((obb[1] - 20.0).abs() < EPS, "cy: {}", obb[1]);
        assert!((obb[2] - 40.0).abs() < EPS, "w: {}", obb[2]);
        assert!((obb[3] - 20.0).abs() < EPS, "h: {}", obb[3]);
        assert!(obb[4].abs() < EPS, "angle: {}", obb[4]);

        // Category auto-discovered
        assert_eq!(result.categories.len(), 1);
        assert_eq!(result.categories[0].name, "vehicle");
    }
}
