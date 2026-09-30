//! DOTA format conversion (oriented bounding boxes).
//!
//! DOTA uses one `.txt` file per image with lines:
//! `x1 y1 x2 y2 x3 y3 x4 y4 category difficulty`
//! where (x1,y1)...(x4,y4) are the four corner points of the rotated rectangle.
//! Files may start with `imagesource:`/`gsd:` metadata lines, which are skipped.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::geometry::{corners_to_obb, obb_to_aabb, obb_to_corners};
use crate::types::{Annotation, Category, Dataset, Image};

use super::{ConvertError, anns_by_image, file_stem, line_err};

/// Statistics returned by [`coco_to_dota`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DotaStats {
    /// Images written (one `.txt` label file per image).
    pub images: usize,
    /// Annotations written.
    pub annotations: usize,
    /// Annotations skipped because they have no oriented bounding box (`obb`).
    pub skipped_no_obb: usize,
}

/// Convert a COCO dataset with OBB annotations to DOTA text format.
///
/// Creates one `.txt` file per image in `output_dir`, named by the image
/// filename stem. Each line contains the 4 corner points, category name, and
/// difficulty (1 for crowd, 0 otherwise).
///
/// # Errors
///
/// Returns [`ConvertError::Io`] on filesystem errors,
/// [`ConvertError::StemCollision`] if two images share a filename stem (their
/// label files would overwrite each other), or
/// [`ConvertError::UnknownCategory`] if an annotation references a category id
/// that is not in the dataset.
pub fn coco_to_dota(dataset: &Dataset, output_dir: &Path) -> Result<DotaStats, ConvertError> {
    super::check_unique_stems(dataset)?;
    fs::create_dir_all(output_dir)?;

    let cat_map = crate::types::cat_id_to_name(dataset);

    let grouped = anns_by_image(dataset);
    let mut stats = DotaStats::default();

    for img in &dataset.images {
        let stem = file_stem(&img.file_name);
        let path = output_dir.join(format!("{stem}.txt"));
        let mut file = fs::File::create(&path)?;

        if let Some(anns) = grouped.get(&img.id) {
            for ann in anns {
                let obb = match ann.obb.as_deref() {
                    Some(o) => o,
                    None => {
                        stats.skipped_no_obb += 1;
                        continue;
                    }
                };

                let cat_name =
                    *cat_map
                        .get(&ann.category_id)
                        .ok_or(ConvertError::UnknownCategory {
                            ann_id: ann.id,
                            category_id: ann.category_id,
                        })?;
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
/// `x1 y1 x2 y2 x3 y3 x4 y4 category difficulty` (difficulty optional);
/// `imagesource:`/`gsd:` metadata lines and `#` comments are skipped.
///
/// When `categories` is provided it fixes the category list and its ID order
/// (DOTA has no sidecar category file to read them from); labels found in the
/// files but absent from the list are appended in first-use order. Without it,
/// categories are auto-discovered in first-use order.
///
/// `image_dims` maps filename stems (or stems with common image extensions) to
/// `(width, height)`. DOTA corners are absolute pixels, so dimensions are pure
/// metadata here: an image missing from the map is recorded as `0×0` (unknown)
/// and its geometry is unaffected. Each image's `file_name` is the label
/// file's stem — DOTA does not record the image extension, and none is
/// invented.
///
/// # Errors
///
/// Returns [`ConvertError::ParseError`] — naming the file and line — for a
/// line with fewer than 9 or more than 10 fields, an unparsable corner
/// coordinate or difficulty, or a label file with a non-UTF-8 name.
pub fn dota_to_coco(
    label_dir: &Path,
    categories: Option<Vec<String>>,
    image_dims: &HashMap<String, (u32, u32)>,
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
                ..Default::default()
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
        let stem = super::utf8_stem(&path)?.to_string();

        // Dimensions are metadata only (corners are absolute pixels); unknown
        // stays 0×0 per the module's missing-dimensions policy.
        let (width, height) = super::lookup_image_dims(image_dims, &stem).unwrap_or((0, 0));

        images.push(Image {
            id: img_id,
            file_name: stem,
            width,
            height,
            ..Default::default()
        });

        let file = fs::File::open(&path)?;
        let reader = BufReader::new(file);

        for (line_idx, line) in reader.lines().enumerate() {
            let line_no = line_idx + 1;
            let line = line?;
            let line = line.trim();
            // DOTA files may begin with metadata lines; `#` comments are ours.
            if line.is_empty()
                || line.starts_with('#')
                || line.starts_with("imagesource:")
                || line.starts_with("gsd:")
            {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 9 || parts.len() > 10 {
                return Err(line_err(
                    &path,
                    line_no,
                    format!(
                        "expected `x1 y1 x2 y2 x3 y3 x4 y4 category [difficulty]` (9 or 10 fields), got {} in: {line}",
                        parts.len()
                    ),
                ));
            }

            // Parse 8 corner coordinates
            let coords: Vec<f64> = parts[..8]
                .iter()
                .map(|s| {
                    s.parse::<f64>().map_err(|_| {
                        line_err(&path, line_no, format!("invalid corner coordinate: {s}"))
                    })
                })
                .collect::<Result<_, _>>()?;

            let cat_name = parts[8].to_string();
            let difficulty: u8 = match parts.get(9) {
                Some(s) => s
                    .parse()
                    .map_err(|_| line_err(&path, line_no, format!("invalid difficulty: {s}")))?,
                None => 0,
            };

            // Auto-discover category
            let category_id = if let Some(&id) = cat_name_to_id.get(&cat_name) {
                id
            } else {
                let id = (cat_list.len() + 1) as u64;
                cat_name_to_id.insert(cat_name.clone(), id);
                cat_list.push(Category {
                    id,
                    name: cat_name,
                    ..Default::default()
                });
                id
            };

            // `coords` always holds exactly 8 values here, so the length-check
            // error arm is unreachable in practice.
            let obb =
                corners_to_obb(&coords).map_err(|e| line_err(&path, line_no, e.to_string()))?;

            annotations.push(Annotation {
                id: ann_id,
                image_id: img_id,
                category_id,
                bbox: Some(obb_to_aabb(&obb)),
                area: Some(obb[2] * obb[3]),
                iscrowd: difficulty > 0,
                obb: Some(Box::new(obb)),
                ..Default::default()
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
        let obb = corners_to_obb(&coords).unwrap();
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
                ..Default::default()
            }],
            annotations: vec![Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 40.0, 20.0]),
                area: Some(800.0),
                obb: Some(Box::new([30.0, 20.0, 40.0, 20.0, 0.0])),
                ..Default::default()
            }],
            categories: vec![Category {
                id: 1,
                name: "vehicle".into(),
                ..Default::default()
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
        let result = dota_to_coco(&label_dir, None, &dims).unwrap();

        assert_eq!(result.annotations.len(), 1);
        let ann = &result.annotations[0];
        let obb = ann.obb.as_deref().unwrap();

        // Check round-trip accuracy (limited by .1 decimal formatting)
        assert!((obb[0] - 30.0).abs() < EPS, "cx: {}", obb[0]);
        assert!((obb[1] - 20.0).abs() < EPS, "cy: {}", obb[1]);
        assert!((obb[2] - 40.0).abs() < EPS, "w: {}", obb[2]);
        assert!((obb[3] - 20.0).abs() < EPS, "h: {}", obb[3]);
        assert!(obb[4].abs() < EPS, "angle: {}", obb[4]);

        // Category auto-discovered; file_name is the bare stem (no invented
        // extension), and dims came from the lookup map.
        assert_eq!(result.categories.len(), 1);
        assert_eq!(result.categories[0].name, "vehicle");
        assert_eq!(result.images[0].file_name, "test");
        assert_eq!(
            (result.images[0].width, result.images[0].height),
            (800, 600)
        );
    }
}
