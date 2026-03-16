use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::types::{Annotation, Category, Dataset, Image};

/// Statistics returned by [`coco_to_yolo`].
#[derive(Debug, Clone)]
pub struct YoloStats {
    /// Total number of images processed (one `.txt` written per image).
    pub images: usize,
    /// Total number of annotations written.
    pub annotations: usize,
    /// Crowd annotations skipped (`iscrowd == true`).
    pub skipped_crowd: usize,
    /// Annotations skipped because they had no `bbox`.
    pub missing_bbox: usize,
}

/// Errors that can occur during format conversion.
#[derive(Debug)]
pub enum ConvertError {
    /// An I/O error occurred while reading or writing files.
    Io(io::Error),
    /// An image has `width == 0` or `height == 0`, preventing normalization.
    MissingImageDimensions(u64),
    /// No `data.yaml` found in the YOLO directory.
    MissingDataYaml,
    /// A label file or `data.yaml` could not be parsed.
    ParseError(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::Io(e) => write!(f, "I/O error: {e}"),
            ConvertError::MissingImageDimensions(id) => {
                write!(f, "image id={id} has zero width or height")
            }
            ConvertError::MissingDataYaml => write!(f, "data.yaml not found in YOLO directory"),
            ConvertError::ParseError(s) => write!(f, "parse error: {s}"),
        }
    }
}

impl std::error::Error for ConvertError {}

impl From<io::Error> for ConvertError {
    fn from(e: io::Error) -> Self {
        ConvertError::Io(e)
    }
}

/// Convert a COCO dataset to YOLO label format.
///
/// Writes one `.txt` label file per image (named by the image filename stem) into
/// `output_dir`, plus a `data.yaml` file describing the category list.
///
/// Each label line has the format `class_idx cx cy w h` where all coordinates are
/// normalized to `[0, 1]` by the image dimensions. Categories are sorted by COCO ID
/// and assigned 0-indexed YOLO class IDs in that order.
///
/// # Skipping rules
///
/// - Crowd annotations (`iscrowd == true`) are skipped; counted in [`YoloStats::skipped_crowd`].
/// - Annotations without a `bbox` are skipped; counted in [`YoloStats::missing_bbox`].
/// - Images with no annotations still produce an empty `.txt` file (YOLO convention).
///
/// # Errors
///
/// Returns [`ConvertError::MissingImageDimensions`] for any image with `width == 0`
/// or `height == 0`, since normalized coordinates cannot be computed.
pub fn coco_to_yolo(dataset: &Dataset, output_dir: &Path) -> Result<YoloStats, ConvertError> {
    fs::create_dir_all(output_dir)?;

    // Sort categories by ID → 0-indexed YOLO class IDs
    let mut sorted_cats = dataset.categories.clone();
    sorted_cats.sort_by_key(|c| c.id);
    let cat_id_to_idx: HashMap<u64, usize> = sorted_cats
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id, i))
        .collect();

    // Group annotations by image_id
    let mut anns_by_image: HashMap<u64, Vec<&Annotation>> = HashMap::new();
    for ann in &dataset.annotations {
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }

    let mut total_annotations = 0usize;
    let mut skipped_crowd = 0usize;
    let mut missing_bbox = 0usize;

    for img in &dataset.images {
        if img.width == 0 || img.height == 0 {
            return Err(ConvertError::MissingImageDimensions(img.id));
        }

        let path = Path::new(&img.file_name);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(img.file_name.as_str())
            .to_string();

        let txt_path = output_dir.join(format!("{stem}.txt"));
        let mut file = fs::File::create(&txt_path)?;

        let w = img.width as f64;
        let h = img.height as f64;

        if let Some(anns) = anns_by_image.get(&img.id) {
            for ann in anns {
                if ann.iscrowd {
                    skipped_crowd += 1;
                    continue;
                }
                let bbox = match ann.bbox {
                    Some(b) => b,
                    None => {
                        missing_bbox += 1;
                        continue;
                    }
                };
                let class_idx = match cat_id_to_idx.get(&ann.category_id) {
                    Some(&idx) => idx,
                    None => continue,
                };

                let [x, y, bw, bh] = bbox;
                let cx = (x + bw / 2.0) / w;
                let cy = (y + bh / 2.0) / h;
                let nw = bw / w;
                let nh = bh / h;

                writeln!(file, "{class_idx} {cx:.6} {cy:.6} {nw:.6} {nh:.6}")?;
                total_annotations += 1;
            }
        }
    }

    // Write data.yaml (hand-rolled; no serde_yaml dep needed for this two-field format)
    let yaml_path = output_dir.join("data.yaml");
    let mut yaml_file = fs::File::create(&yaml_path)?;
    writeln!(yaml_file, "nc: {}", sorted_cats.len())?;
    let names_csv: Vec<&str> = sorted_cats.iter().map(|c| c.name.as_str()).collect();
    writeln!(yaml_file, "names: [{}]", names_csv.join(", "))?;

    Ok(YoloStats {
        images: dataset.images.len(),
        annotations: total_annotations,
        skipped_crowd,
        missing_bbox,
    })
}

/// Convert a YOLO label directory back to COCO format.
///
/// Reads `data.yaml` from `yolo_dir` for the category list, then parses every `.txt`
/// label file in the directory. Returns a [`Dataset`] with sequential image and
/// annotation IDs starting at 1.
///
/// # Image dimensions
///
/// `image_dims` maps filename stems (e.g. `"000042"`) or stems with extensions
/// (e.g. `"000042.jpg"`) to `(width, height)` in pixels. Both forms are tried;
/// common image extensions (`jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`) are also
/// checked as fallbacks. If a stem cannot be found, the image is stored with
/// `width=0, height=0` and bounding boxes are computed as zero-pixel values.
///
/// # Errors
///
/// Returns [`ConvertError::MissingDataYaml`] if no `data.yaml` is present.
/// Returns [`ConvertError::ParseError`] if a label line does not have exactly
/// 5 fields or contains an out-of-range `class_idx`.
pub fn yolo_to_coco(
    yolo_dir: &Path,
    image_dims: &HashMap<String, (u32, u32)>,
) -> Result<Dataset, ConvertError> {
    let yaml_path = yolo_dir.join("data.yaml");
    if !yaml_path.exists() {
        return Err(ConvertError::MissingDataYaml);
    }
    let yaml_content = fs::read_to_string(&yaml_path)?;
    let names = parse_data_yaml(&yaml_content)?;

    let categories: Vec<Category> = names
        .iter()
        .enumerate()
        .map(|(i, name)| Category {
            id: (i + 1) as u64,
            name: name.clone(),
            supercategory: None,
            skeleton: None,
            keypoints: None,
            frequency: None,
        })
        .collect();

    // Collect and sort .txt files for deterministic ordering
    let mut txt_files: Vec<PathBuf> = fs::read_dir(yolo_dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("txt") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    txt_files.sort();

    let mut images: Vec<Image> = Vec::new();
    let mut annotations: Vec<Annotation> = Vec::new();
    let mut img_id = 1u64;
    let mut ann_id = 1u64;

    for txt_path in &txt_files {
        let stem = txt_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        let (width, height) = lookup_image_dims(image_dims, &stem);

        images.push(Image {
            id: img_id,
            file_name: stem.clone(),
            width,
            height,
            license: None,
            coco_url: None,
            flickr_url: None,
            date_captured: None,
            neg_category_ids: vec![],
            not_exhaustive_category_ids: vec![],
        });

        let content = fs::read_to_string(txt_path)?;
        let w = width as f64;
        let h = height as f64;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 5 {
                return Err(ConvertError::ParseError(format!(
                    "expected 5 fields, got {} in: {line}",
                    parts.len()
                )));
            }
            let class_idx: usize = parts[0].parse().map_err(|_| {
                ConvertError::ParseError(format!("invalid class_idx: {}", parts[0]))
            })?;
            let cx: f64 = parts[1]
                .parse()
                .map_err(|_| ConvertError::ParseError(format!("invalid cx: {}", parts[1])))?;
            let cy: f64 = parts[2]
                .parse()
                .map_err(|_| ConvertError::ParseError(format!("invalid cy: {}", parts[2])))?;
            let bw: f64 = parts[3]
                .parse()
                .map_err(|_| ConvertError::ParseError(format!("invalid width: {}", parts[3])))?;
            let bh: f64 = parts[4]
                .parse()
                .map_err(|_| ConvertError::ParseError(format!("invalid height: {}", parts[4])))?;

            if class_idx >= categories.len() {
                return Err(ConvertError::ParseError(format!(
                    "class_idx {class_idx} out of range (nc={})",
                    categories.len()
                )));
            }

            let category_id = (class_idx + 1) as u64;
            let px = (cx - bw / 2.0) * w;
            let py = (cy - bh / 2.0) * h;
            let pw = bw * w;
            let ph = bh * h;

            annotations.push(Annotation {
                id: ann_id,
                image_id: img_id,
                category_id,
                bbox: Some([px, py, pw, ph]),
                area: Some(pw * ph),
                segmentation: None,
                iscrowd: false,
                keypoints: None,
                num_keypoints: None,
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
        categories,
        licenses: vec![],
    })
}

/// Look up image dimensions by stem; try common extensions as fallback.
fn lookup_image_dims(image_dims: &HashMap<String, (u32, u32)>, stem: &str) -> (u32, u32) {
    if let Some(&dims) = image_dims.get(stem) {
        return dims;
    }
    for ext in &["jpg", "jpeg", "png", "bmp", "tif", "tiff"] {
        let key = format!("{stem}.{ext}");
        if let Some(&dims) = image_dims.get(&key) {
            return dims;
        }
    }
    (0, 0)
}

/// Parse `data.yaml` produced by [`coco_to_yolo`] and return the names list.
///
/// Expects a `names: [a, b, c]` line in flow sequence format.
fn parse_data_yaml(content: &str) -> Result<Vec<String>, ConvertError> {
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("names:") {
            let rest = rest.trim();
            if let Some(inner) = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
                let names: Vec<String> = inner
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                return Ok(names);
            }
        }
    }
    Err(ConvertError::ParseError(
        "no 'names' field found in data.yaml".into(),
    ))
}
