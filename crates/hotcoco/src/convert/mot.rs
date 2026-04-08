use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::BufRead;
use std::path::Path;

use crate::types::{Annotation, Category, Dataset, Image, Track, Video};

use super::ConvertError;

/// Statistics from a MOTChallenge format conversion.
#[derive(Debug, Clone)]
pub struct MotStats {
    /// Number of sequences (videos) found.
    pub sequences: usize,
    /// Total number of frames.
    pub frames: usize,
    /// Total number of annotations.
    pub annotations: usize,
    /// Total number of unique tracks.
    pub tracks: usize,
}

/// Convert a MOTChallenge `gt.txt` file to a COCO tracking dataset.
///
/// Parses the standard MOT format:
/// `frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility`
///
/// All annotations are placed in a single video sequence.
pub fn mot_to_coco(gt_path: &Path) -> Result<(Dataset, MotStats), ConvertError> {
    let content = fs::read_to_string(gt_path)?;
    parse_mot_csv(&content, gt_path.file_name().and_then(|n| n.to_str()))
}

/// Convert a MOTChallenge sequence directory to a COCO tracking dataset.
///
/// Expects the directory structure:
/// ```text
/// seq_dir/
///   seqinfo.ini    (optional — provides seqName, imWidth, imHeight)
///   gt/gt.txt      (ground truth annotations)
/// ```
pub fn mot_seq_to_coco(seq_dir: &Path) -> Result<(Dataset, MotStats), ConvertError> {
    let gt_path = seq_dir.join("gt").join("gt.txt");
    if !gt_path.exists() {
        return Err(ConvertError::ParseError(format!(
            "gt/gt.txt not found in {}",
            seq_dir.display()
        )));
    }

    let content = fs::read_to_string(&gt_path)?;

    // Try to parse seqinfo.ini for metadata.
    let seqinfo_path = seq_dir.join("seqinfo.ini");
    let (seq_name, width, height) = if seqinfo_path.exists() {
        parse_seqinfo(&seqinfo_path)?
    } else {
        let name = seq_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        (name.to_string(), None, None)
    };

    let (mut dataset, stats) = parse_mot_csv(&content, Some(&seq_name))?;

    // Apply metadata from seqinfo.ini.
    if let Some(video) = dataset.videos.first_mut() {
        video.name = Some(seq_name);
        video.width = width;
        video.height = height;
    }
    if let (Some(w), Some(h)) = (width, height) {
        for img in &mut dataset.images {
            if img.width == 0 {
                img.width = w;
            }
            if img.height == 0 {
                img.height = h;
            }
        }
    }

    Ok((dataset, stats))
}

/// Parse MOT CSV content into a COCO Dataset.
fn parse_mot_csv(
    content: &str,
    seq_name: Option<&str>,
) -> Result<(Dataset, MotStats), ConvertError> {
    let mut rows: Vec<MotRow> = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            return Err(ConvertError::ParseError(format!(
                "line {}: expected at least 6 fields, got {}",
                line_num + 1,
                parts.len()
            )));
        }

        macro_rules! field {
            ($idx:expr, $name:expr) => {
                parts[$idx].trim().parse().map_err(|_| {
                    ConvertError::ParseError(format!(
                        "line {}: invalid {}: {}",
                        line_num + 1,
                        $name,
                        parts[$idx]
                    ))
                })
            };
        }

        let frame: u32 = field!(0, "frame")?;
        let track_id: u64 = field!(1, "track_id")?;
        let bb_left: f64 = field!(2, "bb_left")?;
        let bb_top: f64 = field!(3, "bb_top")?;
        let bb_width: f64 = field!(4, "bb_width")?;
        let bb_height: f64 = field!(5, "bb_height")?;

        // Optional fields: conf (index 6), class (index 7), visibility (index 8)
        let conf: f64 = parts
            .get(6)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1.0);
        let class: u64 = parts
            .get(7)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1);

        // MOT convention: conf <= 0 means "ignore this annotation"
        if conf <= 0.0 {
            continue;
        }

        rows.push(MotRow {
            frame,
            track_id,
            bb_left,
            bb_top,
            bb_width,
            bb_height,
            class,
        });
    }

    // Build dataset structures.
    let mut frame_set: BTreeSet<u32> = BTreeSet::new();
    let mut track_set: BTreeSet<u64> = BTreeSet::new();
    let mut class_set: BTreeSet<u64> = BTreeSet::new();

    for row in &rows {
        frame_set.insert(row.frame);
        track_set.insert(row.track_id);
        class_set.insert(row.class);
    }

    // Map frame numbers to image IDs (1-indexed).
    let frame_to_img_id: HashMap<u32, u64> = frame_set
        .iter()
        .enumerate()
        .map(|(i, &f)| (f, (i + 1) as u64))
        .collect();

    let video_id = 1u64;

    let images: Vec<Image> = frame_set
        .iter()
        .enumerate()
        .map(|(i, &frame)| Image {
            id: (i + 1) as u64,
            file_name: format!("{:06}.jpg", frame),
            height: 0,
            width: 0,
            license: None,
            coco_url: None,
            flickr_url: None,
            date_captured: None,
            neg_category_ids: vec![],
            not_exhaustive_category_ids: vec![],
            video_id: Some(video_id),
            frame_index: Some(frame),
        })
        .collect();

    let mut ann_id = 1u64;
    let annotations: Vec<Annotation> = rows
        .iter()
        .map(|row| {
            let id = ann_id;
            ann_id += 1;
            let area = row.bb_width * row.bb_height;
            Annotation {
                id,
                image_id: frame_to_img_id[&row.frame],
                category_id: row.class,
                bbox: Some([row.bb_left, row.bb_top, row.bb_width, row.bb_height]),
                area: Some(area),
                segmentation: None,
                iscrowd: false,
                keypoints: None,
                num_keypoints: None,
                obb: None,
                score: None,
                is_group_of: None,
                track_id: Some(row.track_id),
                video_id: Some(video_id),
            }
        })
        .collect();

    let categories: Vec<Category> = class_set
        .iter()
        .map(|&id| Category {
            id,
            name: format!("class_{id}"),
            supercategory: None,
            skeleton: None,
            keypoints: None,
            frequency: None,
        })
        .collect();

    // Build track-to-category mapping from first annotation per track.
    let mut track_cat: HashMap<u64, u64> = HashMap::new();
    for row in &rows {
        track_cat.entry(row.track_id).or_insert(row.class);
    }

    let tracks: Vec<Track> = track_set
        .iter()
        .map(|&tid| Track {
            id: tid,
            category_id: *track_cat.get(&tid).unwrap_or(&1),
            video_id: Some(video_id),
        })
        .collect();

    let videos = vec![Video {
        id: video_id,
        name: seq_name.map(str::to_string),
        width: None,
        height: None,
    }];

    let stats = MotStats {
        sequences: 1,
        frames: images.len(),
        annotations: annotations.len(),
        tracks: tracks.len(),
    };

    let dataset = Dataset {
        images,
        annotations,
        categories,
        videos,
        tracks,
        ..Default::default()
    };

    Ok((dataset, stats))
}

struct MotRow {
    frame: u32,
    track_id: u64,
    bb_left: f64,
    bb_top: f64,
    bb_width: f64,
    bb_height: f64,
    class: u64,
}

/// Parse `seqinfo.ini` for sequence metadata.
fn parse_seqinfo(path: &Path) -> Result<(String, Option<u32>, Option<u32>), ConvertError> {
    let file = fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut name = String::new();
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if let Some((key, value)) = trimmed.split_once('=') {
            match key.trim().to_lowercase().as_str() {
                "name" | "seqname" => name = value.trim().to_string(),
                "imwidth" => width = value.trim().parse().ok(),
                "imheight" => height = value.trim().parse().ok(),
                _ => {}
            }
        }
    }

    if name.is_empty() {
        name = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
    }

    Ok((name, width, height))
}
