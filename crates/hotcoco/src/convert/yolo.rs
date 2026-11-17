use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::types::{Annotation, Category, Dataset, Image};

use super::{ConvertError, line_err, parse_err};

/// Statistics returned by [`coco_to_yolo`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct YoloStats {
    /// Images written (one `.txt` label file per image).
    pub images: usize,
    /// Annotations written.
    pub annotations: usize,
    /// Crowd annotations skipped (`iscrowd == true`; YOLO has no crowd concept).
    pub skipped_crowd: usize,
    /// Annotations skipped because they have no `bbox`.
    pub skipped_no_bbox: usize,
}

/// Convert a COCO dataset to YOLO label format.
///
/// Writes one `.txt` label file per image (named by the image filename stem) into
/// `output_dir`, plus a `data.yaml` file describing the category list.
///
/// Each label line has the format `class_idx cx cy w h` where all coordinates are
/// normalized to `[0, 1]` by the image dimensions. Categories are sorted by COCO ID
/// and assigned 0-indexed YOLO class IDs in that order. Category names that would
/// break the YAML flow list (commas, quotes, `#`, ...) are single-quoted.
///
/// # Skipping rules
///
/// - Crowd annotations (`iscrowd == true`) are skipped; counted in [`YoloStats::skipped_crowd`].
/// - Annotations without a `bbox` are skipped; counted in [`YoloStats::skipped_no_bbox`].
/// - Images with no annotations still produce an empty `.txt` file (YOLO convention).
///
/// # Errors
///
/// Returns [`ConvertError::MissingImageDimensions`] for any image with `width == 0`
/// or `height == 0` (normalized coordinates cannot be computed),
/// [`ConvertError::StemCollision`] if two images share a filename stem (their
/// label files would overwrite each other), or
/// [`ConvertError::UnknownCategory`] if an annotation references a category id
/// that is not in the dataset.
pub fn coco_to_yolo(dataset: &Dataset, output_dir: &Path) -> Result<YoloStats, ConvertError> {
    super::check_unique_stems(dataset)?;
    fs::create_dir_all(output_dir).map_err(super::io_at(output_dir))?;

    let mut sorted_cats: Vec<&Category> = dataset.categories.iter().collect();
    sorted_cats.sort_by_key(|c| c.id);
    let cat_id_to_idx: HashMap<u64, usize> = sorted_cats
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id, i))
        .collect();

    let anns_by_image = super::anns_by_image(dataset);

    let mut stats = YoloStats {
        images: dataset.images.len(),
        ..Default::default()
    };

    for img in &dataset.images {
        if img.width == 0 || img.height == 0 {
            return Err(ConvertError::MissingImageDimensions(format!(
                "{} (id {})",
                img.file_name, img.id
            )));
        }

        let stem = super::file_stem(&img.file_name);

        let txt_path = output_dir.join(format!("{stem}.txt"));
        let mut file = fs::File::create(&txt_path).map_err(super::io_at(&txt_path))?;

        let w = img.width as f64;
        let h = img.height as f64;

        if let Some(anns) = anns_by_image.get(&img.id) {
            for ann in anns {
                if ann.iscrowd {
                    stats.skipped_crowd += 1;
                    continue;
                }
                let bbox = match ann.bbox {
                    Some(b) => b,
                    None => {
                        stats.skipped_no_bbox += 1;
                        continue;
                    }
                };
                let class_idx =
                    *cat_id_to_idx
                        .get(&ann.category_id)
                        .ok_or(ConvertError::UnknownCategory {
                            ann_id: ann.id,
                            category_id: ann.category_id,
                        })?;

                let [x, y, bw, bh] = bbox;
                let cx = (x + bw / 2.0) / w;
                let cy = (y + bh / 2.0) / h;
                let nw = bw / w;
                let nh = bh / h;

                writeln!(file, "{class_idx} {cx:.6} {cy:.6} {nw:.6} {nh:.6}")?;
                stats.annotations += 1;
            }
        }
    }

    // Write data.yaml (hand-rolled; no serde_yaml dep needed for this two-field format)
    let yaml_path = output_dir.join("data.yaml");
    let mut yaml_file = fs::File::create(&yaml_path).map_err(super::io_at(&yaml_path))?;
    writeln!(yaml_file, "nc: {}", sorted_cats.len())?;
    let names_csv: Vec<Cow<'_, str>> = sorted_cats
        .iter()
        .map(|c| yaml_scalar(c.name.as_str()))
        .collect();
    writeln!(yaml_file, "names: [{}]", names_csv.join(", "))?;

    Ok(stats)
}

/// Convert a YOLO label directory back to COCO format.
///
/// Reads `data.yaml` from `yolo_dir` for the category list, then parses every `.txt`
/// label file in the directory. Returns a [`Dataset`] with sequential image and
/// annotation IDs starting at 1. Each image's `file_name` is the label file's
/// stem — YOLO does not record the image extension, and none is invented.
///
/// # Image dimensions
///
/// `image_dims` maps filename stems such as `"000042"`, or stems with extensions
/// such as `"000042.jpg"`, to `(width, height)` in pixels. Both forms are tried;
/// common image extensions (`jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`) are also
/// checked as fallbacks. YOLO coordinates are normalized, so a stem with no
/// usable dimensions is [`ConvertError::MissingImageDimensions`] — denormalizing
/// by an unknown size would silently produce zero-size boxes.
///
/// # Errors
///
/// Returns [`ConvertError::MissingDataYaml`] if no `data.yaml` is present, and
/// [`ConvertError::ParseError`] — naming the file and line — if a label line
/// does not have exactly 5 numeric fields or contains an out-of-range
/// `class_idx`, or if `data.yaml` has no parseable `names` entry.
pub fn yolo_to_coco(
    yolo_dir: &Path,
    image_dims: &HashMap<String, (u32, u32)>,
) -> Result<Dataset, ConvertError> {
    let yaml_path = yolo_dir.join("data.yaml");
    if !yaml_path.exists() {
        return Err(ConvertError::MissingDataYaml);
    }
    let yaml_content = fs::read_to_string(&yaml_path).map_err(super::io_at(&yaml_path))?;
    let names = parse_data_yaml(&yaml_path, &yaml_content)?;

    let categories: Vec<Category> = names
        .iter()
        .enumerate()
        .map(|(i, name)| Category {
            id: (i + 1) as u64,
            name: name.clone(),
            ..Default::default()
        })
        .collect();

    // Collect and sort .txt files for deterministic ordering. The extension is
    // compared as bytes so a non-UTF-8 name is caught by `utf8_stem` below
    // rather than silently filtered out here.
    let mut txt_files: Vec<PathBuf> = fs::read_dir(yolo_dir)
        .map_err(super::io_at(yolo_dir))?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension() == Some(OsStr::new("txt")) {
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
        let stem = super::utf8_stem(txt_path)?.to_string();

        let (width, height) = super::lookup_image_dims(image_dims, &stem)
            .ok_or_else(|| ConvertError::MissingImageDimensions(stem.clone()))?;

        images.push(Image {
            id: img_id,
            file_name: stem.clone(),
            width,
            height,
            ..Default::default()
        });

        let content = fs::read_to_string(txt_path).map_err(super::io_at(txt_path))?;
        let w = f64::from(width);
        let h = f64::from(height);

        for (line_idx, line) in content.lines().enumerate() {
            let line_no = line_idx + 1;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 5 {
                return Err(line_err(
                    txt_path,
                    line_no,
                    format!("expected 5 fields, got {} in: {line}", parts.len()),
                ));
            }
            let field = |idx: usize, name: &str| -> Result<f64, ConvertError> {
                parts[idx].parse::<f64>().map_err(|_| {
                    line_err(txt_path, line_no, format!("invalid {name}: {}", parts[idx]))
                })
            };
            let class_idx: usize = parts[0].parse().map_err(|_| {
                line_err(
                    txt_path,
                    line_no,
                    format!("invalid class_idx: {}", parts[0]),
                )
            })?;
            let cx = field(1, "cx")?;
            let cy = field(2, "cy")?;
            let bw = field(3, "width")?;
            let bh = field(4, "height")?;

            if class_idx >= categories.len() {
                return Err(line_err(
                    txt_path,
                    line_no,
                    format!(
                        "class_idx {class_idx} out of range (nc={})",
                        categories.len()
                    ),
                ));
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
        categories,
        licenses: vec![],
    })
}

/// Quote a category name for the `names:` flow list when writing it bare would
/// change the YAML structure — the flow-list mirror of `oid.rs`'s `csv_field`.
/// A comma written bare would split the name and shift every later class index.
fn yaml_scalar(name: &str) -> Cow<'_, str> {
    let needs_quoting = name.is_empty()
        || name != name.trim()
        || name.starts_with('-')
        || name.contains([
            ',', ':', '#', '[', ']', '{', '}', '\'', '"', '\n', '&', '*', '?', '|', '>', '!', '%',
            '@', '`',
        ]);
    if needs_quoting {
        Cow::Owned(format!("'{}'", name.replace('\'', "''")))
    } else {
        Cow::Borrowed(name)
    }
}

/// Strip one level of YAML quoting from a scalar.
///
/// Handles the single-quoted form (with `''` escapes) this module writes and
/// the double-quoted form (with `\"` / `\\` escapes) other tools may write.
fn yaml_unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('\'') && s.ends_with('\'') {
        s[1..s.len() - 1].replace("''", "'")
    } else if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
    } else {
        s.to_string()
    }
}

/// Split the inside of a YAML flow list on commas, honoring quoted items so a
/// name like `'Camera, still'` stays one item.
fn split_yaml_flow(inner: &str) -> Vec<String> {
    let mut items: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for c in inner.chars() {
        match quote {
            Some(q) => {
                current.push(c);
                if c == q {
                    // `''` inside a single-quoted scalar re-opens the quote on
                    // the next iteration, which is exactly right: the escape
                    // stays intact for `yaml_unquote` to resolve.
                    quote = None;
                }
            }
            None => match c {
                '\'' | '"' => {
                    current.push(c);
                    quote = Some(c);
                }
                ',' => items.push(std::mem::take(&mut current)),
                _ => current.push(c),
            },
        }
    }
    items.push(current);

    items
        .iter()
        .map(|item| yaml_unquote(item))
        .filter(|item| !item.is_empty())
        .collect()
}

/// Parse `data.yaml` and return the class names in index order.
///
/// Accepts every form Ultralytics tooling reads or writes:
///
/// ```yaml
/// names: [person, bicycle]      # flow list (what coco_to_yolo writes)
/// names:
///   0: person                   # block mapping (Ultralytics default)
///   1: bicycle
/// names:
///   - person                    # block list
///   - bicycle
/// ```
fn parse_data_yaml(yaml_path: &Path, content: &str) -> Result<Vec<String>, ConvertError> {
    let lines: Vec<&str> = content.lines().collect();
    for (i, raw) in lines.iter().enumerate() {
        let trimmed = raw.trim();
        let Some(rest) = trimmed.strip_prefix("names:") else {
            continue;
        };
        let rest = rest.trim();
        if rest.is_empty() || rest.starts_with('#') {
            let indent = raw.len() - raw.trim_start().len();
            return parse_block_names(yaml_path, &lines, i + 1, indent);
        }
        return parse_flow_names(yaml_path, i + 1, rest);
    }
    Err(parse_err(yaml_path, "no `names` field found in data.yaml"))
}

/// Parse the flow form `names: [a, b, 'c, d']`.
fn parse_flow_names(
    yaml_path: &Path,
    line_no: usize,
    rest: &str,
) -> Result<Vec<String>, ConvertError> {
    let inner = rest
        .strip_prefix('[')
        .and_then(|r| r.strip_suffix(']'))
        .ok_or_else(|| {
            line_err(
                yaml_path,
                line_no,
                format!("expected a flow list after `names:`, got `{rest}`"),
            )
        })?;
    Ok(split_yaml_flow(inner))
}

/// Parse the block forms following a bare `names:` line — either the
/// `index: name` mapping Ultralytics writes by default, or a `- name` list.
///
/// Mapping indices must be exactly `0..n` with no gaps or duplicates, because
/// they become the label files' class indices.
fn parse_block_names(
    yaml_path: &Path,
    lines: &[&str],
    start: usize,
    names_indent: usize,
) -> Result<Vec<String>, ConvertError> {
    let mut list_items: Vec<String> = Vec::new();
    let mut map_items: Vec<(usize, String, usize)> = Vec::new(); // (class_idx, name, line_no)

    for (offset, raw) in lines[start..].iter().enumerate() {
        let line_no = start + offset + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = raw.len() - raw.trim_start().len();
        if indent <= names_indent {
            break; // next top-level key
        }
        if let Some(item) = trimmed.strip_prefix('-') {
            list_items.push(yaml_unquote(item));
        } else if let Some((key, value)) = trimmed.split_once(':') {
            let idx: usize = key.trim().parse().map_err(|_| {
                line_err(
                    yaml_path,
                    line_no,
                    format!("expected a class index before `:`, got `{key}`"),
                )
            })?;
            map_items.push((idx, yaml_unquote(value), line_no));
        } else {
            return Err(line_err(
                yaml_path,
                line_no,
                format!("cannot parse names entry `{trimmed}`"),
            ));
        }
    }

    match (list_items.is_empty(), map_items.is_empty()) {
        (false, true) => Ok(list_items),
        (true, false) => {
            map_items.sort_by_key(|&(idx, ..)| idx);
            let mut names = Vec::with_capacity(map_items.len());
            for (expected, (idx, name, line_no)) in map_items.into_iter().enumerate() {
                if idx != expected {
                    return Err(line_err(
                        yaml_path,
                        line_no,
                        format!(
                            "names indices must be 0..n without gaps or duplicates; expected {expected}, found {idx}"
                        ),
                    ));
                }
                names.push(name);
            }
            Ok(names)
        }
        (true, true) => Err(parse_err(yaml_path, "`names:` has no entries")),
        (false, false) => Err(parse_err(
            yaml_path,
            "`names:` mixes `- name` list entries and `index: name` mapping entries",
        )),
    }
}
