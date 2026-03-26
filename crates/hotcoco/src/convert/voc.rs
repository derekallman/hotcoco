use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;

use quick_xml::events::{BytesEnd, BytesStart, Event};
use quick_xml::reader::Reader;
use quick_xml::writer::Writer;

use crate::types::{Annotation, Category, Dataset, Image};

use super::ConvertError;

/// Statistics returned by [`coco_to_voc`].
#[derive(Debug, Clone)]
pub struct VocStats {
    /// Total number of images processed (one `.xml` written per image).
    pub images: usize,
    /// Total number of annotations written.
    pub annotations: usize,
    /// Crowd annotations written with `<difficult>1</difficult>`.
    pub crowd_as_difficult: usize,
    /// Annotations skipped because they had no `bbox`.
    pub missing_bbox: usize,
}

/// Convert a COCO dataset to Pascal VOC annotation format.
///
/// Writes one XML file per image into `output_dir/Annotations/`, named by the image
/// filename stem (e.g. `000042.xml`). Also writes a `labels.txt` file listing all
/// category names sorted by COCO category ID.
///
/// # Field mapping
///
/// - COCO `bbox` `[x, y, w, h]` → VOC `<xmin>/<ymin>/<xmax>/<ymax>` (rounded to integers)
/// - COCO `iscrowd` → VOC `<difficult>1</difficult>` (approximate mapping)
/// - COCO segmentation, keypoints → not exported (bbox-only)
///
/// # Errors
///
/// Returns [`ConvertError::Io`] on filesystem errors or [`ConvertError::XmlError`]
/// on XML writing failures.
pub fn coco_to_voc(dataset: &Dataset, output_dir: &Path) -> Result<VocStats, ConvertError> {
    let ann_dir = output_dir.join("Annotations");
    fs::create_dir_all(&ann_dir)?;

    let cat_name: HashMap<u64, &str> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();

    let anns_by_image = super::anns_by_image(dataset);

    let mut total_annotations = 0usize;
    let mut crowd_as_difficult = 0usize;
    let mut missing_bbox = 0usize;

    for img in &dataset.images {
        let stem = super::file_stem(&img.file_name);

        let xml_path = ann_dir.join(format!("{stem}.xml"));
        let file = fs::File::create(&xml_path)?;
        let buf = BufWriter::new(file);
        let mut writer = Writer::new_with_indent(buf, b' ', 2);

        writer.write_event(Event::Start(BytesStart::new("annotation")))?;

        write_text_element(&mut writer, "folder", "Annotations")?;
        write_text_element(&mut writer, "filename", &img.file_name)?;

        writer.write_event(Event::Start(BytesStart::new("size")))?;
        write_text_element(&mut writer, "width", &img.width.to_string())?;
        write_text_element(&mut writer, "height", &img.height.to_string())?;
        write_text_element(&mut writer, "depth", "3")?;
        writer.write_event(Event::End(BytesEnd::new("size")))?;

        write_text_element(&mut writer, "segmented", "0")?;

        if let Some(anns) = anns_by_image.get(&img.id) {
            for ann in anns {
                let bbox = match ann.bbox {
                    Some(b) => b,
                    None => {
                        missing_bbox += 1;
                        continue;
                    }
                };
                let name = match cat_name.get(&ann.category_id) {
                    Some(n) => *n,
                    None => continue,
                };

                let [x, y, w, h] = bbox;
                let xmin = x.round() as i64;
                let ymin = y.round() as i64;
                let xmax = (x + w).round() as i64;
                let ymax = (y + h).round() as i64;

                writer.write_event(Event::Start(BytesStart::new("object")))?;
                write_text_element(&mut writer, "name", name)?;
                write_text_element(&mut writer, "pose", "Unspecified")?;
                write_text_element(&mut writer, "truncated", "0")?;
                write_text_element(
                    &mut writer,
                    "difficult",
                    if ann.iscrowd { "1" } else { "0" },
                )?;

                if ann.iscrowd {
                    crowd_as_difficult += 1;
                }

                writer.write_event(Event::Start(BytesStart::new("bndbox")))?;
                write_text_element(&mut writer, "xmin", &xmin.to_string())?;
                write_text_element(&mut writer, "ymin", &ymin.to_string())?;
                write_text_element(&mut writer, "xmax", &xmax.to_string())?;
                write_text_element(&mut writer, "ymax", &ymax.to_string())?;
                writer.write_event(Event::End(BytesEnd::new("bndbox")))?;

                writer.write_event(Event::End(BytesEnd::new("object")))?;
                total_annotations += 1;
            }
        }

        writer.write_event(Event::End(BytesEnd::new("annotation")))?;
    }

    // labels.txt drives category-ID ordering in voc_to_coco round-trips
    let mut sorted_cats: Vec<&Category> = dataset.categories.iter().collect();
    sorted_cats.sort_by_key(|c| c.id);
    let labels: Vec<&str> = sorted_cats.iter().map(|c| c.name.as_str()).collect();
    fs::write(output_dir.join("labels.txt"), labels.join("\n") + "\n")?;

    Ok(VocStats {
        images: dataset.images.len(),
        annotations: total_annotations,
        crowd_as_difficult,
        missing_bbox,
    })
}

/// Convert a Pascal VOC annotation directory to COCO format.
///
/// Scans for `*.xml` files in `voc_dir/Annotations/` (falls back to `voc_dir/`
/// directly). If `labels.txt` exists in `voc_dir`, uses it for canonical category
/// ordering; otherwise categories are sorted alphabetically.
///
/// # Field mapping
///
/// - VOC `<xmin>/<ymin>/<xmax>/<ymax>` → COCO `bbox` `[xmin, ymin, xmax-xmin, ymax-ymin]`
/// - VOC `<difficult>` → dropped (not equivalent to COCO `iscrowd`)
/// - VOC `<truncated>`, `<pose>` → dropped
///
/// # Errors
///
/// Returns [`ConvertError::XmlError`] on malformed XML or [`ConvertError::ParseError`]
/// if required elements are missing.
pub fn voc_to_coco(voc_dir: &Path) -> Result<Dataset, ConvertError> {
    let ann_dir = {
        let sub = voc_dir.join("Annotations");
        if sub.is_dir() {
            sub
        } else {
            voc_dir.to_path_buf()
        }
    };

    let mut xml_files: Vec<std::path::PathBuf> = fs::read_dir(&ann_dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("xml") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    xml_files.sort();

    if xml_files.is_empty() {
        return Ok(Dataset::default());
    }

    let mut parsed_images: Vec<ParsedVocImage> = Vec::new();
    let mut category_names: Vec<String> = Vec::new();
    let mut category_seen: HashSet<String> = HashSet::new();

    for xml_path in &xml_files {
        let file = fs::File::open(xml_path)?;
        let parsed = parse_voc_xml(BufReader::new(file))?;
        for obj in &parsed.objects {
            if category_seen.insert(obj.name.clone()) {
                category_names.push(obj.name.clone());
            }
        }
        parsed_images.push(parsed);
    }

    // Load labels.txt for canonical ordering, or sort alphabetically
    let labels_path = voc_dir.join("labels.txt");
    if labels_path.exists() {
        let file = fs::File::open(&labels_path)?;
        let reader = BufReader::new(file);
        let labels: Vec<String> = reader
            .lines()
            .filter_map(|line| {
                let line = line.ok()?;
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            })
            .collect();
        let mut ordered = labels;
        let ordered_set: HashSet<String> = ordered.iter().cloned().collect();
        for name in &category_names {
            if !ordered_set.contains(name) {
                ordered.push(name.clone());
            }
        }
        category_names = ordered;
    } else {
        category_names.sort();
    }

    let categories: Vec<Category> = category_names
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

    let name_to_id: HashMap<&str, u64> =
        categories.iter().map(|c| (c.name.as_str(), c.id)).collect();

    let mut images: Vec<Image> = Vec::new();
    let mut annotations: Vec<Annotation> = Vec::new();
    let mut img_id = 1u64;
    let mut ann_id = 1u64;

    for parsed in &parsed_images {
        images.push(Image {
            id: img_id,
            file_name: parsed.filename.clone(),
            width: parsed.width,
            height: parsed.height,
            license: None,
            coco_url: None,
            flickr_url: None,
            date_captured: None,
            neg_category_ids: vec![],
            not_exhaustive_category_ids: vec![],
        });

        for obj in &parsed.objects {
            let category_id = match name_to_id.get(obj.name.as_str()) {
                Some(&id) => id,
                None => continue,
            };

            let x = obj.xmin as f64;
            let y = obj.ymin as f64;
            let w = (obj.xmax - obj.xmin) as f64;
            let h = (obj.ymax - obj.ymin) as f64;

            annotations.push(Annotation {
                id: ann_id,
                image_id: img_id,
                category_id,
                bbox: Some([x, y, w, h]),
                area: Some(w * h),
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

// ── Internal types ───────────────────────────────────────────────────────────

use super::write_text_element;

struct ParsedVocImage {
    filename: String,
    width: u32,
    height: u32,
    objects: Vec<ParsedVocObject>,
}

struct ParsedVocObject {
    name: String,
    xmin: i64,
    ymin: i64,
    xmax: i64,
    ymax: i64,
}

/// Parse a single VOC XML annotation file from a reader.
fn parse_voc_xml<R: std::io::BufRead>(reader: R) -> Result<ParsedVocImage, ConvertError> {
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut filename = String::new();
    let mut width: u32 = 0;
    let mut height: u32 = 0;
    let mut objects: Vec<ParsedVocObject> = Vec::new();

    let mut current_tag: Vec<u8> = Vec::new();
    let mut in_size = false;
    let mut in_object = false;
    let mut in_bndbox = false;
    // VOC2012 person objects contain <part> sub-elements with their own <name>;
    // track depth to skip them (parts can theoretically nest).
    let mut part_depth: u32 = 0;

    let mut obj_name = String::new();
    let mut xmin: i64 = 0;
    let mut ymin: i64 = 0;
    let mut xmax: i64 = 0;
    let mut ymax: i64 = 0;

    let mut buf = Vec::new();
    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name = e.name();
                let tag = name.as_ref();
                match tag {
                    b"size" => in_size = true,
                    b"object" => {
                        in_object = true;
                        obj_name.clear();
                        xmin = 0;
                        ymin = 0;
                        xmax = 0;
                        ymax = 0;
                    }
                    b"bndbox" => in_bndbox = true,
                    b"part" => part_depth += 1,
                    _ => {}
                }
                current_tag = tag.to_vec();
            }
            Ok(Event::End(ref e)) => {
                let name = e.name();
                match name.as_ref() {
                    b"size" => in_size = false,
                    b"object" => {
                        if in_object {
                            objects.push(ParsedVocObject {
                                name: std::mem::take(&mut obj_name),
                                xmin,
                                ymin,
                                xmax,
                                ymax,
                            });
                        }
                        in_object = false;
                    }
                    b"bndbox" => in_bndbox = false,
                    b"part" => part_depth = part_depth.saturating_sub(1),
                    _ => {}
                }
                current_tag.clear();
            }
            Ok(Event::Text(ref e)) => {
                let text = e
                    .unescape()
                    .map_err(|err| ConvertError::XmlError(format!("invalid XML text: {err}")))?;
                let text = text.trim();

                if !in_object && !in_size && current_tag == b"filename" {
                    filename = text.to_string();
                } else if in_size && !in_object {
                    match current_tag.as_slice() {
                        b"width" => {
                            width = text.parse().map_err(|_| {
                                ConvertError::ParseError(format!("invalid width: {text}"))
                            })?;
                        }
                        b"height" => {
                            height = text.parse().map_err(|_| {
                                ConvertError::ParseError(format!("invalid height: {text}"))
                            })?;
                        }
                        _ => {}
                    }
                } else if in_object && !in_bndbox && part_depth == 0 && current_tag == b"name" {
                    obj_name = text.to_string();
                } else if in_bndbox {
                    let val: i64 = text.parse().map_err(|_| {
                        ConvertError::ParseError(format!("invalid bbox coordinate: {text}",))
                    })?;
                    match current_tag.as_slice() {
                        b"xmin" => xmin = val,
                        b"ymin" => ymin = val,
                        b"xmax" => xmax = val,
                        b"ymax" => ymax = val,
                        _ => {}
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(ConvertError::XmlError(e.to_string())),
            _ => {}
        }
        buf.clear();
    }

    if filename.is_empty() {
        return Err(ConvertError::ParseError(
            "VOC XML missing <filename> element".into(),
        ));
    }

    Ok(ParsedVocImage {
        filename,
        width,
        height,
        objects,
    })
}
