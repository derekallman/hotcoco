use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;

use quick_xml::events::{BytesEnd, BytesStart, Event};
use quick_xml::reader::Reader;
use quick_xml::writer::Writer;

use crate::types::{Annotation, Category, Dataset, Image};

use super::ConvertError;

/// Statistics returned by [`coco_to_voc`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VocStats {
    /// Images written (one `.xml` annotation file per image).
    pub images: usize,
    /// Annotations written.
    pub annotations: usize,
    /// Crowd annotations written with `<difficult>1</difficult>`.
    pub crowd_as_difficult: usize,
    /// Annotations skipped because they have no `bbox`.
    pub skipped_no_bbox: usize,
}

/// Convert a COCO dataset to Pascal VOC annotation format.
///
/// Writes one XML file per image into `output_dir/Annotations/`, named by the image
/// filename stem, for example `000042.xml`. Also writes a `labels.txt` file listing all
/// category names sorted by COCO category ID.
///
/// # Field mapping
///
/// - COCO `bbox` `[x, y, w, h]` → VOC `<xmin>/<ymin>/<xmax>/<ymax>` using the
///   VOC 1-based inclusive convention: `xmin = x + 1`, `xmax = x + w` (rounded
///   to integers). See [`voc_to_coco`] for the convention's rationale — the two
///   directions are exact inverses, so integer boxes round-trip exactly.
/// - COCO `iscrowd` → VOC `<difficult>1</difficult>` (imported back to
///   `iscrowd` by [`voc_to_coco`], so the mapping is bidirectional)
/// - COCO segmentation, keypoints → not exported (bbox-only)
///
/// # Errors
///
/// Returns [`ConvertError::Io`] on filesystem errors,
/// [`ConvertError::XmlError`] on XML writing failures,
/// [`ConvertError::StemCollision`] if two images share a filename stem (their
/// annotation files would overwrite each other), or
/// [`ConvertError::UnknownCategory`] if an annotation references a category id
/// that is not in the dataset.
pub fn coco_to_voc(dataset: &Dataset, output_dir: &Path) -> Result<VocStats, ConvertError> {
    super::check_unique_stems(dataset)?;
    let ann_dir = output_dir.join("Annotations");
    fs::create_dir_all(&ann_dir)?;

    let cat_name = crate::types::cat_id_to_name(dataset);

    let anns_by_image = super::anns_by_image(dataset);

    let mut stats = VocStats {
        images: dataset.images.len(),
        ..Default::default()
    };

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
                        stats.skipped_no_bbox += 1;
                        continue;
                    }
                };
                let name =
                    *cat_name
                        .get(&ann.category_id)
                        .ok_or(ConvertError::UnknownCategory {
                            ann_id: ann.id,
                            category_id: ann.category_id,
                        })?;

                // Inverse of the import mapping: COCO x = xmin - 1 and
                // w = xmax - xmin + 1, so xmin = x + 1 and xmax = x + w.
                let [x, y, w, h] = bbox;
                let xmin = (x + 1.0).round() as i64;
                let ymin = (y + 1.0).round() as i64;
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
                    stats.crowd_as_difficult += 1;
                }

                writer.write_event(Event::Start(BytesStart::new("bndbox")))?;
                write_text_element(&mut writer, "xmin", &xmin.to_string())?;
                write_text_element(&mut writer, "ymin", &ymin.to_string())?;
                write_text_element(&mut writer, "xmax", &xmax.to_string())?;
                write_text_element(&mut writer, "ymax", &ymax.to_string())?;
                writer.write_event(Event::End(BytesEnd::new("bndbox")))?;

                writer.write_event(Event::End(BytesEnd::new("object")))?;
                stats.annotations += 1;
            }
        }

        writer.write_event(Event::End(BytesEnd::new("annotation")))?;
    }

    // labels.txt drives category-ID ordering in voc_to_coco round-trips
    let mut sorted_cats: Vec<&Category> = dataset.categories.iter().collect();
    sorted_cats.sort_by_key(|c| c.id);
    let labels: Vec<&str> = sorted_cats.iter().map(|c| c.name.as_str()).collect();
    fs::write(output_dir.join("labels.txt"), labels.join("\n") + "\n")?;

    Ok(stats)
}

/// Convert a Pascal VOC annotation directory to COCO format.
///
/// Scans for `*.xml` files in `voc_dir/Annotations/` (falls back to `voc_dir/`
/// directly). If `labels.txt` exists in `voc_dir`, uses it for canonical category
/// ordering; otherwise categories are sorted alphabetically.
///
/// # Field mapping
///
/// - VOC `<xmin>/<ymin>/<xmax>/<ymax>` → COCO `bbox`
///   `[xmin - 1, ymin - 1, xmax - xmin + 1, ymax - ymin + 1]`. VOC coordinates
///   are 1-based pixel indexes and the box includes both endpoints — the VOC
///   devkit computes box area as `(xmax - xmin + 1) * (ymax - ymin + 1)`, and
///   pycocotools-adjacent tooling (Detectron2's `pascal_voc.py`, mmdetection's
///   VOC loader) applies the same `-1` / `+1` when converting to 0-based
///   half-open COCO coordinates. Coordinates are parsed as floats, so files
///   with values like `156.00` import fine.
/// - VOC `<difficult>` → COCO `iscrowd` (the inverse of [`coco_to_voc`]'s
///   export mapping; approximate — VOC "difficult" marks hard examples the
///   challenge ignores in scoring, the closest VOC analogue of a crowd region)
/// - VOC `<truncated>`, `<pose>`, `<part>` → dropped
///
/// # Errors
///
/// Returns [`ConvertError::XmlError`] on malformed XML or
/// [`ConvertError::ParseError`] if required elements are missing or a value is
/// unparsable — including an `<object>` with an empty `<name>`, no `<bndbox>`,
/// a missing coordinate, or an inverted box (`xmax < xmin` or `ymax < ymin`).
/// Errors name the file and the byte position where available.
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
        let parsed = parse_voc_xml(BufReader::new(file)).map_err(|e| e.with_path(xml_path))?;
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
            ..Default::default()
        })
        .collect();

    let name_to_id = crate::types::cat_name_to_id(&categories);

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
            ..Default::default()
        });

        for obj in &parsed.objects {
            let category_id = match name_to_id.get(obj.name.as_str()) {
                Some(&id) => id,
                None => continue,
            };

            // VOC 1-based inclusive → COCO 0-based half-open (see the
            // field-mapping doc above).
            let x = obj.xmin - 1.0;
            let y = obj.ymin - 1.0;
            let w = obj.xmax - obj.xmin + 1.0;
            let h = obj.ymax - obj.ymin + 1.0;

            annotations.push(Annotation {
                id: ann_id,
                image_id: img_id,
                category_id,
                bbox: Some([x, y, w, h]),
                area: Some(w * h),
                iscrowd: obj.difficult,
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

// ── Internal types ───────────────────────────────────────────────────────────

use super::{at_byte, write_text_element};

struct ParsedVocImage {
    filename: String,
    width: u32,
    height: u32,
    objects: Vec<ParsedVocObject>,
}

/// A validated `<object>`: a non-empty class name and a complete,
/// non-inverted box.
struct ParsedVocObject {
    name: String,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    difficult: bool,
}

/// An `<object>` while it is being read. Every field is optional here so a
/// missing one is detected when the object closes, rather than defaulting to
/// `0.0` and producing a box with negative width or height.
#[derive(Default)]
struct PartialVocObject {
    name: String,
    has_bndbox: bool,
    xmin: Option<f64>,
    ymin: Option<f64>,
    xmax: Option<f64>,
    ymax: Option<f64>,
    difficult: bool,
}

impl PartialVocObject {
    /// Assign one `<bndbox>` coordinate by its tag; an unrecognized tag is
    /// ignored.
    fn set_coord(&mut self, tag: &[u8], val: f64) {
        match tag {
            b"xmin" => self.xmin = Some(val),
            b"ymin" => self.ymin = Some(val),
            b"xmax" => self.xmax = Some(val),
            b"ymax" => self.ymax = Some(val),
            _ => {}
        }
    }

    /// Check the object is complete. A missing `<name>`, `<bndbox>`, or
    /// coordinate, or a box whose max is below its min, is malformed input.
    fn finish(self) -> Result<ParsedVocObject, ConvertError> {
        if self.name.is_empty() {
            return Err(ConvertError::ParseError(
                "<object> has a missing or empty <name>".into(),
            ));
        }
        if !self.has_bndbox {
            return Err(ConvertError::ParseError(format!(
                "<object> `{}` has no <bndbox>",
                self.name
            )));
        }
        let coord = |val: Option<f64>, tag: &str| {
            val.ok_or_else(|| {
                ConvertError::ParseError(format!(
                    "<object> `{}`: <bndbox> is missing <{tag}>",
                    self.name
                ))
            })
        };
        let (xmin, ymin) = (coord(self.xmin, "xmin")?, coord(self.ymin, "ymin")?);
        let (xmax, ymax) = (coord(self.xmax, "xmax")?, coord(self.ymax, "ymax")?);
        if xmax < xmin || ymax < ymin {
            return Err(ConvertError::ParseError(format!(
                "<object> `{}`: inverted <bndbox> (xmin {xmin}, xmax {xmax}, ymin {ymin}, ymax {ymax})",
                self.name
            )));
        }
        Ok(ParsedVocObject {
            name: self.name,
            xmin,
            ymin,
            xmax,
            ymax,
            difficult: self.difficult,
        })
    }
}

/// The image-level fields being filled, so the text router takes one target
/// instead of three out-parameters.
#[derive(Default)]
struct VocFields {
    filename: String,
    width: u32,
    height: u32,
}

/// Where in a VOC document the parser is, for the purpose of routing text.
///
/// VOC identifies a field by its *enclosing* element, not by tag name alone:
/// `<name>` is an object's class in one place and a body part's name in another,
/// and `<xmin>` appears inside both an object's `<bndbox>` and a part's. One
/// value naming the position is what keeps those apart — the boolean-per-element
/// form this replaced required the same guard to be restated at every use site,
/// and a missing `part_depth == 0` on the coordinate branch made every VOC2012
/// person annotation report its last part's box.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    /// Directly inside `<annotation>` — `<filename>`.
    Root,
    /// Inside `<size>` — `<width>` and `<height>`.
    Size,
    /// Inside `<object>` but outside its `<bndbox>` — `<name>`, `<difficult>`.
    Object,
    /// Inside an object's own `<bndbox>` — the four coordinates.
    ObjectBox,
    /// Anywhere inside a `<part>`. Parts describe sub-regions of an object — a
    /// person's head, hand or foot — and carry their own `<name>` and `<bndbox>`.
    /// None of it belongs to the object, so all of it is dropped.
    Part,
}

/// Parse a single VOC XML annotation file from a reader.
fn parse_voc_xml<R: std::io::BufRead>(reader: R) -> Result<ParsedVocImage, ConvertError> {
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut fields = VocFields::default();
    let mut objects: Vec<ParsedVocObject> = Vec::new();

    let mut section = Section::Root;
    // `<part>` nesting depth. Parts can in principle nest, so the section returns
    // to the enclosing object only when the outermost one closes.
    let mut part_depth: u32 = 0;
    let mut current_tag: Vec<u8> = Vec::new();
    // The object currently being filled; `None` outside any `<object>`.
    let mut object: Option<PartialVocObject> = None;

    let mut buf = Vec::new();
    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name = e.name();
                let tag = name.as_ref();
                match tag {
                    b"part" => {
                        part_depth += 1;
                        section = Section::Part;
                    }
                    // Inside a part nothing else moves the section — this arm is
                    // what stops a part's `<bndbox>` becoming the object's.
                    _ if section == Section::Part => {}
                    b"size" => section = Section::Size,
                    b"object" => {
                        section = Section::Object;
                        object = Some(PartialVocObject::default());
                    }
                    b"bndbox" => {
                        section = Section::ObjectBox;
                        if let Some(obj) = object.as_mut() {
                            obj.has_bndbox = true;
                        }
                    }
                    _ => {}
                }
                current_tag = tag.to_vec();
            }
            Ok(Event::End(ref e)) => {
                let name = e.name();
                match name.as_ref() {
                    b"part" => {
                        part_depth = part_depth.saturating_sub(1);
                        if part_depth == 0 {
                            section = Section::Object;
                        }
                    }
                    _ if section == Section::Part => {}
                    b"size" => section = Section::Root,
                    b"object" => {
                        // A stray `</object>` with no open object contributes
                        // nothing.
                        if let Some(obj) = object.take() {
                            let obj = obj
                                .finish()
                                .map_err(|err| at_byte(err, xml.buffer_position()))?;
                            objects.push(obj);
                        }
                        section = Section::Root;
                    }
                    b"bndbox" => section = Section::Object,
                    _ => {}
                }
                current_tag.clear();
            }
            Ok(Event::Text(ref e)) => {
                let text = e
                    .decode()
                    .map_err(|err| ConvertError::XmlError(format!("invalid XML text: {err}")))?;
                route_text(
                    section,
                    &current_tag,
                    text.trim(),
                    &mut fields,
                    object.as_mut(),
                )
                .map_err(|err| at_byte(err, xml.buffer_position()))?;
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(ConvertError::XmlError(format!(
                    "near byte {}: {e}",
                    xml.error_position()
                )));
            }
            _ => {}
        }
        buf.clear();
    }

    if fields.filename.is_empty() {
        return Err(ConvertError::ParseError(
            "VOC XML missing <filename> element".into(),
        ));
    }

    Ok(ParsedVocImage {
        filename: fields.filename,
        width: fields.width,
        height: fields.height,
        objects,
    })
}

/// Route one text node to the field its position names.
///
/// This is the payoff of naming the position: with [`Section`] already decided,
/// the destination is a flat table from (where, tag), and there is no per-branch
/// "am I somewhere that counts?" guard left to get wrong.
fn route_text(
    section: Section,
    tag: &[u8],
    text: &str,
    fields: &mut VocFields,
    object: Option<&mut PartialVocObject>,
) -> Result<(), ConvertError> {
    match (section, tag) {
        (Section::Root, b"filename") => fields.filename = text.to_string(),
        (Section::Size, b"width") => {
            fields.width = text
                .parse()
                .map_err(|_| ConvertError::ParseError(format!("invalid width: {text}")))?;
        }
        (Section::Size, b"height") => {
            fields.height = text
                .parse()
                .map_err(|_| ConvertError::ParseError(format!("invalid height: {text}")))?;
        }
        (Section::Object, b"name") => {
            if let Some(obj) = object {
                obj.name = text.to_string();
            }
        }
        (Section::Object, b"difficult") => {
            let val: i64 = text
                .parse()
                .map_err(|_| ConvertError::ParseError(format!("invalid difficult: {text}")))?;
            if let Some(obj) = object {
                obj.difficult = val != 0;
            }
        }
        // Every text node in a `<bndbox>` is parsed as a coordinate, named or
        // not, so a non-numeric child is an error rather than a silent no-op.
        // Parsed as f64: real-world VOC files carry values like `156.00`.
        (Section::ObjectBox, coord) => {
            let val: f64 = text.parse().map_err(|_| {
                ConvertError::ParseError(format!("invalid bbox coordinate: {text}"))
            })?;
            if let Some(obj) = object {
                obj.set_coord(coord, val);
            }
        }
        _ => {}
    }
    Ok(())
}
