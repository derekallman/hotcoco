use std::collections::HashSet;
use std::fs;
use std::io::{BufReader, BufWriter};

use quick_xml::events::{BytesEnd, BytesStart, Event};
use quick_xml::reader::Reader;
use quick_xml::writer::Writer;

use crate::types::{Annotation, Category, Dataset, Image, Segmentation};

use super::ConvertError;

/// Statistics returned by [`coco_to_cvat`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CvatStats {
    /// Images written (every image in the dataset gets an `<image>` element).
    pub images: usize,
    /// `<box>` elements written.
    pub boxes: usize,
    /// `<polygon>` elements written.
    pub polygons: usize,
    /// Annotations that produced no element: neither bbox nor usable polygon.
    pub skipped_no_geometry: usize,
    /// Polygons skipped because they have fewer than 3 points (CVAT rejects
    /// them on import, so writing them would produce a file that cannot
    /// round-trip). Counted per polygon, not per annotation.
    pub skipped_degenerate: usize,
}

/// Statistics returned by [`cvat_to_coco`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CvatImportStats {
    /// `<image>` elements read.
    pub images: usize,
    /// `<box>` shapes imported.
    pub boxes: usize,
    /// `<polygon>` shapes imported.
    pub polygons: usize,
    /// Polygons skipped because they have fewer than 3 points.
    pub skipped_degenerate: usize,
    /// Shapes of kinds COCO cannot express (`<polyline>`, `<points>`,
    /// `<cuboid>`, `<mask>`, ...), skipped together with their children.
    pub skipped_unsupported: usize,
}

/// Convert a COCO dataset to CVAT for Images 1.1 XML format.
///
/// Writes a single XML file at `output_path` containing all images and annotations.
///
/// # Field mapping
///
/// - COCO `bbox [x, y, w, h]` → CVAT `<box xtl="x" ytl="y" xbr="x+w" ybr="y+h">`
/// - COCO `Segmentation::Polygon` → CVAT `<polygon points="x0,y0;x1,y1;...">`
///   (one `<polygon>` per polygon with at least 3 points)
/// - Annotations with a usable polygon get `<polygon>` elements and no `<box>`;
///   annotations whose polygons are all degenerate fall back to their bbox
/// - Annotations with neither are counted in [`CvatStats::skipped_no_geometry`]
///
/// # Errors
///
/// Returns [`ConvertError::Io`] on filesystem errors,
/// [`ConvertError::XmlError`] on XML writing failures, or
/// [`ConvertError::UnknownCategory`] if an annotation references a category id
/// that is not in the dataset.
pub fn coco_to_cvat(
    dataset: &Dataset,
    output_path: &std::path::Path,
) -> Result<CvatStats, ConvertError> {
    let cat_name = crate::types::cat_id_to_name(dataset);

    let anns_by_image = super::anns_by_image(dataset);

    let file = fs::File::create(output_path)?;
    let buf = BufWriter::new(file);
    let mut writer = Writer::new_with_indent(buf, b' ', 2);

    // <annotations>
    writer.write_event(Event::Start(BytesStart::new("annotations")))?;
    write_text_element(&mut writer, "version", "1.1")?;

    // <meta><task><labels>...</labels></task></meta>
    writer.write_event(Event::Start(BytesStart::new("meta")))?;
    writer.write_event(Event::Start(BytesStart::new("task")))?;
    writer.write_event(Event::Start(BytesStart::new("labels")))?;
    let mut sorted_cats: Vec<&Category> = dataset.categories.iter().collect();
    sorted_cats.sort_by_key(|c| c.id);
    for cat in &sorted_cats {
        writer.write_event(Event::Start(BytesStart::new("label")))?;
        write_text_element(&mut writer, "name", &cat.name)?;
        writer.write_event(Event::End(BytesEnd::new("label")))?;
    }
    writer.write_event(Event::End(BytesEnd::new("labels")))?;
    writer.write_event(Event::End(BytesEnd::new("task")))?;
    writer.write_event(Event::End(BytesEnd::new("meta")))?;

    let mut stats = CvatStats {
        images: dataset.images.len(),
        ..Default::default()
    };

    for img in &dataset.images {
        let mut img_elem = BytesStart::new("image");
        img_elem.push_attribute(("id", img.id.to_string().as_str()));
        img_elem.push_attribute(("name", img.file_name.as_str()));
        img_elem.push_attribute(("width", img.width.to_string().as_str()));
        img_elem.push_attribute(("height", img.height.to_string().as_str()));
        writer.write_event(Event::Start(img_elem))?;

        if let Some(anns) = anns_by_image.get(&img.id) {
            for ann in anns {
                let label =
                    *cat_name
                        .get(&ann.category_id)
                        .ok_or(ConvertError::UnknownCategory {
                            ann_id: ann.id,
                            category_id: ann.category_id,
                        })?;

                // Prefer polygon segmentation if available.
                let mut wrote_shape = false;
                if let Some(polys) = ann.segmentation.as_ref().and_then(Segmentation::polygons) {
                    for poly in polys.iter() {
                        if poly.len() < 6 {
                            // Fewer than 3 points — the importer (ours and
                            // CVAT's) rejects these, so don't write them.
                            stats.skipped_degenerate += 1;
                            continue;
                        }
                        let points_str = poly
                            .chunks_exact(2)
                            .map(|p| format!("{:.2},{:.2}", p[0], p[1]))
                            .collect::<Vec<_>>()
                            .join(";");
                        let mut elem = BytesStart::new("polygon");
                        elem.push_attribute(("label", label));
                        elem.push_attribute(("points", points_str.as_str()));
                        elem.push_attribute(("occluded", "0"));
                        writer.write_event(Event::Empty(elem))?;
                        stats.polygons += 1;
                        wrote_shape = true;
                    }
                }
                if !wrote_shape {
                    if let Some([x, y, w, h]) = ann.bbox {
                        let mut elem = BytesStart::new("box");
                        elem.push_attribute(("label", label));
                        elem.push_attribute(("xtl", format!("{:.2}", x).as_str()));
                        elem.push_attribute(("ytl", format!("{:.2}", y).as_str()));
                        elem.push_attribute(("xbr", format!("{:.2}", x + w).as_str()));
                        elem.push_attribute(("ybr", format!("{:.2}", y + h).as_str()));
                        elem.push_attribute(("occluded", "0"));
                        writer.write_event(Event::Empty(elem))?;
                        stats.boxes += 1;
                    } else {
                        stats.skipped_no_geometry += 1;
                    }
                }
            }
        }

        writer.write_event(Event::End(BytesEnd::new("image")))?;
    }

    writer.write_event(Event::End(BytesEnd::new("annotations")))?;

    Ok(stats)
}

/// Convert a CVAT for Images 1.1 XML file to COCO format.
///
/// Reads a single XML file at `cvat_path`. Category ordering comes from the
/// `<meta><task><labels>` block if present; otherwise categories are sorted
/// alphabetically.
///
/// # Field mapping
///
/// - CVAT `<box>` `xtl,ytl,xbr,ybr` → COCO `bbox` `[xtl, ytl, xbr-xtl, ybr-ytl]`
/// - CVAT `<polygon>` `points` → COCO `Segmentation::Polygon` + computed bbox and area
/// - Shapes are read whether self-closing or written as open/close pairs (CVAT
///   uses the latter whenever a shape has `<attribute>` children; the
///   attributes themselves are not imported)
/// - `<polyline>`, `<points>`, `<cuboid>`, and other unsupported shape kinds →
///   skipped, counted in [`CvatImportStats::skipped_unsupported`]
/// - Polygons with fewer than 3 points → skipped, counted in
///   [`CvatImportStats::skipped_degenerate`]
///
/// # Errors
///
/// Returns [`ConvertError::XmlError`] on malformed XML or
/// [`ConvertError::ParseError`] if a required attribute (an image's `name`,
/// `width`, or `height`; a shape's `label` or coordinates) is missing or
/// unparsable. Errors name the file and the byte position where available.
pub fn cvat_to_coco(
    cvat_path: &std::path::Path,
) -> Result<(Dataset, CvatImportStats), ConvertError> {
    let file = fs::File::open(cvat_path)?;
    let parsed = parse_cvat_xml(BufReader::new(file)).map_err(|e| e.with_path(cvat_path))?;

    let (boxes, polygons) = parsed.images.iter().flat_map(|img| &img.shapes).fold(
        (0usize, 0usize),
        |(boxes, polygons), s| match s.kind {
            ShapeKind::Box { .. } => (boxes + 1, polygons),
            ShapeKind::Polygon { .. } => (boxes, polygons + 1),
        },
    );
    let stats = CvatImportStats {
        images: parsed.images.len(),
        boxes,
        polygons,
        skipped_degenerate: parsed.skipped_degenerate,
        skipped_unsupported: parsed.skipped_unsupported,
    };

    let names = derive_category_names(parsed.meta_labels, &parsed.images);
    Ok((build_dataset(parsed.images, names), stats))
}

/// What one CVAT XML file yields: the declared label list, the images, and the
/// per-shape skip counts.
struct ParsedCvat {
    /// Labels from the `<meta><task><labels>` block, in the order declared there.
    /// Empty when the file has no `<meta>` block.
    meta_labels: Vec<String>,
    images: Vec<ParsedCvatImage>,
    /// Polygons dropped for having fewer than 3 points.
    skipped_degenerate: usize,
    /// Shapes of kinds COCO cannot express, dropped with their children.
    skipped_unsupported: usize,
}

/// Read a CVAT for Images 1.1 document into its labels and images.
///
/// Shapes arrive either self-closing (`<box .../>`) or as open/close pairs —
/// CVAT writes the pair form whenever a shape carries `<attribute>` children.
/// Both are handled: geometry always lives in the element's own attributes, so
/// the children are skipped wholesale. Unsupported shape kinds cost their
/// annotation (counted), not the file.
fn parse_cvat_xml<R: std::io::BufRead>(reader: R) -> Result<ParsedCvat, ConvertError> {
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut meta_labels: Vec<String> = Vec::new();
    let mut images: Vec<ParsedCvatImage> = Vec::new();
    let mut skipped_degenerate = 0usize;
    let mut skipped_unsupported = 0usize;

    // The `<meta><task><labels><label>` path, tracked one level at a time so a
    // `<label>` outside that path cannot contribute a category name.
    let mut in_meta = false;
    let mut in_task = false;
    let mut in_labels = false;
    let mut in_label = false;
    let mut current_tag: Vec<u8> = Vec::new();
    let mut label_name = String::new();

    // The `<image>` currently being filled; shapes arrive as its children.
    let mut current_image: Option<ParsedCvatImage> = None;

    let mut buf = Vec::new();
    // Scratch for skipping a shape's children (`<attribute>` etc.).
    let mut skip_buf = Vec::new();
    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name = e.name();
                let tag = name.as_ref();
                match tag {
                    b"meta" => in_meta = true,
                    b"task" if in_meta => in_task = true,
                    b"labels" if in_task => in_labels = true,
                    b"label" if in_labels => {
                        in_label = true;
                        label_name.clear();
                    }
                    b"image" => current_image = Some(parse_image_attrs(e)?),
                    // CVAT writes a shape as an open/close pair whenever it has
                    // `<attribute>` children. The geometry lives on the element
                    // itself, so parse it here and skip the children.
                    b"box" | b"polygon" => {
                        let pos = xml.buffer_position();
                        let img = current_image
                            .as_mut()
                            .ok_or_else(|| shape_outside_image(tag, pos))?;
                        match parse_shape_attrs(tag, e).map_err(|err| at_byte(err, pos))? {
                            Some(shape) => img.shapes.push(shape),
                            None => skipped_degenerate += 1,
                        }
                        let end = e.to_end().into_owned();
                        skip_buf.clear();
                        xml.read_to_end_into(end.name(), &mut skip_buf)?;
                    }
                    // Any other element inside an <image> is a shape kind COCO
                    // cannot express — skipped with its children, and counted.
                    _ if current_image.is_some() => {
                        skipped_unsupported += 1;
                        let end = e.to_end().into_owned();
                        skip_buf.clear();
                        xml.read_to_end_into(end.name(), &mut skip_buf)?;
                    }
                    _ => {}
                }
                current_tag = tag.to_vec();
            }
            Ok(Event::Empty(ref e)) => {
                let name = e.name();
                let tag = name.as_ref();
                match tag {
                    // A self-closing image has no shapes.
                    b"image" => images.push(parse_image_attrs(e)?),
                    b"box" | b"polygon" => {
                        let pos = xml.buffer_position();
                        let img = current_image
                            .as_mut()
                            .ok_or_else(|| shape_outside_image(tag, pos))?;
                        match parse_shape_attrs(tag, e).map_err(|err| at_byte(err, pos))? {
                            Some(shape) => img.shapes.push(shape),
                            None => skipped_degenerate += 1,
                        }
                    }
                    _ if current_image.is_some() => skipped_unsupported += 1,
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let name = e.name();
                match name.as_ref() {
                    b"meta" => in_meta = false,
                    b"task" => in_task = false,
                    b"labels" => in_labels = false,
                    b"label" if in_label => {
                        if !label_name.is_empty() {
                            meta_labels.push(std::mem::take(&mut label_name));
                        }
                        in_label = false;
                    }
                    b"image" => {
                        if let Some(img) = current_image.take() {
                            images.push(img);
                        }
                    }
                    _ => {}
                }
                current_tag.clear();
            }
            Ok(Event::Text(ref e)) => {
                if in_label && current_tag == b"name" {
                    let text = e
                        .decode()
                        .map_err(|err| ConvertError::XmlError(format!("invalid text: {err}")))?;
                    label_name = text.trim().to_string();
                }
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

    Ok(ParsedCvat {
        meta_labels,
        images,
        skipped_degenerate,
        skipped_unsupported,
    })
}

/// Category names in the order they become ids (the first is id 1).
///
/// A `<meta>` label list is authoritative when present — it carries the project's
/// own label order, so ids follow it, and any label used by a shape but missing
/// from `<meta>` is appended in first-use order. With no `<meta>` block there is
/// no authored order to preserve, so discovered labels are sorted instead, which
/// is what keeps ids reproducible across runs.
fn derive_category_names(meta_labels: Vec<String>, images: &[ParsedCvatImage]) -> Vec<String> {
    let had_meta = !meta_labels.is_empty();
    let mut seen: HashSet<String> = meta_labels.iter().cloned().collect();
    let mut names = meta_labels;

    for shape in images.iter().flat_map(|img| &img.shapes) {
        if seen.insert(shape.label.clone()) {
            names.push(shape.label.clone());
        }
    }

    if !had_meta {
        names.sort();
    }
    names
}

/// Assemble the COCO dataset, numbering images and annotations from 1.
///
/// A shape whose label is not in `category_names` is dropped — that can only
/// happen if the name list did not come from [`derive_category_names`], which
/// collects every label it sees.
fn build_dataset(parsed_images: Vec<ParsedCvatImage>, category_names: Vec<String>) -> Dataset {
    let categories: Vec<Category> = category_names
        .into_iter()
        .enumerate()
        .map(|(i, name)| Category {
            id: (i + 1) as u64,
            name,
            ..Default::default()
        })
        .collect();

    let name_to_id = crate::types::cat_name_to_id(&categories);

    let mut images: Vec<Image> = Vec::new();
    let mut annotations: Vec<Annotation> = Vec::new();

    for (i, parsed) in parsed_images.iter().enumerate() {
        let img_id = (i + 1) as u64;
        images.push(Image {
            id: img_id,
            file_name: parsed.name.clone(),
            width: parsed.width,
            height: parsed.height,
            ..Default::default()
        });

        for shape in &parsed.shapes {
            let Some(&category_id) = name_to_id.get(shape.label.as_str()) else {
                continue;
            };
            let ann_id = (annotations.len() + 1) as u64;
            annotations.push(shape_to_annotation(
                ann_id,
                img_id,
                category_id,
                &shape.kind,
            ));
        }
    }

    Dataset {
        info: None,
        images,
        annotations,
        categories,
        licenses: vec![],
    }
}

/// One CVAT shape as a COCO annotation.
///
/// A box carries its extent directly; a polygon has its bbox and area derived
/// from the points and keeps them as its segmentation too. Every other field is
/// absent — a CVAT shape has no crowd flag, keypoints, or score.
fn shape_to_annotation(id: u64, image_id: u64, category_id: u64, kind: &ShapeKind) -> Annotation {
    let (bbox, area, segmentation) = match kind {
        ShapeKind::Box { xtl, ytl, xbr, ybr } => {
            let (w, h) = (xbr - xtl, ybr - ytl);
            ([*xtl, *ytl, w, h], w * h, None)
        }
        ShapeKind::Polygon { points } => (
            crate::geometry::polygon_bbox(points),
            crate::geometry::polygon_area(points),
            Some(Segmentation::Polygon(vec![
                points.iter().flat_map(|&(x, y)| [x, y]).collect(),
            ])),
        ),
    };

    Annotation {
        id,
        image_id,
        category_id,
        bbox: Some(bbox),
        area: Some(area),
        segmentation,
        ..Default::default()
    }
}

// ── Internal types ───────────────────────────────────────────────────────────

struct ParsedCvatImage {
    name: String,
    width: u32,
    height: u32,
    shapes: Vec<ParsedCvatShape>,
}

struct ParsedCvatShape {
    label: String,
    kind: ShapeKind,
}

enum ShapeKind {
    Box {
        xtl: f64,
        ytl: f64,
        xbr: f64,
        ybr: f64,
    },
    Polygon {
        points: Vec<(f64, f64)>,
    },
}

// ── Internal helpers ─────────────────────────────────────────────────────────

use super::{at_byte, write_text_element};

/// The error for a shape encountered outside any `<image>` — the signature of a
/// CVAT for *video* export, whose shapes live under `<track>` elements.
fn shape_outside_image(tag: &[u8], pos: u64) -> ConvertError {
    ConvertError::ParseError(format!(
        "near byte {pos}: found <{}> outside an <image> element — is this a CVAT for video export? Only CVAT for Images 1.1 is supported",
        String::from_utf8_lossy(tag)
    ))
}

/// Parse one `<box>` or `<polygon>` element's attributes.
///
/// `Ok(None)` is a degenerate polygon (fewer than 3 points) — the caller counts
/// it, so one bad shape costs itself, not the file.
fn parse_shape_attrs(tag: &[u8], e: &BytesStart) -> Result<Option<ParsedCvatShape>, ConvertError> {
    if tag == b"box" {
        parse_box_attrs(e).map(Some)
    } else {
        parse_polygon_attrs(e)
    }
}

/// Parse `<image>` element attributes. `name`, `width`, and `height` are all
/// required — CVAT always writes them, and the dimensions are the only record
/// of the image size the file carries.
fn parse_image_attrs(e: &BytesStart) -> Result<ParsedCvatImage, ConvertError> {
    let mut name: Option<String> = None;
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;

    for attr in e.attributes().flatten() {
        let val = String::from_utf8_lossy(&attr.value);
        match attr.key.as_ref() {
            b"name" => name = Some(val.to_string()),
            b"width" => {
                width = Some(val.parse().map_err(|_| {
                    ConvertError::ParseError(format!("invalid image width: {val}"))
                })?);
            }
            b"height" => {
                height = Some(val.parse().map_err(|_| {
                    ConvertError::ParseError(format!("invalid image height: {val}"))
                })?);
            }
            _ => {}
        }
    }

    let missing =
        |attr: &str| ConvertError::ParseError(format!("CVAT <image> missing `{attr}` attribute"));
    Ok(ParsedCvatImage {
        name: name
            .filter(|n| !n.is_empty())
            .ok_or_else(|| missing("name"))?,
        width: width.ok_or_else(|| missing("width"))?,
        height: height.ok_or_else(|| missing("height"))?,
        shapes: Vec::new(),
    })
}

/// Parse `<box>` element attributes. The label and all four coordinates are
/// required; a missing coordinate is an error, never a silent `0.0`.
fn parse_box_attrs(e: &BytesStart) -> Result<ParsedCvatShape, ConvertError> {
    const COORDS: [&str; 4] = ["xtl", "ytl", "xbr", "ybr"];
    let mut label: Option<String> = None;
    let mut coords: [Option<f64>; 4] = [None; 4];

    for attr in e.attributes().flatten() {
        let val = String::from_utf8_lossy(&attr.value);
        let key = attr.key.as_ref();
        if key == b"label" {
            label = Some(val.to_string());
        } else if let Some(i) = COORDS.iter().position(|c| c.as_bytes() == key) {
            coords[i] =
                Some(val.parse().map_err(|_| {
                    ConvertError::ParseError(format!("invalid {}: {val}", COORDS[i]))
                })?);
        }
    }

    let label = label
        .filter(|l| !l.is_empty())
        .ok_or_else(|| ConvertError::ParseError("CVAT <box> missing `label` attribute".into()))?;
    let coord = |i: usize| {
        coords[i].ok_or_else(|| {
            ConvertError::ParseError(format!("CVAT <box> missing `{}` attribute", COORDS[i]))
        })
    };

    Ok(ParsedCvatShape {
        label,
        kind: ShapeKind::Box {
            xtl: coord(0)?,
            ytl: coord(1)?,
            xbr: coord(2)?,
            ybr: coord(3)?,
        },
    })
}

/// Parse `<polygon>` element attributes.
///
/// A missing `label` or `points` attribute is an error; a *present* points list
/// with fewer than 3 points is degenerate and returns `Ok(None)` for the caller
/// to count.
fn parse_polygon_attrs(e: &BytesStart) -> Result<Option<ParsedCvatShape>, ConvertError> {
    let mut label: Option<String> = None;
    let mut points_str: Option<String> = None;

    for attr in e.attributes().flatten() {
        match attr.key.as_ref() {
            b"label" => label = Some(String::from_utf8_lossy(&attr.value).to_string()),
            b"points" => points_str = Some(String::from_utf8_lossy(&attr.value).to_string()),
            _ => {}
        }
    }

    let label = label.filter(|l| !l.is_empty()).ok_or_else(|| {
        ConvertError::ParseError("CVAT <polygon> missing `label` attribute".into())
    })?;
    let points_str = points_str.ok_or_else(|| {
        ConvertError::ParseError("CVAT <polygon> missing `points` attribute".into())
    })?;

    let points = parse_cvat_points(&points_str)?;
    if points.len() < 3 {
        return Ok(None);
    }

    Ok(Some(ParsedCvatShape {
        label,
        kind: ShapeKind::Polygon { points },
    }))
}

/// Parse CVAT points string `"x1,y1;x2,y2;..."` into coordinate pairs.
fn parse_cvat_points(s: &str) -> Result<Vec<(f64, f64)>, ConvertError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(';')
        .map(|pair| {
            let parts: Vec<&str> = pair.split(',').collect();
            if parts.len() != 2 {
                return Err(ConvertError::ParseError(format!(
                    "invalid point pair: {pair}"
                )));
            }
            let x: f64 = parts[0]
                .trim()
                .parse()
                .map_err(|_| ConvertError::ParseError(format!("invalid x: {}", parts[0])))?;
            let y: f64 = parts[1]
                .trim()
                .parse()
                .map_err(|_| ConvertError::ParseError(format!("invalid y: {}", parts[1])))?;
            Ok((x, y))
        })
        .collect()
}
