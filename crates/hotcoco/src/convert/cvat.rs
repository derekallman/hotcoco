use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, BufWriter};

use quick_xml::events::{BytesEnd, BytesStart, Event};
use quick_xml::reader::Reader;
use quick_xml::writer::Writer;

use crate::types::{Annotation, Category, Dataset, Image, Segmentation};

use super::ConvertError;

/// Statistics returned by [`coco_to_cvat`].
#[derive(Debug, Clone)]
pub struct CvatStats {
    /// Total number of images written.
    pub images: usize,
    /// Number of `<box>` elements written.
    pub boxes: usize,
    /// Number of `<polygon>` elements written.
    pub polygons: usize,
    /// Annotations skipped because they had neither bbox nor segmentation.
    pub skipped_no_geometry: usize,
}

/// Convert a COCO dataset to CVAT for Images 1.1 XML format.
///
/// Writes a single XML file at `output_path` containing all images and annotations.
///
/// # Field mapping
///
/// - COCO `bbox [x, y, w, h]` → CVAT `<box xtl="x" ytl="y" xbr="x+w" ybr="y+h">`
/// - COCO `Segmentation::Polygon` → CVAT `<polygon points="x0,y0;x1,y1;...">`
///   (one `<polygon>` per polygon in the segmentation)
/// - Annotations with a polygon get both `<polygon>` and no separate `<box>`
/// - Annotations with only a bbox get a `<box>`
/// - Annotations with neither are skipped
///
/// # Errors
///
/// Returns [`ConvertError::Io`] on filesystem errors or [`ConvertError::XmlError`]
/// on XML writing failures.
pub fn coco_to_cvat(
    dataset: &Dataset,
    output_path: &std::path::Path,
) -> Result<CvatStats, ConvertError> {
    let cat_name: HashMap<u64, &str> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();

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

    let mut total_boxes = 0usize;
    let mut total_polygons = 0usize;
    let mut skipped_no_geometry = 0usize;

    for img in &dataset.images {
        let mut img_elem = BytesStart::new("image");
        img_elem.push_attribute(("id", img.id.to_string().as_str()));
        img_elem.push_attribute(("name", img.file_name.as_str()));
        img_elem.push_attribute(("width", img.width.to_string().as_str()));
        img_elem.push_attribute(("height", img.height.to_string().as_str()));
        writer.write_event(Event::Start(img_elem))?;

        if let Some(anns) = anns_by_image.get(&img.id) {
            for ann in anns {
                let label = match cat_name.get(&ann.category_id) {
                    Some(n) => *n,
                    None => continue,
                };

                // Prefer polygon segmentation if available
                if let Some(Segmentation::Polygon(ref polys)) = ann.segmentation {
                    for poly in polys {
                        if poly.len() < 4 {
                            continue; // need at least 2 points
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
                        total_polygons += 1;
                    }
                } else if let Some([x, y, w, h]) = ann.bbox {
                    // Bbox only — no segmentation
                    let mut elem = BytesStart::new("box");
                    elem.push_attribute(("label", label));
                    elem.push_attribute(("xtl", format!("{:.2}", x).as_str()));
                    elem.push_attribute(("ytl", format!("{:.2}", y).as_str()));
                    elem.push_attribute(("xbr", format!("{:.2}", x + w).as_str()));
                    elem.push_attribute(("ybr", format!("{:.2}", y + h).as_str()));
                    elem.push_attribute(("occluded", "0"));
                    writer.write_event(Event::Empty(elem))?;
                    total_boxes += 1;
                } else {
                    skipped_no_geometry += 1;
                }
            }
        }

        writer.write_event(Event::End(BytesEnd::new("image")))?;
    }

    writer.write_event(Event::End(BytesEnd::new("annotations")))?;

    Ok(CvatStats {
        images: dataset.images.len(),
        boxes: total_boxes,
        polygons: total_polygons,
        skipped_no_geometry,
    })
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
/// - `<polyline>`, `<points>`, `<cuboid>` → skipped
///
/// # Errors
///
/// Returns [`ConvertError::XmlError`] on malformed XML or [`ConvertError::ParseError`]
/// if required attributes are missing.
pub fn cvat_to_coco(cvat_path: &std::path::Path) -> Result<Dataset, ConvertError> {
    let file = fs::File::open(cvat_path)?;
    let mut xml = Reader::from_reader(BufReader::new(file));
    xml.config_mut().trim_text(true);

    let mut meta_labels: Vec<String> = Vec::new();
    let mut parsed_images: Vec<ParsedCvatImage> = Vec::new();

    // State machine
    let mut in_meta = false;
    let mut in_task = false;
    let mut in_labels = false;
    let mut in_label = false;
    let mut current_tag: Vec<u8> = Vec::new();
    let mut label_name = String::new();

    // Current image being parsed
    let mut current_image: Option<ParsedCvatImage> = None;

    let mut buf = Vec::new();
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
                    b"image" => {
                        current_image = Some(parse_image_attrs(e)?);
                    }
                    _ => {}
                }
                current_tag = tag.to_vec();
            }
            Ok(Event::Empty(ref e)) => {
                let name = e.name();
                match name.as_ref() {
                    b"box" => {
                        if let Some(ref mut img) = current_image {
                            img.shapes.push(parse_box_attrs(e)?);
                        }
                    }
                    b"polygon" => {
                        if let Some(ref mut img) = current_image {
                            img.shapes.push(parse_polygon_attrs(e)?);
                        }
                    }
                    // Skip unsupported shape types
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
                            parsed_images.push(img);
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
            Err(e) => return Err(ConvertError::XmlError(e.to_string())),
            _ => {}
        }
        buf.clear();
    }

    // Build category list — prefer meta labels, fall back to collecting from shapes
    let category_names: Vec<String> = if !meta_labels.is_empty() {
        let mut names = meta_labels;
        let mut seen: HashSet<String> = names.iter().cloned().collect();
        for img in &parsed_images {
            for shape in &img.shapes {
                if seen.insert(shape.label.clone()) {
                    names.push(shape.label.clone());
                }
            }
        }
        names
    } else {
        let mut names = Vec::new();
        let mut seen = HashSet::new();
        for img in &parsed_images {
            for shape in &img.shapes {
                if seen.insert(shape.label.clone()) {
                    names.push(shape.label.clone());
                }
            }
        }
        names.sort();
        names
    };

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
            file_name: parsed.name.clone(),
            width: parsed.width,
            height: parsed.height,
            license: None,
            coco_url: None,
            flickr_url: None,
            date_captured: None,
            neg_category_ids: vec![],
            not_exhaustive_category_ids: vec![],
            video_id: None,
            frame_index: None,
        });

        for shape in &parsed.shapes {
            let category_id = match name_to_id.get(shape.label.as_str()) {
                Some(&id) => id,
                None => continue,
            };

            match &shape.kind {
                ShapeKind::Box { xtl, ytl, xbr, ybr } => {
                    let w = xbr - xtl;
                    let h = ybr - ytl;
                    annotations.push(Annotation {
                        id: ann_id,
                        image_id: img_id,
                        category_id,
                        bbox: Some([*xtl, *ytl, w, h]),
                        area: Some(w * h),
                        segmentation: None,
                        iscrowd: false,
                        keypoints: None,
                        num_keypoints: None,
                        obb: None,
                        score: None,
                        is_group_of: None,
                        track_id: None,
                        video_id: None,
                    });
                    ann_id += 1;
                }
                ShapeKind::Polygon { points } => {
                    let bbox = polygon_bbox(points);
                    let area = polygon_area(points);
                    let flat: Vec<f64> = points.iter().flat_map(|&(x, y)| [x, y]).collect();
                    annotations.push(Annotation {
                        id: ann_id,
                        image_id: img_id,
                        category_id,
                        bbox: Some(bbox),
                        area: Some(area),
                        segmentation: Some(Segmentation::Polygon(vec![flat])),
                        iscrowd: false,
                        keypoints: None,
                        num_keypoints: None,
                        obb: None,
                        score: None,
                        is_group_of: None,
                        track_id: None,
                        video_id: None,
                    });
                    ann_id += 1;
                }
            }
        }
        img_id += 1;
    }

    Ok(Dataset {
        info: None,
        images,
        annotations,
        categories,
        licenses: vec![],
        videos: vec![],
        tracks: vec![],
    })
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

use super::write_text_element;

/// Parse `<image>` element attributes.
fn parse_image_attrs(e: &BytesStart) -> Result<ParsedCvatImage, ConvertError> {
    let mut name = String::new();
    let mut width: u32 = 0;
    let mut height: u32 = 0;

    for attr in e.attributes().flatten() {
        match attr.key.as_ref() {
            b"name" => {
                name = String::from_utf8_lossy(&attr.value).to_string();
            }
            b"width" => {
                width = String::from_utf8_lossy(&attr.value)
                    .parse()
                    .map_err(|_| ConvertError::ParseError("invalid image width".into()))?;
            }
            b"height" => {
                height = String::from_utf8_lossy(&attr.value)
                    .parse()
                    .map_err(|_| ConvertError::ParseError("invalid image height".into()))?;
            }
            _ => {}
        }
    }

    if name.is_empty() {
        return Err(ConvertError::ParseError(
            "CVAT <image> missing 'name' attribute".into(),
        ));
    }

    Ok(ParsedCvatImage {
        name,
        width,
        height,
        shapes: Vec::new(),
    })
}

/// Parse `<box>` element attributes.
fn parse_box_attrs(e: &BytesStart) -> Result<ParsedCvatShape, ConvertError> {
    let mut label = String::new();
    let mut xtl: f64 = 0.0;
    let mut ytl: f64 = 0.0;
    let mut xbr: f64 = 0.0;
    let mut ybr: f64 = 0.0;

    for attr in e.attributes().flatten() {
        let val = String::from_utf8_lossy(&attr.value);
        match attr.key.as_ref() {
            b"label" => label = val.to_string(),
            b"xtl" => {
                xtl = val
                    .parse()
                    .map_err(|_| ConvertError::ParseError(format!("invalid xtl: {val}")))?;
            }
            b"ytl" => {
                ytl = val
                    .parse()
                    .map_err(|_| ConvertError::ParseError(format!("invalid ytl: {val}")))?;
            }
            b"xbr" => {
                xbr = val
                    .parse()
                    .map_err(|_| ConvertError::ParseError(format!("invalid xbr: {val}")))?;
            }
            b"ybr" => {
                ybr = val
                    .parse()
                    .map_err(|_| ConvertError::ParseError(format!("invalid ybr: {val}")))?;
            }
            _ => {}
        }
    }

    if label.is_empty() {
        return Err(ConvertError::ParseError(
            "CVAT <box> missing 'label' attribute".into(),
        ));
    }

    Ok(ParsedCvatShape {
        label,
        kind: ShapeKind::Box { xtl, ytl, xbr, ybr },
    })
}

/// Parse `<polygon>` element attributes.
fn parse_polygon_attrs(e: &BytesStart) -> Result<ParsedCvatShape, ConvertError> {
    let mut label = String::new();
    let mut points_str = String::new();

    for attr in e.attributes().flatten() {
        match attr.key.as_ref() {
            b"label" => label = String::from_utf8_lossy(&attr.value).to_string(),
            b"points" => points_str = String::from_utf8_lossy(&attr.value).to_string(),
            _ => {}
        }
    }

    if label.is_empty() {
        return Err(ConvertError::ParseError(
            "CVAT <polygon> missing 'label' attribute".into(),
        ));
    }

    let points = parse_cvat_points(&points_str)?;

    // A polygon needs at least 3 vertices (6 coordinate values when flattened)
    if points.len() < 3 {
        return Err(ConvertError::ParseError(format!(
            "CVAT <polygon> has {} points, need at least 3",
            points.len()
        )));
    }

    Ok(ParsedCvatShape {
        label,
        kind: ShapeKind::Polygon { points },
    })
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

/// Compute polygon area using the shoelace formula.
fn polygon_area(points: &[(f64, f64)]) -> f64 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i].0 * points[j].1;
        area -= points[j].0 * points[i].1;
    }
    area.abs() / 2.0
}

/// Compute the bounding rectangle of a polygon as `[x, y, w, h]`.
fn polygon_bbox(points: &[(f64, f64)]) -> [f64; 4] {
    if points.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let mut xmin = f64::MAX;
    let mut ymin = f64::MAX;
    let mut xmax = f64::MIN;
    let mut ymax = f64::MIN;
    for &(x, y) in points {
        xmin = xmin.min(x);
        ymin = ymin.min(y);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
    }
    [xmin, ymin, xmax - xmin, ymax - ymin]
}
