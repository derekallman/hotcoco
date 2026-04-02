mod cvat;
mod dota;
mod voc;
mod yolo;

pub use cvat::{CvatStats, coco_to_cvat, cvat_to_coco};
pub use dota::{DotaStats, coco_to_dota, dota_to_coco};
pub use voc::{VocStats, coco_to_voc, voc_to_coco};
pub use yolo::{YoloStats, coco_to_yolo, yolo_to_coco};

use std::collections::HashMap;
use std::io;
use std::path::Path;

use crate::types::{Annotation, Dataset};

/// Errors that can occur during format conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    /// An I/O error occurred while reading or writing files.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// An image has `width == 0` or `height == 0`, preventing normalization.
    #[error("image id={0} has zero width or height")]
    MissingImageDimensions(u64),
    /// No `data.yaml` found in the YOLO directory.
    #[error("data.yaml not found in YOLO directory")]
    MissingDataYaml,
    /// A label file or `data.yaml` could not be parsed.
    #[error("parse error: {0}")]
    ParseError(String),
    /// An XML parsing or writing error occurred.
    #[error("XML error: {0}")]
    XmlError(String),
}

impl From<quick_xml::Error> for ConvertError {
    fn from(e: quick_xml::Error) -> Self {
        ConvertError::XmlError(e.to_string())
    }
}

/// Group annotations by `image_id`.
pub(crate) fn anns_by_image(dataset: &Dataset) -> HashMap<u64, Vec<&Annotation>> {
    let mut map: HashMap<u64, Vec<&Annotation>> = HashMap::new();
    for ann in &dataset.annotations {
        map.entry(ann.image_id).or_default().push(ann);
    }
    map
}

/// Extract the filename stem (without extension) from a file path string.
pub(crate) fn file_stem(file_name: &str) -> &str {
    Path::new(file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file_name)
}

/// Look up image dimensions by stem; try common extensions as fallback.
pub(crate) fn lookup_image_dims(
    image_dims: &HashMap<String, (u32, u32)>,
    stem: &str,
) -> (u32, u32) {
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

/// Write a simple `<tag>text</tag>` XML element.
pub(crate) fn write_text_element<W: std::io::Write>(
    writer: &mut quick_xml::writer::Writer<W>,
    tag: &str,
    text: &str,
) -> Result<(), quick_xml::Error> {
    use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
    writer.write_event(Event::Start(BytesStart::new(tag)))?;
    writer.write_event(Event::Text(BytesText::new(text)))?;
    writer.write_event(Event::End(BytesEnd::new(tag)))?;
    Ok(())
}
