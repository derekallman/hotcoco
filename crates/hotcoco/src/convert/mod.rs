//! Format converters: COCO ↔ YOLO, Pascal VOC, CVAT, DOTA, and Open Images.
//!
//! Five formats, one design. Every converter follows the same policies, so
//! learning one is learning all of them.
//!
//! # Malformed input is an error; inexpressible input is a counted skip
//!
//! A record that is *wrong* — an unparsable number, a missing required
//! attribute or column, the wrong field count — fails the conversion with a
//! [`ConvertError`] naming the file and, where one exists, the line or byte
//! position. A record the target format *cannot express* — an
//! unsupported CVAT shape kind, a polygon with fewer than three points, a crowd
//! annotation in YOLO — is skipped and counted in the returned stats struct.
//! Nothing is ever dropped without either a counter or an error. An annotation
//! whose `category_id` is not in the dataset is a data-integrity error
//! ([`ConvertError::UnknownCategory`]), not a skip.
//!
//! # Missing image dimensions are an error where geometry depends on them
//!
//! YOLO and Open Images store coordinates normalized by the image size, so
//! converting them in either direction must scale by the true width/height: a
//! zero or unknown dimension is [`ConvertError::MissingImageDimensions`]. The
//! one documented exception is [`oid_to_coco`], which keeps normalized
//! coordinates against a 1×1 image when no dimensions are supplied — IoU and
//! IoA (and therefore Open Images AP) are unaffected, only absolute areas lose
//! meaning. Where dimensions are pure metadata (DOTA's corners are absolute
//! pixels), an unknown size is recorded as `0×0` and never multiplied into
//! geometry.
//!
//! # `file_name` is never invented
//!
//! CVAT and VOC record the image file name, and it round-trips verbatim. YOLO,
//! DOTA, and Open Images record only a file stem or image ID, so their
//! importers use that stem as `file_name` without fabricating an extension
//! (dimension lookups accept both the bare stem and stem + common image
//! extensions). Exporters key per-image output by the stem of `file_name`; two
//! images whose stems collide (`train/img.jpg` vs `val/img.jpg`) would
//! silently overwrite each other's output, so that is
//! [`ConvertError::StemCollision`].
//!
//! # One shape for signatures and stats
//!
//! Importers are `*_to_coco(source_path, <format-specific options>,
//! image_dims)` — source first, `image_dims` last, and only present when the
//! format does not record dimensions itself. Exporters are
//! `coco_to_*(dataset, output_path)` and return a `*Stats` struct with shared
//! vocabulary: `images` and `annotations` count what was written, `skipped_*`
//! fields count records the format cannot express. Every field is documented
//! on its struct.

mod cvat;
mod dota;
mod oid;
mod voc;
mod yolo;

pub use cvat::{CvatImportStats, CvatStats, coco_to_cvat, cvat_to_coco};
pub use dota::{DotaStats, coco_to_dota, dota_to_coco};
pub use oid::{OidStats, coco_to_oid, oid_results_to_anns, oid_to_coco, read_class_descriptions};
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
    /// An image has zero or unknown dimensions where the conversion must scale
    /// coordinates by the image size. The payload identifies the image.
    #[error(
        "image `{0}` has zero or unknown width/height, which this conversion needs to scale coordinates"
    )]
    MissingImageDimensions(String),
    /// No `data.yaml` found in the YOLO directory.
    #[error("data.yaml not found in YOLO directory")]
    MissingDataYaml,
    /// A label file, CSV, or `data.yaml` could not be parsed. The message names
    /// the file and, where one exists, the line or position.
    #[error("parse error: {0}")]
    ParseError(String),
    /// An XML parsing or writing error occurred.
    #[error("XML error: {0}")]
    XmlError(String),
    /// Two images share a file stem, so per-image outputs keyed by stem would
    /// silently overwrite each other.
    #[error(
        "images `{first}` and `{second}` share the file stem `{stem}`; per-image outputs are keyed by stem and would overwrite each other — rename one image"
    )]
    StemCollision {
        /// The colliding stem.
        stem: String,
        /// `file_name` of the image seen first.
        first: String,
        /// `file_name` of the image seen second.
        second: String,
    },
    /// An annotation references a `category_id` that is not in the dataset's
    /// category list.
    #[error(
        "annotation {ann_id} references category id {category_id}, which is not in the dataset's categories"
    )]
    UnknownCategory {
        /// The annotation's id.
        ann_id: u64,
        /// The dangling category id it references.
        category_id: u64,
    },
}

/// Append what an entity reference stands for — `&amp;`, `&#233;` — to the
/// text being collected.
///
/// quick-xml hands element text over in pieces, split at each reference:
/// `salt &amp; pepper` is `Text("salt ")`, `GeneralRef("amp")`,
/// `Text(" pepper")`. A reader that keeps one `Text` event keeps ` pepper`,
/// which is how a category named `salt & pepper` came back from its own
/// export as `pepper`.
pub(crate) fn push_entity(
    out: &mut String,
    entity: &quick_xml::events::BytesRef<'_>,
) -> Result<(), ConvertError> {
    let invalid =
        |e: &dyn std::fmt::Display| ConvertError::XmlError(format!("invalid entity: {e}"));
    if let Some(ch) = entity.resolve_char_ref().map_err(|e| invalid(&e))? {
        out.push(ch);
        return Ok(());
    }
    let name = entity.decode().map_err(|e| invalid(&e))?;
    let text = quick_xml::escape::resolve_predefined_entity(&name)
        .ok_or_else(|| ConvertError::XmlError(format!("unknown entity `&{name};`")))?;
    out.push_str(text);
    Ok(())
}

/// An attribute's value with its entity references resolved: `R&amp;D.jpg`
/// in the file is the name `R&D.jpg`.
pub(crate) fn attr_text(
    attr: &quick_xml::events::attributes::Attribute<'_>,
) -> Result<String, ConvertError> {
    attr.normalized_value(quick_xml::XmlVersion::Implicit1_0)
        .map(std::borrow::Cow::into_owned)
        .map_err(|e| ConvertError::XmlError(format!("invalid attribute value: {e}")))
}

/// The error for a file that ends with an element still open: a truncated
/// file is malformed, not a shorter dataset.
pub(crate) fn unclosed_at_eof() -> ConvertError {
    ConvertError::XmlError("the file ends inside an element that is never closed".into())
}

impl From<quick_xml::Error> for ConvertError {
    fn from(e: quick_xml::Error) -> Self {
        ConvertError::XmlError(e.to_string())
    }
}

impl ConvertError {
    /// Prefix a parse/XML error's message with the file it came from.
    ///
    /// This is the single place file context is attached, so parsers that take
    /// a reader (and cannot know the path) stay path-agnostic while every error
    /// that leaves the module still names its file.
    pub(crate) fn with_path(self, path: &Path) -> Self {
        match self {
            ConvertError::ParseError(msg) => {
                ConvertError::ParseError(format!("{}: {msg}", path.display()))
            }
            ConvertError::XmlError(msg) => {
                ConvertError::XmlError(format!("{}: {msg}", path.display()))
            }
            // A filesystem error keeps its kind, so Python still raises
            // `FileNotFoundError`, and gains the path it was missing. Bytes
            // that are not UTF-8 are bad content, not a filesystem problem.
            ConvertError::Io(e) if e.kind() == io::ErrorKind::InvalidData => {
                ConvertError::ParseError(format!("{}: {e}", path.display()))
            }
            ConvertError::Io(e) => {
                ConvertError::Io(io::Error::new(e.kind(), format!("{}: {e}", path.display())))
            }
            other => other,
        }
    }
}

/// `map_err` for a bare filesystem call: the error with `path` attached, as
/// [`ConvertError::with_path`] attaches it.
pub(crate) fn io_at(path: &Path) -> impl Fn(io::Error) -> ConvertError + '_ {
    move |e| ConvertError::Io(e).with_path(path)
}

/// A parse error carrying its file.
pub(crate) fn parse_err(path: &Path, detail: impl std::fmt::Display) -> ConvertError {
    ConvertError::ParseError(format!("{}: {detail}", path.display()))
}

/// A parse error carrying its file and 1-based line number.
pub(crate) fn line_err(
    path: &Path,
    line_no: usize,
    detail: impl std::fmt::Display,
) -> ConvertError {
    ConvertError::ParseError(format!("{}: line {line_no}: {detail}", path.display()))
}

/// Group annotations by `image_id`.
pub(crate) fn anns_by_image(dataset: &Dataset) -> HashMap<u64, Vec<&Annotation>> {
    let mut map: HashMap<u64, Vec<&Annotation>> = HashMap::new();
    for ann in &dataset.annotations {
        map.entry(ann.image_id).or_default().push(ann);
    }
    map
}

/// Attach the reader's byte offset to a parse error — XML carries no line
/// numbers a streaming reader can cheaply report, but a byte position still
/// pins the failing element.
pub(crate) fn at_byte(err: ConvertError, pos: u64) -> ConvertError {
    match err {
        ConvertError::ParseError(msg) => {
            ConvertError::ParseError(format!("near byte {pos}: {msg}"))
        }
        other => other,
    }
}

/// Extract the filename stem (without extension) from a file path string.
pub(crate) fn file_stem(file_name: &str) -> &str {
    Path::new(file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file_name)
}

/// The UTF-8 stem of an on-disk label file, or an error naming the file.
///
/// A non-UTF-8 file name cannot become a COCO `file_name`; silently mapping it
/// to an empty stem would fabricate colliding image records.
pub(crate) fn utf8_stem(path: &Path) -> Result<&str, ConvertError> {
    path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
        ConvertError::ParseError(format!(
            "label file `{}` has a non-UTF-8 name; cannot derive an image file name from it",
            path.display()
        ))
    })
}

/// Verify every image's file stem is unique before writing per-image outputs.
///
/// Exporters that write one file (or CSV image id) per image key it by the stem
/// of `file_name`; `train/img.jpg` and `val/img.jpg` share the stem `img` and
/// the second would silently overwrite the first.
pub(crate) fn check_unique_stems(dataset: &Dataset) -> Result<(), ConvertError> {
    let mut seen: HashMap<&str, &str> = HashMap::new();
    for img in &dataset.images {
        let stem = file_stem(&img.file_name);
        if let Some(first) = seen.insert(stem, &img.file_name) {
            return Err(ConvertError::StemCollision {
                stem: stem.to_string(),
                first: first.to_string(),
                second: img.file_name.clone(),
            });
        }
    }
    Ok(())
}

/// Image file extensions tried when a bare stem has no direct hit, and the
/// files [`read_image_dims`] reads.
pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff"];

/// The `(width, height)` of every image directly in `dir` whose extension is
/// in [`IMAGE_EXTENSIONS`], in any case, keyed by file stem: the
/// `image_dims` map the YOLO, DOTA, and Open Images importers take, since
/// none of the three formats records a pixel size. Each size comes from the
/// file's header, as Pillow's `Image.size` reports it, with nothing decoded.
/// The format comes from the content, as Pillow detects it, so a GIF, WebP,
/// ICO, PNM, TGA, PSD, QOI, or DDS
/// file under one of those extensions reads too. Two files with one stem keep
/// the size of whichever `read_dir` lists last, and a file whose stem is not
/// UTF-8, which no label file can name, is left out.
///
/// # Errors
///
/// [`ConvertError::Io`], naming the path, when `dir` or a file in it cannot
/// be read, and [`ConvertError::ParseError`], naming the file, for one whose
/// header is none of those formats'.
pub fn read_image_dims(dir: &Path) -> Result<HashMap<String, (u32, u32)>, ConvertError> {
    use crate::primitives::sim::ForkSafe;
    use rayon::prelude::*;

    let images: Vec<(String, std::path::PathBuf)> = std::fs::read_dir(dir)
        .map_err(io_at(dir))?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let ext = path.extension()?.to_str()?.to_lowercase();
            if !IMAGE_EXTENSIONS.contains(&ext.as_str()) {
                return None;
            }
            let stem = path.file_stem()?.to_str()?.to_owned();
            Some((stem, path))
        })
        .collect();
    let size = |(stem, path): &(String, std::path::PathBuf)| {
        let (width, height) = image_size(path)?;
        let side = |v: usize| {
            u32::try_from(v)
                .map_err(|_| parse_err(path, format_args!("an image side of {v} pixels")))
        };
        Ok((stem.clone(), (side(width)?, side(height)?)))
    };
    // Parallel even for a few images: each item opens a file, thousands of
    // times the work `worth_parallel`'s threshold is calibrated on. Each
    // result is kept in `read_dir` order, so the last of two files with one
    // stem still wins and the error raised is the first failing file's, as a
    // serial loop raises it: collected straight into a `Result`, rayon returns
    // whichever worker fails first, and the kind and the file named varied
    // from run to run.
    type Sized = Result<(String, (u32, u32)), ConvertError>;
    let sizes: Vec<Sized> = images.par_iter().fork_safe().map(size).collect();
    sizes.into_iter().collect()
}

/// `(width, height)` from `path`'s header, as Pillow's `Image.size` reads
/// it. The format comes from the content, not the extension, so a PNG named
/// `.jpg` still reads.
fn image_size(path: &Path) -> Result<(usize, usize), ConvertError> {
    let bad = |e: imagesize::ImageError| match e {
        // A file shorter than its header is bad content too.
        imagesize::ImageError::IoError(e) if e.kind() != io::ErrorKind::UnexpectedEof => {
            io_at(path)(e)
        }
        e => parse_err(path, format_args!("cannot read its image size: {e}")),
    };
    let mut reader = io::BufReader::new(std::fs::File::open(path).map_err(io_at(path))?);
    let kind = imagesize::reader_type(&mut reader).map_err(bad)?;
    // The formats whose `imagesize` reader disagrees with Pillow on files
    // Pillow reads get readers of their own.
    match kind {
        imagesize::ImageType::Bmp => bmp_size(&mut reader).map_err(bad),
        imagesize::ImageType::Gif => gif_size(&mut reader).map_err(bad),
        imagesize::ImageType::Jpeg => jpeg_size(&mut reader).map_err(bad),
        imagesize::ImageType::Png => png_size(&mut reader).map_err(bad),
        imagesize::ImageType::Webp => webp_size(&mut reader).map_err(bad),
        kind => {
            let dims = kind.reader_size(&mut reader).map_err(bad)?;
            Ok((dims.width, dims.height))
        }
    }
}

/// A PNG's `(width, height)` from its `IHDR` chunk, found by walking the
/// chunks as Pillow does. `imagesize` reads the bytes where `IHDR` sits when
/// it is first, so a file with another chunk first (Apple's `CgBI`) came back
/// at a size of its other chunk's bytes.
fn png_size<R: io::Read + io::Seek>(reader: &mut R) -> imagesize::ImageResult<(usize, usize)> {
    reader.seek(io::SeekFrom::Start(8))?;
    loop {
        let mut head = [0u8; 8];
        reader.read_exact(&mut head)?;
        let len = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
        match &head[4..] {
            b"IHDR" => {
                let mut sides = [0u8; 8];
                reader.read_exact(&mut sides)?;
                let side = |i: usize| {
                    u32::from_be_bytes([sides[i], sides[i + 1], sides[i + 2], sides[i + 3]])
                        as usize
                };
                return Ok((side(0), side(4)));
            }
            // The image data or its end before any header: Pillow gives up.
            b"IDAT" | b"IEND" => return Err(imagesize::ImageError::CorruptedImage),
            // Another chunk: its data, then its CRC.
            _ => {
                reader.seek(io::SeekFrom::Current(i64::from(len) + 4))?;
            }
        }
    }
}

/// A GIF's `(width, height)` as Pillow reports it: the logical screen,
/// grown to the first frame where the frame reaches past it. `imagesize`
/// reads the logical screen alone.
fn gif_size<R: io::Read + io::Seek>(reader: &mut R) -> imagesize::ImageResult<(usize, usize)> {
    fn byte<R: io::Read>(reader: &mut R) -> io::Result<u8> {
        let mut b = [0u8; 1];
        reader.read_exact(&mut b)?;
        Ok(b[0])
    }
    let mut screen = [0u8; 13];
    reader.seek(io::SeekFrom::Start(0))?;
    reader.read_exact(&mut screen)?;
    let u16_at = |b: &[u8], i: usize| usize::from(u16::from_le_bytes([b[i], b[i + 1]]));
    let (screen_w, screen_h) = (u16_at(&screen, 6), u16_at(&screen, 8));
    let flags = screen[10];
    if flags & 0x80 != 0 {
        // The global color table.
        reader.seek(io::SeekFrom::Current(3 << ((flags & 7) + 1)))?;
    }
    loop {
        match byte(reader)? {
            // An extension: its label, then data sub-blocks up to an empty one.
            b'!' => {
                byte(reader)?;
                loop {
                    let n = byte(reader)?;
                    if n == 0 {
                        break;
                    }
                    reader.seek(io::SeekFrom::Current(i64::from(n)))?;
                }
            }
            // The first frame: left, top, width, height.
            b',' => {
                let mut frame = [0u8; 8];
                reader.read_exact(&mut frame)?;
                let x1 = u16_at(&frame, 0) + u16_at(&frame, 4);
                let y1 = u16_at(&frame, 2) + u16_at(&frame, 6);
                return Ok((x1.max(screen_w), y1.max(screen_h)));
            }
            // The trailer before any frame: no image.
            b';' => return Err(imagesize::ImageError::CorruptedImage),
            // A stray byte, which Pillow skips.
            _ => {}
        }
    }
}

/// A WebP's `(width, height)`. A lossy (`VP8 `) file's 16-bit sides carry
/// two scale bits on top, which libwebp, and so Pillow, masks off and
/// `imagesize` does not; the other kinds read as `imagesize` reads them.
fn webp_size<R: io::BufRead + io::Seek>(reader: &mut R) -> imagesize::ImageResult<(usize, usize)> {
    let mut head = [0u8; 16];
    reader.seek(io::SeekFrom::Start(0))?;
    reader.read_exact(&mut head)?;
    if &head[12..16] == b"VP8 " {
        // The chunk size, the frame tag, the start code, then the sides.
        let mut frame = [0u8; 14];
        reader.read_exact(&mut frame)?;
        let side = |i: usize| usize::from(u16::from_le_bytes([frame[i], frame[i + 1]]) & 0x3FFF);
        return Ok((side(10), side(12)));
    }
    let dims = imagesize::ImageType::Webp.reader_size(reader)?;
    Ok((dims.width, dims.height))
}

/// A JPEG's `(width, height)`, read as Pillow's `JpegImageFile._open` reads
/// it: the markers walked from the start of image to the first start of
/// scan, any byte between segments that does not start a marker skipped,
/// `0xFF` fill bytes before a marker skipped, and the size taken from the
/// last frame header (`SOFn`) before the scan. `imagesize` stops at a byte
/// between segments that is not `0xFF` and reads a fill byte as a marker
/// with a length, so it failed on files libjpeg and Pillow read.
fn jpeg_size<R: io::Read + io::Seek>(reader: &mut R) -> imagesize::ImageResult<(usize, usize)> {
    fn byte<R: io::Read>(reader: &mut R) -> io::Result<u8> {
        let mut b = [0u8; 1];
        reader.read_exact(&mut b)?;
        Ok(b[0])
    }
    // A segment's length counts its own two bytes; Pillow reads one shorter
    // than that as empty.
    fn segment_len<R: io::Read>(reader: &mut R) -> io::Result<usize> {
        let mut len = [0u8; 2];
        reader.read_exact(&mut len)?;
        Ok(usize::from(u16::from_be_bytes(len)).saturating_sub(2))
    }
    // Past `FF D8 FF`, which `reader_type` matched: the third byte starts
    // the first marker.
    reader.seek(io::SeekFrom::Start(3))?;
    let mut lead = 0xFF;
    let mut size = (0, 0);
    loop {
        if lead != 0xFF {
            lead = byte(reader)?;
            continue;
        }
        match byte(reader)? {
            // A fill byte: the next one may still be the marker's.
            0xFF => continue,
            0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF | 0xDE => {
                let mut frame = vec![0u8; segment_len(reader)?];
                reader.read_exact(&mut frame)?;
                // Precision, then the height and the width.
                let be16 = |i: usize| -> imagesize::ImageResult<usize> {
                    let b = frame
                        .get(i..i + 2)
                        .ok_or(imagesize::ImageError::CorruptedImage)?;
                    Ok(usize::from(u16::from_be_bytes([b[0], b[1]])))
                };
                size = (be16(3)?, be16(1)?);
            }
            // The segments with a length; the start of scan ends the header.
            marker @ (0xC4 | 0xCC | 0xDA..=0xDD | 0xDF | 0xE0..=0xEF | 0xFE) => {
                let len = segment_len(reader)?;
                reader.seek(io::SeekFrom::Current(len as i64))?;
                if marker == 0xDA {
                    return Ok(size);
                }
            }
            // `FF 00`, an escaped `0xFF`, and the markers with no segment:
            // JPG, RSTn, SOI, EOI, JPGn.
            0x00 | 0xC8 | 0xD0..=0xD9 | 0xF0..=0xFD => {}
            // `FF 01` to `FF BF`: no marker Pillow knows, and it gives up.
            _ => return Err(imagesize::ImageError::CorruptedImage),
        }
        lead = byte(reader)?;
    }
}

/// A BMP's `(width, height)` from its DIB header, read as Pillow reads it:
/// the 16-bit sides of an OS/2 header, and otherwise signed sides with the
/// height's sign dropped. `imagesize` reads the height as unsigned, so a
/// top-down BMP, whose height is stored negative, came back about 2³² rows
/// tall.
fn bmp_size<R: io::Read + io::Seek>(reader: &mut R) -> imagesize::ImageResult<(usize, usize)> {
    let mut header = [0u8; 26];
    reader.seek(io::SeekFrom::Start(0))?;
    reader.read_exact(&mut header)?;
    let u16_at = |i: usize| usize::from(u16::from_le_bytes([header[i], header[i + 1]]));
    let i32_at =
        |i: usize| i32::from_le_bytes([header[i], header[i + 1], header[i + 2], header[i + 3]]);
    // The DIB header's size: 12 for OS/2's BITMAPCOREHEADER.
    if i32_at(14) == 12 {
        return Ok((u16_at(18), u16_at(20)));
    }
    let width = usize::try_from(i32_at(18)).map_err(|_| imagesize::ImageError::CorruptedImage)?;
    Ok((width, i32_at(22).unsigned_abs() as usize))
}

/// Look up image dimensions by stem; try common extensions as fallback.
///
/// Returns `None` when the stem is absent or maps to a zero dimension — both
/// mean "unknown" to callers, which apply the missing-dimensions policy
/// documented in the module docs.
pub(crate) fn lookup_image_dims(
    image_dims: &HashMap<String, (u32, u32)>,
    stem: &str,
) -> Option<(u32, u32)> {
    let found = image_dims.get(stem).or_else(|| {
        IMAGE_EXTENSIONS
            .iter()
            .find_map(|ext| image_dims.get(&format!("{stem}.{ext}")))
    });
    match found {
        Some(&(w, h)) if w > 0 && h > 0 => Some((w, h)),
        _ => None,
    }
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
