//! Class-id label maps from PNG files.
//!
//! A semantic label map is a single-channel image whose sample *is* the class
//! id: 8-bit grayscale (Cityscapes `labelTrainIds`, ADE20K), 16-bit grayscale
//! for more than 256 classes, or a palette image whose palette index is the
//! class (Pascal VOC, whose border color sits at index 255). The palette is
//! never expanded, since the index is the label and the color is only how a
//! viewer shows it. An RGB file is refused: it is either a rendering, which
//! carries no ids, or a COCO panoptic file, whose colors encode *segment* ids
//! and which [`crate::panoptic`] reads.

use std::borrow::Cow;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::error::Error;
use crate::metrics::semantic::LabelMap;

/// One image's class ids, row-major, with its size. An 8-bit file stays one
/// byte per pixel, the common case, rather than being widened on the way in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ClassMap {
    pub h: u32,
    pub w: u32,
    pub labels: LabelMap<'static>,
}

/// Decode a label map, as described in the [module docs](self).
pub(super) fn read_class_map(path: &Path) -> crate::error::Result<ClassMap> {
    let file = File::open(path)
        .map_err(|e| Error::io(e, format_args!("cannot open label map {}", path.display())))?;
    // No transformations: a palette index must stay an index, and a 16-bit
    // sample must keep its high byte.
    let decoder = png::Decoder::new(BufReader::new(file));
    // A read that fails keeps its I/O kind; a file that is not a label map
    // is bad input. A file that ends before its PNG stream does comes back
    // as an end-of-file error, which is about the content, so bad input too.
    let fail = |e: png::DecodingError| match e {
        png::DecodingError::IoError(e) if e.kind() != std::io::ErrorKind::UnexpectedEof => {
            Error::io(e, format_args!("cannot read label map {}", path.display()))
        }
        e => Error::InvalidInput(format!("cannot decode label map {}: {e}", path.display())),
    };
    let mut reader = decoder.read_info().map_err(fail)?;
    let size = reader.output_buffer_size().ok_or_else(|| {
        Error::InvalidInput(format!(
            "label map {} is too large to decode",
            path.display()
        ))
    })?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(fail)?;
    let (h, w) = (info.height, info.width);
    let n = h as usize * w as usize;
    let rows = || buf.chunks_exact(info.line_size).take(h as usize);
    // `png` sizes the buffer as `line_size × height`, and every branch below
    // reads it as that many whole rows; a file that decodes to any other
    // size is refused, not assumed.
    if buf.len() != info.line_size * h as usize {
        return Err(Error::InvalidInput(format!(
            "label map {} decodes to {} bytes for {w}×{h} pixels",
            path.display(),
            buf.len()
        )));
    }

    let labels = match (info.color_type, info.bit_depth) {
        (png::ColorType::Grayscale | png::ColorType::Indexed, png::BitDepth::Eight) => {
            // One byte per pixel and rows laid end to end: the buffer is the
            // map: `line_size` is `width` at this depth, so the guard above
            // has checked it holds `h × w` bytes, what the counting trusts.
            LabelMap::U8(Cow::Owned(buf))
        }
        (png::ColorType::Grayscale, png::BitDepth::Sixteen) => {
            let mut v = Vec::with_capacity(n);
            for row in rows() {
                v.extend(
                    row[..w as usize * 2]
                        .chunks_exact(2)
                        .map(|px| u16::from_be_bytes([px[0], px[1]])),
                );
            }
            LabelMap::U16(Cow::Owned(v))
        }
        (png::ColorType::Grayscale | png::ColorType::Indexed, depth) => {
            // 1, 2, or 4 bits per sample, packed high bit first.
            let bits = depth as u32;
            let per_byte = 8 / bits;
            let mask = (1u8 << bits) - 1;
            let mut v = Vec::with_capacity(n);
            for row in rows() {
                v.extend((0..w).map(|x| {
                    let byte = row[(x / per_byte) as usize];
                    let shift = 8 - bits * (x % per_byte + 1);
                    (byte >> shift) & mask
                }));
            }
            LabelMap::U8(Cow::Owned(v))
        }
        (other, _) => {
            return Err(Error::InvalidInput(format!(
                "label map {} is {other:?}; a class-id map is single-channel (grayscale or \
                 palette). A COCO panoptic PNG, whose colors encode segment ids, goes through \
                 hotcoco.panoptic",
                path.display()
            )));
        }
    };
    Ok(ClassMap { h, w, labels })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(
        path: &Path,
        w: u32,
        h: u32,
        color: png::ColorType,
        depth: png::BitDepth,
        data: &[u8],
    ) {
        let file = File::create(path).expect("valid test input");
        let mut enc = png::Encoder::new(file, w, h);
        enc.set_color(color);
        enc.set_depth(depth);
        if color == png::ColorType::Indexed {
            enc.set_palette(vec![0u8; 3 * (1 << (depth as usize).min(8))]);
        }
        enc.write_header()
            .expect("png header")
            .write_image_data(data)
            .expect("valid test input");
    }

    fn labels(path: &Path) -> Vec<u16> {
        match read_class_map(path).expect("a label map").labels {
            LabelMap::U8(v) => v.iter().copied().map(u16::from).collect(),
            LabelMap::U16(v) => v.into_owned(),
            LabelMap::U64(_) | LabelMap::I64(_) => unreachable!("a PNG sample is at most 16 bits"),
        }
    }

    #[test]
    fn grayscale_8_bit_is_read_as_is() {
        let dir = tempfile::tempdir().expect("valid test input");
        let p = dir.path().join("a.png");
        write(
            &p,
            3,
            2,
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            &[0, 1, 255, 7, 7, 2],
        );
        let m = read_class_map(&p).expect("valid test input");
        assert_eq!((m.h, m.w), (2, 3));
        assert_eq!(labels(&p), vec![0, 1, 255, 7, 7, 2]);
    }

    #[test]
    fn palette_index_is_the_label_not_the_color() {
        let dir = tempfile::tempdir().expect("valid test input");
        let p = dir.path().join("voc.png");
        write(
            &p,
            2,
            1,
            png::ColorType::Indexed,
            png::BitDepth::Eight,
            &[3, 255],
        );
        assert_eq!(labels(&p), vec![3, 255]);
    }

    #[test]
    fn packed_low_bit_depths_unpack_high_bit_first() {
        let dir = tempfile::tempdir().expect("valid test input");
        let p = dir.path().join("packed.png");
        // 4-bit indexed, three pixels: 0x1, 0xA, 0xF → bytes 0x1A, 0xF0.
        write(
            &p,
            3,
            1,
            png::ColorType::Indexed,
            png::BitDepth::Four,
            &[0x1A, 0xF0],
        );
        assert_eq!(labels(&p), vec![1, 10, 15]);
        let p = dir.path().join("bits.png");
        // 1-bit grayscale, five pixels 1,0,1,1,0 → 0b10110000.
        write(
            &p,
            5,
            1,
            png::ColorType::Grayscale,
            png::BitDepth::One,
            &[0b1011_0000],
        );
        assert_eq!(labels(&p), vec![1, 0, 1, 1, 0]);
    }

    #[test]
    fn sixteen_bit_keeps_the_high_byte() {
        let dir = tempfile::tempdir().expect("valid test input");
        let p = dir.path().join("wide.png");
        write(
            &p,
            2,
            1,
            png::ColorType::Grayscale,
            png::BitDepth::Sixteen,
            &[0x01, 0x02, 0xFF, 0xFF],
        );
        assert_eq!(labels(&p), vec![258, 65535]);
    }

    #[test]
    fn rgb_is_refused_and_points_at_panoptic() {
        let dir = tempfile::tempdir().expect("valid test input");
        let p = dir.path().join("rgb.png");
        write(
            &p,
            1,
            1,
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[1, 2, 3],
        );
        let err = match read_class_map(&p) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        assert!(
            err.contains("Rgb") && err.contains("hotcoco.panoptic"),
            "{err}"
        );
    }
}
