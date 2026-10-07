//! Label maps: from a COCO panoptic PNG, or painted from segment masks.
//!
//! Both producers emit the same thing — a flat `u32` map in **column-major**
//! order, the order RLE runs come in, so painting a mask is a contiguous
//! fill. A PNG decodes row-major and is transposed once here; the overlap
//! histogram downstream is order-free, so only the two sides' agreement
//! matters, and this module is the one place that agreement is made.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::types::{Rle, Segmentation};

/// One image's segment ids, one per pixel, column-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LabelMap {
    pub h: u32,
    pub w: u32,
    pub labels: Vec<u32>,
}

/// Decode a COCO panoptic PNG: each pixel's color `(R, G, B)` is the segment
/// id `R + 256·G + 256²·B`, panopticapi's `rgb2id`. Alpha, if any, is ignored.
pub(super) fn read_png(path: &Path) -> crate::error::Result<LabelMap> {
    let file = File::open(path)
        .map_err(|e| format!("cannot open panoptic PNG {}: {e}", path.display()))?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    // Palette and low-bit-depth files expand to 8-bit samples; 16-bit files
    // are reduced to 8, which is what PIL hands panopticapi as well.
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let fail =
        |e: png::DecodingError| format!("cannot decode panoptic PNG {}: {e}", path.display());
    let mut reader = decoder.read_info().map_err(fail)?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| format!("panoptic PNG {} is too large to decode", path.display()))?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(fail)?;
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => {
            return Err(format!(
                "panoptic PNG {} is {other:?}; the COCO panoptic format encodes segment ids as RGB",
                path.display()
            )
            .into());
        }
    };
    let (h, w) = (info.height as usize, info.width as usize);
    let mut labels = vec![0u32; h * w];
    for (y, row) in buf.chunks_exact(info.line_size).take(h).enumerate() {
        for (x, px) in row.chunks_exact(channels).take(w).enumerate() {
            let id = u32::from(px[0]) + (u32::from(px[1]) << 8) + (u32::from(px[2]) << 16);
            labels[x * h + y] = id;
        }
    }
    Ok(LabelMap {
        h: info.height,
        w: info.width,
        labels,
    })
}

/// Paint segment masks onto an `h × w` canvas, `labels[i]` for the `i`th
/// mask, in order — a later mask over an earlier one where they overlap.
/// Unpainted pixels stay `0`, void.
///
/// Each mask is rasterized at `h × w`; an RLE that is some other size is an
/// error naming its label, since a mask drawn on a different canvas is not a
/// mask of this image.
pub(super) fn paint(
    masks: &[(u32, &Segmentation)],
    h: u32,
    w: u32,
) -> crate::error::Result<LabelMap> {
    let n = h as usize * w as usize;
    let mut out = vec![0u32; n];
    for &(label, seg) in masks {
        let rle: Rle = seg.to_rle(h, w)?;
        if rle.h != h || rle.w != w {
            return Err(format!(
                "segment {label} has a {}×{} mask on a {h}×{w} image",
                rle.h, rle.w
            )
            .into());
        }
        let mut pos = 0usize;
        let mut foreground = false;
        for &run in &rle.counts {
            let end = (pos + run as usize).min(n);
            if foreground {
                out[pos..end].fill(label);
            }
            pos = end;
            foreground = !foreground;
        }
    }
    Ok(LabelMap { h, w, labels: out })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a row-major `(h, w)` id map the way the COCO panoptic PNG files are
    /// written: RGB, `id2rgb`.
    pub(crate) fn write_png(path: &Path, h: u32, w: u32, ids_row_major: &[u32], alpha: bool) {
        let file = File::create(path).expect("create png");
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        enc.set_color(if alpha {
            png::ColorType::Rgba
        } else {
            png::ColorType::Rgb
        });
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().expect("header");
        let mut data = Vec::with_capacity(ids_row_major.len() * 4);
        for &id in ids_row_major {
            data.extend_from_slice(&[
                (id & 0xff) as u8,
                ((id >> 8) & 0xff) as u8,
                ((id >> 16) & 0xff) as u8,
            ]);
            if alpha {
                data.push(255);
            }
        }
        writer.write_image_data(&data).expect("write");
    }

    #[test]
    fn png_round_trips_ids_and_transposes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("a.png");
        // 2 rows × 3 columns, row-major.
        let ids = [1, 2, 3, 4, 5, 70000];
        write_png(&path, 2, 3, &ids, false);
        let map = read_png(&path).expect("decode");
        assert_eq!((map.h, map.w), (2, 3));
        // Column-major: column 0 is (1, 4), column 1 is (2, 5), column 2 is (3, 70000).
        assert_eq!(map.labels, vec![1, 4, 2, 5, 3, 70000]);

        write_png(&path, 2, 3, &ids, true);
        assert_eq!(
            read_png(&path).expect("decode rgba").labels,
            vec![1, 4, 2, 5, 3, 70000]
        );
    }

    #[test]
    fn paint_fills_runs_and_later_masks_win() {
        // 2×2 canvas, column-major: pixel order (0,0) (1,0) (0,1) (1,1).
        let a = Segmentation::UncompressedRle {
            size: [2, 2],
            counts: vec![0, 3, 1], // first three pixels
        };
        let b = Segmentation::UncompressedRle {
            size: [2, 2],
            counts: vec![2, 2], // last two pixels
        };
        let map = paint(&[(7, &a), (9, &b)], 2, 2).expect("paint");
        assert_eq!(map.labels, vec![7, 7, 9, 9]);
        let map = paint(&[(9, &b), (7, &a)], 2, 2).expect("paint");
        assert_eq!(map.labels, vec![7, 7, 7, 9]);
    }

    #[test]
    fn paint_rejects_a_mask_of_another_size() {
        let a = Segmentation::UncompressedRle {
            size: [3, 2],
            counts: vec![0, 6],
        };
        let err = paint(&[(7, &a)], 2, 2).expect_err("size mismatch");
        assert!(err.to_string().contains("3×2 mask on a 2×2 image"), "{err}");
    }

    #[test]
    fn paint_rasterizes_polygons_on_the_canvas() {
        let square = Segmentation::Polygon(vec![vec![0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]]);
        let map = paint(&[(5, &square)], 4, 4).expect("paint");
        assert_eq!(map.labels.iter().filter(|&&l| l == 5).count(), 4);
    }
}
