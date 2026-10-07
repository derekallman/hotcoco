//! Label maps: from a COCO panoptic PNG, or painted from segment masks.
//!
//! A painted map is a flat `u32` map in **column-major** order, the order
//! RLE runs come in, so painting a mask is a contiguous fill. A PNG stays as
//! its decoded bytes ([`Png`]) and yields ids row-major on demand: the
//! overlap histogram is order-free, so two PNG sides are scanned straight
//! off their bytes, and only a PNG paired with a painted map is transposed
//! into column-major to agree with it. This module is the one place that
//! agreement is made.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::primitives::panoptic::{Overlaps, OverlapsBuilder};
use crate::types::{Rle, Segmentation};

/// panopticapi's `rgb2id` for one pixel's leading three samples.
#[inline]
fn rgb2id(px: &[u8]) -> u32 {
    u32::from(px[0]) | (u32::from(px[1]) << 8) | (u32::from(px[2]) << 16)
}

/// One image's segment ids, one per pixel, column-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LabelMap {
    pub h: u32,
    pub w: u32,
    pub labels: Vec<u32>,
}

/// A decoded COCO panoptic PNG, as 8-bit RGB or RGBA rows.
pub(super) struct Png {
    pub h: u32,
    pub w: u32,
    channels: usize,
    line_size: usize,
    bytes: Vec<u8>,
}

impl Png {
    /// The pixel rows, each `w` samples of `channels` bytes.
    fn rows(&self) -> impl Iterator<Item = &[u8]> + '_ {
        let (h, w, channels) = (self.h as usize, self.w as usize, self.channels);
        self.bytes
            .chunks_exact(self.line_size)
            .take(h)
            .map(move |row| &row[..w * channels])
    }

    /// Segment ids pixel by pixel, row-major: each color `(R, G, B)` is
    /// `R + 256·G + 256²·B`, panopticapi's `rgb2id`. Alpha is ignored.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        let channels = self.channels;
        self.rows()
            .flat_map(move |row| row.chunks_exact(channels).map(rgb2id))
    }

    /// The overlap histogram of two PNG sides of the same size, read
    /// straight off their bytes in row-major order.
    pub fn overlaps(&self, other: &Png) -> Overlaps {
        debug_assert_eq!((self.h, self.w), (other.h, other.w));
        let mut builder = OverlapsBuilder::default();
        for (a, b) in self.rows().zip(other.rows()) {
            for (pa, pb) in a
                .chunks_exact(self.channels)
                .zip(b.chunks_exact(other.channels))
            {
                builder.push(rgb2id(pa), rgb2id(pb));
            }
        }
        builder.finish()
    }

    /// The ids as a column-major map, the order a painted map has.
    pub fn into_label_map(self) -> LabelMap {
        let (h, w) = (self.h as usize, self.w as usize);
        let mut labels = vec![0u32; h * w];
        for (i, id) in self.ids().enumerate() {
            let (y, x) = (i / w, i % w);
            labels[x * h + y] = id;
        }
        LabelMap {
            h: self.h,
            w: self.w,
            labels,
        }
    }
}

/// Decode a COCO panoptic PNG.
pub(super) fn read_png(path: &Path) -> crate::error::Result<Png> {
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
    Ok(Png {
        h: info.height,
        w: info.width,
        channels,
        line_size: info.line_size,
        bytes: buf,
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
        let png = read_png(&path).expect("decode");
        assert_eq!((png.h, png.w), (2, 3));
        assert_eq!(png.ids().collect::<Vec<_>>(), ids, "row-major, as stored");
        // Column-major: column 0 is (1, 4), column 1 is (2, 5), column 2 is (3, 70000).
        assert_eq!(png.into_label_map().labels, vec![1, 4, 2, 5, 3, 70000]);

        write_png(&path, 2, 3, &ids, true);
        let png = read_png(&path).expect("decode rgba");
        assert_eq!(png.ids().collect::<Vec<_>>(), ids, "alpha is ignored");
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
