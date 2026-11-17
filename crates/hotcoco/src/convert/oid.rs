//! Open Images CSV conversion.
//!
//! Open Images ships annotations as CSV with coordinates normalized to `[0, 1]`
//! and a column order that is *not* the usual one:
//!
//! ```text
//! ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
//! ```
//!
//! `XMax` precedes `YMin`. The challenge subset keeps only
//! `ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf`, and detection CSVs add a
//! `Score` column. All three are read by column *name*, never by position — a
//! positional reader that drifted by one field would transpose an axis and still
//! produce plausible boxes.
//!
//! `LabelName` is a Knowledge Graph MID such as `/m/0cmf2`. Passing the
//! `class-descriptions-boxable.csv` shipped alongside the annotations resolves
//! those to human-readable names.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::types::{Annotation, Category, Dataset, Image};

use super::{ConvertError, anns_by_image, file_stem, line_err, parse_err};

/// Statistics returned by [`coco_to_oid`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OidStats {
    /// Images that contributed at least one row (images with no annotations
    /// produce nothing in a row-per-annotation CSV).
    pub images: usize,
    /// Rows (annotations) written.
    pub annotations: usize,
    /// Rows written with `IsGroupOf=1`.
    pub group_of: usize,
    /// Annotations skipped because they have no `bbox`.
    pub skipped_no_bbox: usize,
}

/// Split one CSV line into fields, honoring double-quoted values.
///
/// Open Images label descriptions contain commas inside quotes (`"Salt and
/// pepper shakers"` is safe, but `/m/0dv5r,"Camera, still"` is not), so a plain
/// `split(',')` corrupts the name map.
fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                current.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => fields.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
    }
    fields.push(current);
    fields
}

/// Quote a field for output when it would otherwise break the row.
///
/// Category names reach the writer from whatever loaded them, and Open Images'
/// own descriptions include values like `Camera, still`. Writing that bare
/// produces a row with one field too many and a truncated label — a corruption
/// that survives re-import as a plausible category called "Camera".
fn csv_field(value: &str) -> std::borrow::Cow<'_, str> {
    if value.contains([',', '"', '\n']) {
        std::borrow::Cow::Owned(format!("\"{}\"", value.replace('"', "\"\"")))
    } else {
        std::borrow::Cow::Borrowed(value)
    }
}

/// Column positions resolved from a CSV header row.
struct Columns {
    image_id: usize,
    label: usize,
    xmin: usize,
    xmax: usize,
    ymin: usize,
    ymax: usize,
    is_group_of: Option<usize>,
    score: Option<usize>,
}

impl Columns {
    /// Resolve required and optional columns by name.
    ///
    /// `Score` and `Confidence` are deliberately different things: detection CSVs
    /// carry `Score`, while ground truth V6 carries a `Confidence` column that is
    /// always 1. Only `Score` is read as a detection score.
    fn from_header(header: &str) -> Result<Self, ConvertError> {
        let names: Vec<String> = split_csv_line(header)
            .into_iter()
            .map(|f| f.trim().to_ascii_lowercase())
            .collect();
        let find = |want: &str| names.iter().position(|n| n == want);
        let require = |want: &str| {
            find(want).ok_or_else(|| {
                ConvertError::ParseError(format!(
                    "Open Images CSV is missing the `{want}` column (header: {header})"
                ))
            })
        };

        Ok(Columns {
            image_id: require("imageid")?,
            label: require("labelname")?,
            xmin: require("xmin")?,
            xmax: require("xmax")?,
            ymin: require("ymin")?,
            ymax: require("ymax")?,
            is_group_of: find("isgroupof"),
            score: find("score"),
        })
    }

    /// Highest column index this layout reads, used to reject short rows.
    fn max_index(&self) -> usize {
        [
            self.image_id,
            self.label,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        ]
        .into_iter()
        .chain(self.is_group_of)
        .chain(self.score)
        .max()
        .unwrap_or(0)
    }
}

/// One parsed CSV row, still in normalized coordinates.
struct Row {
    image_id: String,
    label: String,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    is_group_of: bool,
    score: Option<f64>,
}

/// Parse every data row of an Open Images CSV.
fn read_rows(csv_path: &Path) -> Result<Vec<Row>, ConvertError> {
    let file = fs::File::open(csv_path).map_err(super::io_at(csv_path))?;
    let mut lines = BufReader::new(file).lines();

    let header = lines
        .next()
        .transpose()
        .map_err(super::io_at(csv_path))?
        .ok_or_else(|| parse_err(csv_path, "Open Images CSV is empty"))?;
    let cols = Columns::from_header(&header).map_err(|e| e.with_path(csv_path))?;
    let min_fields = cols.max_index() + 1;

    let mut rows = Vec::new();
    for (n, line) in lines.enumerate() {
        // Header is line 1, so data line `n` is file line `n + 2`.
        let line_no = n + 2;
        let line = line.map_err(super::io_at(csv_path))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_csv_line(&line);
        if fields.len() < min_fields {
            return Err(line_err(
                csv_path,
                line_no,
                format!(
                    "expected at least {min_fields} fields, got {}",
                    fields.len()
                ),
            ));
        }

        let coord = |idx: usize, name: &str| -> Result<f64, ConvertError> {
            fields[idx].trim().parse::<f64>().map_err(|_| {
                line_err(
                    csv_path,
                    line_no,
                    format!("invalid {name}: {}", fields[idx]),
                )
            })
        };

        rows.push(Row {
            image_id: fields[cols.image_id].trim().to_string(),
            label: fields[cols.label].trim().to_string(),
            xmin: coord(cols.xmin, "XMin")?,
            xmax: coord(cols.xmax, "XMax")?,
            ymin: coord(cols.ymin, "YMin")?,
            ymax: coord(cols.ymax, "YMax")?,
            is_group_of: cols
                .is_group_of
                .is_some_and(|i| fields[i].trim().starts_with('1')),
            score: match cols.score {
                Some(i) => Some(coord(i, "Score")?),
                None => None,
            },
        });
    }

    Ok(rows)
}

/// Load `class-descriptions-boxable.csv` as a MID → display-name map.
///
/// The file is headerless: each line is `/m/0cmf2,Beer`. A header row, if
/// present, is recognized by its first field not starting with `/` and skipped.
pub fn read_class_descriptions(path: &Path) -> Result<HashMap<String, String>, ConvertError> {
    let file = fs::File::open(path).map_err(super::io_at(path))?;
    let mut map = HashMap::new();

    for (line_idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(super::io_at(path))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_csv_line(&line);
        if fields.len() < 2 {
            return Err(line_err(
                path,
                line_idx + 1,
                format!("expected `MID,DisplayName`, got: {line}"),
            ));
        }
        let mid = fields[0].trim();
        // A header row would map "LabelName" to "DisplayName"; MIDs start with '/'.
        if !mid.starts_with('/') {
            continue;
        }
        map.insert(mid.to_string(), fields[1].trim().to_string());
    }

    Ok(map)
}

/// Scale factors and recorded dimensions for one image.
///
/// Open Images coordinates are normalized, so denormalizing needs pixel sizes the
/// CSV does not carry. Without them the boxes stay in `[0, 1]` against a 1×1
/// image: IoU and IoA are ratios of areas scaled identically on both axes, so
/// Open Images AP is unaffected, but absolute areas — and therefore the
/// small/medium/large ranges — are not meaningful.
fn image_scale(image_dims: &HashMap<String, (u32, u32)>, image_id: &str) -> (f64, f64, u32, u32) {
    match super::lookup_image_dims(image_dims, image_id) {
        Some((w, h)) => (f64::from(w), f64::from(h), w, h),
        None => (1.0, 1.0, 1, 1),
    }
}

/// Convert an Open Images annotation CSV to a COCO dataset.
///
/// Reads both the full V6 layout and the challenge subset. `IsGroupOf` becomes
/// [`Annotation::is_group_of`], which is what Open Images evaluation matches with
/// IoA rather than IoU.
///
/// Categories are derived from the `LabelName` values present in the file, sorted
/// for deterministic IDs. When `class_descriptions` is supplied, category names
/// are the human-readable labels; otherwise they remain MIDs. Classes absent from
/// the split are not invented — they would score `-1` and drop out of the mean
/// either way.
///
/// `image_dims` maps image ID (or file name) to `(width, height)`. Images missing
/// from it keep normalized coordinates against a 1×1 image. IoU and IoA are
/// ratios of areas scaled identically on both axes, so Open Images AP is
/// unaffected — but absolute areas, and therefore the small/medium/large ranges,
/// are not meaningful in that mode.
pub fn oid_to_coco(
    csv_path: &Path,
    class_descriptions: Option<&Path>,
    image_dims: &HashMap<String, (u32, u32)>,
) -> Result<Dataset, ConvertError> {
    let rows = read_rows(csv_path)?;
    let names = match class_descriptions {
        Some(p) => read_class_descriptions(p)?,
        None => HashMap::new(),
    };

    // Deterministic IDs: sort the distinct image IDs and labels rather than
    // numbering them in file order, so two splits of the same data agree.
    let mut image_ids: Vec<&str> = rows
        .iter()
        .map(|r| r.image_id.as_str())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    image_ids.sort_unstable();

    let mut labels: Vec<&str> = rows
        .iter()
        .map(|r| r.label.as_str())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    labels.sort_unstable();

    let img_index: HashMap<&str, u64> = image_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, (i + 1) as u64))
        .collect();
    let cat_index: HashMap<&str, u64> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (*l, (i + 1) as u64))
        .collect();

    let categories: Vec<Category> = labels
        .iter()
        .map(|label| Category {
            id: cat_index[label],
            name: names
                .get(*label)
                .cloned()
                .unwrap_or_else(|| (*label).into()),
            ..Default::default()
        })
        .collect();

    // Resolved once per unique image here, and reused below for every
    // annotation on that image — `image_scale`'s miss path allocates a
    // `format!`-ed extension string per call, so it must not run per row.
    let mut scale_by_img: HashMap<u64, (f64, f64)> = HashMap::with_capacity(image_ids.len());
    let images: Vec<Image> = image_ids
        .iter()
        .map(|id| {
            let (sx, sy, w, h) = image_scale(image_dims, id);
            let img_id = img_index[id];
            scale_by_img.insert(img_id, (sx, sy));
            Image {
                id: img_id,
                // The CSV records only the image ID; per the module policy the
                // ID becomes `file_name` verbatim, with no invented extension.
                file_name: (*id).to_string(),
                width: w,
                height: h,
                ..Default::default()
            }
        })
        .collect();

    let mut annotations = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let image_id = img_index[row.image_id.as_str()];
        let (sx, sy) = scale_by_img[&image_id];
        let x = row.xmin * sx;
        let y = row.ymin * sy;
        let w = (row.xmax - row.xmin) * sx;
        let h = (row.ymax - row.ymin) * sy;

        annotations.push(Annotation {
            id: (i + 1) as u64,
            image_id,
            category_id: cat_index[row.label.as_str()],
            bbox: Some([x, y, w, h]),
            area: Some(w * h),
            score: row.score,
            is_group_of: Some(row.is_group_of),
            ..Default::default()
        });
    }

    Ok(Dataset {
        images,
        annotations,
        categories,
        ..Default::default()
    })
}

/// Parse an Open Images detection CSV against an existing ground-truth dataset.
///
/// Detections must resolve to the same image and category IDs as the ground
/// truth, so this reads `gt` rather than numbering independently: `ImageID` is
/// matched against image file-name stems and `LabelName` against category names
/// (resolved through `class_descriptions` first, when supplied, to match however
/// the ground truth was loaded).
///
/// An unresolvable ID is an error rather than a skipped row. Dropping detections
/// silently changes recall, which is exactly the kind of quiet metric shift that
/// is impossible to notice downstream.
pub fn oid_results_to_anns(
    gt: &Dataset,
    csv_path: &Path,
    class_descriptions: Option<&Path>,
) -> Result<Vec<Annotation>, ConvertError> {
    let rows = read_rows(csv_path)?;
    let names = match class_descriptions {
        Some(p) => read_class_descriptions(p)?,
        None => HashMap::new(),
    };

    let img_index: HashMap<&str, &Image> = gt
        .images
        .iter()
        .map(|img| (file_stem(&img.file_name), img))
        .collect();
    let cat_index = crate::types::cat_name_to_id(&gt.categories);

    let mut annotations = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let img = img_index.get(row.image_id.as_str()).ok_or_else(|| {
            ConvertError::ParseError(format!(
                "detection references ImageID `{}`, which is not in the ground truth",
                row.image_id
            ))
        })?;

        let name = names
            .get(&row.label)
            .map_or(row.label.as_str(), String::as_str);
        let category_id = *cat_index.get(name).ok_or_else(|| {
            ConvertError::ParseError(format!(
                "detection references LabelName `{}`, which is not a ground-truth category",
                row.label
            ))
        })?;

        let (sx, sy) = if img.width == 0 || img.height == 0 {
            (1.0, 1.0)
        } else {
            (f64::from(img.width), f64::from(img.height))
        };
        let x = row.xmin * sx;
        let y = row.ymin * sy;
        let w = (row.xmax - row.xmin) * sx;
        let h = (row.ymax - row.ymin) * sy;

        annotations.push(Annotation {
            id: (i + 1) as u64,
            image_id: img.id,
            category_id,
            bbox: Some([x, y, w, h]),
            area: Some(w * h),
            score: Some(row.score.unwrap_or(1.0)),
            ..Default::default()
        });
    }

    Ok(annotations)
}

/// Convert a COCO dataset to an Open Images challenge-format annotation CSV.
///
/// Writes `ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf` with coordinates
/// normalized back to `[0, 1]`. `LabelName` is the COCO category name, so a
/// dataset imported with `class_descriptions` round-trips to display names rather
/// than to the MIDs it came from.
///
/// Detection scores, when present, are written as an extra trailing `Score`
/// column so that results files round-trip too.
///
/// # Errors
///
/// Returns [`ConvertError::MissingImageDimensions`] if an annotated image has
/// zero width/height (normalizing coordinates needs the true size),
/// [`ConvertError::StemCollision`] if two images share a filename stem (their
/// rows would merge under one `ImageID`), or
/// [`ConvertError::UnknownCategory`] if an annotation references a category id
/// that is not in the dataset.
pub fn coco_to_oid(dataset: &Dataset, output_csv: &Path) -> Result<OidStats, ConvertError> {
    super::check_unique_stems(dataset)?;
    if let Some(parent) = output_csv.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(super::io_at(parent))?;
        }
    }

    let cat_map = crate::types::cat_id_to_name(dataset);
    let grouped = anns_by_image(dataset);
    let scored = dataset.annotations.iter().any(|a| a.score.is_some());

    let mut file = fs::File::create(output_csv).map_err(super::io_at(output_csv))?;
    if scored {
        writeln!(
            file,
            "ImageID,LabelName,Score,XMin,XMax,YMin,YMax,IsGroupOf"
        )?;
    } else {
        writeln!(file, "ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf")?;
    }

    let mut stats = OidStats::default();
    for img in &dataset.images {
        let Some(anns) = grouped.get(&img.id) else {
            continue;
        };

        if img.width == 0 || img.height == 0 {
            return Err(ConvertError::MissingImageDimensions(format!(
                "{} (id {})",
                img.file_name, img.id
            )));
        }
        let (w, h) = (f64::from(img.width), f64::from(img.height));
        let stem = csv_field(file_stem(&img.file_name));
        let mut wrote = false;

        for ann in anns {
            let Some(bbox) = ann.bbox else {
                stats.skipped_no_bbox += 1;
                continue;
            };
            let label = csv_field(cat_map.get(&ann.category_id).ok_or(
                ConvertError::UnknownCategory {
                    ann_id: ann.id,
                    category_id: ann.category_id,
                },
            )?);
            let group_of = ann.is_group_of.unwrap_or(false);
            let (xmin, ymin) = (bbox[0] / w, bbox[1] / h);
            let (xmax, ymax) = ((bbox[0] + bbox[2]) / w, (bbox[1] + bbox[3]) / h);

            if scored {
                writeln!(
                    file,
                    "{stem},{label},{:.6},{xmin:.6},{xmax:.6},{ymin:.6},{ymax:.6},{}",
                    ann.score.unwrap_or(1.0),
                    i32::from(group_of),
                )?;
            } else {
                writeln!(
                    file,
                    "{stem},{label},{xmin:.6},{xmax:.6},{ymin:.6},{ymax:.6},{}",
                    i32::from(group_of),
                )?;
            }

            stats.annotations += 1;
            stats.group_of += usize::from(group_of);
            wrote = true;
        }

        if wrote {
            stats.images += 1;
        }
    }

    Ok(stats)
}
