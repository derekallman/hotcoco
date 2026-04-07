use serde::{Deserialize, Deserializer, Serialize};

/// Top-level COCO dataset structure.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Dataset {
    #[serde(default)]
    pub info: Option<Info>,
    #[serde(default)]
    pub images: Vec<Image>,
    #[serde(default)]
    pub annotations: Vec<Annotation>,
    #[serde(default)]
    pub categories: Vec<Category>,
    #[serde(default)]
    pub licenses: Vec<License>,
}

/// Dataset metadata (version, description, date, etc.).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Info {
    #[serde(default)]
    pub year: Option<u32>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub contributor: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub date_created: Option<String>,
}

/// A single image in the dataset.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Image {
    pub id: u64,
    #[serde(default)]
    pub file_name: String,
    pub height: u32,
    pub width: u32,
    #[serde(default)]
    pub license: Option<u64>,
    #[serde(default)]
    pub coco_url: Option<String>,
    #[serde(default)]
    pub flickr_url: Option<String>,
    #[serde(default)]
    pub date_captured: Option<String>,
    /// LVIS: categories confirmed absent in this image (unmatched DTs are FP).
    #[serde(default)]
    pub neg_category_ids: Vec<u64>,
    /// LVIS: categories not exhaustively checked in this image (unmatched DTs are ignored).
    #[serde(default)]
    pub not_exhaustive_category_ids: Vec<u64>,
}

/// A single object annotation (ground truth or detection result).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Annotation {
    #[serde(default)]
    pub id: u64,
    pub image_id: u64,
    pub category_id: u64,
    #[serde(default)]
    pub bbox: Option<[f64; 4]>,
    #[serde(default)]
    pub area: Option<f64>,
    #[serde(default)]
    pub segmentation: Option<Segmentation>,
    #[serde(default, deserialize_with = "deserialize_iscrowd")]
    pub iscrowd: bool,
    #[serde(default)]
    pub keypoints: Option<Vec<f64>>,
    #[serde(default)]
    pub num_keypoints: Option<u32>,
    /// Oriented bounding box as `[cx, cy, w, h, angle]` where angle is in radians.
    /// Used for rotated detection evaluation (aerial imagery, document analysis, scene text).
    #[serde(default)]
    pub obb: Option<[f64; 5]>,
    /// Detection score (present only in result annotations).
    #[serde(default)]
    pub score: Option<f64>,
    /// Open Images group-of flag. When true, the annotation represents a group of objects
    /// rather than a single instance. Distinct from `iscrowd` — different matching semantics.
    #[serde(default)]
    pub is_group_of: Option<bool>,
}

/// Deserialize `iscrowd` from either a boolean or an integer (0/1).
///
/// COCO JSON files use both representations, so we accept either.
fn deserialize_iscrowd<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum IsCrowd {
        Bool(bool),
        Int(u8),
    }
    match IsCrowd::deserialize(deserializer)? {
        IsCrowd::Bool(b) => Ok(b),
        IsCrowd::Int(i) => Ok(i != 0),
    }
}

/// Segmentation mask in one of three COCO formats.
///
/// Uses `#[serde(untagged)]` to auto-detect the format from JSON structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Segmentation {
    /// Polygon format: list of polygons, each a flat list of [x, y, x, y, ...] coordinates.
    Polygon(Vec<Vec<f64>>),
    /// Compressed RLE format (as stored in COCO JSON results).
    CompressedRle { size: [u32; 2], counts: String },
    /// Uncompressed RLE format.
    UncompressedRle { size: [u32; 2], counts: Vec<u32> },
}

/// An object category (e.g. "person", "car").
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Category {
    pub id: u64,
    pub name: String,
    #[serde(default)]
    pub supercategory: Option<String>,
    #[serde(default)]
    pub skeleton: Option<Vec<[u32; 2]>>,
    #[serde(default)]
    pub keypoints: Option<Vec<String>>,
    /// LVIS frequency bucket: "r" (rare), "c" (common), "f" (frequent).
    #[serde(default)]
    pub frequency: Option<String>,
}

/// Image license information.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct License {
    #[serde(default)]
    pub id: u64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
}

/// Summary statistics (min/max/mean/median) for a numeric field.
#[derive(Debug, Clone)]
pub struct SummaryStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
}

/// Per-category dataset statistics.
#[derive(Debug, Clone)]
pub struct CategoryStats {
    pub id: u64,
    pub name: String,
    pub ann_count: usize,
    pub img_count: usize,
    pub crowd_count: usize,
}

/// Dataset health-check statistics returned by [`crate::COCO::stats`].
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub image_count: usize,
    pub annotation_count: usize,
    pub category_count: usize,
    pub crowd_count: usize,
    /// Per-category breakdown, sorted by `ann_count` descending.
    pub per_category: Vec<CategoryStats>,
    pub image_width: SummaryStats,
    pub image_height: SummaryStats,
    /// Summary over annotations that have an `area` value.
    pub annotation_area: SummaryStats,
}

/// Run-length encoding for masks.
#[derive(Debug, Clone, PartialEq)]
pub struct Rle {
    pub h: u32,
    pub w: u32,
    /// Run counts: alternating runs of 0s and 1s, starting with 0s.
    pub counts: Vec<u32>,
}

impl Rle {
    /// Create a new RLE with a debug assertion that counts sum to `h * w`.
    pub fn new(h: u32, w: u32, counts: Vec<u32>) -> Self {
        let sum: u64 = counts.iter().map(|&c| c as u64).sum();
        let expected = h as u64 * w as u64;
        debug_assert_eq!(
            sum, expected,
            "RLE counts must sum to h*w ({h} * {w} = {expected}), got {sum}",
        );
        Self { h, w, counts }
    }
}
