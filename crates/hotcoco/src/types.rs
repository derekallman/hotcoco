use std::borrow::Cow;
use std::collections::HashMap;
use std::marker::PhantomData;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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

/// Dataset metadata: version, description, date, and the rest of the `info` block.
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
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Image {
    #[serde(deserialize_with = "deserialize_uint")]
    pub id: u64,
    #[serde(default)]
    pub file_name: String,
    /// Optional on load, as on the Python dict path: box-only evaluation
    /// never reads it, and TorchMetrics emits bare `{"id": i}` records.
    #[serde(default, deserialize_with = "deserialize_uint")]
    pub height: u32,
    #[serde(default, deserialize_with = "deserialize_uint")]
    pub width: u32,
    #[serde(default, deserialize_with = "deserialize_opt_uint")]
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
    /// Keys not in the COCO schema, preserved verbatim so
    /// load → filter/split/merge → save round-trips user metadata
    /// (pycocotools keeps unknown keys because it stores raw dicts).
    #[serde(flatten, skip_serializing_if = "Extra::is_empty")]
    pub extra: Extra,
}

/// The keys of a record that are not in the COCO schema, in file order.
///
/// A small map: lookups are linear over what is nearly always nothing or a
/// handful of entries, and an empty one allocates nothing. A
/// `serde_json::Map` is a B-tree whose first entry allocates a node of about
/// 600 bytes, three times the record itself, and detector outputs often
/// carry one unknown key on every detection.
#[derive(Clone, Debug, Default)]
pub struct Extra(Box<[(String, serde_json::Value)]>);

/// Set `key` in `entries`, replacing an existing entry in place; the
/// previous value, if any.
fn upsert(
    entries: &mut Vec<(String, serde_json::Value)>,
    key: String,
    value: serde_json::Value,
) -> Option<serde_json::Value> {
    match entries.iter_mut().find(|(k, _)| *k == key) {
        Some(slot) => Some(std::mem::replace(&mut slot.1, value)),
        None => {
            entries.reserve_exact(1);
            entries.push((key, value));
            None
        }
    }
}

impl Extra {
    /// No keys.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether there are no keys.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The number of keys.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// The value under `key`.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.index_of(key).map(|at| &self.0[at].1)
    }

    /// The value under `key`, mutably.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut serde_json::Value> {
        self.index_of(key).map(|at| &mut self.0[at].1)
    }

    /// Whether `key` is present.
    pub fn contains_key(&self, key: &str) -> bool {
        self.index_of(key).is_some()
    }

    fn index_of(&self, key: &str) -> Option<usize> {
        self.0.iter().position(|(k, _)| k == key)
    }

    /// Run `f` over the entries as a vector, then box them again.
    fn edit<R>(&mut self, f: impl FnOnce(&mut Vec<(String, serde_json::Value)>) -> R) -> R {
        let mut entries = Vec::from(std::mem::take(&mut self.0));
        let out = f(&mut entries);
        self.0 = entries.into_boxed_slice();
        out
    }

    /// Set `key` to `value`; the previous value, if any. An existing key
    /// keeps its position.
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Option<serde_json::Value> {
        let key = key.into();
        self.edit(|entries| upsert(entries, key, value))
    }

    /// Remove `key`; its value, if it was present.
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        let at = self.index_of(key)?;
        Some(self.edit(|entries| entries.remove(at).1))
    }

    /// The `(key, value)` entries, in file order.
    pub fn iter(&self) -> std::slice::Iter<'_, (String, serde_json::Value)> {
        self.0.iter()
    }
}

impl PartialEq for Extra {
    /// Equal as maps: the same keys with the same values, in any order.
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

impl<'a> IntoIterator for &'a Extra {
    type Item = &'a (String, serde_json::Value);
    type IntoIter = std::slice::Iter<'a, (String, serde_json::Value)>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for Extra {
    type Item = (String, serde_json::Value);
    type IntoIter = std::vec::IntoIter<(String, serde_json::Value)>;

    fn into_iter(self) -> Self::IntoIter {
        Vec::from(self.0).into_iter()
    }
}

impl FromIterator<(String, serde_json::Value)> for Extra {
    /// A repeated key keeps its last value, as a JSON object does.
    fn from_iter<I: IntoIterator<Item = (String, serde_json::Value)>>(iter: I) -> Self {
        let mut entries = Vec::new();
        for (key, value) in iter {
            upsert(&mut entries, key, value);
        }
        Extra(entries.into_boxed_slice())
    }
}

impl From<serde_json::Map<String, serde_json::Value>> for Extra {
    fn from(map: serde_json::Map<String, serde_json::Value>) -> Self {
        map.into_iter().collect()
    }
}

impl From<Extra> for serde_json::Map<String, serde_json::Value> {
    fn from(extra: Extra) -> Self {
        extra.into_iter().collect()
    }
}

impl Serialize for Extra {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.iter().map(|(k, v)| (k, v)))
    }
}

impl<'de> Deserialize<'de> for Extra {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ExtraVisitor;

        impl<'de> Visitor<'de> for ExtraVisitor {
            type Value = Extra;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an object")
            }

            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Extra, A::Error> {
                std::iter::from_fn(|| map.next_entry::<String, serde_json::Value>().transpose())
                    .collect()
            }
        }

        deserializer.deserialize_map(ExtraVisitor)
    }
}

/// A single object annotation (ground truth or detection result).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Annotation {
    #[serde(default, deserialize_with = "deserialize_uint")]
    pub id: u64,
    #[serde(deserialize_with = "deserialize_uint")]
    pub image_id: u64,
    #[serde(deserialize_with = "deserialize_uint")]
    pub category_id: u64,
    #[serde(default)]
    pub bbox: Option<[f64; 4]>,
    #[serde(default)]
    pub area: Option<f64>,
    #[serde(default)]
    pub segmentation: Option<Segmentation>,
    #[serde(default, deserialize_with = "deserialize_flag")]
    pub iscrowd: bool,
    #[serde(default, deserialize_with = "deserialize_exact_opt")]
    pub keypoints: Option<Vec<f64>>,
    /// Count of labeled keypoints (visibility `> 0`). Optional in the wild;
    /// read it through [`Annotation::num_visible_keypoints`], which derives it
    /// from `keypoints` when absent instead of treating absence as zero.
    #[serde(default, deserialize_with = "deserialize_opt_uint")]
    pub num_keypoints: Option<u32>,
    /// Oriented bounding box as `[cx, cy, w, h, angle]` where angle is in radians.
    /// Used for rotated detection evaluation (aerial imagery, document analysis, scene text).
    ///
    /// Boxed: every record pays for this field and few carry one.
    #[serde(default)]
    pub obb: Option<Box<[f64; 5]>>,
    /// Detection score (present only in result annotations).
    #[serde(default)]
    pub score: Option<f64>,
    /// Open Images group-of flag. When true, the annotation represents a group of objects
    /// rather than a single instance. Distinct from `iscrowd` — different matching semantics.
    #[serde(default, deserialize_with = "deserialize_opt_flag")]
    pub is_group_of: Option<bool>,
    /// Keys not in the COCO schema, preserved verbatim so
    /// load → filter/split/merge → save round-trips user metadata
    /// (pycocotools keeps unknown keys because it stores raw dicts).
    #[serde(flatten, skip_serializing_if = "Extra::is_empty")]
    pub extra: Extra,
}

impl Annotation {
    /// The number of labeled keypoints: `num_keypoints` when the record has
    /// it, otherwise the count of `(x, y, v)` triplets in `keypoints` with
    /// `v > 0`, which is what the field means.
    ///
    /// The only reader the keypoint ignore rule goes through. Reading the
    /// raw field with a zero default marked every ground truth in a file
    /// that omits `num_keypoints` as ignored, and keypoint AP came out of a
    /// dataset with no scorable objects — plausible numbers, wrong ones.
    pub fn num_visible_keypoints(&self) -> u32 {
        match (self.num_keypoints, &self.keypoints) {
            (Some(n), _) => n,
            (None, Some(k)) => k.chunks_exact(3).filter(|t| t[2] > 0.0).count() as u32,
            (None, None) => 0,
        }
    }
}

/// A 0/1 flag read from JSON: `true`/`false`, any integer, or an integral
/// float.
///
/// COCO files spell `iscrowd` as both `0`/`1` and `true`/`false`; a file
/// written through pandas spells it `0.0`. The Python dict path applies the
/// same rule in the bindings' `extract_flag`. A fractional float is an error
/// naming the value, not "truthy".
struct Flag(bool);

impl<'de> Deserialize<'de> for Flag {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FlagVisitor;

        impl Visitor<'_> for FlagVisitor {
            type Value = Flag;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a bool or a 0/1 flag")
            }

            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Flag, E> {
                Ok(Flag(v))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Flag, E> {
                Ok(Flag(v != 0))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Flag, E> {
                Ok(Flag(v != 0))
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Flag, E> {
                if v.fract() == 0.0 {
                    Ok(Flag(v != 0.0))
                } else {
                    Err(E::custom(format!("expected a bool or 0/1 flag, got {v}")))
                }
            }
        }

        deserializer.deserialize_any(FlagVisitor)
    }
}

fn deserialize_flag<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    Flag::deserialize(deserializer).map(|f| f.0)
}

fn deserialize_opt_flag<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<bool>, D::Error> {
    Option::<Flag>::deserialize(deserializer).map(|o| o.map(|f| f.0))
}

/// A non-negative integer read from JSON that may be spelled as an integral
/// float.
///
/// `"image_id": 1.0` is how a JSON written through pandas or a numpy-backed
/// encoder reads back, and pycocotools accepts it because `1.0 == 1`. A
/// fractional or negative value is still an error naming the value. The
/// Python dict path applies the same rule in the bindings' `extract_int`.
/// A streaming visitor rather than `#[serde(untagged)]`, for the reason the
/// [`Segmentation`] deserializer gives: ids are the most numerous scalars in
/// a file and must not be buffered.
struct Uint<T>(T);

impl<'de, T: TryFrom<u64>> Deserialize<'de> for Uint<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct UintVisitor<T>(PhantomData<T>);

        impl<T: TryFrom<u64>> Visitor<'_> for UintVisitor<T> {
            type Value = Uint<T>;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a non-negative integer")
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Uint<T>, E> {
                T::try_from(v)
                    .map(Uint)
                    .map_err(|_| E::custom(format!("integer {v} is out of range for this field")))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Uint<T>, E> {
                u64::try_from(v)
                    .map_err(|_| E::custom(format!("expected a non-negative integer, got {v}")))
                    .and_then(|u| self.visit_u64(u))
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Uint<T>, E> {
                if v.fract() == 0.0 && v >= 0.0 && v <= u64::MAX as f64 {
                    self.visit_u64(v as u64)
                } else {
                    Err(E::custom(format!(
                        "expected a non-negative integer, got {v}"
                    )))
                }
            }
        }

        deserializer.deserialize_any(UintVisitor(PhantomData))
    }
}

fn deserialize_uint<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: TryFrom<u64>,
{
    Uint::deserialize(deserializer).map(|u| u.0)
}

fn deserialize_opt_uint<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: TryFrom<u64>,
{
    Option::<Uint<T>>::deserialize(deserializer).map(|o| o.map(|u| u.0))
}

/// A `Vec<f64>` read from JSON with its capacity equal to its length.
///
/// serde cannot size a JSON array before reading it, so `Vec<f64>` grows by
/// doubling and keeps the slack: a 51-value keypoint list holds 64 slots and
/// val2017's polygons carry 8 MB of slack on 14 MB of coordinates. The
/// values are read into a thread-local buffer and copied out at exact size.
struct ExactVec(Vec<f64>);

thread_local! {
    static EXACT_SCRATCH: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl<'de> Deserialize<'de> for ExactVec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        EXACT_SCRATCH.with(|scratch| {
            let mut buf = scratch.borrow_mut();
            // Stable serde API, hidden from its docs: overwrites the existing
            // elements, then truncates or pushes, keeping the capacity.
            Vec::deserialize_in_place(deserializer, &mut buf)?;
            Ok(ExactVec(buf.to_vec()))
        })
    }
}

fn deserialize_exact_opt<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<f64>>, D::Error> {
    Option::<ExactVec>::deserialize(deserializer).map(|o| o.map(|v| v.0))
}

/// Segmentation mask in one of three COCO formats.
///
/// `#[serde(untagged)]` auto-detects the format when *serializing* (it just
/// writes the variant's content, which is the COCO wire shape). Deserialization
/// is hand-written below instead of untagged: untagged buffers the entire value
/// into serde's internal `Content` tree and then tries each variant against it,
/// which materializes every polygon coordinate twice — measured as the dominant
/// cost of loading a polygon-heavy GT file.
/// The visitor streams instead: a JSON array is a polygon list, a JSON object
/// is an RLE whose variant is decided by the type of its `counts` value.
///
/// Marked `#[non_exhaustive]`: later families are expected to add formats
/// (a panoptic segment, a mask file), and downstream code must not be broken
/// by that. Match with a `_` arm; [`Segmentation::polygons`] reads whichever
/// variants are polygons.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum Segmentation {
    /// Polygon format: list of polygons, each a flat list of [x, y, x, y, ...] coordinates.
    Polygon(Vec<Vec<f64>>),
    /// The four-corner polygon of a box, stored as the box `[x, y, w, h]`.
    ///
    /// What `load_res` gives a box result that came without a segmentation, as
    /// pycocotools' `loadRes` does. Wherever a polygon list is expected — JSON,
    /// Python, CVAT export, rasterization — it is the single polygon
    /// [`Segmentation::rect_corners`] returns; [`Segmentation::polygons`] reads
    /// either variant. Storing the box keeps a box result off the heap.
    #[serde(serialize_with = "serialize_rect")]
    Rect([f64; 4]),
    /// Compressed RLE format (as stored in COCO JSON results).
    CompressedRle { size: [u32; 2], counts: String },
    /// Uncompressed RLE format.
    UncompressedRle { size: [u32; 2], counts: Vec<u32> },
}

impl Segmentation {
    /// The polygon list of a `Polygon` or `Rect` segmentation; `None` for an RLE.
    pub fn polygons(&self) -> Option<Cow<'_, [Vec<f64>]>> {
        match self {
            Segmentation::Polygon(polys) => Some(Cow::Borrowed(polys)),
            Segmentation::Rect(bbox) => Some(Cow::Owned(vec![Self::rect_corners(bbox).to_vec()])),
            Segmentation::CompressedRle { .. } | Segmentation::UncompressedRle { .. } => None,
        }
    }

    /// The corners of a `[x, y, w, h]` box as one flat polygon, in the order
    /// pycocotools' `loadRes` writes them.
    pub fn rect_corners(bbox: &[f64; 4]) -> [f64; 8] {
        let [x1, y1, bw, bh] = *bbox;
        let (x2, y2) = (x1 + bw, y1 + bh);
        [x1, y1, x1, y2, x2, y2, x2, y1]
    }
}

fn serialize_rect<S: Serializer>(bbox: &[f64; 4], serializer: S) -> Result<S::Ok, S::Error> {
    [Segmentation::rect_corners(bbox)].serialize(serializer)
}

impl<'de> Deserialize<'de> for Segmentation {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// `counts` value: a compressed-RLE string or an uncompressed run list,
        /// decided by the token serde hands the visitor — no buffering.
        enum Counts {
            Str(String),
            Ints(Vec<u32>),
        }

        impl<'de> Deserialize<'de> for Counts {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct CountsVisitor;
                impl<'de> serde::de::Visitor<'de> for CountsVisitor {
                    type Value = Counts;

                    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        f.write_str("an RLE counts string or an array of run lengths")
                    }

                    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Counts, E> {
                        Ok(Counts::Str(v.to_owned()))
                    }

                    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Counts, E> {
                        Ok(Counts::Str(v))
                    }

                    fn visit_seq<A: serde::de::SeqAccess<'de>>(
                        self,
                        mut seq: A,
                    ) -> Result<Counts, A::Error> {
                        let mut v = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                        while let Some(c) = seq.next_element()? {
                            v.push(c);
                        }
                        v.shrink_to_fit();
                        Ok(Counts::Ints(v))
                    }
                }
                deserializer.deserialize_any(CountsVisitor)
            }
        }

        struct SegVisitor;
        impl<'de> serde::de::Visitor<'de> for SegVisitor {
            type Value = Segmentation;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a list of polygons or an RLE object with `size` and `counts`")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Segmentation, A::Error> {
                // Nearly every segmentation is one polygon, and the list is
                // sized to what it holds like the polygons themselves.
                let mut polys = Vec::with_capacity(seq.size_hint().unwrap_or(1));
                while let Some(ExactVec(p)) = seq.next_element()? {
                    polys.push(p);
                }
                polys.shrink_to_fit();
                Ok(Segmentation::Polygon(polys))
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Segmentation, A::Error> {
                let mut size: Option<[u32; 2]> = None;
                let mut counts: Option<Counts> = None;
                while let Some(key) = map.next_key::<std::borrow::Cow<'_, str>>()? {
                    match key.as_ref() {
                        "size" => size = Some(map.next_value()?),
                        "counts" => counts = Some(map.next_value()?),
                        _ => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                let size = size.ok_or_else(|| serde::de::Error::missing_field("size"))?;
                match counts.ok_or_else(|| serde::de::Error::missing_field("counts"))? {
                    Counts::Str(counts) => Ok(Segmentation::CompressedRle { size, counts }),
                    Counts::Ints(counts) => Ok(Segmentation::UncompressedRle { size, counts }),
                }
            }
        }

        deserializer.deserialize_any(SegVisitor)
    }
}

/// An object category, such as "person" or "car".
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Category {
    #[serde(deserialize_with = "deserialize_uint")]
    pub id: u64,
    /// Display name. A category loaded without one gets the
    /// [`placeholder_cat_name`](crate::COCO::placeholder_cat_name) at index
    /// time, so this is never empty on an indexed dataset.
    #[serde(default)]
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
    /// Keys not in the COCO schema, preserved verbatim so
    /// load → filter/split/merge → save round-trips user metadata
    /// (pycocotools keeps unknown keys because it stores raw dicts).
    #[serde(flatten, skip_serializing_if = "Extra::is_empty")]
    pub extra: Extra,
}

/// Map `category_id -> name`.
///
/// The single owner of this reduction — `convert`'s exporters that write
/// categories by name, `detection::hierarchy`, and `quality::healthcheck` all
/// call this rather than re-collecting `dataset.categories` themselves.
pub(crate) fn cat_id_to_name(dataset: &Dataset) -> HashMap<u64, &str> {
    dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect()
}

/// Map `name -> category_id`.
///
/// Takes a category slice rather than a [`Dataset`] — some callers resolve
/// names against categories they are still assembling, before a `Dataset`
/// exists to hold them.
pub(crate) fn cat_name_to_id(categories: &[Category]) -> HashMap<&str, u64> {
    categories.iter().map(|c| (c.name.as_str(), c.id)).collect()
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

/// Run-length encoding for masks.
#[derive(Debug, Clone, PartialEq)]
pub struct Rle {
    pub h: u32,
    pub w: u32,
    /// Run counts: alternating runs of 0s and 1s, starting with 0s.
    pub counts: Vec<u32>,
}

impl Rle {
    /// Validated constructor: errors unless `counts` sums to exactly `h * w`.
    ///
    /// Validation happens in release builds too — use this at untrusted
    /// boundaries. Internal code that produces RLEs it already knows to be
    /// well-formed (the codecs in [`crate::mask`]) constructs the struct
    /// directly instead; the fields stay public for that reason.
    pub fn new(h: u32, w: u32, counts: Vec<u32>) -> crate::error::Result<Self> {
        let sum: u64 = counts.iter().map(|&c| c as u64).sum();
        let expected = h as u64 * w as u64;
        if sum != expected {
            return Err(
                format!("RLE counts must sum to h*w ({h} * {w} = {expected}), got {sum}").into(),
            );
        }
        Ok(Self { h, w, counts })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// The record is what every detection costs: 200 bytes with `obb` and
    /// `extra` behind pointers (248 inline), and `Rect` must fit the enum
    /// without growing it.
    #[test]
    fn record_layout_stays_slim() {
        assert!(std::mem::size_of::<Annotation>() <= 200);
        assert_eq!(
            std::mem::size_of::<Segmentation>(),
            std::mem::size_of::<Option<Segmentation>>()
        );
        assert!(std::mem::size_of::<Segmentation>() <= 40);
    }

    #[test]
    fn a_rect_reads_and_writes_as_the_polygon_load_res_used_to_build() {
        let bbox = [10.5, 20.25, 30.0, 40.125];
        let rect = Segmentation::Rect(bbox);
        let corners = vec![10.5, 20.25, 10.5, 60.375, 40.5, 60.375, 40.5, 20.25];
        let poly = Segmentation::Polygon(vec![corners.clone()]);

        assert_eq!(rect.polygons().unwrap().as_ref(), &[corners]);
        assert_eq!(
            serde_json::to_string(&rect).unwrap(),
            serde_json::to_string(&poly).unwrap()
        );
        assert!(matches!(
            serde_json::from_str::<Segmentation>(&serde_json::to_string(&rect).unwrap()).unwrap(),
            Segmentation::Polygon(p) if p.len() == 1 && p[0].len() == 8
        ));
        assert!(
            Segmentation::CompressedRle {
                size: [1, 1],
                counts: String::new()
            }
            .polygons()
            .is_none()
        );
    }

    /// Unknown keys survive a load in file order, a repeated key keeps its
    /// last value, an empty map serializes to nothing, and equality is a
    /// map's.
    #[test]
    fn extra_keys_round_trip_as_a_small_map() {
        let json = r#"{"image_id": 1, "category_id": 1, "zeta": 1, "alpha": [2], "zeta": "last"}"#;
        let ann: Annotation = serde_json::from_str(json).unwrap();
        assert_eq!(ann.extra.len(), 2);
        assert_eq!(ann.extra.get("zeta"), Some(&serde_json::json!("last")));
        assert_eq!(
            ann.extra
                .iter()
                .map(|(k, _)| k.as_str())
                .collect::<Vec<_>>(),
            ["zeta", "alpha"]
        );
        let out = serde_json::to_value(&ann).unwrap();
        assert_eq!(out["zeta"], "last");
        assert_eq!(out["alpha"], serde_json::json!([2]));

        let mut other = Extra::new();
        other.insert("alpha", serde_json::json!([2]));
        other.insert("zeta", serde_json::json!("last"));
        assert_eq!(ann.extra, other);
        assert_eq!(other.remove("alpha"), Some(serde_json::json!([2])));
        assert_ne!(ann.extra, other);
        other.remove("zeta");
        assert!(other.is_empty());

        let plain: Annotation =
            serde_json::from_str(r#"{"image_id": 1, "category_id": 1}"#).unwrap();
        assert!(!serde_json::to_string(&plain).unwrap().contains("extra"));
        assert!(plain.extra.is_empty());
    }

    /// The per-record heap cost is the coordinates themselves, not serde's
    /// growth slack: every parsed polygon and keypoint list is sized exactly.
    #[test]
    fn parsed_float_lists_carry_no_capacity_slack() {
        let json = r#"{"image_id": 1, "category_id": 1,
            "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5, 2.5, 2.5], [1, 1, 2, 1, 2, 2]],
            "keypoints": [1, 2, 2, 3, 4, 2, 5, 6, 0, 7, 8, 1, 9, 10, 2, 11, 12, 2, 13, 14, 1]}"#;
        let ann: Annotation = serde_json::from_str(json).unwrap();
        let Some(Segmentation::Polygon(polys)) = &ann.segmentation else {
            panic!("expected polygons");
        };
        assert_eq!(polys.capacity(), 2);
        assert_eq!(
            polys[0],
            vec![0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0, 2.5, 2.5]
        );
        assert_eq!(polys[1].len(), 6);
        assert!(polys.iter().all(|p| p.capacity() == p.len()));
        let kpts = ann.keypoints.unwrap();
        assert_eq!(kpts.len(), 21);
        assert_eq!(kpts.capacity(), 21);
        assert_eq!(kpts[2], 2.0);
    }
}
