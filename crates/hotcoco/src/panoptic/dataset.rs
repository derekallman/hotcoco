//! The panoptic data model: one record per image, each a list of segments.
//!
//! COCO panoptic JSON differs from the detection schema in one way — an
//! `annotation` is an image, not an object, and its `segments_info` list
//! names the segments found in that image's PNG. [`PanopticDataset`] is that
//! schema. [`PanopticDataset::from_dataset`] builds the same shape from a
//! detection-style [`Dataset`] whose annotations carry masks, which is the
//! path that needs no PNG files.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::types::{
    Category, Dataset, Image, Info, License, Segmentation, deserialize_flag, deserialize_opt_uint,
    deserialize_uint,
};

/// One segment of one image.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SegmentInfo {
    /// The segment's id: its color in the PNG (`R + 256 G + 256² B`), or the
    /// annotation id when the dataset came from a [`Dataset`].
    #[serde(deserialize_with = "deserialize_uint")]
    pub id: u64,
    #[serde(deserialize_with = "deserialize_uint")]
    pub category_id: u64,
    /// Pixel count as the file states it. For ground truth this is the area
    /// the matching uses, as panopticapi reads it; when absent, the pixels
    /// found in the mask stand in. For predictions it is ignored — the mask
    /// decides, as in the reference.
    #[serde(
        default,
        deserialize_with = "deserialize_opt_uint",
        skip_serializing_if = "Option::is_none"
    )]
    pub area: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f64; 4]>,
    /// Ground truth only. Never matched and never a miss; absorbs unmatched
    /// predictions of its category that mostly lie on it.
    #[serde(default, deserialize_with = "deserialize_flag")]
    pub iscrowd: bool,
    /// The segment's mask, for a dataset built without PNG files. `None` in
    /// the COCO panoptic format, where the PNG holds every mask.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segmentation: Option<Segmentation>,
}

/// One image's segments.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PanopticAnnotation {
    #[serde(deserialize_with = "deserialize_uint")]
    pub image_id: u64,
    /// The PNG holding this image's label map, relative to the dataset's
    /// folder. `None` when the segments carry their own masks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_name: Option<String>,
    /// Required, empty or not: its absence is how a detection-style file
    /// handed to [`from_file`](PanopticDataset::from_file) is told apart
    /// from a panoptic one, instead of parsing as images with no segments.
    pub segments_info: Vec<SegmentInfo>,
}

/// A COCO panoptic JSON file, plus where its PNG files live.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PanopticDataset {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<Info>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub licenses: Vec<License>,
    #[serde(default)]
    pub images: Vec<Image>,
    #[serde(default)]
    pub annotations: Vec<PanopticAnnotation>,
    /// Empty in a prediction file; the ground truth's categories are the
    /// ones evaluation uses, as in panopticapi.
    #[serde(default)]
    pub categories: Vec<Category>,
    /// Directory of the PNG label maps. An annotation with a `file_name` is
    /// read from `folder / file_name` when this is `Some`; any other
    /// annotation is painted from its segments' own `segmentation`.
    #[serde(skip)]
    pub folder: Option<PathBuf>,
}

impl PanopticDataset {
    /// Load a COCO panoptic JSON file.
    ///
    /// The PNG folder defaults to the file's path without its `.json`
    /// extension, panopticapi's convention (`panoptic_val2017.json` beside
    /// `panoptic_val2017/`); [`with_folder`](Self::with_folder) overrides it.
    pub fn from_file(path: &Path) -> crate::error::Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read panoptic JSON {}: {e}", path.display()))?;
        let mut dataset: PanopticDataset = serde_json::from_slice(&bytes)
            .map_err(|e| format!("cannot parse panoptic JSON {}: {e}", path.display()))?;
        dataset.folder = Some(path.with_extension(""));
        Ok(dataset)
    }

    /// Read the PNG files from `folder` instead of the default beside the JSON.
    #[must_use]
    pub fn with_folder(mut self, folder: impl Into<PathBuf>) -> Self {
        self.folder = Some(folder.into());
        self
    }

    /// A panoptic dataset from detection-style annotations, one segment per
    /// annotation, with no PNG files.
    ///
    /// Every annotation must carry a `segmentation` — polygon or RLE — and
    /// its image must have a `height` and `width`; `run()` reports the ones
    /// that do not. The segment id is the annotation id, `iscrowd` carries
    /// over, and `area` is left to be counted from the mask: a panoptic map
    /// is a partition, so the painted pixels are the segment, and a stale
    /// `area` field must not be able to move an IoU. Annotations are painted
    /// in dataset order, a later one over an earlier one where masks
    /// overlap. Images with no annotations get an empty record, so a
    /// prediction with nothing in an image is still a prediction for it.
    pub fn from_dataset(dataset: &Dataset) -> Self {
        let mut by_image: BTreeMap<u64, Vec<SegmentInfo>> = dataset
            .images
            .iter()
            .map(|img| (img.id, Vec::new()))
            .collect();
        for ann in &dataset.annotations {
            by_image.entry(ann.image_id).or_default().push(SegmentInfo {
                id: ann.id,
                category_id: ann.category_id,
                area: None,
                bbox: ann.bbox,
                iscrowd: ann.iscrowd,
                segmentation: ann.segmentation.clone(),
            });
        }
        PanopticDataset {
            info: dataset.info.clone(),
            licenses: dataset.licenses.clone(),
            images: dataset.images.clone(),
            annotations: by_image
                .into_iter()
                .map(|(image_id, segments_info)| PanopticAnnotation {
                    image_id,
                    file_name: None,
                    segments_info,
                })
                .collect(),
            categories: dataset.categories.clone(),
            folder: None,
        }
    }

    /// Serialize to JSON — the `annotations` and `categories` a prediction
    /// file needs, plus whatever else this dataset holds.
    pub fn to_json(&self) -> crate::error::Result<String> {
        Ok(serde_json::to_string(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_coco_panoptic_shape() {
        let json = r#"{
            "images": [{"id": 139, "file_name": "000000000139.jpg", "height": 426, "width": 640}],
            "annotations": [{
                "image_id": 139, "file_name": "000000000139.png",
                "segments_info": [
                    {"id": 3226956, "category_id": 1, "iscrowd": 0, "bbox": [413, 158, 53, 138], "area": 2840},
                    {"id": 6979964, "category_id": 184, "iscrowd": 0, "bbox": [0, 0, 640, 426], "area": 83000}
                ]
            }],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "person", "isthing": 1, "color": [220, 20, 60]},
                {"id": 184, "name": "floor-other-merged", "isthing": 0, "color": [96, 36, 108]}
            ]
        }"#;
        let ds: PanopticDataset = serde_json::from_str(json).expect("parses");
        assert_eq!(ds.annotations[0].segments_info[1].area, Some(83000));
        assert_eq!(ds.categories[0].isthing, Some(true));
        assert_eq!(ds.categories[1].isthing, Some(false));
        assert!(
            ds.folder.is_none(),
            "a parsed dataset has no folder until told"
        );
        // `isthing` writes back as a bool, which panopticapi's `== 1` reads as 1/0.
        let back = serde_json::to_string(&ds.categories[0]).expect("serializes");
        assert!(back.contains(r#""isthing":true"#), "{back}");
    }

    #[test]
    fn prediction_file_needs_only_annotations() {
        let ds: PanopticDataset = serde_json::from_str(
            r#"{"annotations": [{"image_id": 1, "file_name": "1.png", "segments_info": []}]}"#,
        )
        .expect("parses");
        assert!(ds.categories.is_empty());
        assert_eq!(ds.annotations[0].segments_info.len(), 0);
    }
}
