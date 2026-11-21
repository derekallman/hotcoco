//! Regression tests for the 1.0 preflight converter fixes.
//!
//! Each test pins one behavioral fix: CVAT open/close-pair shapes, YOLO
//! block-form `names:`, comma-safe YAML names, VOC float coordinates and the
//! 1-based inclusive convention, stem-collision detection, and the unified
//! missing-dimensions / malformed-input policies.

use std::collections::HashMap;

use hotcoco::convert::{
    ConvertError, coco_to_dota, coco_to_oid, coco_to_voc, coco_to_yolo, cvat_to_coco, dota_to_coco,
    voc_to_coco, yolo_to_coco,
};
use hotcoco::types::{Annotation, Category, Dataset, Image, Segmentation};

/// A one-image dataset with the given file name and simple bbox annotations.
fn one_image_dataset(file_name: &str, cats: &[&str], bboxes: &[(u64, [f64; 4])]) -> Dataset {
    Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: file_name.into(),
            width: 640,
            height: 480,
            ..Default::default()
        }],
        annotations: bboxes
            .iter()
            .enumerate()
            .map(|(i, &(cat_id, bbox))| Annotation {
                id: (i + 1) as u64,
                image_id: 1,
                category_id: cat_id,
                bbox: Some(bbox),
                area: Some(bbox[2] * bbox[3]),
                ..Default::default()
            })
            .collect(),
        categories: cats
            .iter()
            .enumerate()
            .map(|(i, name)| Category {
                id: (i + 1) as u64,
                name: (*name).into(),
                ..Default::default()
            })
            .collect(),
        licenses: vec![],
    }
}

// ── CVAT ─────────────────────────────────────────────────────────────────────

/// CVAT writes a shape as an open/close pair whenever it has `<attribute>`
/// children. Those shapes must import, not vanish — a real CVAT export can
/// otherwise read as zero annotations with no error.
#[test]
fn cvat_start_event_shapes_with_attribute_children_import() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xml_path = dir.path().join("annotations.xml");
    let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><labels>
    <label><name>person</name></label>
    <label><name>region</name></label>
  </labels></task></meta>
  <image id="0" name="test.jpg" width="640" height="480">
    <box label="person" xtl="100" ytl="50" xbr="300" ybr="400" occluded="0">
      <attribute name="pose">standing</attribute>
      <attribute name="verified">true</attribute>
    </box>
    <polygon label="region" points="10.0,20.0;50.0,20.0;50.0,80.0" occluded="0">
      <attribute name="kind">roi</attribute>
    </polygon>
  </image>
</annotations>"#;
    std::fs::write(&xml_path, xml).expect("write xml");

    let (dataset, stats) = cvat_to_coco(&xml_path).expect("cvat_to_coco");

    assert_eq!(dataset.annotations.len(), 2, "both pair-form shapes import");
    assert_eq!(stats.boxes, 1);
    assert_eq!(stats.polygons, 1);
    assert_eq!(stats.skipped_unsupported, 0, "<attribute> is not a shape");

    let box_ann = &dataset.annotations[0];
    assert_eq!(box_ann.bbox.expect("bbox"), [100.0, 50.0, 200.0, 350.0]);
    let poly_ann = &dataset.annotations[1];
    assert!(matches!(
        poly_ann.segmentation,
        Some(Segmentation::Polygon(_))
    ));
}

/// A `<box>` missing a coordinate attribute must be an error, never a silent
/// `0.0` default producing a plausible wrong box.
#[test]
fn cvat_missing_coordinate_attribute_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xml_path = dir.path().join("annotations.xml");
    let xml = r#"<annotations>
  <image id="0" name="a.jpg" width="10" height="10">
    <box label="thing" xtl="1" ytl="2" xbr="3" occluded="0"/>
  </image>
</annotations>"#;
    std::fs::write(&xml_path, xml).expect("write xml");

    let err = cvat_to_coco(&xml_path).expect_err("missing ybr must fail");
    let msg = err.to_string();
    assert!(msg.contains("ybr"), "error names the attribute: {msg}");
    assert!(
        msg.contains("annotations.xml"),
        "error names the file: {msg}"
    );
}

/// A missing `points` attribute is malformed input (error); a *present* points
/// list with fewer than 3 points is degenerate — skipped and counted, keeping
/// the rest of the file.
#[test]
fn cvat_degenerate_polygon_is_skipped_and_counted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xml_path = dir.path().join("annotations.xml");
    let xml = r#"<annotations>
  <image id="0" name="a.jpg" width="100" height="100">
    <polygon label="thing" points="10,10;50,50" occluded="0"/>
    <box label="thing" xtl="10" ytl="10" xbr="50" ybr="50" occluded="0"/>
  </image>
</annotations>"#;
    std::fs::write(&xml_path, xml).expect("write xml");

    let (dataset, stats) = cvat_to_coco(&xml_path).expect("file must survive");
    assert_eq!(dataset.annotations.len(), 1, "the box still imports");
    assert_eq!(
        stats.skipped_degenerate, 1,
        "the 2-point polygon is counted"
    );

    let no_points = r#"<annotations>
  <image id="0" name="a.jpg" width="100" height="100">
    <polygon label="thing" occluded="0"/>
  </image>
</annotations>"#;
    std::fs::write(&xml_path, no_points).expect("write xml");
    let err = cvat_to_coco(&xml_path).expect_err("missing points attribute");
    assert!(err.to_string().contains("points"), "error was: {err}");
}

/// A CVAT for *video* export nests shapes under `<track>`, not `<image>`.
/// Importing one used to yield an empty dataset with no error.
#[test]
fn cvat_video_export_is_a_clear_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xml_path = dir.path().join("annotations.xml");
    let xml = r#"<annotations>
  <version>1.1</version>
  <track id="0" label="person">
    <box frame="0" xtl="1" ytl="2" xbr="3" ybr="4" outside="0"/>
  </track>
</annotations>"#;
    std::fs::write(&xml_path, xml).expect("write xml");

    let err = cvat_to_coco(&xml_path).expect_err("video export must not import as empty");
    assert!(err.to_string().contains("video"), "error was: {err}");
}

/// Export: an annotation whose polygons are all degenerate falls back to its
/// bbox instead of vanishing; the degenerate polygons are counted.
#[test]
fn cvat_export_counts_degenerate_polygons_and_falls_back_to_bbox() {
    let mut dataset = one_image_dataset("a.jpg", &["thing"], &[(1, [10.0, 10.0, 20.0, 20.0])]);
    dataset.annotations[0].segmentation =
        Some(Segmentation::Polygon(vec![vec![1.0, 1.0, 2.0, 2.0]])); // 2 points

    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out.xml");
    let stats = hotcoco::convert::coco_to_cvat(&dataset, &out).expect("coco_to_cvat");
    assert_eq!(stats.skipped_degenerate, 1);
    assert_eq!(stats.boxes, 1, "bbox fallback");
    assert_eq!(stats.polygons, 0);
    assert_eq!(stats.skipped_no_geometry, 0);

    let (recovered, _) = cvat_to_coco(&out).expect("re-import");
    assert_eq!(recovered.annotations.len(), 1);
    assert_eq!(
        recovered.annotations[0].bbox.expect("bbox"),
        [10.0, 10.0, 20.0, 20.0]
    );
}

// ── YOLO ─────────────────────────────────────────────────────────────────────

/// The block/mapping form of `names:` is what Ultralytics writes by default;
/// rejecting it means rejecting most real data.yaml files.
#[test]
fn yolo_block_mapping_names_parse() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("data.yaml"),
        "path: ../datasets/coco\ntrain: images/train\nnc: 2\nnames:\n  0: person\n  1: bicycle\nextra: 1\n",
    )
    .expect("write data.yaml");
    std::fs::write(dir.path().join("img.txt"), "1 0.5 0.5 0.2 0.2\n").expect("write label");

    let dims: HashMap<String, (u32, u32)> = [("img".to_string(), (100u32, 100u32))].into();
    let ds = yolo_to_coco(dir.path(), &dims).expect("yolo_to_coco");

    assert_eq!(ds.categories.len(), 2);
    assert_eq!(ds.categories[0].name, "person");
    assert_eq!(ds.categories[1].name, "bicycle");
    assert_eq!(ds.annotations[0].category_id, 2, "class 1 → bicycle");
}

/// The block list form (`- person`) also appears in the wild.
#[test]
fn yolo_block_list_names_parse() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("data.yaml"),
        "nc: 2\nnames:\n  - person\n  - bicycle\n",
    )
    .expect("write data.yaml");
    std::fs::write(dir.path().join("img.txt"), "0 0.5 0.5 0.2 0.2\n").expect("write label");

    let dims: HashMap<String, (u32, u32)> = [("img".to_string(), (100u32, 100u32))].into();
    let ds = yolo_to_coco(dir.path(), &dims).expect("yolo_to_coco");
    assert_eq!(ds.categories.len(), 2);
    assert_eq!(ds.categories[0].name, "person");
}

/// Gaps or duplicates in the mapping indices would silently shift class ids.
#[test]
fn yolo_block_mapping_with_gap_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("data.yaml"),
        "names:\n  0: person\n  2: bicycle\n",
    )
    .expect("write data.yaml");

    let err = yolo_to_coco(dir.path(), &HashMap::new()).expect_err("gap in indices");
    assert!(err.to_string().contains("gaps"), "error was: {err}");
}

/// A comma inside a category name must not split the flow list and shift every
/// later class index.
#[test]
fn yolo_names_containing_commas_round_trip() {
    let dataset = one_image_dataset(
        "img.jpg",
        &["Camera, still", "dog"],
        &[(1, [10.0, 10.0, 20.0, 20.0]), (2, [40.0, 40.0, 20.0, 20.0])],
    );
    let dir = tempfile::tempdir().expect("tempdir");
    coco_to_yolo(&dataset, dir.path()).expect("coco_to_yolo");

    let yaml = std::fs::read_to_string(dir.path().join("data.yaml")).expect("data.yaml");
    assert!(
        yaml.contains("'Camera, still'"),
        "comma name is quoted: {yaml}"
    );

    let dims: HashMap<String, (u32, u32)> = [("img".to_string(), (640u32, 480u32))].into();
    let recovered = yolo_to_coco(dir.path(), &dims).expect("yolo_to_coco");
    assert_eq!(
        recovered.categories.len(),
        2,
        "comma did not split the list"
    );
    assert_eq!(recovered.categories[0].name, "Camera, still");
    assert_eq!(recovered.categories[1].name, "dog");
}

/// Importing a YOLO label with no known image size used to multiply by 0×0 and
/// produce silent zero-size boxes; it is now an error.
#[test]
fn yolo_import_missing_dims_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("data.yaml"), "nc: 1\nnames: [thing]\n")
        .expect("write data.yaml");
    std::fs::write(dir.path().join("img.txt"), "0 0.5 0.5 0.2 0.2\n").expect("write label");

    let err = yolo_to_coco(dir.path(), &HashMap::new()).expect_err("no dims for img");
    assert!(
        matches!(err, ConvertError::MissingImageDimensions(_)),
        "got: {err}"
    );
}

/// Label parse errors carry the file and line.
#[test]
fn yolo_label_errors_name_file_and_line() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("data.yaml"), "nc: 1\nnames: [thing]\n")
        .expect("write data.yaml");
    std::fs::write(
        dir.path().join("img.txt"),
        "0 0.5 0.5 0.2 0.2\n0 0.5 oops 0.2 0.2\n",
    )
    .expect("write label");

    let dims: HashMap<String, (u32, u32)> = [("img".to_string(), (100u32, 100u32))].into();
    let err = yolo_to_coco(dir.path(), &dims).expect_err("bad cy");
    let msg = err.to_string();
    assert!(msg.contains("img.txt"), "names the file: {msg}");
    assert!(msg.contains("line 2"), "names the line: {msg}");
}

// ── VOC ──────────────────────────────────────────────────────────────────────

/// Real-world VOC files carry float coordinates like `156.00`; parsing them as
/// integers failed the file and aborted the whole directory import.
#[test]
fn voc_float_coordinates_parse() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ann_dir = dir.path().join("Annotations");
    std::fs::create_dir_all(&ann_dir).expect("mkdir");
    let xml = r"<annotation>
  <filename>a.jpg</filename>
  <size><width>640</width><height>480</height></size>
  <object>
    <name>person</name>
    <bndbox><xmin>156.00</xmin><ymin>50.5</ymin><xmax>300.00</xmax><ymax>400.0</ymax></bndbox>
  </object>
</annotation>";
    std::fs::write(ann_dir.join("a.xml"), xml).expect("write xml");

    let ds = voc_to_coco(dir.path()).expect("float coords must parse");
    assert_eq!(ds.annotations.len(), 1);
    let bbox = ds.annotations[0].bbox.expect("bbox");
    // 1-based inclusive: x = 156 - 1, w = 300 - 156 + 1.
    assert!((bbox[0] - 155.0).abs() < 1e-9, "x: {}", bbox[0]);
    assert!((bbox[2] - 145.0).abs() < 1e-9, "w: {}", bbox[2]);
}

/// iscrowd → `<difficult>` → iscrowd survives a full round trip, and integer
/// boxes round-trip exactly under the 1-based inclusive convention.
#[test]
fn voc_round_trip_is_exact_and_keeps_iscrowd() {
    let mut dataset = one_image_dataset(
        "img.jpg",
        &["thing"],
        &[(1, [10.0, 20.0, 30.0, 40.0]), (1, [0.0, 0.0, 200.0, 150.0])],
    );
    dataset.annotations[0].iscrowd = true;

    let dir = tempfile::tempdir().expect("tempdir");
    coco_to_voc(&dataset, dir.path()).expect("coco_to_voc");
    let recovered = voc_to_coco(dir.path()).expect("voc_to_coco");

    assert_eq!(recovered.annotations.len(), 2);
    // Sorted output preserves annotation order within the single image.
    assert_eq!(
        recovered.annotations[0].bbox.expect("bbox"),
        [10.0, 20.0, 30.0, 40.0],
        "integer boxes round-trip exactly"
    );
    assert_eq!(
        recovered.annotations[1].bbox.expect("bbox"),
        [0.0, 0.0, 200.0, 150.0]
    );
    assert!(recovered.annotations[0].iscrowd, "iscrowd survives");
    assert!(!recovered.annotations[1].iscrowd);
}

/// VOC parse errors name the failing file.
#[test]
fn voc_errors_name_the_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ann_dir = dir.path().join("Annotations");
    std::fs::create_dir_all(&ann_dir).expect("mkdir");
    let xml = r"<annotation>
  <filename>a.jpg</filename>
  <size><width>640</width><height>480</height></size>
  <object>
    <name>person</name>
    <bndbox><xmin>oops</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox>
  </object>
</annotation>";
    std::fs::write(ann_dir.join("bad.xml"), xml).expect("write xml");

    let err = voc_to_coco(dir.path()).expect_err("bad coordinate");
    assert!(err.to_string().contains("bad.xml"), "error was: {err}");
}

// ── Stem collisions ──────────────────────────────────────────────────────────

/// `train/img.jpg` and `val/img.jpg` share the stem `img`; every exporter that
/// keys output by stem must refuse rather than silently overwrite.
#[test]
fn exporters_reject_colliding_file_stems() {
    let mut dataset = one_image_dataset("train/img.jpg", &["thing"], &[(1, [1.0, 1.0, 2.0, 2.0])]);
    dataset.images.push(Image {
        id: 2,
        file_name: "val/img.jpg".into(),
        width: 640,
        height: 480,
        ..Default::default()
    });

    let dir = tempfile::tempdir().expect("tempdir");
    let is_collision = |e: &ConvertError| matches!(e, ConvertError::StemCollision { .. });

    let err = coco_to_yolo(&dataset, dir.path()).expect_err("yolo");
    assert!(is_collision(&err), "yolo: {err}");
    let err = coco_to_voc(&dataset, dir.path()).expect_err("voc");
    assert!(is_collision(&err), "voc: {err}");
    let err = coco_to_dota(&dataset, dir.path()).expect_err("dota");
    assert!(is_collision(&err), "dota: {err}");
    let err = coco_to_oid(&dataset, &dir.path().join("out.csv")).expect_err("oid");
    assert!(is_collision(&err), "oid: {err}");
}

// ── DOTA ─────────────────────────────────────────────────────────────────────

/// Malformed DOTA lines used to be skipped silently and uncounted; they are now
/// errors naming the file and line.
#[test]
fn dota_malformed_line_is_an_error_with_context() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("P0001.txt"),
        "0.0 0.0 4.0 0.0 4.0 3.0 0.0 3.0 plane 0\n1.0 2.0 nonsense\n",
    )
    .expect("write label");

    let err = dota_to_coco(dir.path(), None, &HashMap::new()).expect_err("short line");
    let msg = err.to_string();
    assert!(msg.contains("P0001.txt"), "names the file: {msg}");
    assert!(msg.contains("line 2"), "names the line: {msg}");

    std::fs::write(
        dir.path().join("P0001.txt"),
        "0.0 0.0 4.0 0.0 4.0 3.0 0.0 3.0 plane notanumber\n",
    )
    .expect("write label");
    let err = dota_to_coco(dir.path(), None, &HashMap::new()).expect_err("bad difficulty");
    assert!(err.to_string().contains("difficulty"), "error was: {err}");
}

/// Real DOTA labels start with `imagesource:`/`gsd:` metadata lines; the
/// stricter malformed-line policy must not reject them.
#[test]
fn dota_metadata_header_lines_are_skipped() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("P0001.txt"),
        "imagesource:GoogleEarth\ngsd:0.146343\n0.0 0.0 4.0 0.0 4.0 3.0 0.0 3.0 plane 0\n",
    )
    .expect("write label");

    let ds = dota_to_coco(dir.path(), None, &HashMap::new()).expect("metadata lines skipped");
    assert_eq!(ds.annotations.len(), 1);
    assert_eq!(ds.categories[0].name, "plane");
    // Dims are metadata for DOTA: unknown stays 0×0, and file_name is the stem.
    assert_eq!(ds.images[0].file_name, "P0001");
    assert_eq!((ds.images[0].width, ds.images[0].height), (0, 0));
}

// ── Open Images ──────────────────────────────────────────────────────────────

/// OID CSV errors name the file as well as the line.
#[test]
fn oid_errors_name_the_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let csv = dir.path().join("boxes.csv");
    std::fs::write(
        &csv,
        "ImageID,LabelName,XMin,XMax,YMin,YMax\nimg1,/m/01,oops,0.5,0.3,0.4\n",
    )
    .expect("write csv");

    let err = hotcoco::convert::oid_to_coco(&csv, None, &HashMap::new()).expect_err("bad XMin");
    let msg = err.to_string();
    assert!(msg.contains("boxes.csv"), "names the file: {msg}");
    assert!(msg.contains("line 2"), "names the line: {msg}");
}

/// Exporting an annotated image with unknown dimensions used to skip its rows
/// with only a counter; normalization needs the true size, so it is an error.
#[test]
fn coco_to_oid_missing_dims_is_an_error() {
    let mut dataset = one_image_dataset("img.jpg", &["thing"], &[(1, [1.0, 1.0, 2.0, 2.0])]);
    dataset.images[0].width = 0;

    let dir = tempfile::tempdir().expect("tempdir");
    let err = coco_to_oid(&dataset, &dir.path().join("out.csv")).expect_err("zero width");
    assert!(
        matches!(err, ConvertError::MissingImageDimensions(_)),
        "got: {err}"
    );
}

// ── Cross-format policies ────────────────────────────────────────────────────

/// A dangling `category_id` is a data-integrity error on export, not a silent
/// skip (or a fabricated "unknown" label).
#[test]
fn exporters_reject_unknown_category_ids() {
    let mut dataset = one_image_dataset("img.jpg", &["thing"], &[(1, [1.0, 1.0, 2.0, 2.0])]);
    dataset.annotations[0].category_id = 99;

    let dir = tempfile::tempdir().expect("tempdir");
    let err = coco_to_yolo(&dataset, dir.path()).expect_err("dangling category");
    assert!(
        matches!(
            err,
            ConvertError::UnknownCategory {
                ann_id: 1,
                category_id: 99
            }
        ),
        "got: {err}"
    );
    let err = coco_to_voc(&dataset, dir.path()).expect_err("dangling category");
    assert!(
        matches!(err, ConvertError::UnknownCategory { .. }),
        "voc: {err}"
    );
    let err = coco_to_oid(&dataset, &dir.path().join("out.csv")).expect_err("dangling category");
    assert!(
        matches!(err, ConvertError::UnknownCategory { .. }),
        "oid: {err}"
    );
}

// ── 1.4.1 hunt: XML text, truncated files, paths in I/O errors ──────────────

/// Names that need escaping in XML. quick-xml hands text over in pieces split
/// at each entity reference, and the readers kept the last piece: `salt &
/// pepper` came back from its own export as a new category `pepper`, with the
/// annotations moved onto it.
fn awkward_names() -> Dataset {
    one_image_dataset(
        "R&D \"x\".jpg",
        &["salt & pepper", "a<b"],
        &[(1, [10.0, 20.0, 30.0, 40.0]), (2, [50.0, 60.0, 30.0, 40.0])],
    )
}

fn names_and_labels(ds: &Dataset) -> (String, Vec<String>, Vec<u64>) {
    (
        ds.images[0].file_name.clone(),
        ds.categories.iter().map(|c| c.name.clone()).collect(),
        ds.annotations.iter().map(|a| a.category_id).collect(),
    )
}

#[test]
fn voc_round_trips_names_that_need_escaping() {
    let dataset = awkward_names();
    let dir = tempfile::tempdir().expect("tempdir");
    coco_to_voc(&dataset, dir.path()).expect("coco_to_voc");
    let recovered = voc_to_coco(dir.path()).expect("voc_to_coco");
    assert_eq!(names_and_labels(&recovered), names_and_labels(&dataset));
}

#[test]
fn cvat_round_trips_names_that_need_escaping() {
    let dataset = awkward_names();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("annotations.xml");
    hotcoco::convert::coco_to_cvat(&dataset, &path).expect("coco_to_cvat");
    let (recovered, _) = cvat_to_coco(&path).expect("cvat_to_coco");
    assert_eq!(names_and_labels(&recovered), names_and_labels(&dataset));
}

/// A character reference and CDATA are text too.
#[test]
fn voc_reads_character_references_and_cdata() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xml = r"<annotation>
  <filename>a.jpg</filename>
  <size><width>640</width><height>480</height></size>
  <object><name>ros&#233;</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
  <object><name><![CDATA[a&b]]></name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
</annotation>";
    std::fs::write(dir.path().join("a.xml"), xml).expect("write xml");
    let recovered = voc_to_coco(dir.path()).expect("voc_to_coco");
    let names: Vec<&str> = recovered
        .categories
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(names, ["a&b", "rosé"]);
}

/// A file cut short used to import as a smaller dataset: one object where the
/// file held two, one image where it held two.
#[test]
fn truncated_xml_is_an_error_not_a_smaller_dataset() {
    let dataset = one_image_dataset(
        "a.jpg",
        &["thing"],
        &[(1, [10.0, 20.0, 30.0, 40.0]), (1, [50.0, 60.0, 30.0, 40.0])],
    );
    let cut_after = |path: &std::path::Path, marker: &str, nth: usize| {
        let text = std::fs::read_to_string(path).expect("read");
        let at = text.match_indices(marker).nth(nth).expect("marker").0 + marker.len();
        std::fs::write(path, &text[..at]).expect("write");
    };

    let dir = tempfile::tempdir().expect("tempdir");
    coco_to_voc(&dataset, dir.path()).expect("coco_to_voc");
    let voc_xml = dir.path().join("Annotations").join("a.xml");
    cut_after(&voc_xml, "</object>", 0);
    let err = voc_to_coco(dir.path()).expect_err("truncated VOC");
    assert!(
        err.to_string().contains("a.xml") && err.to_string().contains("never closed"),
        "{err}"
    );

    let mut two_images = dataset.clone();
    two_images.images.push(Image {
        id: 2,
        file_name: "b.jpg".into(),
        width: 640,
        height: 480,
        ..Default::default()
    });
    let cvat = dir.path().join("annotations.xml");
    hotcoco::convert::coco_to_cvat(&two_images, &cvat).expect("coco_to_cvat");
    cut_after(&cvat, "</image>", 0);
    let err = cvat_to_coco(&cvat).expect_err("truncated CVAT");
    assert!(err.to_string().contains("never closed"), "{err}");
}

/// A file that is not CVAT XML used to import as an empty dataset, so
/// `coco convert --from cvat` on a COCO JSON reported success.
#[test]
fn a_file_that_is_not_cvat_xml_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("x.xml");
    for content in [
        "",
        "hello",
        "{\"images\": []}",
        "<annotation><filename>a</filename></annotation>",
    ] {
        std::fs::write(&path, content).expect("write");
        let err = cvat_to_coco(&path).expect_err(content);
        assert!(
            err.to_string().contains("x.xml") && err.to_string().contains("annotations"),
            "{content:?}: {err}"
        );
    }
}

/// A box narrower than a pixel rounded to `xmax = xmin - 1`, which the
/// importer rejects as inverted: one such box made the export unreadable.
#[test]
fn voc_exports_a_sub_pixel_box_it_can_read_back() {
    let dataset = one_image_dataset(
        "a.jpg",
        &["thing"],
        &[(1, [10.0, 10.0, 0.0, 0.0]), (1, [10.0, 10.0, 0.4, 0.4])],
    );
    let dir = tempfile::tempdir().expect("tempdir");
    coco_to_voc(&dataset, dir.path()).expect("coco_to_voc");
    let recovered = voc_to_coco(dir.path()).expect("voc_to_coco");
    assert_eq!(recovered.annotations.len(), 2);
}

/// A missing input is `NotFound` with the path in the message; the bare
/// "No such file or directory (os error 2)" named nothing.
#[test]
fn a_missing_input_names_its_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let nope = dir.path().join("nope");
    let dims = HashMap::new();
    let errors = [
        voc_to_coco(&nope).expect_err("voc"),
        cvat_to_coco(&nope).expect_err("cvat"),
        dota_to_coco(&nope, None, &dims).expect_err("dota"),
        hotcoco::convert::oid_to_coco(&nope, None, &dims).expect_err("oid"),
    ];
    for err in errors {
        match &err {
            ConvertError::Io(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "{err}");
                assert!(e.to_string().contains("nope"), "{err}");
            }
            other => panic!("expected Io, got {other:?}"),
        }
    }
}

/// Bytes that are not UTF-8 are bad content: a parse error naming the file,
/// where it was a bare I/O error, "stream did not contain valid UTF-8".
#[test]
fn a_label_file_that_is_not_utf8_is_a_parse_error_naming_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("bad.txt"), b"\xff\xfe 1 2 3").expect("write");
    let err = dota_to_coco(dir.path(), None, &HashMap::new()).expect_err("dota");
    assert!(
        matches!(err, ConvertError::ParseError(_)) && err.to_string().contains("bad.txt"),
        "{err:?}"
    );
}

/// A `w × h` grayscale PNG, the way the semantic family's PNG reader is fed.
fn write_png(path: &std::path::Path, w: u32, h: u32) {
    let file = std::fs::File::create(path).expect("create png");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    writer
        .write_image_data(&vec![0; (w * h) as usize])
        .expect("png data");
}

/// `read_image_dims` keys `(width, height)` by stem, reads an extension in
/// any case, skips files that are not images by name, and names the file
/// whose header is no image. `tests/test_image_dims.py` holds every format to
/// Pillow's size.
#[test]
fn read_image_dims_reads_headers_by_stem() {
    use hotcoco::convert::{ConvertError, read_image_dims};
    let dir = tempfile::tempdir().expect("tempdir");
    write_png(&dir.path().join("wide.png"), 37, 21);
    write_png(&dir.path().join("TALL.PNG"), 5, 40);
    std::fs::write(dir.path().join("notes.txt"), "not an image").expect("write");
    let dims = read_image_dims(dir.path()).expect("readable");
    assert_eq!(dims.len(), 2);
    assert_eq!(dims["wide"], (37, 21));
    assert_eq!(dims["TALL"], (5, 40));

    std::fs::write(dir.path().join("broken.jpg"), b"\x00\x01 not a jpeg").expect("write");
    match read_image_dims(dir.path()) {
        Err(ConvertError::ParseError(m)) => assert!(m.contains("broken.jpg"), "{m}"),
        other => panic!("{other:?}"),
    }
    match read_image_dims(&dir.path().join("nowhere")) {
        Err(ConvertError::Io(e)) => {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
            assert!(e.to_string().contains("nowhere"), "{e}");
        }
        other => panic!("{other:?}"),
    }
}
