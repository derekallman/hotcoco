//! Regression tests for the 2026-10 review fixes in the converters and the
//! healthcheck: CVAT label names vs. attribute names, VOC objects with missing
//! fields, and out-of-bounds boxes past the top/left edge.

use hotcoco::convert::{ConvertError, cvat_to_coco, voc_to_coco};
use hotcoco::quality;
use hotcoco::types::{Annotation, Category, Dataset, Image};

// ── CVAT ─────────────────────────────────────────────────────────────────────

/// A label's `<attributes><attribute><name>` must not overwrite the label's own
/// `<name>`: the categories follow the declared labels, by name and order.
#[test]
fn cvat_label_attribute_name_does_not_replace_label_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xml_path = dir.path().join("annotations.xml");
    std::fs::write(
        &xml_path,
        r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label>
          <name>car</name>
          <attributes>
            <attribute>
              <name>model</name>
              <input_type>text</input_type>
            </attribute>
          </attributes>
        </label>
        <label>
          <name>person</name>
        </label>
      </labels>
    </task>
  </meta>
  <image id="0" name="a.jpg" width="640" height="480">
    <box label="car" xtl="10" ytl="10" xbr="50" ybr="50"/>
  </image>
</annotations>
"#,
    )
    .expect("write xml");

    let (dataset, _) = cvat_to_coco(&xml_path).expect("cvat_to_coco");
    let cats: Vec<(u64, &str)> = dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();
    assert_eq!(cats, vec![(1, "car"), (2, "person")]);
    assert_eq!(dataset.annotations.len(), 1);
    assert_eq!(dataset.annotations[0].category_id, 1);
}

// ── VOC ──────────────────────────────────────────────────────────────────────

/// Write one VOC annotation file whose single `<object>` body is `object_body`.
fn voc_with_object(object_body: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("a.xml"),
        format!(
            "<annotation><filename>a.jpg</filename>\
             <size><width>640</width><height>480</height></size>\
             <object>{object_body}</object></annotation>"
        ),
    )
    .expect("write xml");
    dir
}

fn assert_voc_parse_error(object_body: &str, needle: &str) {
    let dir = voc_with_object(object_body);
    match voc_to_coco(dir.path()) {
        Err(ConvertError::ParseError(msg)) => {
            assert!(msg.contains(needle), "expected `{needle}` in: {msg}");
            assert!(msg.contains("a.xml"), "error must name the file: {msg}");
        }
        Ok(ds) => panic!(
            "expected a parse error, got bboxes {:?}",
            ds.annotations.iter().map(|a| a.bbox).collect::<Vec<_>>()
        ),
        Err(other) => panic!("expected ParseError, got {other:?}"),
    }
}

#[test]
fn voc_object_without_bndbox_is_an_error() {
    assert_voc_parse_error("<name>car</name>", "<bndbox>");
}

#[test]
fn voc_bndbox_missing_ymax_is_an_error() {
    assert_voc_parse_error(
        "<name>car</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax></bndbox>",
        "ymax",
    );
}

#[test]
fn voc_object_with_empty_name_is_an_error() {
    assert_voc_parse_error(
        "<name></name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>",
        "<name>",
    );
    assert_voc_parse_error(
        "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>",
        "<name>",
    );
}

#[test]
fn voc_inverted_bndbox_is_an_error() {
    assert_voc_parse_error(
        "<name>car</name><bndbox><xmin>10</xmin><ymin>50</ymin><xmax>50</xmax><ymax>10</ymax></bndbox>",
        "ymax",
    );
}

/// The validation must not reject a well-formed object.
#[test]
fn voc_complete_object_still_imports() {
    let dir = voc_with_object(
        "<name>car</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>40</ymax></bndbox>",
    );
    let ds = voc_to_coco(dir.path()).expect("voc_to_coco");
    assert_eq!(ds.annotations.len(), 1);
    assert_eq!(ds.annotations[0].bbox, Some([9.0, 9.0, 41.0, 31.0]));
}

// ── Healthcheck ──────────────────────────────────────────────────────────────

/// A box past the top/left edge is out of bounds just like one past the
/// bottom/right edge.
#[test]
fn healthcheck_flags_bbox_past_top_left_edge() {
    let bboxes = [
        (1, [-50.0, -20.0, 30.0, 30.0]), // past top-left
        (2, [-5.0, 10.0, 30.0, 30.0]),   // past left only
        (3, [10.0, -5.0, 30.0, 30.0]),   // past top only
        (4, [10.0, 10.0, 30.0, 30.0]),   // inside
        (5, [0.0, 0.0, 640.0, 480.0]),   // exactly the image
        (6, [620.0, 10.0, 30.0, 30.0]),  // past right
    ];
    let dataset = Dataset {
        images: vec![Image {
            id: 1,
            file_name: "a.jpg".into(),
            width: 640,
            height: 480,
            ..Default::default()
        }],
        annotations: bboxes
            .iter()
            .map(|&(id, bbox)| Annotation {
                id,
                image_id: 1,
                category_id: 1,
                bbox: Some(bbox),
                area: Some(bbox[2] * bbox[3]),
                ..Default::default()
            })
            .collect(),
        categories: vec![Category {
            id: 1,
            name: "car".into(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let report = quality::healthcheck(&dataset);
    let oob = report
        .warnings
        .iter()
        .find(|f| f.code == "bbox_out_of_bounds")
        .expect("bbox_out_of_bounds warning");
    let mut ids = oob.affected_ids.clone();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3, 6]);
}
