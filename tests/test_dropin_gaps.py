"""Regression tests for the 1.0 drop-in compatibility gaps (PyO3 layer).

Each test here pins a fix from the 1.0 preflight review: derived datasets
carrying the full API, custom-key round-tripping, pycocotools-format annToRLE,
pycocotools stats semantics, real Python warnings, and error types. None of
them need the gitignored data/ directory.
"""

import warnings

import hotcoco
import numpy as np
import pytest
from hotcoco import COCO, COCOeval, LVISeval, LVISResults, mask


def tiny_dataset():
    return {
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "b.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
    }


def tiny_dt(gt):
    return gt.load_res([{"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9}])


# ---------------------------------------------------------------------------
# Derived datasets carry the full API (subclass-loss fix)
# ---------------------------------------------------------------------------


class TestDerivedDatasetsKeepFullApi:
    def test_split_results_have_browse(self):
        coco = COCO(tiny_dataset())
        train, val = coco.split(val_frac=0.5)
        assert hasattr(type(train), "browse")
        assert callable(type(val).browse)

    def test_all_derivation_paths_have_browse(self):
        coco = COCO(tiny_dataset())
        derived = [coco.filter(cat_ids=[1]), coco.sample(n=1), tiny_dt(coco), COCO.merge([coco])]
        for d in derived:
            assert hasattr(type(d), "browse"), f"{type(d)} lost browse()"

    def test_browse_without_deps_or_dir_raises_cleanly(self):
        """browse() reaches the Python implementation (not AttributeError)."""
        coco = COCO(tiny_dataset())
        with pytest.raises((ValueError, ImportError)):
            coco.browse()


# ---------------------------------------------------------------------------
# Custom keys survive the Rust round-trip
# ---------------------------------------------------------------------------


class TestCustomKeysRoundTrip:
    def test_annotation_image_category_extras_survive(self):
        ds = tiny_dataset()
        ds["annotations"][0]["confidence_source"] = "human"
        ds["annotations"][0]["tags"] = ["hard", {"nested": 1}]
        ds["images"][0]["weather"] = "rainy"
        ds["categories"][0]["taxonomy_id"] = 42

        coco = COCO(ds)
        out = coco.dataset
        ann = next(a for a in out["annotations"] if a["id"] == 1)
        assert ann["confidence_source"] == "human"
        assert ann["tags"] == ["hard", {"nested": 1}]
        assert next(i for i in out["images"] if i["id"] == 1)["weather"] == "rainy"
        assert next(c for c in out["categories"] if c["id"] == 1)["taxonomy_id"] == 42

    def test_extras_survive_filter(self):
        ds = tiny_dataset()
        ds["annotations"][0]["custom"] = {"a": 1}
        coco = COCO(ds).filter(cat_ids=[1])
        anns = coco.dataset["annotations"]
        assert anns and anns[0]["custom"] == {"a": 1}

    def test_extras_visible_in_load_anns_and_anns(self):
        ds = tiny_dataset()
        ds["annotations"][1]["reviewer"] = "alice"
        coco = COCO(ds)
        assert coco.load_anns(2)[0]["reviewer"] == "alice"
        assert coco.anns[2]["reviewer"] == "alice"

    def test_non_serializable_extra_raises(self):
        ds = tiny_dataset()
        ds["annotations"][0]["bad"] = object()
        with pytest.raises(TypeError):
            COCO(ds)

    def test_slice_by_callable_sees_custom_image_keys(self):
        ds = tiny_dataset()
        ds["images"][0]["weather"] = "rainy"
        ds["images"][1]["weather"] = "sunny"
        gt = COCO(ds)
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        ev.evaluate()
        result = ev.slice_by(lambda img: img.get("weather"))
        assert "rainy" in result and "sunny" in result
        assert result["rainy"]["num_images"] == 1


# ---------------------------------------------------------------------------
# Every known dict key survives decode (candidate G: interned get_item keys)
#
# convert.rs's decode macros (opt!/req!/opt_with!/req_with!) and the standalone
# get_item calls in py_to_annotation/py_to_segmentation/py_to_rle now fetch
# each key through pyo3::intern! instead of a bare `&str` literal, to avoid
# allocating a fresh PyString per key per record. A typo in one of those
# literals breaks silently: a required key (id, image_id) raises "dict missing
# '<real name>'" because the interned typo never matches, and an optional key
# (score, is_group_of, ...) just vanishes instead of raising. This round-trips
# every field of every record type through COCO to catch either failure mode.
# ---------------------------------------------------------------------------


class TestKnownKeysRoundTrip:
    def test_every_annotation_field_survives(self):
        ds = tiny_dataset()
        ds["annotations"] = [
            {
                "id": 7,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "area": 12.5,
                "segmentation": {"size": [10, 10], "counts": [100]},
                "iscrowd": 1,
                "keypoints": [1.0, 2.0, 2.0],
                "num_keypoints": 1,
                "obb": [5.0, 5.0, 2.0, 2.0, 0.3],
                "score": 0.75,
                "is_group_of": 1,
            }
        ]
        coco = COCO(ds)

        ann = coco.dataset["annotations"][0]

        assert ann["id"] == 7
        assert ann["image_id"] == 1
        assert ann["category_id"] == 1
        assert ann["bbox"] == [1.0, 2.0, 3.0, 4.0]
        assert ann["area"] == 12.5
        assert ann["segmentation"] == {"size": [10, 10], "counts": [100]}
        assert ann["iscrowd"] == 1
        assert ann["keypoints"] == [1.0, 2.0, 2.0]
        assert ann["num_keypoints"] == 1
        assert ann["obb"] == [5.0, 5.0, 2.0, 2.0, 0.3]
        assert ann["score"] == 0.75
        assert ann["is_group_of"] is True

    def test_every_image_field_survives(self):
        ds = tiny_dataset()
        ds["images"] = [
            {
                "id": 9,
                "file_name": "x.jpg",
                "height": 100,
                "width": 100,
                "license": 3,
                "coco_url": "http://a",
                "flickr_url": "http://b",
                "date_captured": "2020-01-01",
                "neg_category_ids": [5, 6],
                "not_exhaustive_category_ids": [7],
            }
        ]
        ds["annotations"] = [ds["annotations"][0] | {"image_id": 9}]
        coco = COCO(ds)

        img = next(i for i in coco.dataset["images"] if i["id"] == 9)

        assert img["file_name"] == "x.jpg"
        assert img["height"] == 100
        assert img["width"] == 100
        assert img["license"] == 3
        assert img["coco_url"] == "http://a"
        assert img["flickr_url"] == "http://b"
        assert img["date_captured"] == "2020-01-01"
        assert img["neg_category_ids"] == [5, 6]
        assert img["not_exhaustive_category_ids"] == [7]

    def test_every_category_field_survives(self):
        ds = tiny_dataset()
        ds["categories"] = [
            {
                "id": 1,
                "name": "person",
                "supercategory": "animal",
                "skeleton": [[0, 1], [1, 2]],
                "keypoints": ["nose", "eye"],
                "frequency": "f",
            },
            ds["categories"][1],
        ]
        coco = COCO(ds)

        cat = next(c for c in coco.dataset["categories"] if c["id"] == 1)

        assert cat["name"] == "person"
        assert cat["supercategory"] == "animal"
        assert cat["skeleton"] == [[0, 1], [1, 2]]
        assert cat["keypoints"] == ["nose", "eye"]
        assert cat["frequency"] == "f"


# ---------------------------------------------------------------------------
# annToRLE returns pycocotools format
# ---------------------------------------------------------------------------


class TestAnnToRle:
    def test_ann_to_rle_pycocotools_format(self):
        ds = tiny_dataset()
        ds["annotations"][0]["segmentation"] = [[10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]]
        coco = COCO(ds)
        rle = coco.ann_to_rle(coco.anns[1])
        assert set(rle) == {"size", "counts"}
        assert rle["size"] == [100, 100]
        assert isinstance(rle["counts"], bytes)
        # Feeds straight back into the mask module, like pycocotools
        assert mask.decode(rle).sum() == mask.area(rle)
        assert coco.annToRLE(coco.anns[1]) == rle


# ---------------------------------------------------------------------------
# ev.stats: [] before summarize, ndarray after — and params copy docs hold
# ---------------------------------------------------------------------------


class TestStatsSemantics:
    def test_stats_empty_list_before_summarize(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        assert ev.stats == []

    def test_stats_ndarray_after_summarize(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        ev.run()
        stats = ev.stats
        assert isinstance(stats, np.ndarray)
        assert stats.dtype == np.float64
        assert stats.shape == (12,)
        # The pycocotools idiom this shape exists for:
        assert stats[0] >= 0.0

    def test_params_max_dets_is_a_list(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        assert ev.params.maxDets == [1, 10, 100]
        assert isinstance(ev.params.maxDets, list)
        # Whole-attribute assignment is the documented mutation path.
        ev.params.maxDets = [1, 10, 100, 200]
        assert ev.params.maxDets == [1, 10, 100, 200]


# ---------------------------------------------------------------------------
# Guards emit real Python warnings, not fd-2 writes
# ---------------------------------------------------------------------------


class TestGuardsWarn:
    def test_accumulate_before_evaluate_warns(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with pytest.warns(UserWarning, match="accumulate"):
            ev.accumulate()

    def test_summarize_before_accumulate_warns(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with pytest.warns(UserWarning, match="summarize"):
            ev.summarize()

    def test_f_scores_before_accumulate_warns(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with pytest.warns(UserWarning, match="f_scores"):
            assert ev.f_scores() == {}

    def test_no_warning_on_correct_order(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ev.run()


# ---------------------------------------------------------------------------
# Error types: converters route through to_pyerr
# ---------------------------------------------------------------------------


class TestErrorTypes:
    def test_from_voc_missing_dir_is_oserror(self):
        with pytest.raises(OSError):
            COCO.from_voc("/nonexistent/voc-dir")

    def test_from_yolo_missing_data_yaml_is_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="data.yaml"):
            COCO.from_yolo(str(tmp_path))

    def test_from_cvat_malformed_xml_is_value_error(self, tmp_path):
        bad = tmp_path / "annotations.xml"
        bad.write_text("<annotations><image name='a.jpg'")
        with pytest.raises(ValueError):
            COCO.from_cvat(str(bad))

    def test_compare_mismatched_params_is_value_error(self):
        gt = COCO(tiny_dataset())
        ev_a = COCOeval(gt, tiny_dt(gt), "bbox")
        ev_a.evaluate()
        ev_b = COCOeval(gt, tiny_dt(gt), "bbox")
        ev_b.params.iouThrs = [0.5]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # deliberate non-reference config
            ev_b.evaluate()
        with pytest.raises(ValueError):
            hotcoco.compare(ev_a, ev_b)


# ---------------------------------------------------------------------------
# mask module round-trip and dict-in/dict-out
# ---------------------------------------------------------------------------


class TestMaskSurface:
    def test_encode_decode_round_trip(self):
        m = np.zeros((10, 12), dtype=np.uint8, order="F")
        m[2:5, 3:7] = 1
        rle = mask.encode(m)
        assert isinstance(rle["counts"], bytes)
        assert (mask.decode(rle) == m).all()

    def test_encode_3d_round_trip(self):
        m = np.zeros((10, 12, 2), dtype=np.uint8, order="F")
        m[2:5, 3:7, 0] = 1
        m[6:9, 1:4, 1] = 1
        rles = mask.encode(m)
        assert isinstance(rles, list) and len(rles) == 2
        assert (mask.decode(rles) == m).all()

    def test_fr_py_objects_dict_in_dict_out(self):
        uncompressed = {"size": [10, 10], "counts": [50, 10, 40]}
        out = mask.frPyObjects(uncompressed, 10, 10)
        assert isinstance(out, dict), "dict input must give dict output (pycocotools parity)"
        outs = mask.frPyObjects([uncompressed], 10, 10)
        assert isinstance(outs, list) and len(outs) == 1


# ---------------------------------------------------------------------------
# LVIS drop-in objects construct the documented types
# ---------------------------------------------------------------------------


class TestLvisDropIn:
    def test_lviseval_returns_cocoeval(self):
        gt = COCO(tiny_dataset())
        ev = LVISeval(gt, tiny_dt(gt), "bbox")
        assert isinstance(ev, COCOeval)

    def test_lvisresults_returns_coco(self):
        gt = COCO(tiny_dataset())
        res = LVISResults(gt, [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5], "score": 0.5}])
        assert isinstance(res, COCO)


# ---------------------------------------------------------------------------
# load_warnings
# ---------------------------------------------------------------------------


class TestLoadWarnings:
    def test_clean_load_has_no_warnings(self):
        assert COCO(tiny_dataset()).load_warnings == []

    def test_duplicate_ann_ids_are_reported(self):
        ds = tiny_dataset()
        ds["annotations"][1]["id"] = 1  # collide with the first
        coco = COCO(ds)
        assert any("annotation" in w for w in coco.load_warnings)


# ---------------------------------------------------------------------------
# Issue #5: findings from adopting hotcoco through TorchMetrics
# ---------------------------------------------------------------------------


def _one_box_dataset():
    return {
        "images": [{"id": 0, "height": 10, "width": 10}],
        "categories": [{"id": 1, "name": "1"}],
        "annotations": [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0}],
    }


def _square_mask(dtype=np.uint8):
    m = np.zeros((10, 10), dtype=dtype)
    m[2:6, 2:6] = 1
    return np.asfortranarray(m)


class TestIssue5BytesCounts:
    """The reported symptom: identical masks scoring segm AP 0.0 when `counts` is bytes.

    The round-trip itself is pinned by `tests/test_parity.py`; this is the
    evaluation-level check that nothing else covers.
    """

    def test_segm_ap_is_one_for_identical_masks(self):
        rle = mask.encode(_square_mask())
        gt_ds, dt_ds = _one_box_dataset(), _one_box_dataset()
        gt_ds["annotations"][0]["segmentation"] = dict(rle)
        dt_ds["annotations"][0]["segmentation"] = dict(rle)
        dt_ds["annotations"][0]["score"] = 0.9
        ev = COCOeval(COCO(gt_ds), COCO(dt_ds), iou_type="segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        # A lone true positive's AP is what ``metrics.average_precision`` says
        # it is (pycocotools' guard term puts it an ulp under 1.0), to within
        # the summary mean's rounding.
        lone_tp = hotcoco.metrics.average_precision([1.0], [True], 1)
        assert ev.stats[0] == pytest.approx(lone_tp, abs=np.finfo(float).eps)


class TestIssue5MaskEncodeDtype:
    """What `tests/test_parity.py` does not pin: sliced views and `int8`."""

    def test_sliced_view_matches_contiguous(self):
        f_order = _square_mask()
        wide = np.zeros((10, 20), np.uint8)
        wide[:, 5:15] = f_order
        assert mask.encode(wide[:, 5:15]) == mask.encode(f_order)

    def test_int8_is_nonzero_foreground(self):
        m = _square_mask(np.int8)
        m[2, 2] = -1
        assert mask.encode(m) == mask.encode(_square_mask())


class TestIssue5MissingCategoryName:
    def test_missing_name_gets_placeholder_and_warns(self):
        ds = tiny_dataset()
        ds["categories"][0] = {"id": 1}
        coco = COCO(ds)
        assert coco.loadCats(1)[0]["name"] == "cat_1"
        assert any("without a name" in w for w in coco.load_warnings)


class TestIssue5SummarizeOutput:
    """`summarize()` prints through `sys.stdout`, so Python-level redirection works,
    and its parameter warnings are Python warnings — emitted once, not also on fd 2."""

    @staticmethod
    def _evaluated(max_dets=None):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        if max_dets:
            ev.params.maxDets = max_dets
        ev.evaluate()
        ev.accumulate()
        return ev

    def test_table_goes_through_sys_stdout(self, capsys):
        # capsys swaps sys.stdout, exactly as contextlib.redirect_stdout does.
        self._evaluated().summarize()
        out = capsys.readouterr().out.splitlines()
        assert len(out) == 12
        assert out[0].startswith(" Average Precision (AP) @[ IoU=0.50:0.95")

    def test_param_warning_is_a_python_warning_only(self, capfd):
        ev = self._evaluated(max_dets=[1, 10, 500])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ev.summarize()
        assert [w for w in caught if "max_dets" in str(w.message)]
        assert "max_dets" not in capfd.readouterr().err


# ---------------------------------------------------------------------------
# Found by tests/fuzz_dropin.py: spellings pycocotools accepts
# ---------------------------------------------------------------------------


def _kp_dataset(num_keypoints=True, as_array=False):
    kps = [0.0] * 51
    for k in range(5):
        kps[k * 3 : k * 3 + 3] = [10.0 + k, 10.0 + k, 2.0]
    ann = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20], "area": 400, "iscrowd": 0}
    ann["keypoints"] = np.asarray(kps).reshape(-1, 3) if as_array else kps
    if num_keypoints:
        ann["num_keypoints"] = 5
    return {
        "images": [{"id": 1, "width": 64, "height": 64}],
        "categories": [{"id": 1, "name": "person", "keypoints": [f"k{i}" for i in range(17)], "skeleton": []}],
        "annotations": [ann],
    }


def _kp_stats(gt_ds):
    gt = COCO(gt_ds)
    det = dict(gt_ds["annotations"][0])
    det["score"] = 0.9
    det["keypoints"] = np.asarray(det["keypoints"]).ravel().tolist()
    dt = gt.loadRes([det])
    ev = COCOeval(gt, dt, "keypoints")
    ev.evaluate()
    ev.accumulate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.summarize()
    return list(ev.stats)


class TestFuzzDropinFindings:
    """Dict-path spellings only; the `num_keypoints` rule itself is pinned in Rust."""

    def test_keypoints_as_nx3_array(self):
        assert _kp_stats(_kp_dataset(as_array=True)) == _kp_stats(_kp_dataset())

    def test_integral_float_ids_load(self):
        ds = tiny_dataset()
        for img in ds["images"]:
            img["id"] = float(img["id"])
        for ann in ds["annotations"]:
            ann["id"], ann["image_id"], ann["category_id"] = (
                float(ann["id"]),
                float(ann["image_id"]),
                float(ann["category_id"]),
            )
            ann["iscrowd"] = 0.0
        for cat in ds["categories"]:
            cat["id"] = float(cat["id"])
        coco = COCO(ds)
        assert coco.getImgIds() == [1, 2]
        assert coco.loadAnns(1)[0]["category_id"] == 1
        assert coco.loadRes(
            [{"image_id": 1.0, "category_id": 1.0, "bbox": [1, 1, 2, 2], "score": 0.5}]
        ).getAnnIds() == [1]

    def test_fractional_id_is_rejected(self):
        ds = tiny_dataset()
        ds["images"][0]["id"] = 1.5
        with pytest.raises(TypeError, match="non-negative integer, got 1.5"):
            COCO(ds)

    def test_out_of_range_int_names_the_overflow(self):
        ds = tiny_dataset()
        ds["images"][0]["height"] = 2**40
        with pytest.raises(OverflowError, match="out of range"):
            COCO(ds)

    @pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, np.int64(1), np.bool_(True)])
    def test_flags_share_one_reader(self, value):
        ds = tiny_dataset()
        ds["annotations"][0]["iscrowd"] = value
        ds["annotations"][0]["is_group_of"] = value
        ann = COCO(ds).loadAnns(1)[0]
        assert ann["iscrowd"] == int(bool(value))
        assert ann["is_group_of"] is bool(value)
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        ev.params.useCats = value
        assert ev.params.useCats is bool(value)
        ev.params.use_cats = value
        assert ev.params.use_cats is bool(value)
