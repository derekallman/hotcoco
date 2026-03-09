"""Tests for .imgs/.anns/.cats getters, CocoEvaluator, CocoDetection, and COCO(dict)."""

import json
import os
import tempfile

import numpy as np
import pytest

from hotcoco import COCO, CocoDetection, CocoEvaluator

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
GT_FILE = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")
DT_FILE = os.path.join(DATA_DIR, "bbox_val2017_results.json")


@pytest.fixture(scope="module")
def coco_gt():
    return COCO(GT_FILE)


# ---------------------------------------------------------------------------
# .imgs / .anns / .cats getters
# ---------------------------------------------------------------------------


class TestImgsAnnsCats:
    def test_imgs_keys_are_ids(self, coco_gt):
        imgs = coco_gt.imgs
        assert isinstance(imgs, dict)
        assert len(imgs) == 5000
        for img_id, img_dict in list(imgs.items())[:5]:
            assert img_dict["id"] == img_id
            assert "file_name" in img_dict
            assert "height" in img_dict
            assert "width" in img_dict

    def test_anns_keys_are_ids(self, coco_gt):
        anns = coco_gt.anns
        assert isinstance(anns, dict)
        assert len(anns) == 36781
        for ann_id, ann_dict in list(anns.items())[:5]:
            assert ann_dict["id"] == ann_id
            assert "image_id" in ann_dict
            assert "category_id" in ann_dict

    def test_cats_keys_are_ids(self, coco_gt):
        cats = coco_gt.cats
        assert isinstance(cats, dict)
        assert len(cats) == 80
        for cat_id, cat_dict in list(cats.items())[:5]:
            assert cat_dict["id"] == cat_id
            assert "name" in cat_dict

    def test_imgs_matches_get_img_ids(self, coco_gt):
        assert set(coco_gt.imgs.keys()) == set(coco_gt.get_img_ids())

    def test_anns_matches_get_ann_ids(self, coco_gt):
        assert set(coco_gt.anns.keys()) == set(coco_gt.get_ann_ids())

    def test_cats_matches_get_cat_ids(self, coco_gt):
        assert set(coco_gt.cats.keys()) == set(coco_gt.get_cat_ids())

    def test_empty_coco(self):
        empty = COCO()
        assert empty.imgs == {}
        assert empty.anns == {}
        assert empty.cats == {}


# ---------------------------------------------------------------------------
# CocoEvaluator
# ---------------------------------------------------------------------------


class FakeTensor:
    """Duck-typed tensor for testing without torch."""

    def __init__(self, data):
        self._data = np.array(data)

    def tolist(self):
        return self._data.tolist()

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return FakeTensor(self._data[idx])


class TestCocoEvaluator:
    def test_string_iou_type_wrapped(self, coco_gt):
        ev = CocoEvaluator(coco_gt, "bbox")
        assert ev.iou_types == ["bbox"]

    def test_empty_results_warns(self, coco_gt, capsys):
        ev = CocoEvaluator(coco_gt, ["bbox"])
        ev.accumulate()
        captured = capsys.readouterr()
        assert "no results" in captured.out.lower()

    def test_bbox_eval_from_dicts(self, coco_gt):
        """Full pipeline using pre-computed detection dicts (no torch needed)."""
        ev = CocoEvaluator(coco_gt, ["bbox"])

        # Load detections as COCO-format dicts and feed directly into results
        import json

        with open(DT_FILE) as f:
            det_dicts = json.load(f)

        # Bypass update() (which expects tensors) — push dicts directly
        ev.results["bbox"] = det_dicts

        ev.accumulate()
        ev.summarize()

        results = ev.get_results()
        assert "bbox" in results
        bbox = results["bbox"]

        # Known val2017 bbox AP values for this detection file
        assert abs(bbox["AP"] - 0.578) < 0.001
        assert abs(bbox["AP50"] - 0.861) < 0.001
        assert abs(bbox["AP75"] - 0.600) < 0.001

    def test_get_results_empty_before_accumulate(self, coco_gt):
        ev = CocoEvaluator(coco_gt, ["bbox"])
        assert ev.get_results() == {}

    def test_synchronize_noop_without_dist(self, coco_gt):
        """synchronize_between_processes is a no-op without torch.distributed."""
        ev = CocoEvaluator(coco_gt, ["bbox"])
        ev.results["bbox"] = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]
        ev.synchronize_between_processes()
        assert len(ev.results["bbox"]) == 1

    def test_update_bbox(self, coco_gt):
        """update() converts XYXY tensors to XYWH COCO dicts."""
        ev = CocoEvaluator(coco_gt, ["bbox"])
        img_id = sorted(coco_gt.get_img_ids())[0]
        predictions = {
            img_id: {
                "boxes": FakeTensor([[10.0, 20.0, 50.0, 80.0]]),  # XYXY
                "scores": FakeTensor([0.95]),
                "labels": FakeTensor([1]),
            }
        }
        ev.update(predictions)
        assert len(ev.results["bbox"]) == 1
        r = ev.results["bbox"][0]
        assert r["image_id"] == img_id
        assert r["category_id"] == 1
        assert r["score"] == pytest.approx(0.95)
        # XYXY (10,20,50,80) → XYWH (10,20,40,60)
        assert r["bbox"] == pytest.approx([10.0, 20.0, 40.0, 60.0])

    def test_update_keypoints(self, coco_gt):
        """update() flattens (N, K, 3) keypoints."""
        ev = CocoEvaluator(coco_gt, ["keypoints"])
        img_id = sorted(coco_gt.get_img_ids())[0]
        # 1 detection, 2 keypoints, each (x, y, v)
        kpts = [[[100.0, 200.0, 2.0], [150.0, 250.0, 1.0]]]
        predictions = {
            img_id: {
                "keypoints": FakeTensor(kpts),
                "scores": FakeTensor([0.8]),
                "labels": FakeTensor([1]),
            }
        }
        ev.update(predictions)
        assert len(ev.results["keypoints"]) == 1
        r = ev.results["keypoints"][0]
        assert r["keypoints"] == pytest.approx([100.0, 200.0, 2.0, 150.0, 250.0, 1.0])

    def test_update_segm(self, coco_gt):
        """update() encodes binary masks to RLE."""
        ev = CocoEvaluator(coco_gt, ["segm"])
        img_id = sorted(coco_gt.get_img_ids())[0]
        # 1 detection, mask shape (1, 1, 10, 10), all ones
        mask = np.ones((1, 1, 10, 10), dtype=np.float32)
        predictions = {
            img_id: {
                "masks": FakeTensor(mask),
                "scores": FakeTensor([0.9]),
                "labels": FakeTensor([1]),
            }
        }
        ev.update(predictions)
        assert len(ev.results["segm"]) == 1
        r = ev.results["segm"][0]
        assert "segmentation" in r
        seg = r["segmentation"]
        assert "size" in seg
        assert "counts" in seg
        assert seg["size"] == [10, 10]

    def test_update_multiple_batches(self, coco_gt):
        """Multiple update() calls accumulate results."""
        ev = CocoEvaluator(coco_gt, ["bbox"])
        ids = sorted(coco_gt.get_img_ids())
        for i in range(3):
            predictions = {
                ids[i]: {
                    "boxes": FakeTensor([[0.0, 0.0, 10.0, 10.0]]),
                    "scores": FakeTensor([0.5]),
                    "labels": FakeTensor([1]),
                }
            }
            ev.update(predictions)
        assert len(ev.results["bbox"]) == 3

    def test_update_empty_scores_skipped(self, coco_gt):
        """Images with no detections produce no results."""
        ev = CocoEvaluator(coco_gt, ["bbox"])
        img_id = sorted(coco_gt.get_img_ids())[0]
        predictions = {
            img_id: {
                "boxes": FakeTensor(np.empty((0, 4))),
                "scores": FakeTensor([]),
                "labels": FakeTensor([]),
            }
        }
        ev.update(predictions)
        assert len(ev.results["bbox"]) == 0


# ---------------------------------------------------------------------------
# CocoDetection
# ---------------------------------------------------------------------------


class TestCocoDetection:
    def test_len(self, coco_gt):
        ds = CocoDetection(root="/tmp", ann_file=GT_FILE)
        assert len(ds) == 5000

    def test_ids_sorted(self):
        ds = CocoDetection(root="/tmp", ann_file=GT_FILE)
        assert ds.ids == sorted(ds.ids)

    def test_repr(self):
        ds = CocoDetection(root="/tmp", ann_file=GT_FILE)
        r = repr(ds)
        assert "CocoDetection" in r
        assert "5000" in r

    def test_getitem_no_pil_raises(self):
        """__getitem__ requires PIL — import error is clear if missing."""
        # We can't truly test missing PIL since it may be installed,
        # but we can verify __getitem__ runs the code path up to image load.
        ds = CocoDetection(root="/nonexistent", ann_file=GT_FILE)
        # Should fail on file-not-found (PIL installed) or ImportError (PIL missing)
        with pytest.raises((FileNotFoundError, ImportError, OSError)):
            ds[0]

    def test_coco_attr(self):
        ds = CocoDetection(root="/tmp", ann_file=GT_FILE)
        assert hasattr(ds, "coco")
        assert len(ds.coco.get_cat_ids()) == 80

    def test_getitem_loads_image_and_annotations(self):
        """__getitem__ returns (PIL.Image, list[dict]) with a real temp image."""
        from PIL import Image

        # First image in val2017: id=139, file_name="000000000139.jpg"
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy JPEG matching the expected filename
            img = Image.new("RGB", (640, 426), color=(128, 64, 32))
            img.save(os.path.join(tmpdir, "000000000139.jpg"))

            ds = CocoDetection(root=tmpdir, ann_file=GT_FILE)
            image, target = ds[0]

            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"
            assert image.size == (640, 426)
            assert isinstance(target, list)
            assert len(target) > 0
            assert "category_id" in target[0]

    def test_transforms_applied(self):
        """transform, target_transform, and transforms are all called."""
        from PIL import Image

        transform_called = []
        target_transform_called = []
        joint_called = []

        def my_transform(img):
            transform_called.append(True)
            return img

        def my_target_transform(tgt):
            target_transform_called.append(True)
            return tgt

        def my_transforms(img, tgt):
            joint_called.append(True)
            return img, tgt

        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new("RGB", (640, 426))
            img.save(os.path.join(tmpdir, "000000000139.jpg"))

            ds = CocoDetection(
                root=tmpdir,
                ann_file=GT_FILE,
                transform=my_transform,
                target_transform=my_target_transform,
                transforms=my_transforms,
            )
            ds[0]

            assert len(transform_called) == 1
            assert len(target_transform_called) == 1
            assert len(joint_called) == 1


# ---------------------------------------------------------------------------
# COCO(dict) constructor
# ---------------------------------------------------------------------------


class TestCOCODict:
    def test_coco_from_dict_empty(self):
        """COCO(dict) with empty lists works."""
        c = COCO({"images": [], "annotations": [], "categories": []})
        assert c.imgs == {}
        assert c.anns == {}
        assert c.cats == {}

    def test_coco_from_dict_basic(self):
        """COCO(dict) with images, annotations, categories."""
        dataset = {
            "images": [
                {"id": 1, "width": 640, "height": 480, "file_name": "a.jpg"},
                {"id": 2, "width": 800, "height": 600, "file_name": "b.jpg"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 20, 30, 40],
                    "area": 1200,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 2,
                    "bbox": [50, 60, 70, 80],
                    "area": 5600,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
            ],
        }
        c = COCO(dataset)
        assert len(c.imgs) == 2
        assert len(c.anns) == 2
        assert len(c.cats) == 2
        assert set(c.get_cat_ids()) == {1, 2}
        assert c.cats[1]["name"] == "cat"
        assert c.anns[1]["bbox"] == pytest.approx([10, 20, 30, 40])

    def test_coco_from_dict_load_res(self):
        """COCO(dict) -> load_res works for detection results."""
        dataset = {
            "images": [{"id": 1, "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 20, 30, 40],
                    "area": 1200,
                    "iscrowd": 0,
                },
            ],
            "categories": [{"id": 1, "name": "cat"}],
        }
        gt = COCO(dataset)
        dt = gt.load_res([
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [12, 22, 28, 38],
                "score": 0.9,
            },
        ])
        assert len(dt.anns) == 1

    def test_coco_from_dict_type_error(self):
        """COCO() with invalid type raises TypeError."""
        with pytest.raises(TypeError):
            COCO(42)
