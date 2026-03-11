"""Benchmark: FiftyOne default COCO eval vs hotcoco backend.

Usage (from crates/hotcoco-pyo3/):
    uv run python data/bench_fiftyone.py
"""

import json
import time

import fiftyone as fo

# ---------------------------------------------------------------------------
# Load COCO data
# ---------------------------------------------------------------------------

GT_FILE = "../../data/annotations/instances_val2017.json"
DT_FILE = "../../data/bbox_val2017_results.json"

print("Loading COCO JSON data...")
t0 = time.perf_counter()
with open(GT_FILE) as f:
    gt_data = json.load(f)
with open(DT_FILE) as f:
    dt_data = json.load(f)
print(f"  Loaded in {time.perf_counter() - t0:.2f}s")
print(f"  Images: {len(gt_data['images'])}, GT anns: {len(gt_data['annotations'])}, "
      f"Detections: {len(dt_data)}, Categories: {len(gt_data['categories'])}")

# ---------------------------------------------------------------------------
# Build FiftyOne dataset
# ---------------------------------------------------------------------------

print("\nBuilding FiftyOne dataset...")
t0 = time.perf_counter()

img_map = {img["id"]: img for img in gt_data["images"]}
cat_map = {cat["id"]: cat["name"] for cat in gt_data["categories"]}

gt_by_img = {}
for ann in gt_data["annotations"]:
    gt_by_img.setdefault(ann["image_id"], []).append(ann)

dt_by_img = {}
for det in dt_data:
    dt_by_img.setdefault(det["image_id"], []).append(det)

dataset = fo.Dataset(name="bench_coco_val2017", overwrite=True)

samples = []
for img_id, img_info in img_map.items():
    W = img_info["width"]
    H = img_info["height"]

    sample = fo.Sample(filepath=f"/tmp/{img_info['file_name']}")
    sample.metadata = fo.ImageMetadata(width=W, height=H)

    gt_dets = []
    for ann in gt_by_img.get(img_id, []):
        x, y, w, h = ann["bbox"]
        gt_dets.append(fo.Detection(
            label=cat_map[ann["category_id"]],
            bounding_box=[x / W, y / H, w / W, h / H],
            iscrowd=bool(ann.get("iscrowd", 0)),
        ))
    sample["ground_truth"] = fo.Detections(detections=gt_dets)

    pred_dets = []
    for det in dt_by_img.get(img_id, []):
        x, y, w, h = det["bbox"]
        pred_dets.append(fo.Detection(
            label=cat_map[det["category_id"]],
            bounding_box=[x / W, y / H, w / W, h / H],
            confidence=det["score"],
        ))
    sample["predictions"] = fo.Detections(detections=pred_dets)

    samples.append(sample)

dataset.add_samples(samples)
print(f"  Built dataset with {len(dataset)} samples in {time.perf_counter() - t0:.2f}s")

# ---------------------------------------------------------------------------
# Benchmark: hotcoco backend (run first — faster, catches errors early)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("hotcoco backend (compute_mAP=True)")
print("=" * 60)

from hotcoco.fiftyone import init_as_fiftyone
init_as_fiftyone()

t0 = time.perf_counter()
results_hotcoco = dataset.evaluate_detections(
    "predictions",
    gt_field="ground_truth",
    eval_key="eval_hotcoco",
    method="hotcoco",
    compute_mAP=True,
)
t_hotcoco = time.perf_counter() - t0

print(f"  Time: {t_hotcoco:.2f}s")
try:
    print(f"  mAP:  {results_hotcoco.mAP():.4f}")
except Exception as e:
    print(f"  mAP:  error - {e}")

dataset.delete_evaluation("eval_hotcoco")

# ---------------------------------------------------------------------------
# Benchmark: FiftyOne default COCO method
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("FiftyOne default COCO method (compute_mAP=True)")
print("=" * 60)

t0 = time.perf_counter()
results_default = dataset.evaluate_detections(
    "predictions",
    gt_field="ground_truth",
    eval_key="eval_default",
    method="coco",
    compute_mAP=True,
)
t_default = time.perf_counter() - t0

print(f"  Time: {t_default:.2f}s")
try:
    print(f"  mAP:  {results_default.mAP():.4f}")
except Exception as e:
    print(f"  mAP:  error - {e}")

dataset.delete_evaluation("eval_default")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  FiftyOne default: {t_default:.2f}s")
print(f"  hotcoco backend:  {t_hotcoco:.2f}s")
print(f"  Speedup:          {t_default / t_hotcoco:.1f}x")

dataset.delete()
