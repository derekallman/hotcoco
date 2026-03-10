# Framework Integrations

hotcoco works as a drop-in replacement for pycocotools in any framework that uses it internally. The standard approach is one line at the top of your script — no other changes needed.

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()
```

This patches `sys.modules` so that `from pycocotools.coco import COCO` and similar imports resolve to hotcoco instead. Everything downstream works unchanged.

| Framework | Uses | `init_as_pycocotools()` |
|-----------|------|------------------------|
| Detectron2 | pycocotools | Yes |
| MMDetection | pycocotools (default) | Yes (default path) |
| RF-DETR | pycocotools | Yes |
| Ultralytics YOLO | Internal (custom) | No — see below |

---

## Detectron2

Detectron2's `COCOEvaluator` imports from `pycocotools` directly. Add `init_as_pycocotools()` before any detectron2 imports:

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
# ... your config setup ...

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

For standalone evaluation:

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("coco_2017_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "coco_2017_val")
print(inference_on_dataset(model, val_loader, evaluator))
```

---

## MMDetection / MMDetection3

MMDetection's `CocoMetric` goes through `mmdet.datasets.api_wrappers`, which wraps pycocotools. Add `init_as_pycocotools()` to your entrypoint before `mmdet` is imported:

```python
# tools/train.py (or your custom entrypoint)
from hotcoco import init_as_pycocotools
init_as_pycocotools()

from mmengine.runner import Runner
# ... rest of training script unchanged
```

!!! note
    MMDetection has an optional `use_faster_coco_eval=True` flag on `CocoMetric`. If your config sets this, the patch won't take effect for that metric — remove the flag or leave it unset to use the default pycocotools path.

---

## RF-DETR

RF-DETR imports `pycocotools.cocoeval.COCOeval` directly. Add `init_as_pycocotools()` before instantiating the model:

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()

from rfdetr import RFDETRBase

model = RFDETRBase()
model.train(dataset_dir="coco/", epochs=12)
```

---

## Ultralytics YOLO

Ultralytics implements its own evaluation metrics internally and does not use pycocotools or faster-coco-eval, so `init_as_pycocotools()` has no effect. To get authoritative COCO numbers from a Ultralytics run — for papers, leaderboards, or comparison against other frameworks — export the predictions and evaluate with hotcoco separately:

```python
from hotcoco import COCO, COCOeval

# Run Ultralytics validation to generate a predictions file
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.val(data="coco.yaml", save_json=True)
# → runs/detect/val/predictions.json

# Evaluate with hotcoco
coco_gt = COCO("annotations/instances_val2017.json")
coco_dt = coco_gt.load_res("runs/detect/val/predictions.json")

ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()
ev.get_results()
```

This gives you access to hotcoco's full feature set (TIDE analysis, per-class AP, confusion matrix, results export) on top of Ultralytics predictions.

---

## Any other pycocotools-based pipeline

The same one-line pattern works for any script that imports from `pycocotools`:

```python
from hotcoco import init_as_pycocotools
init_as_pycocotools()

# All subsequent imports route to hotcoco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
```

Add this as early as possible in your entrypoint — before any framework imports — so the patch is in place when those modules are first loaded.

---

## LVIS-based pipelines

For frameworks that use `lvis-api` (Detectron2 LVIS mode, MMDetection LVIS evaluation), use `init_as_lvis()` instead:

```python
from hotcoco import init_as_lvis
init_as_lvis()

# Resolves to hotcoco
from lvis import LVIS, LVISEval, LVISResults
```

See [LVIS Evaluation](evaluation.md#lvis-evaluation) for the full LVIS workflow.
