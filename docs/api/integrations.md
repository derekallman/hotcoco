# PyTorch Integrations

Drop-in replacements for torchvision's detection reference classes, backed by hotcoco instead of pycocotools. No torchvision or pycocotools dependency required.

```python
from hotcoco.integrations import CocoDetection, CocoEvaluator
```

PyTorch and Pillow are optional — only imported when actually used (`CocoDetection.__getitem__` needs Pillow; `synchronize_between_processes` needs `torch.distributed`).

---

## CocoDetection

A COCO-format image dataset compatible with `torch.utils.data.DataLoader`.

```python
CocoDetection(
    root: str,
    ann_file: str,
    transform=None,
    target_transform=None,
    transforms=None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `root` | `str` | Root directory containing images |
| `ann_file` | `str` | Path to COCO-format annotation JSON |
| `transform` | `callable` | Transform applied to the PIL image |
| `target_transform` | `callable` | Transform applied to the annotation list |
| `transforms` | `callable` | Joint transform applied to `(image, target)` after individual transforms |

**Returns** `(image, annotations)` tuples where `annotations` is a list of COCO annotation dicts.

**Example**

```python
from hotcoco.integrations import CocoDetection
from torchvision import transforms

dataset = CocoDetection(
    root="coco/val2017",
    ann_file="coco/annotations/instances_val2017.json",
    transform=transforms.ToTensor(),
)

image, targets = dataset[0]
# image: Tensor(3, H, W)
# targets: list of annotation dicts with bbox, category_id, etc.

loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=lambda b: tuple(zip(*b)))
```

---

## CocoEvaluator

Distributed COCO evaluator for PyTorch training loops. Wraps `COCOeval` with a tensor-friendly `update()` interface and optional distributed synchronization.

```python
CocoEvaluator(
    coco_gt: COCO,
    iou_types: list[str],
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_gt` | `COCO` | Ground-truth COCO object |
| `iou_types` | `list[str]` | IoU types to evaluate, e.g. `["bbox"]` or `["bbox", "segm"]` |

**Example**

```python
from hotcoco import COCO
from hotcoco.integrations import CocoEvaluator

coco_gt = COCO("instances_val2017.json")
evaluator = CocoEvaluator(coco_gt, ["bbox"])

for images, targets in data_loader:
    outputs = model(images)
    predictions = {t["image_id"]: o for t, o in zip(targets, outputs)}
    evaluator.update(predictions)

evaluator.synchronize_between_processes()  # no-op if not distributed
evaluator.accumulate()
evaluator.summarize()
```

### Methods

#### `update(predictions)`

Accumulate predictions from one batch.

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `dict[int, dict]` | Mapping from image ID to prediction dict |

Prediction dict keys by `iou_type`:

| `iou_type` | Required keys | Notes |
|------------|---------------|-------|
| `"bbox"` | `boxes`, `scores`, `labels` | Boxes in **XYXY** format; converted to XYWH internally |
| `"segm"` | `masks`, `scores`, `labels` | `masks` shape `(N, 1, H, W)`; thresholded at 0.5 and RLE-encoded |
| `"keypoints"` | `keypoints`, `scores`, `labels` | `keypoints` shape `(N, K, 3)` |

#### `synchronize_between_processes()`

Gathers results across all distributed ranks via `torch.distributed.all_gather`. No-op when `torch.distributed` is not initialized or not installed.

#### `accumulate()`

Creates `COCOeval` objects for each `iou_type` and runs `evaluate()` + `accumulate()`. All GT image IDs are included so images with zero detections count against recall.

#### `summarize()`

Prints the standard COCO metrics table for each `iou_type`.

#### `get_results()`

Returns metrics as a nested dict.

```python
results = evaluator.get_results()
# {"bbox": {"AP": 0.412, "AP50": 0.623, ...}}
```

---

## Replacing torchvision references

These classes are designed to be swapped in without any other code changes:

```python
# Before
from torchvision.datasets import CocoDetection
from torchvision.models.detection.coco_utils import CocoEvaluator

# After
from hotcoco.integrations import CocoDetection, CocoEvaluator
```

No pycocotools installation required.
