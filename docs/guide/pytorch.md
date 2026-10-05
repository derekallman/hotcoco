# PyTorch integration

hotcoco ships two classes in `hotcoco.integrations` that replace their torchvision equivalents without requiring pycocotools or torchvision to be installed:

- **`CocoDetection`** — a `DataLoader`-compatible dataset that loads images and COCO annotations
- **`CocoEvaluator`** — accumulates batch predictions during a training loop and runs COCO evaluation at the end of each epoch

Signatures and parameter tables are in the [PyTorch integrations reference](../api/integrations.md).

## Installation

```bash
pip install hotcoco
```

PyTorch and Pillow are optional — the [reference](../api/integrations.md) says which class needs which. To load images and train:

```bash
pip install hotcoco torch pillow
```

## CocoDetection

`CocoDetection` is a drop-in replacement for `torchvision.datasets.CocoDetection`. It wraps a COCO annotation file and an image directory, and returns `(image, annotations)` pairs for each sample.

```python
from hotcoco.integrations import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
])

dataset = CocoDetection(
    root="path/to/val2017",
    ann_file="annotations/instances_val2017.json",
    transform=transform,
)

loader = DataLoader(dataset, batch_size=4, collate_fn=lambda x: tuple(zip(*x)))
```

### Return value

Each `__getitem__` call returns `(image, target)` where:

- `image` — a PIL `Image` (or transformed tensor if a `transform` is provided)
- `target` — a list of annotation dicts, one per instance in the image:

```python
[
    {
        "id": 12345,
        "image_id": 42,
        "category_id": 1,
        "bbox": [x, y, width, height],
        "area": 3200.0,
        "iscrowd": 0,
        "segmentation": [...],
    },
    ...
]
```

### Transforms

`CocoDetection` accepts three transform arguments, applied in order. Use `transform` for pixel-level work on the image alone (normalize, resize); `target_transform` for annotation-level filtering or format changes; and `transforms` for geometric augmentations that must move the annotations with the pixels, since it receives `(image, target)` together. Their signatures are under [CocoDetection](../api/integrations.md#cocodetection).

```python
def filter_crowds(anns):
    return [a for a in anns if not a["iscrowd"]]

dataset = CocoDetection(
    root="path/to/val2017",
    ann_file="annotations/instances_val2017.json",
    transform=T.ToTensor(),
    target_transform=filter_crowds,
)
```

### Accessing the COCO API

The underlying `COCO` object is available as `dataset.coco`:

```python
cat_names = {c["id"]: c["name"] for c in dataset.coco.load_cats(dataset.coco.get_cat_ids())}
print(f"Dataset has {len(dataset)} images and {len(cat_names)} categories")
```

## CocoEvaluator

`CocoEvaluator` wraps hotcoco's `COCOeval` with a tensor-friendly `update()` interface. Call `update()` after each batch, then `accumulate()` and `summarize()` at the end of the epoch.

### Basic usage

```python
from hotcoco import COCO
from hotcoco.integrations import CocoEvaluator

coco_gt = COCO("annotations/instances_val2017.json")
evaluator = CocoEvaluator(coco_gt, iou_types=["bbox"])

model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        # Build {image_id: output} dict for each item in the batch
        predictions = {
            int(t["image_id"]): o
            for t, o in zip(targets, outputs)
        }
        evaluator.update(predictions)

evaluator.accumulate()
evaluator.summarize()
```

### Prediction format

The `predictions` dict maps `image_id → output_dict`. The keys and tensor shapes expected in the output dict for each `iou_type` are in the [`update` reference](../api/integrations.md#updatepredictions).

!!! tip "torchvision detection models"
    Standard torchvision models (Faster R-CNN, RetinaNet, FCOS, and others) already return dicts in the expected format. The `update()` call in the preceding example works without modification.

### Multiple iou_types

Evaluate bbox and segm in a single pass:

```python
evaluator = CocoEvaluator(coco_gt, iou_types=["bbox", "segm"])

for images, targets in val_loader:
    outputs = model(images)
    predictions = {int(t["image_id"]): o for t, o in zip(targets, outputs)}
    evaluator.update(predictions)

evaluator.accumulate()
evaluator.summarize()
# Prints results for bbox, then segm
```

### Getting results programmatically

`get_results()` returns one metrics dict per `iou_type` — the shape is in the [`get_results` reference](../api/integrations.md#get_results).

```python
evaluator.accumulate()

results = evaluator.get_results()
ap = results["bbox"]["AP"]
print(f"Validation AP: {ap:.4f}")
```

Log to an experiment tracker:

```python
import wandb

for iou_type, metrics in results.items():
    wandb.log({f"val/{iou_type}/{k}": v for k, v in metrics.items()}, step=epoch)
```

### Distributed training

`CocoEvaluator.synchronize_between_processes()` gathers predictions across all ranks before evaluation, under either the gloo or the NCCL backend. Call it after the last `update()` and before `accumulate()`:

```python
evaluator.update(predictions)              # last batch of the epoch

evaluator.synchronize_between_processes()

evaluator.accumulate()
evaluator.summarize()
```

The same loop runs unchanged on a single GPU — see [`synchronize_between_processes`](../api/integrations.md#synchronize_between_processes) for what it does outside a distributed run.

## Migrating from torchvision

If you're using `torchvision.datasets.CocoDetection` or the `CocoEvaluator` from torchvision's detection reference scripts, replace the imports:

```python
# Before
from torchvision.datasets import CocoDetection
from references.detection.coco_eval import CocoEvaluator  # torchvision reference script

# After
from hotcoco.integrations import CocoDetection, CocoEvaluator
```

No other changes are needed — the API is identical.
