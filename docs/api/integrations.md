# PyTorch integrations

Drop-in replacements for torchvision's detection reference classes, backed by hotcoco instead of pycocotools. No torchvision or pycocotools dependency required.

```python
from hotcoco.integrations import CocoDetection, CocoEvaluator
```

PyTorch and Pillow are optional — only imported when used (`CocoDetection.__getitem__` needs Pillow; `synchronize_between_processes` needs `torch.distributed`).

Worked examples — a `DataLoader` setup, an epoch loop, and the migration from torchvision — are in the [PyTorch integration guide](../guide/pytorch.md).

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

Worked example: [CocoDetection in the guide](../guide/pytorch.md#cocodetection).

---

## CocoEvaluator

Distributed COCO evaluator for PyTorch training loops. Wraps `COCOeval` with a tensor-friendly `update()` interface and optional distributed synchronization.

```python
CocoEvaluator(
    coco_gt: COCO,
    iou_types: str | list[str],
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_gt` | `COCO` | Ground-truth COCO object |
| `iou_types` | <code>str &#124; list[str]</code> | IoU types to evaluate, for example `"bbox"` or `["bbox", "segm"]` |

Worked examples: [an epoch loop](../guide/pytorch.md#basic-usage) and [distributed training](../guide/pytorch.md#distributed-training) in the guide.

### Methods

#### `update(predictions)`

Accumulate predictions from one batch.

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `dict[int, dict]` | Mapping from image ID to prediction dict |

Prediction dict keys by `iou_type`:

| `iou_type` | Required keys | Notes |
|------------|---------------|-------|
| `"bbox"` | `boxes`, `scores`, `labels` | `boxes` shape `(N, 4)` in **XYXY** format; converted to XYWH internally |
| `"segm"` | `masks`, `scores`, `labels` | `masks` shape `(N, 1, H, W)`, float in `[0, 1]`; thresholded at 0.5 and RLE-encoded |
| `"keypoints"` | `keypoints`, `scores`, `labels` | `keypoints` shape `(N, K, 3)` — x, y, visibility |

`scores` has shape `(N,)`; `labels` has shape `(N,)` and holds COCO category IDs.

#### `synchronize_between_processes()`

Gathers results across all distributed ranks via `torch.distributed.all_gather_object`, which works under both the gloo and NCCL backends. No-op when `torch.distributed` is not installed or initialized, or when the world size is 1.

#### `accumulate()`

Creates `COCOeval` objects for each `iou_type` and runs `evaluate()` + `accumulate()`. All GT image IDs are included so images with zero detections count against recall.

#### `summarize()`

Prints the standard COCO metrics table for each `iou_type`.

#### `get_results()`

Returns metrics as a nested dict, one entry per `iou_type`.

```python
results = evaluator.get_results()
# {"bbox": {"AP": 0.412, "AP50": 0.623, ...}}
```

---

## Replacing torchvision references

Both classes swap in for their torchvision equivalents with an import change and no pycocotools install — see [Migrating from torchvision](../guide/pytorch.md#migrating-from-torchvision).
