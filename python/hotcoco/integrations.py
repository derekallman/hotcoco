"""PyTorch ecosystem integrations for hotcoco.

Provides ``CocoEvaluator`` and ``CocoDetection`` as drop-in replacements for
the equivalent classes in torchvision's detection reference scripts. No
torchvision or pycocotools dependency is required.

PyTorch and Pillow are optional — only needed when calling methods that use
them: ``CocoDetection.__getitem__`` needs PIL, and ``synchronize_between_processes``
needs torch.distributed).
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as _np

from . import mask as mask_utils
from .hotcoco import COCO, COCOeval, Params


class CocoDetection:
    """A COCO-format detection dataset compatible with ``torch.utils.data.DataLoader``.

    Drop-in replacement for ``torchvision.datasets.CocoDetection`` backed by
    hotcoco instead of pycocotools. Does not inherit from any torch class —
    duck-typing is sufficient for ``DataLoader``.

    Parameters
    ----------
    root : str
        Root directory of images.
    ann_file : str
        Path to COCO-format annotation JSON file.
    transform : callable, optional
        Transform applied to the PIL image.
    target_transform : callable, optional
        Transform applied to the list of annotation dicts.
    transforms : callable, optional
        Joint transform applied to ``(image, target)``. Applied *after*
        individual transforms.
    """

    def __init__(self, root, ann_file, transform=None, target_transform=None, transforms=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.get_img_ids())
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

    def _load_image(self, img_id):
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for CocoDetection. Install it with: pip install Pillow") from None
        img_info = self.coco.load_imgs([img_id])[0]
        path = os.path.join(self.root, img_info["file_name"])
        return Image.open(path).convert("RGB")

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        target = self.coco.load_anns(ann_ids)
        image = self._load_image(img_id)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return f"{self.__class__.__name__}(root={self.root!r}, num_images={len(self)})"


class CocoEvaluator:
    """Distributed COCO evaluator for PyTorch training loops.

    Drop-in replacement for the ``CocoEvaluator`` in torchvision's detection
    reference scripts. Wraps hotcoco's ``COCOeval`` with a tensor-friendly
    ``update()`` interface and optional distributed synchronization.

    Parameters
    ----------
    coco_gt : COCO
        Ground-truth COCO object.
    iou_types : list[str]
        IoU types to evaluate, such as ``["bbox"]`` or ``["bbox", "segm"]``.

    Example
    -------
    ::

        evaluator = CocoEvaluator(coco_gt, ["bbox"])
        for images, targets in data_loader:
            outputs = model(images)
            predictions = {t["image_id"]: o for t, o in zip(targets, outputs)}
            evaluator.update(predictions)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
    """

    def __init__(self, coco_gt, iou_types):
        if isinstance(iou_types, str):
            iou_types = [iou_types]
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        self.results = defaultdict(list)

        for iou_type in iou_types:
            self.coco_eval[iou_type] = None

    def update(self, predictions):
        """Accumulate predictions from one batch.

        Parameters
        ----------
        predictions : dict[int, dict]
            Mapping from image ID to a dict with keys depending on iou_type:

            - ``"bbox"``: ``{"boxes": Tensor(N,4), "scores": Tensor(N), "labels": Tensor(N)}``
              Boxes are in XYXY format; converted to XYWH internally.
            - ``"segm"``: ``{"masks": Tensor(N,1,H,W), "scores": Tensor(N), "labels": Tensor(N)}``
              Masks are thresholded at 0.5 and RLE-encoded internally.
            - ``"keypoints"``: ``{"keypoints": Tensor(N,K,3), "scores": Tensor(N),
              "labels": Tensor(N)}``
        """
        for iou_type in self.iou_types:
            results = _prepare_for_coco(predictions, iou_type)
            self.results[iou_type].extend(results)

    def synchronize_between_processes(self):
        """Gather results across all distributed processes.

        No-op when ``torch.distributed`` is not initialized or the world
        size is 1. Works with any backend, including NCCL.
        """
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return

        for iou_type in self.iou_types:
            # An object gather, not a byte-tensor gather: NCCL rejects CPU
            # tensors, and all_gather_object stages through the current CUDA
            # device itself under NCCL (torchvision's CocoEvaluator does the same).
            gathered: list[list] = [[] for _ in range(world_size)]
            dist.all_gather_object(gathered, self.results[iou_type])
            all_results = []
            for rank_results in gathered:
                all_results.extend(rank_results)
            self.results[iou_type] = all_results

    def accumulate(self):
        """Create COCOeval objects and run evaluate + accumulate."""
        for iou_type in self.iou_types:
            results = self.results[iou_type]
            if not results:
                print(f"Warning: no results for iou_type={iou_type!r}")
                continue

            coco_dt = self.coco_gt.load_res(results)
            coco_eval = COCOeval(self.coco_gt, coco_dt, iou_type)

            # Set params to use all image IDs from GT so images with no
            # detections are still counted (affects recall).
            params = Params(iou_type=iou_type)
            params.img_ids = sorted(self.coco_gt.get_img_ids())
            coco_eval.params = params

            coco_eval.evaluate()
            coco_eval.accumulate()
            self.coco_eval[iou_type] = coco_eval

    def summarize(self):
        """Print evaluation metrics for each iou_type."""
        for iou_type in self.iou_types:
            ev = self.coco_eval.get(iou_type)
            if ev is None:
                print(f"Warning: no evaluation for iou_type={iou_type!r}")
                continue
            print(f"IoU metric: {iou_type}")
            ev.summarize()

    def get_results(self):
        """Return metrics as ``{iou_type: {metric_name: value}}``."""
        results = {}
        for iou_type in self.iou_types:
            ev = self.coco_eval.get(iou_type)
            if ev is not None:
                results[iou_type] = ev.get_results()
        return results


def _prepare_for_coco(predictions, iou_type):
    """Convert tensor predictions to COCO-format annotation dicts."""
    coco_results = []
    for img_id, prediction in predictions.items():
        if isinstance(img_id, int):
            original_id = img_id
        else:
            original_id = int(img_id)

        if len(prediction.get("scores", [])) == 0:
            continue

        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        if iou_type == "bbox":
            boxes = prediction["boxes"]
            # Convert XYXY → XYWH
            boxes = boxes.tolist()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": labels[i],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": scores[i],
                    }
                )

        elif iou_type == "segm":
            masks = prediction["masks"]
            # masks: (N, 1, H, W) tensor → threshold → binary uint8 → RLE
            masks = masks.cpu().numpy()
            for i in range(masks.shape[0]):
                mask = masks[i, 0]  # (H, W)
                binary = (mask > 0.5).astype("uint8")
                rle = mask_utils.encode(_np.asfortranarray(binary))
                # counts is bytes — decode for JSON serialization
                rle_seg = {
                    "size": rle["size"],
                    "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"],
                }
                coco_results.append(
                    {"image_id": original_id, "category_id": labels[i], "segmentation": rle_seg, "score": scores[i]}
                )

        elif iou_type == "keypoints":
            keypoints = prediction["keypoints"]
            # (N, K, 3) → flat list
            keypoints = keypoints.tolist()
            for i, kpts in enumerate(keypoints):
                flat = [v for point in kpts for v in point]
                coco_results.append(
                    {"image_id": original_id, "category_id": labels[i], "keypoints": flat, "score": scores[i]}
                )

    return coco_results
