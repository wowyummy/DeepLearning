from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from rcnn_experiment.utils.io import load_json


class UnifiedDetectionDataset(Dataset):
    def __init__(self, annotations_path: str | Path) -> None:
        self.annotations = load_json(annotations_path)
        self.images = self.annotations["images"]
        self.classes = ["__background__"] + self.annotations["classes"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict, dict]:
        image_info = self.images[index]
        image = Image.open(image_info["file_name"]).convert("RGB")
        image_tensor = F.to_tensor(image)
        boxes = []
        labels = []
        for ann in image_info["annotations"]:
            x1, y1, x2, y2 = ann["bbox"]
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(ann["bbox"])
            labels.append(self.class_to_idx[ann["label"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        meta = {
            "id": image_info["id"],
            "file_name": str(Path(image_info["file_name"]).resolve()),
            "width": int(image_info["width"]),
            "height": int(image_info["height"]),
        }
        return image_tensor, target, meta


def collate_detection_batch(batch: list[tuple[torch.Tensor, dict, dict]]) -> tuple[list[torch.Tensor], list[dict], list[dict]]:
    images, targets, metas = zip(*batch)
    return list(images), list(targets), list(metas)
