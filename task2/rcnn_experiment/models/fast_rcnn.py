from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import roi_align


@dataclass
class FastRCNNConfig:
    num_classes: int
    backbone: str
    roi_output_size: int = 7


class FastRCNN(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet50", roi_output_size: int = 7) -> None:
        super().__init__()
        if backbone == "resnet18":
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            out_channels = 512
        elif backbone == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.roi_output_size = roi_output_size
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        hidden_dim = 1024 if out_channels >= 1024 else 512
        self.box_head = nn.Sequential(
            nn.Linear(out_channels * roi_output_size * roi_output_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)

    def forward(self, images: torch.Tensor, proposals: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.feature_extractor(images)
        if not proposals:
            empty_logits = torch.zeros((0, self.num_classes), device=images.device)
            empty_bbox = torch.zeros((0, self.num_classes * 4), device=images.device)
            return empty_logits, empty_bbox

        image_height, image_width = images.shape[-2:]
        feature_height, feature_width = feature_map.shape[-2:]
        spatial_scale = min(feature_height / max(image_height, 1), feature_width / max(image_width, 1))
        rois = []
        for batch_idx, boxes in enumerate(proposals):
            if boxes.numel() == 0:
                continue
            batch_column = torch.full((boxes.shape[0], 1), batch_idx, dtype=boxes.dtype, device=boxes.device)
            rois.append(torch.cat([batch_column, boxes], dim=1))
        if not rois:
            empty_logits = torch.zeros((0, self.num_classes), device=images.device)
            empty_bbox = torch.zeros((0, self.num_classes * 4), device=images.device)
            return empty_logits, empty_bbox

        pooled = roi_align(
            input=feature_map,
            boxes=torch.cat(rois, dim=0),
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            aligned=True,
        )
        pooled = pooled.flatten(start_dim=1)
        features = self.box_head(pooled)
        return self.cls_score(features), self.bbox_pred(features)

    def checkpoint_payload(self, classes: list[str]) -> dict:
        return {
            "state_dict": self.state_dict(),
            "classes": classes,
            "backbone": self.backbone_name,
            "roi_output_size": self.roi_output_size,
        }

    @classmethod
    def load_from_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> tuple["FastRCNN", list[str]]:
        payload = torch.load(path, map_location=map_location)
        model = cls(
            num_classes=len(payload["classes"]),
            backbone=payload["backbone"],
            roi_output_size=payload.get("roi_output_size", 7),
        )
        model.load_state_dict(payload["state_dict"])
        return model, payload["classes"]
