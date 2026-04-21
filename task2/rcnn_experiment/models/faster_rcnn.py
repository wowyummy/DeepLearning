from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights


def build_faster_rcnn(num_classes: int) -> torch.nn.Module:
    return fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
    )


def save_faster_rcnn_checkpoint(model: torch.nn.Module, path: str | Path, classes: list[str]) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "backbone": "resnet50_fpn",
        },
        path,
    )


def load_faster_rcnn_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[torch.nn.Module, list[str]]:
    payload = torch.load(path, map_location=map_location)
    model = build_faster_rcnn(num_classes=len(payload["classes"]))
    model.load_state_dict(payload["state_dict"])
    return model, payload["classes"]
