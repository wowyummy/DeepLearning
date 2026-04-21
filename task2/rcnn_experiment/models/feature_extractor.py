from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


@dataclass
class BackboneSpec:
    name: str
    output_dim: int


BACKBONES = {
    "resnet18": BackboneSpec(name="resnet18", output_dim=512),
    "resnet50": BackboneSpec(name="resnet50", output_dim=2048),
}


class CNNFeatureExtractor:
    def __init__(self, backbone: str = "resnet50", device: str = "cpu") -> None:
        if backbone not in BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone_name = backbone
        self.device = torch.device(device)
        self.use_amp = self.device.type == "cuda"
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.model.to(self.device)
        self.model.eval()

    @property
    def output_dim(self) -> int:
        return BACKBONES[self.backbone_name].output_dim

    def extract_from_crops(self, crops: Iterable[Image.Image], batch_size: int = 32) -> np.ndarray:
        tensors = [self.transform(crop.convert("RGB")) for crop in crops]
        if not tensors:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        features = []
        with torch.inference_mode():
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[start : start + batch_size]).to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    output = self.model(batch).flatten(1)
                features.append(output.float().cpu().numpy())
        return np.concatenate(features, axis=0).astype(np.float32)
