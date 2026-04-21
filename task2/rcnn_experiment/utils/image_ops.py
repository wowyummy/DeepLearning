from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import normalize


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_normalized_image_tensor(path: str, device: str | torch.device) -> torch.Tensor:
    image_tensor = TF.to_tensor(Image.open(path).convert("RGB"))
    image_tensor = normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return image_tensor.unsqueeze(0).to(device)
