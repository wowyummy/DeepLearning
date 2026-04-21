from __future__ import annotations

from pathlib import Path

import torch


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO support requires the 'ultralytics' package. Install it with 'pip install ultralytics'."
        ) from exc
    return YOLO


def load_yolo_model(model: str | Path) -> object:
    yolo_cls = _require_ultralytics()
    return yolo_cls(str(model))


def normalize_yolo_device(device: str) -> str | int:
    if device.startswith("cuda:"):
        return int(device.split(":", maxsplit=1)[1])
    if device == "cuda":
        return 0
    if device == "cpu":
        return "cpu"
    return device


def maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
