from __future__ import annotations

from typing import Iterable, List

import numpy as np


def xywh_to_xyxy(bbox: Iterable[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = box.copy()
    clipped[0] = np.clip(clipped[0], 0, width - 1)
    clipped[1] = np.clip(clipped[1], 0, height - 1)
    clipped[2] = np.clip(clipped[2], 0, width - 1)
    clipped[3] = np.clip(clipped[3], 0, height - 1)
    return clipped


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h
    box_area = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        rest = order[1:]
        ious = compute_iou(boxes[current], boxes[rest])
        order = rest[ious < iou_threshold]
    return keep

