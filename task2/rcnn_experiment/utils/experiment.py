from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def summarize_annotations(annotations: dict) -> dict:
    class_counts = Counter()
    num_objects = 0
    num_empty_images = 0
    for image in annotations["images"]:
        image_annotations = image.get("annotations", [])
        if not image_annotations:
            num_empty_images += 1
        for ann in image_annotations:
            class_counts[ann["label"]] += 1
            num_objects += 1
    classes = annotations.get("classes", [])
    return {
        "dataset_name": annotations.get("dataset_name"),
        "split": annotations.get("split"),
        "num_images": len(annotations.get("images", [])),
        "num_objects": num_objects,
        "num_empty_images": num_empty_images,
        "avg_objects_per_image": num_objects / max(len(annotations.get("images", [])), 1),
        "classes": classes,
        "class_distribution": {class_name: int(class_counts.get(class_name, 0)) for class_name in classes},
    }


def summarize_label_indices(labels: Iterable[int], classes: list[str]) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {
        class_name: int(counts.get(class_idx, 0))
        for class_idx, class_name in enumerate(classes)
    }


def summarize_detections(detections: list[dict], classes: list[str]) -> dict:
    counts = Counter(det["label"] for det in detections)
    scores = [float(det["score"]) for det in detections]
    return {
        "num_detections": len(detections),
        "detections_per_class": {class_name: int(counts.get(class_name, 0)) for class_name in classes if class_name != "__background__"},
        "score_summary": {
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
            "mean": sum(scores) / len(scores) if scores else 0.0,
        },
    }


def experiment_name_from_path(path: str | Path) -> str:
    return Path(path).resolve().parent.name
