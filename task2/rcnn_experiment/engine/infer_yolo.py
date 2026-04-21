from __future__ import annotations

import time
from pathlib import Path

from rcnn_experiment.models.yolo import load_yolo_model, normalize_yolo_device
from rcnn_experiment.utils.io import ensure_dir, load_json, save_json
from rcnn_experiment.utils.metrics import evaluate_detections


def infer_yolo(
    annotations_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    device: str,
    score_threshold: float,
    nms_threshold: float,
    img_size: int,
    batch_size: int,
) -> dict:
    annotations = load_json(annotations_path)
    output_dir = ensure_dir(output_dir)
    detector = load_yolo_model(model_path)

    detections: list[dict] = []
    total_time = 0.0
    image_count = 0

    images = annotations["images"]
    for start_idx in range(0, len(images), batch_size):
        batch_images = images[start_idx : start_idx + batch_size]
        image_paths = [image_info["file_name"] for image_info in batch_images]
        start = time.perf_counter()
        results = detector.predict(
            source=image_paths,
            conf=score_threshold,
            iou=nms_threshold,
            imgsz=img_size,
            batch=batch_size,
            device=normalize_yolo_device(device),
            save=False,
            verbose=False,
            stream=False,
        )
        total_time += time.perf_counter() - start
        image_count += len(batch_images)

        for image_info, result in zip(batch_images, results):
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy_boxes = boxes.xyxy.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            for bbox, score, class_idx in zip(xyxy_boxes, scores, class_ids):
                label = names[int(class_idx)]
                detections.append(
                    {
                        "image_id": image_info["id"],
                        "label": label,
                        "score": float(score),
                        "bbox": [float(value) for value in bbox],
                    }
                )

    metrics = evaluate_detections(annotations=annotations, detections=detections)
    metrics.update(
        {
            "FPS": image_count / total_time if total_time > 0 else 0.0,
            "num_images": image_count,
            "num_detections": len(detections),
            "model_path": str(Path(model_path).resolve()),
            "dataset_name": annotations["dataset_name"],
            "split": annotations["split"],
            "model_family": "yolo",
            "img_size": img_size,
            "batch_size": batch_size,
        }
    )
    save_json(detections, output_dir / "detections.json")
    save_json(metrics, output_dir / "metrics.json")
    return metrics
