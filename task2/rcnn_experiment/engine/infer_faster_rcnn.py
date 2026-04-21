from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rcnn_experiment.data.detection_dataset import UnifiedDetectionDataset, collate_detection_batch
from rcnn_experiment.models.faster_rcnn import load_faster_rcnn_checkpoint
from rcnn_experiment.utils.experiment import current_timestamp, experiment_name_from_path, summarize_detections
from rcnn_experiment.utils.io import ensure_dir, save_json
from rcnn_experiment.utils.metrics import evaluate_detections
from rcnn_experiment.utils.runtime import configure_torch_runtime
from rcnn_experiment.utils.visualization import generate_evaluation_visualizations


def infer_faster_rcnn(
    annotations_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    device: str,
    score_threshold: float,
    nms_threshold: float,
    batch_size: int,
    num_workers: int,
) -> dict:
    dataset = UnifiedDetectionDataset(annotations_path)
    annotations = dataset.annotations
    configure_torch_runtime(device)
    use_cuda = torch.device(device).type == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_detection_batch,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )
    model, classes = load_faster_rcnn_checkpoint(model_path, map_location=device)
    model.roi_heads.score_thresh = score_threshold
    model.roi_heads.nms_thresh = nms_threshold
    model.to(device)
    model.eval()
    output_dir = ensure_dir(output_dir)

    detections: list[dict] = []
    total_time = 0.0
    image_count = 0

    with torch.no_grad():
        for images, _, metas in tqdm(loader, desc="Running Faster R-CNN inference"):
            images = [image.to(device, non_blocking=True) for image in images]
            start = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                outputs = model(images)
            total_time += time.perf_counter() - start
            image_count += len(images)

            for output, meta in zip(outputs, metas):
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()
                for box, score, label in zip(boxes, scores, labels):
                    class_idx = int(label.item())
                    if class_idx <= 0 or class_idx >= len(classes):
                        continue
                    detections.append(
                        {
                            "image_id": meta["id"],
                            "label": classes[class_idx],
                            "score": float(score.item()),
                            "bbox": box.tolist(),
                        }
                    )

    metrics = evaluate_detections(annotations=annotations, detections=detections)
    detection_summary = summarize_detections(detections, classes)
    metrics.update(
        {
            "generated_at": current_timestamp(),
            "model_family": "faster_rcnn",
            "experiment_name": experiment_name_from_path(model_path),
            "FPS": image_count / total_time if total_time > 0 else 0.0,
            "num_images": image_count,
            "num_detections": len(detections),
            "model_path": str(Path(model_path).resolve()),
            "dataset_name": annotations["dataset_name"],
            "split": annotations["split"],
            "score_threshold": score_threshold,
            "nms_threshold": nms_threshold,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "classes": classes,
            "detection_summary": detection_summary,
        }
    )
    save_json(detections, output_dir / "detections.json")
    save_json(metrics, output_dir / "metrics.json")
    generate_evaluation_visualizations(
        annotations=annotations,
        detections=detections,
        metrics=metrics,
        output_dir=output_dir,
        score_threshold=score_threshold,
    )
    return metrics
