from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from rcnn_experiment.models.classifier import RCNNClassifierBundle
from rcnn_experiment.models.feature_extractor import CNNFeatureExtractor
from rcnn_experiment.utils.box_ops import nms
from rcnn_experiment.utils.io import ensure_dir, load_json, save_json
from rcnn_experiment.utils.metrics import evaluate_detections


def _crop_boxes(image: Image.Image, boxes: np.ndarray) -> list[Image.Image]:
    crops = []
    width, height = image.size
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        crops.append(image.crop((x1, y1, x2, y2)))
    return crops


def infer_and_evaluate(
    annotations_path: str | Path,
    proposal_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    device: str,
    score_threshold: float,
    nms_threshold: float,
    batch_size: int,
    max_proposals: int,
    proposal_batch_size: int,
) -> dict:
    annotations = load_json(annotations_path)
    bundle = RCNNClassifierBundle.load(model_path)
    extractor = CNNFeatureExtractor(backbone=bundle.backbone, device=device)
    proposal_dir = Path(proposal_dir)
    output_dir = ensure_dir(output_dir)

    detections: list[dict] = []
    total_time = 0.0
    image_count = 0

    for image_info in tqdm(annotations["images"], desc="Running inference"):
        proposal_path = proposal_dir / f"{image_info['id']}.npz"
        if not proposal_path.exists():
            raise FileNotFoundError(f"Missing proposal file: {proposal_path}")
        with np.load(proposal_path) as proposal_payload:
            proposals = proposal_payload["boxes"].astype(np.float32)[:max_proposals]
        if proposals.size == 0:
            image_count += 1
            continue
        image = Image.open(image_info["file_name"]).convert("RGB")
        start = time.perf_counter()
        prob_batches = []
        for start_idx in range(0, len(proposals), proposal_batch_size):
            proposal_batch = proposals[start_idx : start_idx + proposal_batch_size]
            crops = _crop_boxes(image, proposal_batch)
            features = extractor.extract_from_crops(crops, batch_size=batch_size)
            prob_batches.append(bundle.pipeline.predict_proba(features))
        probs = np.concatenate(prob_batches, axis=0)
        total_time += time.perf_counter() - start
        image_count += 1

        for class_idx, class_name in enumerate(bundle.classes[1:], start=1):
            class_scores = probs[:, class_idx]
            keep = np.where(class_scores >= score_threshold)[0]
            if keep.size == 0:
                continue
            class_boxes = proposals[keep]
            class_scores = class_scores[keep]
            nms_keep = nms(class_boxes, class_scores, iou_threshold=nms_threshold)
            for idx in nms_keep:
                detections.append(
                    {
                        "image_id": image_info["id"],
                        "label": class_name,
                        "score": float(class_scores[idx]),
                        "bbox": class_boxes[idx].astype(float).tolist(),
                    }
                )

    metrics = evaluate_detections(annotations=annotations, detections=detections)
    fps = image_count / total_time if total_time > 0 else 0.0
    metrics.update(
        {
            "FPS": fps,
            "num_images": image_count,
            "num_detections": len(detections),
            "model_path": str(Path(model_path).resolve()),
            "proposal_dir": str(proposal_dir.resolve()),
            "dataset_name": annotations["dataset_name"],
            "split": annotations["split"],
        }
    )
    save_json(detections, output_dir / "detections.json")
    save_json(metrics, output_dir / "metrics.json")
    return metrics
