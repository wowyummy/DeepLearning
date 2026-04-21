from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm

from rcnn_experiment.data.detection_dataset import UnifiedDetectionDataset, collate_detection_batch
from rcnn_experiment.models.fast_rcnn import FastRCNN
from rcnn_experiment.utils.detection_ops import clip_boxes_to_image, decode_boxes
from rcnn_experiment.utils.io import ensure_dir, save_json
from rcnn_experiment.utils.metrics import evaluate_detections


def infer_fast_rcnn(
    annotations_path: str | Path,
    proposal_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    image_root: str | Path | None,
    device: str,
    score_threshold: float,
    nms_threshold: float,
    max_proposals: int,
    proposal_batch_size: int,
) -> dict:
    dataset = UnifiedDetectionDataset(annotations_path)
    annotations = dataset.annotations
    proposal_dir = Path(proposal_dir)
    output_dir = ensure_dir(output_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_detection_batch,
    )
    model, classes = FastRCNN.load_from_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()

    detections: list[dict] = []
    total_time = 0.0

    with torch.no_grad():
        for images, _, metas in tqdm(loader, desc="Running Fast R-CNN inference"):
            image_tensor = images[0].unsqueeze(0).to(device)
            meta = metas[0]
            proposal_path = proposal_dir / f"{meta['id']}.npz"
            if not proposal_path.exists():
                raise FileNotFoundError(f"Missing proposal file: {proposal_path}")
            with np.load(proposal_path) as proposal_payload:
                proposals = torch.from_numpy(proposal_payload["boxes"].astype(np.float32)[:max_proposals]).to(device)
            if proposals.numel() == 0:
                continue

            start = time.perf_counter()
            probs_batches = []
            bbox_batches = []
            proposal_batches = []
            for start_idx in range(0, proposals.shape[0], proposal_batch_size):
                proposal_batch = proposals[start_idx : start_idx + proposal_batch_size]
                class_logits, bbox_deltas = model(image_tensor, [proposal_batch])
                probs_batches.append(torch.softmax(class_logits, dim=1))
                bbox_batches.append(bbox_deltas)
                proposal_batches.append(proposal_batch)
            probs = torch.cat(probs_batches, dim=0)
            bbox_deltas = torch.cat(bbox_batches, dim=0)
            proposals = torch.cat(proposal_batches, dim=0)
            total_time += time.perf_counter() - start

            bbox_deltas = bbox_deltas.view(-1, len(classes), 4)
            for class_idx, class_name in enumerate(classes[1:], start=1):
                class_scores = probs[:, class_idx]
                keep = torch.where(class_scores >= score_threshold)[0]
                if keep.numel() == 0:
                    continue

                refined_boxes = decode_boxes(bbox_deltas[keep, class_idx], proposals[keep])
                refined_boxes = clip_boxes_to_image(refined_boxes, height=meta["height"], width=meta["width"])
                keep_after_nms = nms(refined_boxes, class_scores[keep], nms_threshold)
                for det_idx in keep_after_nms.tolist():
                    detections.append(
                        {
                            "image_id": meta["id"],
                            "label": class_name,
                            "score": float(class_scores[keep][det_idx].item()),
                            "bbox": refined_boxes[det_idx].cpu().tolist(),
                        }
                    )

    metrics = evaluate_detections(annotations=annotations, detections=detections)
    image_count = len(dataset)
    metrics.update(
        {
            "FPS": image_count / total_time if total_time > 0 else 0.0,
            "num_images": image_count,
            "num_detections": len(detections),
            "model_path": str(Path(model_path).resolve()),
            "proposal_dir": str(proposal_dir.resolve()),
            "image_root": str(Path(image_root).resolve()) if image_root is not None else None,
            "dataset_name": annotations["dataset_name"],
            "split": annotations["split"],
        }
    )
    save_json(detections, output_dir / "detections.json")
    save_json(metrics, output_dir / "metrics.json")
    return metrics
