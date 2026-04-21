from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from rcnn_experiment.models.classifier import train_classifier_from_chunks
from rcnn_experiment.models.feature_extractor import CNNFeatureExtractor
from rcnn_experiment.utils.box_ops import compute_iou
from rcnn_experiment.utils.io import ensure_dir, load_json


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


def build_training_chunks(
    train_annotations_path: str | Path,
    proposal_dir: str | Path,
    chunk_dir: str | Path,
    backbone: str,
    device: str,
    pos_iou: float = 0.5,
    neg_iou: float = 0.3,
    max_pos_per_image: int = 32,
    max_neg_per_image: int = 96,
    batch_size: int = 32,
    chunk_size: int = 100,
) -> tuple[list[Path], list[str], int, int]:
    train_annotations = load_json(train_annotations_path)
    classes = ["__background__"] + train_annotations["classes"]
    proposal_dir = Path(proposal_dir)
    chunk_dir = ensure_dir(chunk_dir)
    extractor = CNNFeatureExtractor(backbone=backbone, device=device)

    for old_chunk in chunk_dir.glob("chunk_*.npz"):
        old_chunk.unlink()

    chunk_features: list[np.ndarray] = []
    chunk_labels: list[np.ndarray] = []
    chunk_image_count = 0
    chunk_index = 0
    num_positive_samples = 0
    num_negative_samples = 0

    def flush_chunk() -> None:
        nonlocal chunk_features, chunk_labels, chunk_image_count, chunk_index
        if not chunk_features:
            return
        feature_array = np.concatenate(chunk_features, axis=0)
        label_array = np.concatenate(chunk_labels, axis=0)
        np.savez_compressed(
            chunk_dir / f"chunk_{chunk_index:05d}.npz",
            features=feature_array,
            labels=label_array,
        )
        chunk_features = []
        chunk_labels = []
        chunk_image_count = 0
        chunk_index += 1

    for image_info in tqdm(train_annotations["images"], desc="Building training set"):
        proposal_path = proposal_dir / f"{image_info['id']}.npz"
        if not proposal_path.exists():
            raise FileNotFoundError(f"Missing proposal file: {proposal_path}")

        proposals = np.load(proposal_path)["boxes"].astype(np.float32)
        gt_boxes = np.array([ann["bbox"] for ann in image_info["annotations"]], dtype=np.float32)
        gt_labels = [ann["label"] for ann in image_info["annotations"]]
        if gt_boxes.size == 0:
            continue

        assigned_labels = []
        selected_boxes = []
        pos_count = 0
        neg_count = 0

        for proposal in proposals:
            ious = compute_iou(proposal, gt_boxes)
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= pos_iou and pos_count < max_pos_per_image:
                assigned_labels.append(classes.index(gt_labels[best_idx]))
                selected_boxes.append(proposal)
                pos_count += 1
                num_positive_samples += 1
            elif best_iou < neg_iou and neg_count < max_neg_per_image:
                assigned_labels.append(0)
                selected_boxes.append(proposal)
                neg_count += 1
                num_negative_samples += 1

            if pos_count >= max_pos_per_image and neg_count >= max_neg_per_image:
                break

        if not selected_boxes:
            continue

        image = Image.open(image_info["file_name"]).convert("RGB")
        crops = _crop_boxes(image, np.array(selected_boxes, dtype=np.float32))
        features = extractor.extract_from_crops(crops, batch_size=batch_size)
        labels = np.array(assigned_labels, dtype=np.int64)

        chunk_features.append(features)
        chunk_labels.append(labels)
        chunk_image_count += 1

        if chunk_image_count >= chunk_size:
            flush_chunk()

    flush_chunk()

    chunk_paths = sorted(chunk_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise RuntimeError("No training samples were generated. Check proposals and IoU thresholds.")

    return chunk_paths, classes, num_positive_samples, num_negative_samples


def train_rcnn_model(
    train_annotations_path: str | Path,
    proposal_dir: str | Path,
    output_dir: str | Path,
    backbone: str,
    device: str,
    pos_iou: float,
    neg_iou: float,
    max_pos_per_image: int,
    max_neg_per_image: int,
    batch_size: int,
) -> dict:
    output_dir = ensure_dir(output_dir)
    chunk_dir = output_dir / "feature_chunks"

    chunk_paths, classes, num_positive_samples, num_negative_samples = build_training_chunks(
        train_annotations_path=train_annotations_path,
        proposal_dir=proposal_dir,
        chunk_dir=chunk_dir,
        backbone=backbone,
        device=device,
        pos_iou=pos_iou,
        neg_iou=neg_iou,
        max_pos_per_image=max_pos_per_image,
        max_neg_per_image=max_neg_per_image,
        batch_size=batch_size,
    )

    bundle = train_classifier_from_chunks(
        chunk_paths=chunk_paths,
        classes=classes,
        backbone=backbone,
    )

    model_path = output_dir / "rcnn_model.joblib"
    bundle.save(model_path)

    num_samples = num_positive_samples + num_negative_samples
    train_stats = {
        "num_samples": int(num_samples),
        "num_positive_samples": int(num_positive_samples),
        "num_negative_samples": int(num_negative_samples),
        "backbone": backbone,
        "model_path": str(model_path.resolve()),
        "feature_chunk_dir": str(chunk_dir.resolve()),
        "num_feature_chunks": len(chunk_paths),
        "classes": classes,
    }
    return train_stats
