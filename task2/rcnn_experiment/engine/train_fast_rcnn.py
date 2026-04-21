from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from rcnn_experiment.data.detection_dataset import UnifiedDetectionDataset, collate_detection_batch
from rcnn_experiment.models.fast_rcnn import FastRCNN
from rcnn_experiment.utils.detection_ops import encode_boxes
from rcnn_experiment.utils.io import ensure_dir


def _pad_images_for_batch(images: list[torch.Tensor]) -> torch.Tensor:
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)
    padded = []
    for image in images:
        pad_height = max_height - image.shape[1]
        pad_width = max_width - image.shape[2]
        padded.append(F.pad(image, (0, pad_width, 0, pad_height)))
    return torch.stack(padded, dim=0)


def _sample_training_proposals(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pos_iou: float,
    neg_iou: float,
    max_pos_per_image: int,
    max_neg_per_image: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if proposals.numel() == 0 or gt_boxes.numel() == 0:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0, 4), dtype=torch.float32),
        )

    ious = box_iou(proposals, gt_boxes)
    best_iou, best_indices = ious.max(dim=1)
    positive_indices = torch.where(best_iou >= pos_iou)[0][:max_pos_per_image]
    negative_indices = torch.where(best_iou < neg_iou)[0][:max_neg_per_image]
    keep = torch.cat([positive_indices, negative_indices], dim=0)
    if keep.numel() == 0:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0, 4), dtype=torch.float32),
        )

    sampled_boxes = proposals[keep]
    matched_gt = gt_boxes[best_indices[keep]]
    labels = torch.zeros((keep.numel(),), dtype=torch.int64)
    labels[: positive_indices.numel()] = gt_labels[best_indices[positive_indices]]
    regression_targets = torch.zeros((keep.numel(), 4), dtype=torch.float32)
    if positive_indices.numel() > 0:
        regression_targets[: positive_indices.numel()] = encode_boxes(
            reference_boxes=matched_gt[: positive_indices.numel()],
            proposals=sampled_boxes[: positive_indices.numel()],
        )
    return sampled_boxes, labels, regression_targets


def train_fast_rcnn_model(
    train_annotations_path: str | Path,
    proposal_dir: str | Path,
    output_dir: str | Path,
    backbone: str,
    device: str,
    pos_iou: float,
    neg_iou: float,
    max_pos_per_image: int,
    max_neg_per_image: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
) -> dict:
    dataset = UnifiedDetectionDataset(train_annotations_path)
    classes = dataset.classes
    proposal_dir = Path(proposal_dir)
    output_dir = ensure_dir(output_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_detection_batch,
    )
    model = FastRCNN(num_classes=len(classes), backbone=backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_samples = 0
    total_pos = 0
    total_neg = 0
    epoch_losses: list[float] = []

    for _ in range(num_epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        for images, targets, metas in tqdm(loader, desc="Training Fast R-CNN"):
            batch_images = []
            batch_boxes = []
            batch_labels = []
            batch_regression_targets = []

            for image_tensor, target, meta in zip(images, targets, metas):
                proposal_path = proposal_dir / f"{meta['id']}.npz"
                if not proposal_path.exists():
                    raise FileNotFoundError(f"Missing proposal file: {proposal_path}")

                with np.load(proposal_path) as proposal_payload:
                    proposals = torch.from_numpy(proposal_payload["boxes"].astype(np.float32))

                sampled_boxes, labels, regression_targets = _sample_training_proposals(
                    proposals=proposals,
                    gt_boxes=target["boxes"],
                    gt_labels=target["labels"],
                    pos_iou=pos_iou,
                    neg_iou=neg_iou,
                    max_pos_per_image=max_pos_per_image,
                    max_neg_per_image=max_neg_per_image,
                )
                if sampled_boxes.numel() == 0:
                    continue

                batch_images.append(image_tensor)
                batch_boxes.append(sampled_boxes.to(device))
                batch_labels.append(labels.to(device))
                batch_regression_targets.append(regression_targets.to(device))

            if not batch_images:
                continue

            image_tensor = _pad_images_for_batch(batch_images).to(device)
            labels = torch.cat(batch_labels, dim=0)
            regression_targets = torch.cat(batch_regression_targets, dim=0)

            class_logits, bbox_deltas = model(image_tensor, batch_boxes)
            classification_loss = F.cross_entropy(class_logits, labels)

            positive_mask = labels > 0
            box_regression = torch.tensor(0.0, device=device)
            if positive_mask.any():
                bbox_deltas = bbox_deltas.view(-1, len(classes), 4)
                positive_deltas = bbox_deltas[positive_mask, labels[positive_mask]]
                box_regression = F.smooth_l1_loss(
                    positive_deltas,
                    regression_targets[positive_mask],
                    beta=1.0,
                )

            loss = classification_loss + box_regression
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            steps += 1
            total_samples += int(labels.numel())
            total_pos += int((labels > 0).sum().item())
            total_neg += int((labels == 0).sum().item())

        epoch_losses.append(running_loss / max(steps, 1))

    model_path = output_dir / "fast_rcnn_model.pth"
    torch.save(model.checkpoint_payload(classes=classes), model_path)
    return {
        "num_samples": total_samples,
        "num_positive_samples": total_pos,
        "num_negative_samples": total_neg,
        "backbone": backbone,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epoch_losses": epoch_losses,
        "model_path": str(model_path.resolve()),
        "classes": classes,
    }
