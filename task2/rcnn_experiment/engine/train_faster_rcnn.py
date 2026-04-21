from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rcnn_experiment.data.detection_dataset import UnifiedDetectionDataset, collate_detection_batch
from rcnn_experiment.models.faster_rcnn import build_faster_rcnn, save_faster_rcnn_checkpoint
from rcnn_experiment.utils.experiment import current_timestamp, summarize_annotations
from rcnn_experiment.utils.io import ensure_dir, save_json
from rcnn_experiment.utils.runtime import configure_torch_runtime
from rcnn_experiment.utils.visualization import plot_training_history


def train_faster_rcnn_model(
    train_annotations_path: str | Path,
    output_dir: str | Path,
    device: str,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
) -> dict:
    dataset = UnifiedDetectionDataset(train_annotations_path)
    annotation_summary = summarize_annotations(dataset.annotations)
    classes = dataset.classes
    configure_torch_runtime(device)
    use_cuda = torch.device(device).type == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
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
    model = build_faster_rcnn(num_classes=len(classes)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(num_epochs // 3, 1), gamma=0.1)
    output_dir = ensure_dir(output_dir)
    use_amp = use_cuda

    epoch_losses: list[float] = []
    epoch_history: list[dict] = []
    total_samples = 0
    best_loss = float("inf")
    best_epoch = -1

    for epoch_idx in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_loss_components = {
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0,
        }
        steps = 0
        epoch_samples = 0
        for images, targets, _ in tqdm(loader, desc="Training Faster R-CNN"):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            running_loss += float(loss.item())
            for loss_name in running_loss_components:
                running_loss_components[loss_name] += float(loss_dict[loss_name].item())
            steps += 1
            total_samples += len(images)
            epoch_samples += len(images)
        epoch_loss = running_loss / max(steps, 1)
        epoch_losses.append(epoch_loss)
        epoch_record = {
            "epoch": epoch_idx + 1,
            "loss": epoch_loss,
            "loss_classifier": running_loss_components["loss_classifier"] / max(steps, 1),
            "loss_box_reg": running_loss_components["loss_box_reg"] / max(steps, 1),
            "loss_objectness": running_loss_components["loss_objectness"] / max(steps, 1),
            "loss_rpn_box_reg": running_loss_components["loss_rpn_box_reg"] / max(steps, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "steps": steps,
            "num_samples": epoch_samples,
        }
        epoch_history.append(epoch_record)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch_idx + 1
            save_faster_rcnn_checkpoint(model, output_dir / "faster_rcnn_model.pth", classes=classes)
        scheduler.step()

    last_model_path = output_dir / "faster_rcnn_model_last.pth"
    save_faster_rcnn_checkpoint(model, last_model_path, classes=classes)
    save_json(epoch_history, output_dir / "train_history.json")
    plot_training_history(epoch_history, output_dir / "loss_curve.png")
    return {
        "generated_at": current_timestamp(),
        "model_family": "faster_rcnn",
        "dataset_name": annotation_summary["dataset_name"],
        "split": annotation_summary["split"],
        "annotation_summary": annotation_summary,
        "num_samples": total_samples,
        "backbone": "resnet50_fpn",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epoch_losses": epoch_losses,
        "epoch_history": epoch_history,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "model_path": str((output_dir / "faster_rcnn_model.pth").resolve()),
        "last_model_path": str(last_model_path.resolve()),
        "train_history_path": str((output_dir / "train_history.json").resolve()),
        "loss_curve_path": str((output_dir / "loss_curve.png").resolve()),
        "train_annotations_path": str(Path(train_annotations_path).resolve()),
        "classes": classes,
    }
