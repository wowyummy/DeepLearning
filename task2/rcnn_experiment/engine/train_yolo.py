from __future__ import annotations

from pathlib import Path

from rcnn_experiment.data.yolo_export import prepare_yolo_dataset
from rcnn_experiment.models.yolo import load_yolo_model, normalize_yolo_device
from rcnn_experiment.utils.io import ensure_dir


def train_yolo_model(
    train_annotations_path: str | Path,
    output_dir: str | Path,
    model: str,
    device: str,
    img_size: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    num_workers: int,
    amp: bool,
    val_annotations_path: str | Path | None = None,
) -> dict:
    output_dir = ensure_dir(output_dir)
    dataset_info = prepare_yolo_dataset(
        train_annotations_path=train_annotations_path,
        val_annotations_path=val_annotations_path,
        output_dir=output_dir / "yolo_dataset",
    )
    detector = load_yolo_model(model)
    results = detector.train(
        data=dataset_info["data_yaml"],
        epochs=num_epochs,
        imgsz=img_size,
        batch=batch_size,
        workers=num_workers,
        device=normalize_yolo_device(device),
        project=str(output_dir),
        name="train",
        exist_ok=True,
        amp=amp,
        lr0=learning_rate,
        weight_decay=weight_decay,
        verbose=True,
    )

    run_dir = Path(getattr(results, "save_dir", output_dir / "train"))
    best_model_path = run_dir / "weights" / "best.pt"
    last_model_path = run_dir / "weights" / "last.pt"
    results_csv_path = run_dir / "results.csv"

    return {
        "model_family": "yolo",
        "base_model": model,
        "dataset_name": dataset_info["dataset_name"],
        "train_annotations_path": str(Path(train_annotations_path).resolve()),
        "val_annotations_path": str(Path(val_annotations_path).resolve()) if val_annotations_path is not None else None,
        "data_yaml": dataset_info["data_yaml"],
        "classes": dataset_info["classes"],
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "amp": amp,
        "run_dir": str(run_dir.resolve()),
        "model_path": str(best_model_path.resolve()),
        "best_model_path": str(best_model_path.resolve()),
        "last_model_path": str(last_model_path.resolve()),
        "results_csv_path": str(results_csv_path.resolve()) if results_csv_path.exists() else None,
    }
