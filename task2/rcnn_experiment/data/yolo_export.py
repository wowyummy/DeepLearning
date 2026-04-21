from __future__ import annotations

import os
import shutil
from pathlib import Path

from rcnn_experiment.utils.io import ensure_dir, load_json, save_json


def _canonical_split_name(split: str) -> str:
    lowered = split.lower()
    if lowered in {"train", "trainval", "train2017"}:
        return "train"
    if lowered in {"val", "val2017", "test"}:
        return "val"
    return lowered


def _link_or_copy_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def _xyxy_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    box_width = max(x2 - x1, 1e-6)
    box_height = max(y2 - y1, 1e-6)
    center_x = x1 + box_width / 2.0
    center_y = y1 + box_height / 2.0
    return center_x / width, center_y / height, box_width / width, box_height / height


def _export_split(annotations: dict, export_root: Path, split_name: str) -> Path:
    images_dir = ensure_dir(export_root / "images" / split_name)
    labels_dir = ensure_dir(export_root / "labels" / split_name)
    manifest_path = export_root / f"{split_name}.txt"

    manifest_lines: list[str] = []
    class_to_index = {name: idx for idx, name in enumerate(annotations["classes"])}
    for image_info in annotations["images"]:
        src_image = Path(image_info["file_name"]).expanduser().resolve()
        image_ext = src_image.suffix or ".jpg"
        stem = str(image_info["id"])
        linked_image = images_dir / f"{stem}{image_ext}"
        label_path = labels_dir / f"{stem}.txt"

        _link_or_copy_file(src_image, linked_image)
        manifest_lines.append(str(linked_image.absolute()))

        label_lines = []
        for ann in image_info["annotations"]:
            class_idx = class_to_index[ann["label"]]
            x_center, y_center, box_width, box_height = _xyxy_to_yolo(
                bbox=ann["bbox"],
                width=int(image_info["width"]),
                height=int(image_info["height"]),
            )
            label_lines.append(
                f"{class_idx} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    return manifest_path


def prepare_yolo_dataset(
    train_annotations_path: str | Path,
    output_dir: str | Path,
    val_annotations_path: str | Path | None = None,
) -> dict:
    output_dir = ensure_dir(output_dir)
    train_annotations = load_json(train_annotations_path)
    val_annotations = load_json(val_annotations_path) if val_annotations_path is not None else None

    if val_annotations is not None and train_annotations["classes"] != val_annotations["classes"]:
        raise ValueError("Train and validation annotations must share the same class ordering for YOLO export.")

    train_split = _canonical_split_name(train_annotations["split"])
    val_split = _canonical_split_name(val_annotations["split"]) if val_annotations is not None else "val"

    train_manifest = _export_split(train_annotations, output_dir, train_split)
    val_manifest = _export_split(val_annotations, output_dir, val_split) if val_annotations is not None else train_manifest

    names = train_annotations["classes"]
    data_yaml = output_dir / "dataset.yaml"
    yaml_lines = [
        f"path: {output_dir.resolve()}",
        f"train: {train_manifest.resolve()}",
        f"val: {val_manifest.resolve()}",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines.extend([f"  {idx}: {name}" for idx, name in enumerate(names)])
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    metadata = {
        "train_annotations_path": str(Path(train_annotations_path).resolve()),
        "val_annotations_path": str(Path(val_annotations_path).resolve()) if val_annotations_path is not None else None,
        "dataset_name": train_annotations["dataset_name"],
        "train_split": train_annotations["split"],
        "val_split": val_annotations["split"] if val_annotations is not None else None,
        "classes": names,
        "data_yaml": str(data_yaml.resolve()),
    }
    save_json(metadata, output_dir / "dataset_metadata.json")
    return metadata
