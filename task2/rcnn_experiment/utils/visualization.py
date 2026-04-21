from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from rcnn_experiment.utils.box_ops import compute_iou
from rcnn_experiment.utils.io import ensure_dir


def plot_training_history(epoch_history: list[dict], output_path: str | Path) -> None:
    if not epoch_history:
        return

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    epochs = [int(record["epoch"]) for record in epoch_history]
    scalar_keys = [
        key
        for key in epoch_history[0]
        if key not in {"epoch", "steps", "num_samples", "num_positive_samples", "num_negative_samples"}
        and isinstance(epoch_history[0][key], (int, float))
    ]
    if not scalar_keys:
        return

    plt.figure(figsize=(9, 5))
    for key in scalar_keys:
        values = [float(record[key]) for record in epoch_history]
        plt.plot(epochs, values, marker="o", linewidth=1.8, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _prepare_ground_truth(annotations: dict) -> tuple[dict[str, dict], list[str]]:
    gt_by_image: dict[str, dict] = {}
    classes = annotations["classes"]
    for image in annotations["images"]:
        gt_by_image[image["id"]] = {
            "boxes": np.array([ann["bbox"] for ann in image["annotations"]], dtype=np.float32)
            if image["annotations"]
            else np.zeros((0, 4), dtype=np.float32),
            "labels": [ann["label"] for ann in image["annotations"]],
            "file_name": image["file_name"],
        }
    return gt_by_image, classes


def _compute_detection_matches(
    annotations: dict,
    detections: list[dict],
    iou_threshold: float,
) -> tuple[list[dict], int]:
    gt_by_image, _ = _prepare_ground_truth(annotations)
    detections_by_class: dict[str, list[dict]] = defaultdict(list)
    gt_total = 0
    for image_gt in gt_by_image.values():
        gt_total += len(image_gt["labels"])
    for det in detections:
        detections_by_class[det["label"]].append(det)

    matched_records: list[dict] = []
    for class_name, class_dets in detections_by_class.items():
        class_dets = sorted(class_dets, key=lambda item: float(item["score"]), reverse=True)
        matched = {
            image_id: np.zeros(
                sum(1 for label in image_gt["labels"] if label == class_name),
                dtype=bool,
            )
            for image_id, image_gt in gt_by_image.items()
        }
        gt_boxes_per_image = {}
        for image_id, image_gt in gt_by_image.items():
            indices = [idx for idx, label in enumerate(image_gt["labels"]) if label == class_name]
            gt_boxes_per_image[image_id] = (
                image_gt["boxes"][indices] if indices else np.zeros((0, 4), dtype=np.float32)
            )

        for det in class_dets:
            image_id = det["image_id"]
            gt_boxes = gt_boxes_per_image[image_id]
            if gt_boxes.size == 0:
                matched_records.append({**det, "matched": False})
                continue
            ious = compute_iou(np.array(det["bbox"], dtype=np.float32), gt_boxes)
            max_iou_idx = int(np.argmax(ious))
            max_iou = float(ious[max_iou_idx])
            is_match = max_iou >= iou_threshold and not matched[image_id][max_iou_idx]
            if is_match:
                matched[image_id][max_iou_idx] = True
            matched_records.append({**det, "matched": is_match})
    matched_records.sort(key=lambda item: float(item["score"]), reverse=True)
    return matched_records, gt_total


def plot_precision_recall_curve(
    annotations: dict,
    detections: list[dict],
    output_path: str | Path,
    iou_threshold: float = 0.5,
) -> None:
    matched_records, gt_total = _compute_detection_matches(annotations, detections, iou_threshold=iou_threshold)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if gt_total == 0:
        return

    tp = np.array([1.0 if record["matched"] else 0.0 for record in matched_records], dtype=np.float32)
    fp = 1.0 - tp
    if tp.size == 0:
        recalls = np.array([0.0], dtype=np.float32)
        precisions = np.array([1.0], dtype=np.float32)
    else:
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(gt_total, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    plt.figure(figsize=(6.5, 5))
    plt.plot(recalls, precisions, linewidth=2.0, color="#3c6e71")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve @ IoU {iou_threshold:.2f}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_per_class_ap(metrics: dict, output_path: str | Path) -> None:
    per_class = metrics.get("per_class", {})
    class_names = list(per_class.keys())
    if not class_names:
        return

    ap50 = [float(per_class[class_name].get("AP@0.5", 0.0)) for class_name in class_names]
    ap5095 = [float(per_class[class_name].get("AP@0.5:0.95", 0.0)) for class_name in class_names]
    x = np.arange(len(class_names))
    width = 0.38

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plt.figure(figsize=(max(10, len(class_names) * 0.55), 5.5))
    plt.bar(x - width / 2, ap50, width=width, label="AP@0.5", color="#3c6e71")
    plt.bar(x + width / 2, ap5095, width=width, label="AP@0.5:0.95", color="#d9ae61")
    plt.xticks(x, class_names, rotation=35, ha="right")
    plt.ylabel("AP")
    plt.title("Per-Class Average Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    annotations: dict,
    detections: list[dict],
    output_path: str | Path,
    iou_threshold: float = 0.5,
) -> None:
    gt_by_image, classes = _prepare_ground_truth(annotations)
    labels = classes + ["__background__"]
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)

    detections_by_image: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        detections_by_image[det["image_id"]].append(det)

    for image_id, image_gt in gt_by_image.items():
        image_detections = sorted(detections_by_image.get(image_id, []), key=lambda item: float(item["score"]), reverse=True)
        gt_boxes = image_gt["boxes"]
        gt_labels = image_gt["labels"]
        matched_gt = np.zeros(len(gt_labels), dtype=bool)

        for det in image_detections:
            pred_label = det["label"]
            pred_idx = label_to_index.get(pred_label, label_to_index["__background__"])
            if gt_boxes.size == 0:
                matrix[label_to_index["__background__"], pred_idx] += 1
                continue
            ious = compute_iou(np.array(det["bbox"], dtype=np.float32), gt_boxes)
            best_gt_idx = int(np.argmax(ious))
            best_iou = float(ious[best_gt_idx])
            if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
                true_label = gt_labels[best_gt_idx]
                true_idx = label_to_index[true_label]
                matrix[true_idx, pred_idx] += 1
                matched_gt[best_gt_idx] = True
            else:
                matrix[label_to_index["__background__"], pred_idx] += 1

        for gt_idx, gt_label in enumerate(gt_labels):
            if not matched_gt[gt_idx]:
                matrix[label_to_index[gt_label], label_to_index["__background__"]] += 1

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plt.figure(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.55)))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"Confusion Matrix @ IoU {iou_threshold:.2f}")
    if len(labels) <= 25:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                plt.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_detection_visualizations(
    annotations: dict,
    detections: list[dict],
    output_dir: str | Path,
    score_threshold: float = 0.3,
    max_images: int = 8,
) -> None:
    output_dir = ensure_dir(output_dir)
    image_map = {image["id"]: image for image in annotations["images"]}
    detections_by_image: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        if float(det["score"]) >= score_threshold:
            detections_by_image[det["image_id"]].append(det)

    ranked_image_ids = sorted(
        detections_by_image,
        key=lambda image_id: max(float(det["score"]) for det in detections_by_image[image_id]),
        reverse=True,
    )[:max_images]

    for image_id in ranked_image_ids:
        image_info = image_map.get(image_id)
        if image_info is None:
            continue
        image = Image.open(image_info["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(image)

        for ann in image_info["annotations"]:
            x1, y1, x2, y2 = ann["bbox"]
            draw.rectangle((x1, y1, x2, y2), outline=(46, 204, 113), width=3)
            draw.text((x1 + 2, max(0, y1 - 14)), f"GT:{ann['label']}", fill=(46, 204, 113))

        for det in sorted(detections_by_image[image_id], key=lambda item: float(item["score"]), reverse=True):
            x1, y1, x2, y2 = det["bbox"]
            draw.rectangle((x1, y1, x2, y2), outline=(231, 76, 60), width=3)
            draw.text(
                (x1 + 2, min(image.height - 14, y1 + 2)),
                f"P:{det['label']} {float(det['score']):.2f}",
                fill=(231, 76, 60),
            )

        image.save(output_dir / f"{image_id}.png")


def generate_evaluation_visualizations(
    annotations: dict,
    detections: list[dict],
    metrics: dict,
    output_dir: str | Path,
    score_threshold: float = 0.3,
) -> None:
    output_dir = ensure_dir(output_dir)
    plot_precision_recall_curve(annotations, detections, output_dir / "pr_curve.png")
    plot_per_class_ap(metrics, output_dir / "per_class_ap.png")
    plot_confusion_matrix(annotations, detections, output_dir / "confusion_matrix.png")
    save_detection_visualizations(
        annotations,
        detections,
        output_dir / "visualizations",
        score_threshold=score_threshold,
    )
