from __future__ import annotations

import numpy as np

from rcnn_experiment.utils.box_ops import compute_iou


def _prepare_ground_truth(annotations: dict) -> tuple[dict, list[str]]:
    gt_by_image = {}
    classes = annotations["classes"]
    for image in annotations["images"]:
        boxes = []
        labels = []
        for ann in image["annotations"]:
            boxes.append(ann["bbox"])
            labels.append(ann["label"])
        gt_by_image[image["id"]] = {
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            "labels": labels,
        }
    return gt_by_image, classes


def _voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(precisions) - 1, 0, -1):
        precisions[idx - 1] = max(precisions[idx - 1], precisions[idx])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]))


def evaluate_detections(
    annotations: dict,
    detections: list[dict],
    iou_thresholds: list[float] | None = None,
) -> dict:
    if iou_thresholds is None:
        iou_thresholds = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]

    gt_by_image, classes = _prepare_ground_truth(annotations)
    class_to_gt_total = {cls_name: 0 for cls_name in classes}
    for image_gt in gt_by_image.values():
        for label in image_gt["labels"]:
            class_to_gt_total[label] += 1

    aps_by_threshold = {}
    prf = {}
    per_class_details: dict[str, dict] = {
        class_name: {
            "gt_count": int(class_to_gt_total[class_name]),
            "detections": 0,
            "AP@0.5": 0.0,
            "AP@0.5:0.95": 0.0,
            "precision@0.5": 0.0,
            "recall@0.5": 0.0,
            "f1@0.5": 0.0,
        }
        for class_name in classes
    }
    total_gt = sum(class_to_gt_total.values())
    for iou_thr in iou_thresholds:
        per_class_ap: dict[str, float] = {}
        tp_sum = 0
        fp_sum = 0
        fn_sum = 0
        for class_name in classes:
            class_dets = [det for det in detections if det["label"] == class_name]
            class_dets.sort(key=lambda item: item["score"], reverse=True)
            per_class_details[class_name]["detections"] = len(class_dets)
            matched = {
                image_id: np.zeros(len([lab for lab in gt_by_image[image_id]["labels"] if lab == class_name]), dtype=bool)
                for image_id in gt_by_image
            }

            gt_boxes_per_image = {}
            for image_id, info in gt_by_image.items():
                indices = [idx for idx, label in enumerate(info["labels"]) if label == class_name]
                gt_boxes_per_image[image_id] = info["boxes"][indices] if indices else np.zeros((0, 4), dtype=np.float32)

            tp = np.zeros(len(class_dets), dtype=np.float32)
            fp = np.zeros(len(class_dets), dtype=np.float32)
            for det_idx, det in enumerate(class_dets):
                image_id = det["image_id"]
                gt_boxes = gt_boxes_per_image[image_id]
                if gt_boxes.size == 0:
                    fp[det_idx] = 1
                    continue
                ious = compute_iou(np.array(det["bbox"], dtype=np.float32), gt_boxes)
                max_iou_idx = int(np.argmax(ious))
                max_iou = float(ious[max_iou_idx])
                if max_iou >= iou_thr and not matched[image_id][max_iou_idx]:
                    tp[det_idx] = 1
                    matched[image_id][max_iou_idx] = True
                else:
                    fp[det_idx] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            gt_total = class_to_gt_total[class_name]
            recalls = tp_cum / max(gt_total, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            per_class_ap[class_name] = _voc_ap(recalls, precisions) if gt_total > 0 else 0.0

            if abs(iou_thr - 0.5) < 1e-8:
                class_tp = int(tp.sum())
                class_fp = int(fp.sum())
                class_fn = max(gt_total - class_tp, 0)
                tp_sum += class_tp
                fp_sum += class_fp
                fn_sum += class_fn
                precision = class_tp / max(class_tp + class_fp, 1)
                recall = class_tp / max(class_tp + class_fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)
                per_class_details[class_name]["AP@0.5"] = per_class_ap[class_name]
                per_class_details[class_name]["precision@0.5"] = precision
                per_class_details[class_name]["recall@0.5"] = recall
                per_class_details[class_name]["f1@0.5"] = f1

        aps_by_threshold[f"{iou_thr:.2f}"] = per_class_ap
        if abs(iou_thr - 0.5) < 1e-8:
            precision = tp_sum / max(tp_sum + fp_sum, 1)
            recall = tp_sum / max(tp_sum + fn_sum, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            prf = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp_sum,
                "fp": fp_sum,
                "fn": fn_sum,
            }

    map50 = float(np.mean(list(aps_by_threshold["0.50"].values()))) if classes else 0.0
    map5095 = float(
        np.mean([np.mean(list(class_ap.values())) for class_ap in aps_by_threshold.values()])
    ) if classes else 0.0
    for class_name in classes:
        per_class_details[class_name]["AP@0.5:0.95"] = float(
            np.mean([threshold_ap[class_name] for threshold_ap in aps_by_threshold.values()])
        ) if aps_by_threshold else 0.0

    return {
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map5095,
        "Precision": prf.get("precision", 0.0),
        "Recall": prf.get("recall", 0.0),
        "F1": prf.get("f1", 0.0),
        "counts": {
            "tp": prf.get("tp", 0),
            "fp": prf.get("fp", 0),
            "fn": prf.get("fn", 0),
            "num_ground_truth": total_gt,
        },
        "per_class": per_class_details,
        "AP_by_threshold": aps_by_threshold,
    }
