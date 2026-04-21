from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from rcnn_experiment.utils.box_ops import clip_box
from rcnn_experiment.utils.io import ensure_dir, load_json


def _summarize_missing_images(annotations: dict) -> tuple[int, list[str]]:
    missing = []
    for image in annotations["images"]:
        image_path = Path(image["file_name"])
        if not image_path.exists():
            missing.append(str(image_path))
    return len(missing), missing[:3]


class SelectiveSearchProposer:
    def __init__(self, mode: str = "fast", max_proposals: int = 1500, min_size: int = 20) -> None:
        if mode not in {"fast", "quality"}:
            raise ValueError("mode must be 'fast' or 'quality'")
        self.mode = mode
        self.max_proposals = max_proposals
        self.min_size = min_size

    def generate(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        height, width = image.shape[:2]
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        if self.mode == "fast":
            ss.switchToSelectiveSearchFast()
        else:
            ss.switchToSelectiveSearchQuality()
        rects = ss.process()
        proposals = []
        for x, y, w, h in rects:
            if w < self.min_size or h < self.min_size:
                continue
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            box = clip_box(box, width=width, height=height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            proposals.append(box)
            if len(proposals) >= self.max_proposals:
                break
        if not proposals:
            proposals.append(np.array([0, 0, width - 1, height - 1], dtype=np.float32))
        return np.stack(proposals, axis=0)


def generate_from_annotations(
    annotations_path: str | Path,
    output_dir: str | Path,
    mode: str,
    max_proposals: int,
    min_size: int,
    overwrite: bool = False,
) -> None:
    annotations = load_json(annotations_path)
    missing_count, missing_examples = _summarize_missing_images(annotations)
    if missing_count:
        dataset_name = annotations.get("dataset_name", "unknown")
        split = annotations.get("split", "unknown")
        total = len(annotations["images"])
        message = (
            f"Annotation file {Path(annotations_path).resolve()} references {missing_count}/{total} missing images "
            f"for dataset '{dataset_name}' split '{split}'. Example paths: {', '.join(missing_examples)}"
        )
        if dataset_name == "voc" and split == "test" and missing_count == total:
            message += (
                ". This usually means the VOC2007 test JPEGs are not present in JPEGImages "
                "(for example, only the trainval set was downloaded)."
            )
        raise FileNotFoundError(message)
    output_dir = ensure_dir(output_dir)
    proposer = SelectiveSearchProposer(mode=mode, max_proposals=max_proposals, min_size=min_size)
    for image in tqdm(annotations["images"], desc="Generating proposals"):
        target_path = output_dir / f"{image['id']}.npz"
        if target_path.exists() and not overwrite:
            continue
        boxes = proposer.generate(image["file_name"])
        np.savez_compressed(target_path, boxes=boxes)
