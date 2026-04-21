from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from pycocotools.coco import COCO
from tqdm import tqdm

from rcnn_experiment.utils.box_ops import xywh_to_xyxy
from rcnn_experiment.utils.io import save_json


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def prepare_voc(root: str | Path, split: str, output: str | Path) -> None:
    root = Path(root)
    split_file = root / "ImageSets" / "Main" / f"{split}.txt"
    image_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing_images = []
    images = []
    for image_id in tqdm(image_ids, desc=f"Preparing VOC {split}"):
        xml_path = root / "Annotations" / f"{image_id}.xml"
        image_path = root / "JPEGImages" / f"{image_id}.jpg"
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        tree = ET.parse(xml_path)
        root_node = tree.getroot()
        size_node = root_node.find("size")
        width = int(size_node.findtext("width"))
        height = int(size_node.findtext("height"))
        annotations = []
        for obj in root_node.findall("object"):
            difficult = int(obj.findtext("difficult", default="0"))
            if difficult == 1:
                continue
            label = obj.findtext("name")
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
            annotations.append(
                {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "label": label,
                    "category_id": VOC_CLASSES.index(label),
                }
            )
        images.append(
            {
                "id": image_id,
                "file_name": str(image_path.resolve()),
                "width": width,
                "height": height,
                "split": split,
                "annotations": annotations,
            }
        )
    if missing_images:
        sample = ", ".join(missing_images[:3])
        raise FileNotFoundError(
            f"Missing {len(missing_images)} VOC image files under {root / 'JPEGImages'} while preparing split "
            f"'{split}'. Example paths: {sample}"
        )
    save_json({"dataset_name": "voc", "split": split, "classes": VOC_CLASSES, "images": images}, output)


def prepare_coco(root: str | Path, split: str, output: str | Path) -> None:
    root = Path(root)
    ann_path = root / "annotations" / f"instances_{split}.json"
    image_dir = root / split
    coco = COCO(str(ann_path))
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cats = sorted(cats, key=lambda x: x["id"])
    classes = [cat["name"] for cat in cats]
    cat_id_to_index = {cat["id"]: idx for idx, cat in enumerate(cats)}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in cats}
    images = []
    for image_info in tqdm(coco.loadImgs(coco.getImgIds()), desc=f"Preparing COCO {split}"):
        ann_ids = coco.getAnnIds(imgIds=[image_info["id"]], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        annotations = []
        for ann in anns:
            bbox = xywh_to_xyxy(ann["bbox"])
            label = cat_id_to_name[ann["category_id"]]
            annotations.append(
                {
                    "bbox": bbox,
                    "label": label,
                    "category_id": cat_id_to_index[ann["category_id"]],
                }
            )
        images.append(
            {
                "id": str(image_info["id"]),
                "file_name": str((image_dir / image_info["file_name"]).resolve()),
                "width": int(image_info["width"]),
                "height": int(image_info["height"]),
                "split": split,
                "annotations": annotations,
            }
        )
    save_json({"dataset_name": "coco", "split": split, "classes": classes, "images": images}, output)
