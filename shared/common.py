"""Shared data conventions; also used by the PyTorch-free local runner."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

CLASSES = ["bus", "bike", "car", "pedestrian", "truck"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_config(path):
    import yaml

    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    root = Path(cfg.get("path", path.parent))
    if not root.is_absolute():
        root = path.parent / root
    if (path.parent / "manifest.json").is_file():
        root = path.parent  # Prepared datasets remain usable after copying off Kaggle.
    if cfg.get("names") != CLASSES:
        raise ValueError(f"Run prepare_yolo.py first; names must be {CLASSES}")
    manifest = read_json(root / "manifest.json")
    return root, cfg, manifest


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def camera_scene(filename):
    match = re.search(r"camera(\d+)_([MAEN])_", Path(filename).name, re.IGNORECASE)
    return (int(match[1]), match[2].upper()) if match else None


def box_iou(box, boxes):
    """IoU for XYXY pixel coordinates, including empty arrays."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    box = np.asarray(box, dtype=np.float32)
    intersection = np.maximum(
        0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])
    ).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-8)


def class_nms(boxes, scores, labels, threshold=0.6):
    """Score-ordered, per-class NMS with actual IoU (not intersection/area)."""
    keep = []
    for label in np.unique(labels):
        remaining = np.flatnonzero(labels == label)
        remaining = remaining[np.argsort(-scores[remaining], kind="stable")]
        while len(remaining):
            index = int(remaining[0])
            keep.append(index)
            remaining = remaining[1:]
            remaining = remaining[box_iou(boxes[index], boxes[remaining]) <= threshold]
    return np.asarray(sorted(keep, key=lambda i: -float(scores[i])), dtype=np.int64)


def detection_metrics(ground_truth, predictions, confidence=0.5, iou=0.5):
    """Micro precision/recall/F1 from one-to-one matches within image and class."""
    tp = fp = fn = 0
    matched_ious = []
    for truth, prediction in zip(ground_truth, predictions):
        boxes = np.asarray(truth["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(truth["labels"])
        used = set()
        order = np.argsort(-np.asarray(prediction["scores"]), kind="stable")
        for index in order:
            if prediction["scores"][index] < confidence:
                continue
            candidates = [
                i
                for i in np.flatnonzero(labels == prediction["labels"][index])
                if i not in used
            ]
            overlaps = box_iou(prediction["boxes"][index], boxes[candidates])
            if len(overlaps) and overlaps.max() >= iou:
                best = int(overlaps.argmax())
                used.add(candidates[best])
                matched_ious.append(float(overlaps[best]))
                tp += 1
            else:
                fp += 1
        fn += len(boxes) - len(used)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def validate_coco(annotation, images_root):
    """Fail before training on wrong class IDs, invalid boxes or missing images."""
    data = read_json(annotation)
    categories = {int(cat["id"]): cat["name"].lower() for cat in data["categories"]}
    if categories != dict(enumerate(CLASSES)):
        raise ValueError(
            f"{annotation}: expected category IDs 0..4 in order {CLASSES}. Run prepare_data.py --format coco first."
        )
    images = {image["id"]: image for image in data["images"]}
    if not images or len(images) != len(data["images"]):
        raise ValueError(
            f"{annotation}: image IDs must be unique and the split must be nonempty"
        )
    annotation_ids = set()
    for image in images.values():
        path = Path(images_root) / image["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if image["width"] <= 0 or image["height"] <= 0:
            raise ValueError(f"Invalid image dimensions: {path}")
    for ann in data["annotations"]:
        if ann["id"] in annotation_ids:
            raise ValueError(f"Duplicate annotation ID: {ann['id']}")
        annotation_ids.add(ann["id"])
        if ann["image_id"] not in images or ann["category_id"] not in categories:
            raise ValueError(f"Unknown image/category in annotation {ann['id']}")
        x, y, width, height = ann["bbox"]
        image = images[ann["image_id"]]
        if (
            not np.isfinite([x, y, width, height]).all()
            or min(x, y) < 0
            or min(width, height) <= 0
            or x + width > image["width"] + 1e-4
            or y + height > image["height"] + 1e-4
        ):
            raise ValueError(f"Invalid bounding box in annotation {ann['id']}")
    return data
