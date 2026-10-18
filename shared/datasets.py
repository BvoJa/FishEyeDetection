"""Read YOLO/COCO labels and split source records without model dependencies."""

import math
import random
from pathlib import Path

from PIL import Image

from efficient_detection.shared.common import (
    CLASSES,
    EXTENSIONS,
    camera_scene,
    read_json,
)


def clipped_box(box, width, height):
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in [x, y, w, h]) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid bounding box: {box}")
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    return [x1, y1, x2 - x1, y2 - y1] if x2 > x1 and y2 > y1 else None


def load_yolo(image_dirs, label_dirs, confidence=0.5, allow_missing=False):
    if len(image_dirs) != len(label_dirs):
        raise ValueError("Provide one --labels directory for each --images directory")
    records = []
    seen = set()
    for source, (image_dir, label_dir) in enumerate(zip(image_dirs, label_dirs)):
        image_dir, label_dir = Path(image_dir).resolve(), Path(label_dir).resolve()
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Missing image/label directory: {image_dir}, {label_dir}"
            )
        paths = sorted(
            path
            for path in image_dir.rglob("*")
            if path.suffix.lower() in EXTENSIONS and path.is_file()
        )
        for path in paths:
            if path.resolve() in seen:
                raise ValueError(f"Image supplied more than once: {path}")
            seen.add(path.resolve())
            relative = path.relative_to(image_dir)
            label = label_dir / relative.with_suffix(".txt")
            if not label.is_file() and not allow_missing:
                raise FileNotFoundError(
                    f"{label}: use an empty .txt for a negative image, or --allow-missing-labels"
                )
            with Image.open(path) as image:
                width, height = image.size
            boxes = []
            for line_number, line in enumerate(
                label.read_text().splitlines() if label.is_file() else [], 1
            ):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) not in (5, 6):
                    raise ValueError(
                        f"{label}:{line_number}: expected class xc yc width height [score]"
                    )
                values = list(map(float, fields))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{label}:{line_number}: non-finite values")
                category, xc, yc, w, h = values[:5]
                if category != int(category) or not 0 <= category < len(CLASSES):
                    raise ValueError(
                        f"{label}:{line_number}: class ID must be 0..4 in order {CLASSES}"
                    )
                if len(values) == 6 and values[5] < confidence:
                    continue
                box = clipped_box(
                    [
                        (xc - w / 2) * width,
                        (yc - h / 2) * height,
                        w * width,
                        h * height,
                    ],
                    width,
                    height,
                )
                if box:
                    boxes.append(
                        {"category_id": int(category), "bbox": box, "iscrowd": 0}
                    )
            records.append(
                {
                    "path": path,
                    "name": f"source{source}/{relative.as_posix()}",
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                }
            )
    return records


def load_coco(annotation, image_dir, prefix="source0", category_map=None):
    data = read_json(annotation)
    aliases = {name: i for i, name in enumerate(CLASSES)}
    aliases.update({"person": 3, "motorcycle": 1, "bicycle": 1})
    mapping = {}
    for category in data["categories"]:
        source_id = int(category["id"])
        target = (
            category_map.get(str(source_id))
            if category_map is not None
            else aliases.get(category["name"].lower())
        )
        if target is not None and target not in range(len(CLASSES)):
            raise ValueError(f"Invalid target category: {target}")
        mapping[source_id] = target
    if not any(value is not None for value in mapping.values()):
        raise ValueError(
            "No recognized road-object categories. Supply --category-map with source IDs mapped to 0..4."
        )
    grouped = {}
    ids, filenames = set(), set()
    for ann in data["annotations"]:
        if ann["category_id"] not in mapping:
            raise ValueError(f"Undefined category: {ann['category_id']}")
        target = mapping[ann["category_id"]]
        if target is not None:
            grouped.setdefault(ann["image_id"], []).append(
                {
                    "category_id": target,
                    "bbox": ann["bbox"],
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )
    records = []
    for image in data["images"]:
        if image["id"] in ids:
            raise ValueError(f"Duplicate COCO image ID: {image['id']}")
        ids.add(image["id"])
        relative = Path(image["file_name"])
        if relative.as_posix() in filenames:
            raise ValueError(f"Duplicate COCO image filename: {relative}")
        filenames.add(relative.as_posix())
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"COCO file_name must be relative to --images: {relative}")
        path = (Path(image_dir) / relative).resolve()
        with Image.open(path) as pil:
            width, height = pil.size
        if (width, height) != (image["width"], image["height"]):
            raise ValueError(f"COCO image dimensions disagree with file: {path}")
        boxes = []
        for ann in grouped.get(image["id"], []):
            box = clipped_box(ann["bbox"], width, height)
            if box:
                boxes.append({**ann, "bbox": box})
        records.append(
            {
                "path": path,
                "name": f"{prefix}/{relative.as_posix()}",
                "width": width,
                "height": height,
                "boxes": boxes,
            }
        )
    if set(grouped) - ids:
        raise ValueError("COCO annotations reference unknown image IDs")
    return records


def split_records(records, split="camera", val_fraction=0.2, seed=42, val_cameras=None):
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two images are needed")
    rng = random.Random(seed)
    if split == "camera":
        groups = [camera_scene(record["path"].name) for record in records]
        if any(group is None for group in groups):
            raise ValueError(
                "Camera splitting requires camera<ID>_<M/A/E/N>_ filenames. Use --split random for other naming schemes or supply a separate validation set."
            )
        cameras = sorted({group[0] for group in groups})
        if val_cameras is None:
            if len(cameras) < 2:
                raise ValueError("Camera splitting needs at least two cameras")
            rng.shuffle(cameras)
            val_cameras = set(
                cameras[
                    : max(1, min(len(cameras) - 1, round(len(cameras) * val_fraction)))
                ]
            )
        else:
            val_cameras = set(val_cameras)
        is_val = [group[0] in val_cameras for group in groups]
    else:
        indices = list(range(len(records)))
        rng.shuffle(indices)
        selected = set(
            indices[: max(1, min(len(records) - 1, round(len(records) * val_fraction)))]
        )
        is_val = [i in selected for i in range(len(records))]
    train = [record for record, selected in zip(records, is_val) if not selected]
    val = [record for record, selected in zip(records, is_val) if selected]
    if not train or not val:
        raise ValueError("The split must contain both training and validation images")
    return train, val
