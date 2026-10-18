"""Augment only the training split; validation images are never donors."""

import argparse
import copy
import math
import random
from pathlib import Path

from PIL import Image

from efficient_detection.shared.common import (
    box_iou,
    camera_scene,
    validate_coco,
    write_json,
)


def xyxy(box):
    x, y, width, height = box
    return [x, y, x + width, y + height]


def augment(dataset, max_objects=10, seed=42):
    root = Path(dataset) / "images"
    data = validate_coco(Path(dataset) / "train.json", root)
    result = copy.deepcopy(data)
    groups, annotations, all_annotations = {}, {}, {}
    for ann in data["annotations"]:
        all_annotations.setdefault(ann["image_id"], []).append(ann)
        if not ann.get("iscrowd", 0):
            annotations.setdefault(ann["image_id"], []).append(ann)
    for image in data["images"]:
        group = camera_scene(image["file_name"])
        if group:
            groups.setdefault(group, []).append(image)
    next_image = max(image["id"] for image in data["images"]) + 1
    next_ann = max((ann["id"] for ann in data["annotations"]), default=0) + 1
    rng = random.Random(seed)
    generated = 0
    for image in data["images"]:
        donors = [
            donor
            for donor in groups.get(camera_scene(image["file_name"]), [])
            if donor["id"] != image["id"]
            and (donor["width"], donor["height"]) == (image["width"], image["height"])
            and annotations.get(donor["id"])
        ]
        if not donors:
            continue
        with Image.open(root / image["file_name"]) as source:
            canvas = source.convert("RGB")
        originals = all_annotations.get(image["id"], [])
        existing = [xyxy(ann["bbox"]) for ann in originals]
        added = []
        for _ in range(max_objects * 10):
            if len(added) >= max_objects:
                break
            donor = rng.choice(donors)
            ann = rng.choice(annotations[donor["id"]])
            x1, y1, x2, y2 = xyxy(ann["bbox"])
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = (
                min(image["width"], math.ceil(x2)),
                min(image["height"], math.ceil(y2)),
            )
            box = [x1, y1, x2, y2]
            center_x, center_y = image["width"] / 2, image["height"] / 2
            distance = math.hypot((x1 + x2) / 2 - center_x, (y1 + y2) / 2 - center_y)
            probability = min(1.0, 0.3 + distance / math.hypot(center_x, center_y))
            if rng.random() > probability or (
                len(existing) and box_iou(box, existing).max() > 0.1
            ):
                continue
            with Image.open(root / donor["file_name"]) as source:
                canvas.paste(source.convert("RGB").crop(tuple(box)), (x1, y1))
            added.append(
                {
                    "category_id": ann["category_id"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "iscrowd": 0,
                }
            )
            existing.append(box)
        if not added:
            continue
        name = (
            Path("augmented") / Path(image["file_name"]).with_suffix(".png")
        ).as_posix()
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target)
        result["images"].append({**image, "id": next_image, "file_name": name})
        for ann in originals + added:
            result["annotations"].append(
                {
                    "id": next_ann,
                    "image_id": next_image,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["bbox"][2] * ann["bbox"][3],
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            next_ann += 1
        next_image += 1
        generated += 1
    write_json(Path(dataset) / "train_augmented.json", result)
    print(f"Created {generated} augmented training images; validation is unchanged")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_objects < 1:
        parser.error("--max-objects must be positive")
    augment(args.dataset, args.max_objects, args.seed)
