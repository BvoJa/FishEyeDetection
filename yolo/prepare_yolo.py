"""Prepare YOLO/COCO gold data, then merge offline annotations into training only."""

import argparse
import shutil
from pathlib import Path

import yaml

from efficient_detection.shared.common import (
    CLASSES,
    camera_scene,
    dataset_config,
    read_json,
    sha256,
    write_json,
)
from efficient_detection.shared.datasets import load_coco, load_yolo, split_records


def materialize(record, root, split, mode):
    relative = Path(record["name"])
    target = root / "images" / split / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if mode == "symlink":
        target.symlink_to(record["path"].resolve())
    else:
        shutil.copy2(record["path"], target)
    label = root / "labels" / split / relative.with_suffix(".txt")
    if label.exists():
        raise ValueError(f"Two images map to the same YOLO label filename: {label}")
    label.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for ann in record["boxes"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, width, height = ann["bbox"]
        lines.append(
            f"{ann['category_id']} {(x + width / 2) / record['width']:.9f} {(y + height / 2) / record['height']:.9f} {width / record['width']:.9f} {height / record['height']:.9f}"
        )
    label.write_text("\n".join(lines) + ("\n" if lines else ""))
    return {
        "file_name": f"{split}/{relative.as_posix()}",
        "source": str(record["path"].resolve()),
        "sha256": record.get("sha256") or sha256(record["path"]),
        "width": record["width"],
        "height": record["height"],
        "boxes": record["boxes"],
        "origin": record.get("origin", "human"),
        "camera_scene": camera_scene(relative.name),
    }


def write_annotations(root, split, records):
    data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)],
    }
    for index, record in enumerate(records, 1):
        data["images"].append(
            {
                "id": index,
                "file_name": record["file_name"],
                "width": record["width"],
                "height": record["height"],
            }
        )
        for ann in record["boxes"]:
            data["annotations"].append(
                {
                    "id": len(data["annotations"]) + 1,
                    "image_id": index,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "iscrowd": ann.get("iscrowd", 0),
                    "area": ann["bbox"][2] * ann["bbox"][3],
                }
            )
    write_json(root / "annotations" / f"{split}.json", data)


def merge_pseudo(data_path, pseudo_path, mode="symlink"):
    root, _, manifest = dataset_config(data_path)
    pseudo = read_json(Path(pseudo_path) / "annotations.json")
    if pseudo["classes"] != CLASSES:
        raise ValueError(
            "Pseudo-label class mapping does not match the prepared dataset"
        )
    human = [record for rows in manifest["splits"].values() for record in rows]
    seen = {record["sha256"] for record in human}
    heldout_cameras = {
        record["camera_scene"][0]
        for split, rows in manifest["splits"].items()
        if split != "train"
        for record in rows
        if record["camera_scene"]
    }
    accepted = []
    for record in pseudo["images"]:
        group = camera_scene(record["file_name"])
        fingerprint = sha256(record["source"])
        if fingerprint != record["sha256"]:
            raise ValueError(
                f"Pseudo source changed after annotation: {record['source']}"
            )
        if (
            fingerprint in seen
            or (group and group[0] in heldout_cameras)
            or not record["boxes"]
        ):
            continue
        accepted.append(
            {
                "path": Path(record["source"]),
                "name": "pseudo/" + record["file_name"],
                "width": record["width"],
                "height": record["height"],
                "boxes": record["boxes"],
                "sha256": fingerprint,
                "origin": "pseudo",
            }
        )
        seen.add(fingerprint)
    # Validate target names before copying anything.
    names = set()
    for record in accepted:
        key = Path(record["name"]).with_suffix("").as_posix()
        if (
            key in names
            or (
                root / "labels/train" / Path(record["name"]).with_suffix(".txt")
            ).exists()
        ):
            raise ValueError(
                f"Conflicting pseudo label name: {key}; use unique input paths"
            )
        names.add(key)
    for record in accepted:
        manifest["splits"]["train"].append(materialize(record, root, "train", mode))
    if accepted:
        manifest.setdefault("pseudo_sources", []).append(
            {
                "path": str(Path(pseudo_path).resolve()),
                "generator": pseudo["generator"],
                "added": len(accepted),
            }
        )
        write_annotations(root, "train", manifest["splits"]["train"])
        write_json(root / "manifest.json", manifest)
        for cache in (root / "labels").rglob("*.cache"):
            cache.unlink()
    print(
        f"Merged {len(accepted)} pseudo-labeled training images; human labels and held-out splits preserved"
    )
    return manifest


def prepare(args):
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty --output directory")
    if args.format == "yolo":
        if not args.labels:
            raise ValueError("YOLO input requires --labels")
        records = load_yolo(args.images, args.labels, confidence=0)
        val = (
            load_yolo([args.val_images], [args.val_labels], confidence=0)
            if args.val_images and args.val_labels
            else None
        )
        test = (
            load_yolo([args.test_images], [args.test_labels], confidence=0)
            if args.test_images and args.test_labels
            else None
        )
    else:
        if len(args.images) != 1 or not args.annotations:
            raise ValueError(
                "COCO input requires one --images directory and --annotations"
            )
        mapping = read_json(args.category_map) if args.category_map else None
        records = load_coco(args.annotations, args.images[0], category_map=mapping)
        val = (
            load_coco(args.val_annotations, args.val_images, category_map=mapping)
            if args.val_images and args.val_annotations
            else None
        )
        test = (
            load_coco(args.test_annotations, args.test_images, category_map=mapping)
            if args.test_images and args.test_annotations
            else None
        )
    if args.val_images and val is None or args.test_images and test is None:
        raise ValueError(
            "Provide labels/annotations for each explicit validation/test image directory"
        )
    train, val = (
        (records, val)
        if val is not None
        else split_records(
            records, args.split, args.val_fraction, args.seed, args.val_cameras
        )
    )
    splits = {"train": train, "val": val}
    if test is not None:
        splits["test"] = test
    seen = {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty split: {split}")
        names = set()
        for record in rows:
            record["sha256"] = sha256(record["path"])
            if record["sha256"] in seen:
                raise ValueError(
                    f"Duplicate image content across dataset entries: {record['path']} ({seen[record['sha256']]}, {split})"
                )
            seen[record["sha256"]] = split
            key = Path(record["name"]).with_suffix("").as_posix()
            if key in names:
                raise ValueError(f"YOLO label filename collision: {key}")
            names.add(key)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"classes": CLASSES, "splits": {}}
    for split, rows in splits.items():
        manifest["splits"][split] = [
            materialize(record, root, split, args.mode) for record in rows
        ]
        write_annotations(root, split, manifest["splits"][split])
    write_json(root / "manifest.json", manifest)
    cfg = {
        "train": "images/train",
        "val": "images/val",
        "names": CLASSES,
    }
    if test is not None:
        cfg["test"] = "images/test"
    (root / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(
        f"Prepared {len(train)} training and {len(val)} validation images: {root / 'data.yaml'}"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["yolo", "coco"])
    parser.add_argument("--images", type=Path, nargs="+")
    parser.add_argument("--labels", type=Path, nargs="+")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-labels", type=Path)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--test-images", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--test-annotations", type=Path)
    parser.add_argument("--category-map", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=["camera", "random"], default="camera")
    parser.add_argument("--val-cameras", type=int, nargs="+")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument(
        "--data", type=Path, help="Existing data.yaml when merging pseudo labels"
    )
    parser.add_argument("--add-pseudo", type=Path)
    args = parser.parse_args()
    if args.add_pseudo:
        if not args.data:
            parser.error("--add-pseudo requires --data")
        merge_pseudo(args.data, args.add_pseudo, args.mode)
    else:
        if not args.format or not args.images or not args.output:
            parser.error("Initial preparation requires --format, --images and --output")
        prepare(args)


if __name__ == "__main__":
    main()
