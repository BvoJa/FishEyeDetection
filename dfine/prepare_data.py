"""Prepare five-class COCO train/validation splits for the legacy D-FINE workflow."""

import argparse
import shutil
from pathlib import Path

from efficient_detection.shared.common import CLASSES, read_json, write_json
from efficient_detection.shared.datasets import load_coco, load_yolo, split_records


def write_split(records, output, split, mode):
    data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)],
    }
    for image_id, record in enumerate(records, 1):
        target = output / "images" / record["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            target.symlink_to(record["path"])
        else:
            shutil.copy2(record["path"], target)
        data["images"].append(
            {
                "id": image_id,
                "file_name": record["name"],
                "width": record["width"],
                "height": record["height"],
            }
        )
        for box in record["boxes"]:
            data["annotations"].append(
                {
                    **box,
                    "id": len(data["annotations"]) + 1,
                    "image_id": image_id,
                    "area": box["bbox"][2] * box["bbox"][3],
                }
            )
    write_json(output / f"{split}.json", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["yolo", "coco"], required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, nargs="+")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-labels", type=Path)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument(
        "--category-map", type=Path, help='JSON, e.g. {"1": 3, "3": 2, "6": 0, "8": 4}'
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["camera", "random"], default="camera")
    parser.add_argument("--val-cameras", type=int, nargs="+")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pseudo-confidence", type=float, default=0.5)
    parser.add_argument("--allow-missing-labels", action="store_true")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    args = parser.parse_args()
    if not 0 <= args.pseudo_confidence <= 1:
        parser.error("--pseudo-confidence must be between 0 and 1")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(
            "Output must be empty to avoid mixing datasets; choose a new directory"
        )
    if args.format == "yolo":
        if not args.labels:
            parser.error("YOLO input requires --labels")
        records = load_yolo(
            args.images, args.labels, args.pseudo_confidence, args.allow_missing_labels
        )
        if args.val_images:
            if not args.val_labels:
                parser.error("--val-images requires --val-labels")
            val = load_yolo(
                [args.val_images],
                [args.val_labels],
                args.pseudo_confidence,
                args.allow_missing_labels,
            )
            for record in val:
                record["name"] = "validation/" + record["name"]
    else:
        if len(args.images) != 1 or not args.annotations:
            parser.error("COCO input requires one --images directory and --annotations")
        mapping = read_json(args.category_map) if args.category_map else None
        records = load_coco(args.annotations, args.images[0], category_map=mapping)
        if args.val_images:
            if not args.val_annotations:
                parser.error("--val-images requires --val-annotations")
            val = load_coco(
                args.val_annotations, args.val_images, "validation", mapping
            )
    if args.val_images:
        train = records
        if not train or not val:
            parser.error("Training and validation must be nonempty")
        if {record["path"] for record in train} & {record["path"] for record in val}:
            parser.error("The same image appears in training and validation")
    else:
        train, val = split_records(
            records, args.split, args.val_fraction, args.seed, args.val_cameras
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_split(train, args.output, "train", args.mode)
    write_split(val, args.output, "val", args.mode)
    write_json(args.output / "classes.json", CLASSES)
    print(
        f"Prepared {len(train)} training and {len(val)} validation images in {args.output}"
    )


if __name__ == "__main__":
    main()
