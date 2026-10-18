"""Offline D-FINE-S/M/L/X annotation with held-out image/camera exclusion."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from efficient_detection.dfine.annotator import load_annotator
from efficient_detection.dfine.config import DEFAULT_REPO
from efficient_detection.shared.common import (
    CLASSES,
    EXTENSIONS,
    camera_scene,
    dataset_config,
    sha256,
    write_json,
)

COCO_TO_FISHEYE = {5: 0, 1: 1, 3: 1, 2: 2, 0: 3, 7: 4}


def map_detections(
    labels, boxes, scores, width, height, checkpoint_classes=80, confidence=0.5
):
    result = []
    for label, box, score in zip(labels, boxes, scores):
        category = (
            COCO_TO_FISHEYE.get(int(label)) if checkpoint_classes == 80 else int(label)
        )
        if (
            category not in range(5)
            or not np.isfinite(score)
            or score < confidence
            or not np.isfinite(box).all()
        ):
            continue
        x1, y1, x2, y2 = map(float, box)
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 > x1 and y2 > y1:
            result.append(
                {
                    "category_id": category,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(np.clip(score, 0, 1)),
                    "iscrowd": 0,
                }
            )
    return result


def candidates(images, manifest):
    seen = {row["sha256"] for rows in manifest["splits"].values() for row in rows}
    heldout = {
        row["camera_scene"][0]
        for split, rows in manifest["splits"].items()
        if split != "train"
        for row in rows
        if row["camera_scene"]
    }
    for path in sorted(Path(images).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        group = camera_scene(path.name)
        if group and group[0] in heldout:
            continue
        fingerprint = sha256(path)
        if fingerprint not in seen:
            seen.add(fingerprint)
            yield path, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-model",
        choices=["dfine-s", "dfine-m", "dfine-l", "dfine-x"],
        default="dfine-s",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-classes", type=int, choices=[5, 80], default=80)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Prepare gold splits first to exclude human/held-out images",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("Confidence must be between 0 and 1")
    if not args.images.is_dir():
        parser.error("--images must be an image directory")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    _, _, manifest = dataset_config(args.data)
    import torch

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)
    model, postprocessor = load_annotator(
        args.checkpoint,
        args.annotation_model,
        args.checkpoint_classes,
        args.image_size,
        device,
        args.repo_root,
    )
    output = {
        "classes": CLASSES,
        "generator": {
            "model": args.annotation_model,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_classes": args.checkpoint_classes,
            "confidence": args.confidence,
            "image_size": args.image_size,
        },
        "images": [],
    }
    for path, fingerprint in candidates(args.images, manifest):
        with Image.open(path) as source:
            image = source.convert("RGB")
        pixels = (
            np.asarray(
                image.resize(
                    (args.image_size, args.image_size), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            / 255
        )
        tensor = torch.from_numpy(
            np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        ).to(device)
        size = torch.tensor([image.size], dtype=torch.int64, device=device)
        with torch.inference_mode():
            prediction = postprocessor(model(tensor), size)[0]
        boxes = map_detections(
            prediction["labels"].cpu().numpy(),
            prediction["boxes"].cpu().numpy(),
            prediction["scores"].cpu().numpy(),
            *image.size,
            args.checkpoint_classes,
            args.confidence,
        )
        if not boxes:
            continue
        relative = path.relative_to(args.images)
        label = args.output / "labels" / relative.with_suffix(".txt")
        if label.exists():
            raise ValueError(f"Pseudo label filename collision: {label}")
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for ann in boxes:
            x, y, w, h = ann["bbox"]
            lines.append(
                f"{ann['category_id']} {(x + w / 2) / image.width:.9f} {(y + h / 2) / image.height:.9f} {w / image.width:.9f} {h / image.height:.9f}"
            )
        label.write_text("\n".join(lines) + "\n")
        output["images"].append(
            {
                "source": str(path.resolve()),
                "file_name": relative.as_posix(),
                "sha256": fingerprint,
                "width": image.width,
                "height": image.height,
                "boxes": boxes,
            }
        )
        print(f"{relative}: {len(boxes)} accepted labels", flush=True)
    write_json(args.output / "annotations.json", output)
    print(
        f"Saved {len(output['images'])} pseudo-labeled images; no-detection images excluded"
    )


if __name__ == "__main__":
    main()
