"""Run a five-class YOLO11 ONNX model locally on CPU without PyTorch."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from efficient_detection.shared.common import CLASSES, EXTENSIONS, read_json, write_json
from efficient_detection.yolo.runtime import YoloSession


def run(
    model,
    images,
    output,
    threads=4,
    confidence=0.25,
    annotations=None,
    save_images=True,
    optimization="all",
):
    images, output = Path(images), Path(output)
    if not images.exists():
        raise FileNotFoundError(images)
    if images.is_dir() and output.resolve().is_relative_to(images.resolve()):
        raise ValueError("Choose an output outside the input images directory")
    records = read_json(annotations)["images"] if annotations else None
    if records is not None and not images.is_dir():
        raise ValueError(
            "With --annotations, --input must be the annotation image root"
        )
    if records is None:
        paths = (
            sorted(
                path
                for path in images.rglob("*")
                if path.is_file() and path.suffix.lower() in EXTENSIONS
            )
            if images.is_dir()
            else [images]
        )
        records = [
            {
                "id": i,
                "file_name": (
                    path.relative_to(images) if images.is_dir() else Path(path.name)
                ).as_posix(),
            }
            for i, path in enumerate(paths, 1)
        ]
    if not records:
        raise ValueError("No input images")
    session = YoloSession(model, threads, optimization)
    predictions, manifest, timings = [], [], []
    for index, record in enumerate(records, 1):
        relative = Path(record["file_name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Image names must be relative paths within the image root")
        path = images / relative if images.is_dir() else images
        with Image.open(path) as source:
            image = source.convert("RGB")
        if index == 1:
            for _ in range(3):
                session.predict(image, confidence)
        result = session.predict(image, confidence)
        timings.append(result["timing_ms"])
        manifest.append(
            {
                "id": record["id"],
                "file_name": relative.as_posix(),
                "width": image.width,
                "height": image.height,
            }
        )
        draw = ImageDraw.Draw(image)
        for label, box, score in zip(
            result["labels"], result["boxes"], result["scores"]
        ):
            x1, y1, x2, y2 = map(float, box)
            predictions.append(
                {
                    "image_id": record["id"],
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
            if save_images:
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                draw.text(
                    (x1, max(0, y1 - 12)),
                    f"{CLASSES[int(label)]} {score:.2f}",
                    fill="lime",
                )
        if save_images:
            target = output / "images" / relative.with_name(relative.name + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
        print(
            f"{index}/{len(records)} {relative}: {len(result['labels'])} detections",
            flush=True,
        )
    summary = {
        "images": len(records),
        "threads": threads,
        "graph_optimization": optimization,
        "provider": "CPUExecutionProvider",
        "confidence": confidence,
        "timing_scope": "RGB preprocessing + inference + NMS; excludes image I/O and warm-up",
    }
    for stage in timings[0]:
        values = [row[stage] for row in timings]
        summary[stage + "_p50_ms"] = float(np.percentile(values, 50))
        summary[stage + "_p95_ms"] = float(np.percentile(values, 95))
    write_json(output / "predictions.json", predictions)
    write_json(output / "images.json", manifest)
    write_json(output / "summary.json", summary)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--graph-optimization", choices=["off", "all"], default="all")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("Confidence must be in [0,1]")
    run(
        args.model,
        args.input,
        args.output,
        args.threads,
        args.confidence,
        args.annotations,
        not args.no_images,
        args.graph_optimization,
    )
