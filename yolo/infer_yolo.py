"""Run a five-class YOLO11 ONNX model locally on CPU without PyTorch."""

import argparse
import time
from pathlib import Path

from PIL import Image, ImageDraw

from efficient_detection.shared.common import (
    CLASSES,
    EXTENSIONS,
    coco_detection,
    read_json,
    resolve_image,
    write_json,
)
from efficient_detection.shared.performance import (
    FPS_SCOPES,
    FPS_STAGES,
    throughput_fps,
    timing_statistics,
)
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

    def image_path(record):
        return resolve_image(images, record["file_name"]) if images.is_dir() else images

    session = YoloSession(model, threads, optimization)
    with Image.open(image_path(records[0])) as source:
        warmup_image = source.convert("RGB")
    for _ in range(3):
        session.predict(warmup_image, confidence)
    del warmup_image
    predictions, manifest, timings = [], [], []
    dataset_started = time.perf_counter()
    for index, record in enumerate(records, 1):
        request_started = time.perf_counter()
        relative = Path(record["file_name"])
        with Image.open(image_path(record)) as source:
            image = source.convert("RGB")
        result = session.predict(image, confidence)
        request_ms = (time.perf_counter() - request_started) * 1000
        timings.append({**result["timing_ms"], "request": request_ms})
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
            predictions.append(coco_detection(record["id"], label, box, score))
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
    dataset_seconds = time.perf_counter() - dataset_started
    summary = {
        "images": len(records),
        "threads": threads,
        "graph_optimization": optimization,
        "provider": "CPUExecutionProvider",
        "confidence": confidence,
        "warmup": 3,
        "batch_size": 1,
        "dataset_pass_seconds": dataset_seconds,
        "dataset_fps": len(records) / dataset_seconds,
        "save_images": save_images,
        **{
            f"{stage}_fps": throughput_fps(row[stage] for row in timings)
            for stage in FPS_STAGES
        },
        "fps_scope": {
            **FPS_SCOPES,
            "dataset_fps": "Wall-clock image loop including prediction formatting, progress logging and optional image drawing/writing; excludes startup, warm-up and final JSON writing",
        },
        "timing_scope": "Pipeline covers RGB preprocessing + inference + NMS; request additionally includes image reading/decoding; both exclude warm-up and file writing",
    }
    for stage in timings[0]:
        stats = timing_statistics([row[stage] for row in timings])
        for statistic in ["mean_ms", "p50_ms", "p95_ms"]:
            summary[f"{stage}_{statistic}"] = stats[statistic]
    write_json(output / "predictions.json", predictions)
    write_json(output / "images.json", manifest)
    write_json(output / "summary.json", summary)
    print(
        f"FPS: inference={summary['inference_fps']:.2f}, pipeline={summary['pipeline_fps']:.2f}, request={summary['request_fps']:.2f}, dataset pass={summary['dataset_fps']:.2f}",
        flush=True,
    )
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
