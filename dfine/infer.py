"""Standalone local inference: ONNX Runtime, NumPy and Pillow only."""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from efficient_detection.shared.common import (
    CLASSES,
    EXTENSIONS,
    camera_scene,
    class_nms,
    read_json,
    write_json,
)


def preprocess(image, image_size):
    image = image.convert("RGB")
    original_size = np.asarray([image.size], dtype=np.int64)
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]), original_size


def postprocess(
    labels,
    boxes,
    scores,
    width,
    height,
    confidence=0.5,
    nms_iou=0.6,
    heuristics=False,
    filename="",
):
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    scores = np.asarray(scores, dtype=np.float32).reshape(-1).copy()
    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(scores)
        & (labels >= 0)
        & (labels < len(CLASSES))
    )
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    if heuristics:
        scene = camera_scene(filename)
        if scene and scene[1] in {"E", "N"}:
            scores += 0.1
        scores[area < 64**2] += 0.02
        scores[area > 192**2] -= 0.02
    scores = np.clip(scores, 0, 1)
    valid = (scores >= confidence) & (area > 0)
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    keep = (
        class_nms(boxes, scores, labels, nms_iou)
        if nms_iou is not None
        else np.argsort(-scores)
    )
    return labels[keep], boxes[keep], scores[keep]


def run(
    model_path,
    input_path,
    output,
    confidence=0.5,
    nms_iou=0.6,
    heuristics=False,
    threads=4,
    annotations=None,
    save_images=True,
):
    import onnxruntime as ort

    model_path, input_path, output = Path(model_path), Path(input_path), Path(output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output.resolve() == input_path.resolve() or (
        input_path.is_dir() and output.resolve().is_relative_to(input_path.resolve())
    ):
        raise ValueError("Choose an output directory outside the input directory")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {"images", "orig_target_sizes"}:
        raise ValueError(
            "Expected an ONNX model exported by this folder's export_onnx.py"
        )
    shape = inputs["images"].shape
    if (
        len(shape) != 4
        or shape[:2] != [1, 3]
        or not isinstance(shape[2], int)
        or shape[2] != shape[3]
    ):
        raise ValueError(f"Expected a static [1, 3, size, size] model; got {shape}")
    if (
        inputs["images"].type != "tensor(float)"
        or inputs["orig_target_sizes"].type != "tensor(int64)"
    ):
        raise ValueError("Expected float32 images and int64 orig_target_sizes")
    image_size = shape[2]
    sidecar = model_path.with_suffix(".json")
    if sidecar.is_file():
        metadata = read_json(sidecar)
        if metadata["classes"] != CLASSES or metadata["size_order"] != "width_height":
            raise ValueError(
                "Model metadata does not match the expected class/coordinate conventions"
            )
    paths = (
        sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        if input_path.is_dir()
        else [input_path]
    )
    image_ids = {}
    if annotations:
        for image in read_json(annotations)["images"]:
            image_ids[image["file_name"]] = image["id"]
        if input_path.is_dir():
            paths = [input_path / name for name in sorted(image_ids)]
    if not paths:
        raise ValueError("No supported images found")
    output.mkdir(parents=True, exist_ok=True)
    predictions, manifest = [], []
    inference_seconds = 0.0
    started = time.perf_counter()
    for index, path in enumerate(paths, 1):
        relative = (
            path.relative_to(input_path) if input_path.is_dir() else Path(path.name)
        )
        with Image.open(path) as source:
            image = source.convert("RGB")
        pixels, size = preprocess(image, image_size)
        feed = {"images": pixels, "orig_target_sizes": size}
        if index == 1:
            session.run(None, feed)  # Warm up outside inference timing.
        t0 = time.perf_counter()
        labels, boxes, scores = session.run(["labels", "boxes", "scores"], feed)
        inference_seconds += time.perf_counter() - t0
        labels, boxes, scores = postprocess(
            labels,
            boxes,
            scores,
            *image.size,
            confidence,
            nms_iou,
            heuristics,
            path.name,
        )
        image_id = image_ids.get(relative.as_posix())
        if annotations and image_id is None:
            raise ValueError(
                f"{relative}: not found in --annotations; input must be the COCO images root"
            )
        image_id = index if image_id is None else image_id
        manifest.append(
            {
                "id": image_id,
                "file_name": relative.as_posix(),
                "width": image.width,
                "height": image.height,
            }
        )
        draw = ImageDraw.Draw(image)
        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = map(float, box)
            predictions.append(
                {
                    "image_id": image_id,
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
            name = (
                relative
                if relative.suffix.lower() == ".png"
                else relative.with_name(relative.name + ".png")
            )
            target = output / "images" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
        print(f"{index}/{len(paths)} {relative}: {len(labels)} detections", flush=True)
    total_seconds = time.perf_counter() - started
    write_json(output / "predictions.json", predictions)
    write_json(output / "images.json", manifest)
    summary = {
        "images": len(paths),
        "inference_seconds": inference_seconds,
        "inference_fps": len(paths) / max(inference_seconds, 1e-8),
        "pipeline_seconds": total_seconds,
        "provider": "CPUExecutionProvider",
        "confidence": confidence,
        "nms_iou": nms_iou,
        "heuristics": heuristics,
    }
    write_json(output / "summary.json", summary)
    print(f"CPU inference: {summary['inference_fps']:.2f} FPS; results: {output}")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, required=True, help="An image or a directory (recursive)"
    )
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--no-nms", action="store_true")
    parser.add_argument(
        "--heuristics",
        action="store_true",
        help="Opt in to the original night/small/large score adjustments; tune on validation data",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--annotations", type=Path, help="COCO ground truth JSON for matching image IDs"
    )
    parser.add_argument("--no-images", action="store_true", help="Write JSON only")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or not 0 <= args.nms_iou <= 1 or args.threads < 1:
        parser.error("Confidence/NMS must be in [0,1] and threads must be positive")
    run(
        args.model,
        args.input,
        args.output,
        args.confidence,
        None if args.no_nms else args.nms_iou,
        args.heuristics,
        args.threads,
        args.annotations,
        not args.no_images,
    )
