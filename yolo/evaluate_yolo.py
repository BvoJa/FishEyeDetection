"""Evaluate one ONNX model on every COCO image and report CPU inference statistics."""

import argparse
import csv
import os
import platform
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
from PIL import Image, ImageDraw

from efficient_detection.shared.common import (
    CLASSES,
    coco_detection,
    read_json,
    resolve_image,
    sha256,
    validate_coco,
    write_json,
)
from efficient_detection.shared.evaluation import evaluate_predictions
from efficient_detection.shared.metrics import detection_counts, metrics_from_counts
from efficient_detection.shared.performance import (
    FPS_SCOPES,
    FPS_STAGES,
    STAGES,
    cpu_name,
    peak_memory,
    throughput_fps,
    timing_statistics,
    usable_threads,
)
from efficient_detection.yolo.runtime import YoloSession, decode


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary, per_class):
    def percent(value):
        return "n/a" if value is None or value < 0 else f"{value * 100:.2f}"

    metrics = summary["accuracy"]
    fixed = summary["detection_metrics"]
    micro = fixed["micro"]
    lines = [
        "# FishEye8K CPU evaluation",
        "",
        f"Model: `{summary['model']}`; FP32 batch 1; {summary['image_size']} px; {summary['threads']} CPU threads.",
        f"Processed **{summary['images']} / {summary['expected_images']} images**; deployment detections: **{summary['detections']}**.",
        "",
        "| AP50:95 (%) | AP50 (%) | AP75 (%) | AP small (%) |",
        "|---:|---:|---:|---:|",
        f"| {percent(metrics['AP'])} | {percent(metrics['AP50'])} | {percent(metrics['AP75'])} | {percent(metrics['AP_small'])} |",
        "",
        f"Fixed-threshold detection metrics: confidence **{summary['confidence']}**, matching IoU **{fixed['match_iou']}**, same-class score-ordered one-to-one matching.",
        "",
        "| Precision (%) | Recall (%) | F1 micro (%) | F1 macro (%) | Detection accuracy (%) |",
        "|---:|---:|---:|---:|---:|",
        f"| {percent(micro['precision'])} | {percent(micro['recall'])} | {percent(micro['f1'])} | {percent(fixed['macro_f1'])} | {percent(micro['detection_accuracy'])} |",
        "",
        f"TP: **{micro['tp']}**; FP: **{micro['fp']}**; FN: **{micro['fn']}**; ignored predictions: **{micro['ignored_detections']}**.",
        "Detection accuracy is explicitly defined as TP/(TP+FP+FN), a Jaccard-style object detection score. It is not classification accuracy; true negatives are undefined here. F1 = 2TP/(2TP+FP+FN).",
        "",
        "| Stage | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, measured in summary["timing"].items():
        lines.append(
            f"| {name} | {measured['mean_ms']:.2f} | {measured['p50_ms']:.2f} | {measured['p95_ms']:.2f} | {measured['p99_ms']:.2f} |"
        )
    lines += [
        "",
        "| Throughput | FPS (images/s) | Measurement scope |",
        "|---|---:|---|",
        f"| Inference | {summary['inference_fps']:.2f} | {FPS_SCOPES['inference_fps']} |",
        f"| Pipeline | {summary['pipeline_fps']:.2f} | {FPS_SCOPES['pipeline_fps']} |",
        f"| Request | {summary['request_fps']:.2f} | {FPS_SCOPES['request_fps']} |",
        f"| Dataset pass | {summary['dataset_fps']:.2f} | Full image loop, including metric matching, extra AP decoding and optional image saving |",
        "",
        f"Model size: {summary['model_size_mib']:.2f} MiB; parameters: {summary['parameters']:,}; session startup: {summary['startup_ms']:.2f} ms.",
        f"Peak inference-pass RSS: {summary['peak_inference_rss_mib']:.2f} MiB; peak including COCO evaluation: {summary['peak_process_rss_mib']:.2f} MiB.",
        "",
        "| Class | GT objects | Detections | AP50:95 (%) | AP50 (%) | Precision (%) | Recall (%) | F1 (%) | Detection accuracy (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_class:
        lines.append(
            f"| {row['class']} | {row['ground_truth_objects']} | {row['detections']} | {percent(row['AP'])} | {percent(row['AP50'])} | {percent(row['precision'])} | {percent(row['recall'])} | {percent(row['f1'])} | {percent(row['detection_accuracy'])} |"
        )
    lines += [
        "",
        "## Measurement scope",
        "",
        "- Every annotated image is measured once after warm-up; this is not a repeated-image microbenchmark.",
        "- Pipeline timing covers RGB letterbox/normalization, ONNX inference, coordinate restoration and class-aware NMS. It excludes image decoding, file writing, warm-up, startup and COCO evaluation.",
        "- Request timing adds image reading/decoding to the pipeline. Additional AP decoding and output writing are excluded.",
        "- FPS = image count / total measured seconds, equivalent to 1000 / mean latency for inference/pipeline/request. Per-image FPS values are not averaged. All measurements use batch 1 and serial image processing.",
        "- Dataset-pass FPS uses elapsed wall time after warm-up; it includes deployment prediction formatting, metric matching, extra AP processing, progress logging and optional annotated-image writing. It excludes session startup, warm-up, final COCO evaluation and report/JSON/CSV writing; it is not deployment-only FPS.",
        f"- Detection counts use confidence {summary['confidence']}; AP uses confidence {summary['ap_confidence']}, multi-label decoding and standard COCO evaluation (maxDets=100). Both reuse the same model forward pass.",
        "- Precision, recall, F1 and detection accuracy use single-label deployment predictions (max 300/image), separately from low-confidence multi-label COCO AP predictions. Matching and metric computation are outside deployment latency.",
        "- Ignored/crowd ground truths do not contribute FN; predictions matched to those regions are ignored. Macro F1 averages classes with nonignored ground truth or predictions. Zero denominators are reported as zero.",
        "- RSS includes Python, loaded annotations, retained predictions and ONNX Runtime; it is not model-only memory. Sampling interval: 5 ms.",
        "- Results describe this machine, image and model. Containerization is not itself a model acceleration technique.",
        "- If this original FishEye8K test split was used as validation during training, report these numbers as validation results; do not claim an untouched test set.",
        "",
    ]
    if summary.get("fps_statistics_source"):
        lines += [summary["fps_statistics_source"], ""]
    Path(path).write_text("\n".join(lines))


def run(
    model,
    images,
    annotations,
    output,
    threads=4,
    confidence=0.25,
    nms_iou=0.7,
    warmup=20,
    ap_confidence=0.001,
    save_images=False,
    match_iou=0.5,
):
    model, images, annotations, output = map(Path, (model, images, annotations, output))
    if threads < 1 or warmup < 0 or not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError(
            "Use positive threads, nonnegative warm-up and thresholds in [0,1]"
        )
    if not 0 < ap_confidence <= 0.01:
        raise ValueError(
            "AP confidence must be in (0, 0.01] to retain the precision-recall curve"
        )
    if not 0 < match_iou <= 1:
        raise ValueError("Matching IoU must be in (0,1]")
    if output.resolve().is_relative_to(images.resolve()):
        raise ValueError("Choose an output outside the input images directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Choose an empty --output directory to avoid mixing evaluation runs"
        )
    source = read_json(annotations)
    for record in source["images"]:
        resolve_image(images, record["file_name"])
    source = validate_coco(annotations, images)
    if not any(not row.get("iscrowd", 0) for row in source["annotations"]):
        raise ValueError(
            "AP evaluation needs at least one non-crowd ground-truth object"
        )
    requested_threads = threads
    threads, physical, available = usable_threads(threads)
    output.mkdir(parents=True, exist_ok=True)
    predictions, deployment_predictions, rows = [], [], []
    counts_by_class = np.zeros((len(CLASSES), 4), dtype=np.int64)
    annotations_by_image = {}
    for annotation in source["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
    totals = Counter()
    empty_images = 0
    with peak_memory(psutil.Process()) as peak:
        started = time.perf_counter()
        session = YoloSession(model, threads)
        startup_ms = (time.perf_counter() - started) * 1000
        with Image.open(images / source["images"][0]["file_name"]) as image:
            first = image.convert("RGB")
        for _ in range(warmup):
            session.predict(first, confidence, nms_iou)
        del first
        dataset_started = time.perf_counter()
        for index, record in enumerate(source["images"], 1):
            request_started = time.perf_counter()
            with Image.open(images / record["file_name"]) as source_image:
                image = source_image.convert("RGB")
            decoded = time.perf_counter()
            if image.size != (record["width"], record["height"]):
                raise ValueError(
                    f"Image dimensions differ from COCO JSON: {record['file_name']}"
                )
            result = session.predict(image, confidence, nms_iou, return_raw=True)
            completed = time.perf_counter()
            counts = Counter(map(int, result["labels"]))
            totals.update(counts)
            empty_images += not len(result["labels"])
            matched = detection_counts(
                annotations_by_image.get(record["id"], []),
                result["boxes"],
                result["scores"],
                result["labels"],
                match_iou,
            )
            counts_by_class += matched
            rows.append(
                {
                    "image_id": record["id"],
                    "file_name": record["file_name"],
                    "width": image.width,
                    "height": image.height,
                    "detections": len(result["labels"]),
                    **metrics_from_counts(matched.sum(axis=0)),
                    **{
                        f"detections_{name}": counts[category]
                        for category, name in enumerate(CLASSES)
                    },
                    **{
                        f"{name}_ms": value
                        for name, value in result["timing_ms"].items()
                    },
                    "image_decode_ms": (decoded - request_started) * 1000,
                    "request_ms": (completed - request_started) * 1000,
                }
            )
            for label, box, score in zip(
                result["labels"], result["boxes"], result["scores"]
            ):
                deployment_predictions.append(
                    coco_detection(record["id"], label, box, score)
                )
            # Reuse the raw output; AP decoding is outside deployment latency.
            ap_result = decode(
                result["raw"],
                result["geometry"],
                ap_confidence,
                nms_iou,
                multi_label=True,
            )
            for label, box, score in zip(
                ap_result["labels"], ap_result["boxes"], ap_result["scores"]
            ):
                predictions.append(coco_detection(record["id"], label, box, score))
            if save_images:
                draw = ImageDraw.Draw(image)
                for label, box, score in zip(
                    result["labels"], result["boxes"], result["scores"]
                ):
                    coords = list(map(float, box))
                    draw.rectangle(coords, outline="lime", width=2)
                    draw.text(
                        (coords[0], max(0, coords[1] - 12)),
                        f"{CLASSES[int(label)]} {score:.2f}",
                        fill="lime",
                    )
                target = (
                    output / "images" / Path(record["file_name"]).with_suffix(".png")
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target)
            if index % 100 == 0 or index == len(source["images"]):
                print(
                    f"Processed {index}/{len(source['images'])}: {record['file_name']}",
                    flush=True,
                )
        dataset_seconds = time.perf_counter() - dataset_started
        inference_peak = peak[0]
        print("Computing COCO accuracy on all images...", flush=True)
        metrics, per_class = evaluate_predictions(annotations, predictions)
    truth_counts = Counter(int(row["category_id"]) for row in source["annotations"])
    for row in per_class:
        row["ground_truth_objects"] = truth_counts[row["category_id"]]
        row["detections"] = totals[row["category_id"]]
        row.update(metrics_from_counts(counts_by_class[row["category_id"]]))
    active = counts_by_class[:, :3].sum(axis=1) > 0
    fixed_metrics = {
        "confidence": confidence,
        "match_iou": match_iou,
        "matching": "Same class, score descending, best unused ground truth; crowd/ignored matches excluded",
        "micro": metrics_from_counts(counts_by_class.sum(axis=0)),
        "macro_f1": float(
            np.mean([row["f1"] for row, present in zip(per_class, active) if present])
        )
        if np.any(active)
        else 0.0,
        "detection_accuracy_definition": "TP / (TP + FP + FN); Jaccard-style detection score, not classification accuracy; no true-negative count",
        "zero_division": 0,
    }
    timing = {
        name: timing_statistics([row[f"{name}_ms"] for row in rows])
        for name in STAGES
    }
    metadata = session.metadata
    summary = {
        "model": str(model.resolve()),
        "onnx_sha256": sha256(model),
        "annotations_sha256": sha256(annotations),
        "images": len(rows),
        "expected_images": len(source["images"]),
        "ground_truth_objects": len(source["annotations"]),
        "detections": sum(totals.values()),
        "images_without_detections": int(empty_images),
        "detections_by_class": {
            name: totals[index] for index, name in enumerate(CLASSES)
        },
        "image_size": metadata["image_size"],
        "precision": metadata["precision"],
        "batch_size": 1,
        "parameters": metadata.get("parameters", 0),
        "model_size_mib": model.stat().st_size / 1024**2,
        "provider": "CPUExecutionProvider",
        "requested_threads": requested_threads,
        "threads": threads,
        "graph_optimization": "all",
        "confidence": confidence,
        "ap_confidence": ap_confidence,
        "nms_iou": nms_iou,
        "warmup": warmup,
        "save_images": save_images,
        "timed_iterations": len(rows),
        "startup_ms": startup_ms,
        "dataset_pass_seconds": dataset_seconds,
        "dataset_images_per_second": len(rows) / dataset_seconds,
        "dataset_fps": len(rows) / dataset_seconds,
        **{
            f"{stage}_fps": throughput_fps(row[f"{stage}_ms"] for row in rows)
            for stage in FPS_STAGES
        },
        "fps_scope": {
            **FPS_SCOPES,
            "dataset_fps": "Wall-clock image loop including prediction formatting, fixed-threshold matching, extra AP decoding, progress logging and optional image saving; excludes startup, warm-up, final COCO evaluation and report writing",
        },
        "peak_inference_rss_mib": inference_peak / 1024**2,
        "peak_process_rss_mib": peak[0] / 1024**2,
        "accuracy": metrics,
        "detection_metrics": fixed_metrics,
        "timing": timing,
        "timing_scope": "One timed prediction per COCO image after warm-up; pipeline excludes image I/O, startup, extra AP decoding and evaluation; request adds image decoding",
        "memory_scope": "Sampled process RSS, including annotations and retained predictions; total peak includes COCO evaluation; output serialization excluded",
        "dataset_note": "Original FishEye8K test; if used for validation during training, these are validation results",
    }
    environment = {
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "physical_cores": physical,
        "logical_cores": os.cpu_count(),
        "available_logical_cpus": available,
        "python": platform.python_version(),
        "runtime": os.environ.get("EFFICIENT_DETECTION_RUNTIME", "native"),
        "versions": {
            name: version(name)
            for name in ["numpy", "Pillow", "onnxruntime", "psutil", "faster-coco-eval"]
        },
    }
    write_json(output / "predictions.json", predictions)
    write_json(output / "deployment_predictions.json", deployment_predictions)
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment)
    write_csv(output / "per_image.csv", rows)
    write_csv(output / "per_class.csv", per_class)
    write_report(output / "report.md", summary, per_class)
    print(
        f"AP50:95={metrics['AP'] * 100:.2f}%; pipeline p50={timing['pipeline']['p50_ms']:.2f} ms, p95={timing['pipeline']['p95_ms']:.2f} ms",
        flush=True,
    )
    print(
        f"FPS: inference={summary['inference_fps']:.2f}, pipeline={summary['pipeline_fps']:.2f}, request={summary['request_fps']:.2f}, dataset pass={summary['dataset_fps']:.2f}",
        flush=True,
    )
    print(f"Full report: {output / 'report.md'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--ap-confidence", type=float, default=0.001)
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU for fixed-threshold precision/recall/F1/Detection accuracy; distinct from --nms-iou",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
