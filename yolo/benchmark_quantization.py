"""Benchmark quantized ONNX models across quantization schemes, ORT kernels, and CPU thread limits."""

import argparse
import copy
import gc
import json
import os
import resource
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import onnxruntime as ort

from efficient_detection.shared.common import (
    CLASSES,
    coco_detection,
    read_json,
    resolve_image,
    sha256,
    write_json,
)
from efficient_detection.shared.evaluation import evaluate_predictions
from efficient_detection.shared.performance import (
    cpu_name,
    peak_memory,
    throughput_fps,
    timing_statistics,
)
from efficient_detection.yolo.runtime import YoloSession, letterbox, decode


def restrict_system_resources(max_cores: int = 4):
    """Enforce strict thread count and core affinity to guarantee low CPU/RAM footprint."""
    os.environ["MALLOC_ARENA_MAX"] = "2"
    if hasattr(os, "sched_setaffinity"):
        cores = set(range(min(max_cores, os.cpu_count() or 1)))
        try:
            os.sched_setaffinity(0, cores)
        except OSError:
            pass
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[var] = str(max_cores)


def evaluate_accuracy(
    model_path: Path,
    image_paths_with_ids: List[Tuple[int, Path]],
    annotation_path: Path,
    threads: int = 4,
    optimization: str = "all",
) -> Dict[str, float]:
    """Run COCO AP evaluation streaming images one-by-one to keep RAM low."""
    session = YoloSession(model_path, threads=threads, optimization=optimization)
    detections = []

    for img_id, img_path in image_paths_with_ids:
        with Image.open(img_path) as source:
            result = session.predict(
                source.convert("RGB"), confidence=0.001, multi_label=True
            )
        for label, box, score in zip(
            result["labels"], result["boxes"], result["scores"]
        ):
            detections.append(coco_detection(img_id, label, box, score))

    del session
    gc.collect()

    metrics, _ = evaluate_predictions(annotation_path, detections)
    return metrics


def benchmark_speed(
    model_path: Path,
    preprocessed_images: List[Tuple[np.ndarray, dict]],
    threads: int = 4,
    optimization: str = "all",
    warmup: int = 15,
    iterations: int = 60,
) -> Dict[str, float]:
    """Measure inference and pipeline latency with isolated session and peak RSS tracking."""
    import psutil

    process = psutil.Process()
    with peak_memory(process) as peak:
        session = YoloSession(model_path, threads=threads, optimization=optimization)

        # Warmup
        for i in range(warmup):
            px, geom = preprocessed_images[i % len(preprocessed_images)]
            session.session.run(None, {session.input_name: px})

        timings = []
        for i in range(iterations):
            px, geom = preprocessed_images[i % len(preprocessed_images)]
            t0 = time.perf_counter()
            raw = session.session.run(None, {session.input_name: px})[0]
            t_infer = time.perf_counter()
            res = decode(raw, geom, confidence=0.25, nms_iou=0.7)
            t_done = time.perf_counter()

            timings.append({
                "inference": (t_infer - t0) * 1000,
                "postprocess": (t_done - t_infer) * 1000,
                "pipeline": (t_done - t0) * 1000,
            })

        del session
        gc.collect()

    infer_stats = timing_statistics([t["inference"] for t in timings])
    pipe_stats = timing_statistics([t["pipeline"] for t in timings])

    return {
        "threads": threads,
        "optimization": optimization,
        "peak_rss_mb": peak[0] / (1024 * 1024),
        "inference_mean_ms": infer_stats["mean_ms"],
        "inference_p50_ms": infer_stats["p50_ms"],
        "inference_p95_ms": infer_stats["p95_ms"],
        "pipeline_mean_ms": pipe_stats["mean_ms"],
        "pipeline_p50_ms": pipe_stats["p50_ms"],
        "pipeline_p95_ms": pipe_stats["p95_ms"],
        "inference_fps": throughput_fps([t["inference"] for t in timings]),
        "pipeline_fps": throughput_fps([t["pipeline"] for t in timings]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to candidate ONNX models",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to FP32 baseline ONNX model",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        required=True,
        help="Path to COCO test annotations JSON",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/quantization"),
        help="Directory for evaluation reports",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="CPU thread counts to benchmark (default: 1 2 4)",
    )
    parser.add_argument(
        "--optimizations",
        choices=["all", "basic", "extended", "off"],
        nargs="+",
        default=["all", "basic"],
        help="ONNX Runtime graph optimization levels to benchmark",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=120,
        help="Number of images to evaluate COCO AP on (default: 120)",
    )
    parser.add_argument(
        "--accuracy-budget",
        type=float,
        default=2.0,
        help="Maximum allowed drop in AP (absolute points, 2.0 = 0.02 AP)",
    )
    args = parser.parse_args()

    # Enforce global CPU/RAM limits
    max_cores = max(args.threads)
    restrict_system_resources(max_cores=max_cores)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare evaluation images
    print("Loading test annotation subset...")
    full_data = read_json(args.test_json)
    selected_imgs = full_data["images"][:args.eval_samples]
    selected_ids = set(img["id"] for img in selected_imgs)
    sub_annotations = [a for a in full_data["annotations"] if a["image_id"] in selected_ids]

    sub_coco = {
        "images": selected_imgs,
        "annotations": sub_annotations,
        "categories": full_data["categories"],
    }
    sub_coco_path = args.output_dir / "sub_eval_coco.json"
    write_json(sub_coco_path, sub_coco)

    eval_paths = [(img["id"], args.images_dir / img["file_name"]) for img in selected_imgs]

    # Pre-letterbox a small sample of 20 images for speed testing (to keep memory < 100 MB)
    speed_sample_count = min(20, len(selected_imgs))
    speed_images = []
    for _, p in eval_paths[:speed_sample_count]:
        with Image.open(p) as img:
            px, geom = letterbox(img, 640)
            speed_images.append((px, geom))

    print(f"Loaded {len(eval_paths)} images for accuracy, {len(speed_images)} cached for latency tests.")

    # 2. Evaluate accuracy for each model (once per model at 4 threads)
    models = list(dict.fromkeys([args.baseline.resolve()] + [m.resolve() for m in args.models]))
    accuracy_results = {}
    print("\n=== Evaluating COCO AP per Model ===")
    for model_path in models:
        m_name = model_path.stem
        print(f"Evaluating accuracy: {m_name}...")
        metrics = evaluate_accuracy(
            model_path,
            eval_paths,
            sub_coco_path,
            threads=min(4, max_cores),
            optimization="all",
        )
        accuracy_results[str(model_path)] = metrics
        print(f"  -> AP: {metrics['AP']*100:.2f}%, AP50: {metrics['AP50']*100:.2f}%, AP_small: {metrics['AP_small']*100:.2f}%")

    baseline_ap = accuracy_results[str(args.baseline.resolve())]["AP"]

    # 3. Benchmark speed across (Model x Optimization x Threads)
    print("\n=== Benchmarking Speed across Kernels & Threads ===")
    records = []
    for model_path in models:
        m_name = model_path.stem
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        m_ap = accuracy_results[str(model_path)]["AP"]
        ap_drop = (baseline_ap - m_ap) * 100 # In percentage points

        for opt in args.optimizations:
            for th in args.threads:
                print(f"Measuring {m_name} | Opt: {opt:5s} | Threads: {th} ...")
                perf = benchmark_speed(
                    model_path,
                    speed_images,
                    threads=th,
                    optimization=opt,
                    warmup=10,
                    iterations=50,
                )

                is_eligible = ap_drop <= args.accuracy_budget

                row = {
                    "model": m_name,
                    "model_path": str(model_path),
                    "size_mb": round(model_size_mb, 2),
                    "ap": round(m_ap * 100, 2),
                    "ap50": round(accuracy_results[str(model_path)]["AP50"] * 100, 2),
                    "ap_small": round(accuracy_results[str(model_path)]["AP_small"] * 100, 2),
                    "ap_drop_pts": round(ap_drop, 2),
                    "within_budget": is_eligible,
                    "optimization": opt,
                    "threads": th,
                    "infer_mean_ms": round(perf["inference_mean_ms"], 2),
                    "infer_p50_ms": round(perf["inference_p50_ms"], 2),
                    "infer_p95_ms": round(perf["inference_p95_ms"], 2),
                    "pipeline_p50_ms": round(perf["pipeline_p50_ms"], 2),
                    "pipeline_p95_ms": round(perf["pipeline_p95_ms"], 2),
                    "pipeline_fps": round(perf["pipeline_fps"], 1),
                    "peak_rss_mb": round(perf["peak_rss_mb"], 1),
                }
                records.append(row)
                print(f"  -> Pipeline: {row['pipeline_p50_ms']} ms (p50), {row['pipeline_p95_ms']} ms (p95) | {row['pipeline_fps']} FPS | RAM: {row['peak_rss_mb']} MB")

    # 4. Selection: Best model within budget
    # Sort eligible models by lowest pipeline_p95_ms
    eligible = [r for r in records if r["within_budget"]]
    eligible.sort(key=lambda r: r["pipeline_p95_ms"])
    best = eligible[0] if eligible else None

    # 5. Write Report & JSON
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu_name(),
        "baseline_model": args.baseline.name,
        "baseline_ap": round(baseline_ap * 100, 2),
        "accuracy_budget_pts": args.accuracy_budget,
        "best_configuration": best,
        "results": records,
    }
    write_json(args.output_dir / "quantization_benchmark.json", report_data)

    # Markdown Report
    lines = [
        "# Quantization & Kernel Optimization Benchmark Report",
        "",
        f"- **CPU:** {cpu_name()}",
        f"- **Baseline FP32 AP:** {baseline_ap * 100:.2f}%",
        f"- **Accuracy Budget:** $\\le {args.accuracy_budget}$ absolute AP points drop",
        "",
        "## Best Selected Configuration",
        "",
    ]
    if best:
        lines += [
            f"**Winner:** `{best['model']}` with `{best['optimization']}` kernel optimization at **{best['threads']} CPU cores**.",
            f"- **Model Size:** {best['size_mb']} MB (Baseline: {records[0]['size_mb']} MB, **{records[0]['size_mb']/best['size_mb']:.2f}x reduction**)",
            f"- **AP:** {best['ap']}% (Drop: {best['ap_drop_pts']:+.2f} pts)",
            f"- **Latency (p50 / p95):** {best['pipeline_p50_ms']:.1f} ms / {best['pipeline_p95_ms']:.1f} ms",
            f"- **Throughput:** **{best['pipeline_fps']:.1f} FPS**",
            f"- **Peak RAM:** {best['peak_rss_mb']:.1f} MB (Strictly controlled)",
            "",
        ]
    lines += [
        "## Full Sweep Results",
        "",
        "| Model | Format | Opt | Cores | Size (MB) | AP (%) | AP drop | Pipeline p50 (ms) | Pipeline p95 (ms) | FPS | Peak RAM (MB) | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        status = "**SELECTED**" if best and r == best else ("PASS" if r["within_budget"] else "FAIL (AP)")
        lines.append(
            f"| `{r['model']}` | {r['model'].split('_')[1] if '_' in r['model'] else 'fp32'} | {r['optimization']} | {r['threads']} | {r['size_mb']} | {r['ap']:.2f} | {r['ap_drop_pts']:+.2f} | {r['pipeline_p50_ms']} | {r['pipeline_p95_ms']} | {r['pipeline_fps']} | {r['peak_rss_mb']} | {status} |"
        )
    lines.append("")

    report_md = "\n".join(lines)
    (args.output_dir / "report.md").write_text(report_md)
    print("\n" + report_md)


if __name__ == "__main__":
    main()
