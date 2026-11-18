"""Benchmark quantized ONNX models across dtypes, hybrid P/E core affinities, and ORT configs."""

import argparse
import copy
import gc
import json
import os
import resource
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

# Core mapping for Intel Core i7-1360P (4 P-cores / 8 threads: 0-7; 8 E-cores: 8-15)
P_CORES = [0, 2, 4, 6, 1, 3, 5, 7]  # Prefer primary physical P-cores first
E_CORES = [8, 9, 10, 11, 12, 13, 14, 15]


def get_affinity_cpus(mode: str, num_threads: int) -> Optional[Set[int]]:
    """Determine the physical CPU set based on affinity mode and thread count."""
    if mode == "p_core":
        return set(P_CORES[:min(num_threads, len(P_CORES))])
    elif mode == "e_core":
        return set(E_CORES[:min(num_threads, len(E_CORES))])
    elif mode == "os_default":
        return None  # Let OS scheduler place threads freely across all available cores
    else:
        raise ValueError(f"Unknown affinity mode: {mode}")


def apply_affinity(cpus: Optional[Set[int]], num_threads: int):
    """Enforce thread environment and CPU affinity."""
    os.environ["MALLOC_ARENA_MAX"] = "2"
    if hasattr(os, "sched_setaffinity"):
        if cpus is not None:
            try:
                os.sched_setaffinity(0, cpus)
            except OSError:
                pass
        else:
            # Restore all available cores
            all_cores = set(range(os.cpu_count() or 1))
            try:
                os.sched_setaffinity(0, all_cores)
            except OSError:
                pass
    threads_str = str(num_threads)
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[var] = threads_str


def evaluate_accuracy(
    model_path: Path,
    image_paths_with_ids: List[Tuple[int, Path]],
    annotation_path: Path,
    threads: int = 4,
    optimization: str = "all",
) -> Dict[str, float]:
    """Run COCO AP evaluation streaming images one-by-one to keep RAM low (<350MB)."""
    apply_affinity(None, threads)
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
    threads: int = 2,
    affinity_mode: str = "os_default",
    optimization: str = "all",
    warmup: int = 10,
    iterations: int = 40,
) -> Dict[str, float]:
    """Measure inference and pipeline latency under specified affinity and thread count."""
    import psutil

    cpus = get_affinity_cpus(affinity_mode, threads)
    apply_affinity(cpus, threads)

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
        "affinity_mode": affinity_mode,
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
        "--test-json",
        type=Path,
        default=Path("datasets/Fisheye8K/test/test.json"),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("datasets/Fisheye8K/test/images"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/quantization"),
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--accuracy-budget",
        type=float,
        default=2.0,
        help="Max allowable AP drop in percentage points",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load evaluation images
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

    # Pre-letterbox 20 images for speed testing
    speed_images = []
    for _, p in eval_paths[:20]:
        with Image.open(p) as img:
            px, geom = letterbox(img, 640)
            speed_images.append((px, geom))

    print(f"Loaded {len(eval_paths)} images for accuracy, {len(speed_images)} cached for latency tests.")

    # 2. Models to evaluate: Baseline FP32 + INT8 dtypes (U8S8, S8S8, U8U8) + QOperator
    model_entries = [
        ("baseline_fp32", Path("models/baseline_640.onnx"), "FP32", "FP32"),
        ("quant_qdq_u8s8", Path("models/quant_qdq_640_u8s8.onnx"), "QDQ", "U8S8"),
        ("quant_qdq_s8s8", Path("models/quant_qdq_640_s8s8.onnx"), "QDQ", "S8S8"),
        ("quant_qdq_u8u8", Path("models/quant_qdq_640_u8u8.onnx"), "QDQ", "U8U8"),
        ("quant_qop_u8s8", Path("models/quant_qop_640.onnx"), "QOperator", "U8S8"),
    ]

    # Evaluate accuracy once per unique model
    print("\n=== Evaluating COCO AP per Model ===")
    accuracy_results = {}
    for tag, mpath, fmt, dtype in model_entries:
        print(f"Evaluating accuracy: {tag} ({fmt} - {dtype})...")
        metrics = evaluate_accuracy(
            mpath,
            eval_paths,
            sub_coco_path,
            threads=2,
            optimization="all",
        )
        accuracy_results[tag] = metrics
        print(f"  -> AP: {metrics['AP']*100:.2f}%, AP50: {metrics['AP50']*100:.2f}%, AP_small: {metrics['AP_small']*100:.2f}%")

    baseline_ap_val = accuracy_results["baseline_fp32"]["AP"]
    baseline_ap_pct = baseline_ap_val * 100

    # 3. Benchmark speed across (Model x Affinity x Threads)
    # Affinities: P-core only, E-core only, OS default
    # Cores/Threads: 1, 2, 4
    affinities = ["p_core", "e_core", "os_default"]
    threads_list = [1, 2, 4]

    print("\n=== Benchmarking Speed across Dtypes, Hybrid Core Affinities & Threads ===")
    records = []
    for tag, mpath, fmt, dtype in model_entries:
        model_size_mb = mpath.stat().st_size / (1024 * 1024)
        m_ap_val = accuracy_results[tag]["AP"]
        m_ap_pct = m_ap_val * 100
        # Correct delta AP: Model AP minus Baseline AP (negative means drop, positive means gain)
        delta_ap = m_ap_pct - baseline_ap_pct
        is_eligible = delta_ap >= -args.accuracy_budget

        for aff in affinities:
            for th in threads_list:
                print(f"Measuring {tag} ({dtype}) | Affinity: {aff:10s} | Cores: {th} ...")
                perf = benchmark_speed(
                    mpath,
                    speed_images,
                    threads=th,
                    affinity_mode=aff,
                    optimization="all",
                    warmup=10,
                    iterations=40,
                )

                row = {
                    "tag": tag,
                    "format": fmt,
                    "dtype": dtype,
                    "affinity": aff,
                    "threads": th,
                    "size_mb": round(model_size_mb, 2),
                    "ap": round(m_ap_pct, 2),
                    "ap50": round(accuracy_results[tag]["AP50"] * 100, 2),
                    "ap_small": round(accuracy_results[tag]["AP_small"] * 100, 2),
                    "delta_ap": round(delta_ap, 2),
                    "within_budget": is_eligible,
                    "infer_p50_ms": round(perf["inference_p50_ms"], 2),
                    "pipeline_p50_ms": round(perf["pipeline_p50_ms"], 2),
                    "pipeline_p95_ms": round(perf["pipeline_p95_ms"], 2),
                    "pipeline_fps": round(perf["pipeline_fps"], 1),
                    "peak_rss_mb": round(perf["peak_rss_mb"], 1),
                }
                records.append(row)
                print(f"  -> Pipeline: {row['pipeline_p50_ms']} ms (p50), {row['pipeline_p95_ms']} ms (p95) | {row['pipeline_fps']} FPS | RAM: {row['peak_rss_mb']} MB")

    # 4. Selection: Best model within budget
    eligible = [r for r in records if r["within_budget"]]
    eligible.sort(key=lambda r: r["pipeline_p95_ms"])
    best = eligible[0] if eligible else None

    # 5. Output JSON and Markdown
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu_name(),
        "baseline_ap": round(baseline_ap_pct, 2),
        "accuracy_budget_drop_pts": args.accuracy_budget,
        "best_configuration": best,
        "results": records,
    }
    write_json(args.output_dir / "quantization_hybrid_benchmark.json", report_data)

    lines = [
        "# Quantization Benchmark: Dtypes & Intel Hybrid Core Affinities",
        "",
        f"- **CPU:** {cpu_name()} (4 P-cores / 8 threads + 8 E-cores = 16 logical CPUs)",
        f"- **Baseline FP32 AP:** **{baseline_ap_pct:.2f}%**",
        f"- **Accuracy Budget:** Maximum allowed drop $\\le {args.accuracy_budget}$ AP points ($\Delta AP \\ge -{args.accuracy_budget:.2f}$)",
        "",
        "## Best Selected Configuration",
        "",
    ]
    if best:
        lines += [
            f"**Winner:** `{best['tag']}` (Format: **{best['format']}**, Dtype: **{best['dtype']}**)",
            f"- **CPU Affinity:** `{best['affinity']}` at **{best['threads']} cores**",
            f"- **Model Size:** {best['size_mb']} MB (Baseline: {records[0]['size_mb']} MB, **{records[0]['size_mb']/best['size_mb']:.2f}x smaller**)",
            f"- **AP:** {best['ap']:.2f}% (\\(\\Delta AP = {best['delta_ap']:+.2f}\\) points vs baseline)",
            f"- **Pipeline Latency (p50 / p95):** {best['pipeline_p50_ms']:.1f} ms / {best['pipeline_p95_ms']:.1f} ms",
            f"- **Throughput:** **{best['pipeline_fps']:.1f} FPS**",
            f"- **Peak RAM:** {best['peak_rss_mb']:.1f} MB (strictly bounded)",
            "",
        ]

    lines += [
        "## Full Sweep: Dtypes $\\times$ Core Affinity $\\times$ Core Count",
        "",
        "| Model | Dtype | Format | CPU Affinity | Cores | AP (%) | $\\Delta$ AP | Pipeline p50 (ms) | Pipeline p95 (ms) | FPS | Peak RAM (MB) | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        status = "**SELECTED**" if best and r == best else ("PASS" if r["within_budget"] else "FAIL (AP)")
        lines.append(
            f"| `{r['tag']}` | {r['dtype']} | {r['format']} | `{r['affinity']}` | {r['threads']} | {r['ap']:.2f} | {r['delta_ap']:+.2f} | {r['pipeline_p50_ms']} | {r['pipeline_p95_ms']} | {r['pipeline_fps']} | {r['peak_rss_mb']} | {status} |"
        )
    lines.append("")

    report_md = "\n".join(lines)
    (args.output_dir / "report_hybrid.md").write_text(report_md)
    print("\n" + report_md)


if __name__ == "__main__":
    main()
