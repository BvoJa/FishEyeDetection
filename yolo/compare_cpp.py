"""Compare existing Python inference with C++ fresh/reused buffers on identical inputs."""

import argparse
import csv
import os
import platform
import random
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

from efficient_detection.shared.common import (
    CLASSES,
    EXTENSIONS,
    box_iou,
    read_json,
    sha256,
    write_json,
)
from efficient_detection.shared.metrics import detection_counts, metrics_from_counts
from efficient_detection.shared.performance import throughput_fps, timing_statistics
from efficient_detection.yolo.runtime import YoloSession, letterbox

STAGES = [
    "preprocess",
    "inference",
    "postprocess",
    "pipeline",
    "image_decode",
    "request",
]


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    root = (path.parent / config.get("project_root", ".")).resolve()
    for key in ["model", "images", "annotations", "binary", "output", "manifest"]:
        if config.get(key) is not None:
            config[key] = str((root / config[key]).resolve())
    config["project_root"] = str(root)
    return config


def image_path(images, record):
    relative = Path(record["file_name"])
    root = Path(images).resolve()
    path = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.is_relative_to(root)
    ):
        raise ValueError(f"Image name must remain within image root: {relative}")
    return path


def cpu_name():
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def python_worker(config):
    """Use the existing PIL/NumPy/YoloSession path, without preallocated tensors."""
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    records = read_json(config["manifest"])["images"]
    start = time.perf_counter()
    session = YoloSession(config["model"], config["threads"])
    startup_ms = (time.perf_counter() - start) * 1000
    for index in range(config["warmup"]):
        with Image.open(
            image_path(config["images"], records[index % len(records)])
        ) as source:
            session.predict(
                source.convert("RGB"), config["confidence"], config["nms_iou"]
            )
    predictions, rows = [], []
    dataset_started = time.perf_counter()
    for index, record in enumerate(records):
        started = time.perf_counter()
        with Image.open(image_path(config["images"], record)) as source:
            image = source.convert("RGB")
        decoded = time.perf_counter()
        result = session.predict(
            image, config["confidence"], config["nms_iou"], return_raw=True
        )
        completed = time.perf_counter()
        timing = {
            **result["timing_ms"],
            "image_decode": (decoded - started) * 1000,
            "request": (completed - started) * 1000,
        }
        rows.append(
            {
                "image_id": record["id"],
                "file_name": record["file_name"],
                "repeat": 0,
                "detections": len(result["labels"]),
                "timing_ms": timing,
            }
        )
        for box, score, label in zip(
            result["boxes"], result["scores"], result["labels"]
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
        if index < config["dump_tensors"]:
            folder = output / "tensors"
            folder.mkdir(exist_ok=True)
            pixels, _ = letterbox(image, session.image_size)
            pixels.tofile(folder / f"{index}_input.f32")
            result["raw"].tofile(folder / f"{index}_raw.f32")
    dataset_seconds = time.perf_counter() - dataset_started
    summary = {
        "runtime": "python",
        "model": config["model"],
        "onnx_sha256": sha256(config["model"]),
        "images": len(records),
        "timed_iterations": len(rows),
        "threads": config["threads"],
        "warmup": config["warmup"],
        "batch_size": 1,
        "startup_ms": startup_ms,
        "dataset_pass_seconds": dataset_seconds,
        "dataset_fps": len(rows) / dataset_seconds,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "timing": {
            stage: timing_statistics([row["timing_ms"][stage] for row in rows])
            for stage in STAGES
        },
    }
    for stage in ["inference", "pipeline", "request"]:
        summary[f"{stage}_fps"] = throughput_fps(
            row["timing_ms"][stage] for row in rows
        )
    write_json(output / "per_image.json", rows)
    write_json(output / "predictions.json", predictions)
    write_json(output / "summary.json", summary)
    print(
        f"Python: {len(rows)} images; pipeline FPS={summary['pipeline_fps']:.2f}; request FPS={summary['request_fps']:.2f}",
        flush=True,
    )


def prediction_parity(reference, candidate, box_atol=0.01, score_atol=1e-5):
    """Match detections within image/class, independently of serialization order."""
    groups = defaultdict(lambda: [[], []])
    failures = []
    for index, values in enumerate([reference, candidate]):
        for row in values:
            box = np.asarray(row["bbox"], dtype=np.float64)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or not np.isfinite(row["score"])
                or np.any(box[2:] <= 0)
                or row["category_id"] not in range(len(CLASSES))
            ):
                failures.append(
                    f"Invalid detection in {'reference' if index == 0 else 'candidate'}: image {row['image_id']}"
                )
                continue
            groups[(row["image_id"], row["category_id"])][index].append(row)
    max_box_error = max_score_error = 0.0
    min_iou = 1.0
    for key, (truth, predicted) in groups.items():
        if len(truth) != len(predicted):
            failures.append(f"{key}: detection counts {len(truth)} != {len(predicted)}")
            continue
        boxes = [
            np.array([row["bbox"] for row in rows], dtype=np.float64).reshape(-1, 4)
            for rows in [truth, predicted]
        ]
        for values in boxes:
            values[:, 2:] += values[:, :2]
        box_errors = np.abs(boxes[0][:, None] - boxes[1][None, :]).max(axis=2)
        scores = [
            np.array([row["score"] for row in rows]) for rows in [truth, predicted]
        ]
        score_errors = np.abs(scores[0][:, None] - scores[1][None, :])
        allowed = (box_errors <= box_atol) & (score_errors <= score_atol)
        edges = [
            sorted(
                np.flatnonzero(row),
                key=lambda j: (score_errors[i, j], box_errors[i, j]),
            )
            for i, row in enumerate(allowed)
        ]
        owners = {}

        def assign(left, visited, edges, owners):
            for right in edges[left]:
                if right in visited:
                    continue
                visited.add(right)
                if right not in owners or assign(owners[right], visited, edges, owners):
                    owners[right] = left
                    return True
            return False

        # Find a complete matching rather than greedily consuming an ambiguous
        # candidate needed by another reference detection (e.g. equal-score boxes).
        for left in sorted(range(len(truth)), key=lambda i: len(edges[i])):
            if not assign(left, set(), edges, owners):
                failures.append(
                    f"{key}: no complete one-to-one match within box/score tolerances"
                )
                break
        for right, left in owners.items():
            max_box_error = max(max_box_error, float(box_errors[left, right]))
            max_score_error = max(max_score_error, float(score_errors[left, right]))
            min_iou = min(
                min_iou, float(box_iou(boxes[0][left], boxes[1][right : right + 1])[0])
            )
    return {
        "passed": not failures,
        "reference_detections": len(reference),
        "candidate_detections": len(candidate),
        "max_box_abs_error_px": max_box_error,
        "max_score_abs_error": max_score_error,
        "min_matched_iou": min_iou,
        "failures": failures[:20],
        "box_atol_px": box_atol,
        "score_atol": score_atol,
    }


def tensor_parity(reference, candidate, count, config):
    results = []
    for index in range(count):
        row = {"index": index}
        for kind in ["input", "raw"]:
            left = np.fromfile(
                reference / "tensors" / f"{index}_{kind}.f32", dtype=np.float32
            )
            right = np.fromfile(
                candidate / "tensors" / f"{index}_{kind}.f32", dtype=np.float32
            )
            same_shape = left.shape == right.shape and left.size > 0
            finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
            passed = (
                same_shape
                and finite
                and (
                    np.array_equal(left, right)
                    if kind == "input"
                    else np.allclose(
                        left,
                        right,
                        rtol=config["raw_rtol"],
                        atol=config["raw_atol"],
                        equal_nan=False,
                    )
                )
            )
            row[kind] = {
                "passed": bool(passed),
                "elements": int(left.size),
                "candidate_elements": int(right.size),
                "finite": finite,
                "max_abs_error": float(np.abs(left - right).max())
                if same_shape and finite
                else None,
            }
        results.append(row)
    return {
        "passed": all(
            row["input"]["passed"] and row["raw"]["passed"] for row in results
        ),
        "samples": results,
        "input_atol": 0,
        "raw_atol": config["raw_atol"],
        "raw_rtol": config["raw_rtol"],
    }


def fixed_metrics(source, records, predictions):
    ground_truth, detected = defaultdict(list), defaultdict(list)
    for row in source["annotations"]:
        ground_truth[row["image_id"]].append(row)
    for row in predictions:
        detected[row["image_id"]].append(row)
    counts = np.zeros((len(CLASSES), 4), dtype=np.int64)
    for record in records:
        rows = detected[record["id"]]
        boxes = np.array([row["bbox"] for row in rows], dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        counts += detection_counts(
            ground_truth[record["id"]],
            boxes,
            [row["score"] for row in rows],
            [row["category_id"] for row in rows],
        )
    return metrics_from_counts(counts.sum(axis=0))


def write_report(output, summary):
    names = list(summary["backends"])
    lines = [
        "# C++ / Python CPU comparison",
        "",
        f"Same FP32 model, **{summary['images']} images**, **{summary['threads']} threads**, batch 1, **{summary['trials']} trials**.",
        f"Parity: **{'PASS' if summary['parity_passed'] else 'FAIL'}**. See `parity.json` for input/raw-output/detection checks.",
        "",
        "| Backend | Inference FPS | Pipeline FPS | Request FPS | Pipeline p95 (ms) | Peak RSS (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        result = summary["backends"][name]
        lines.append(
            f"| {name} | {result['inference_fps']:.2f} | {result['pipeline_fps']:.2f} | {result['request_fps']:.2f} | {result['timing']['pipeline']['p95_ms']:.2f} | {result['peak_process_rss_mib']:.2f} |"
        )
    lines += [
        "",
        "| Stage | " + " | ".join(name + " mean / p95 (ms)" for name in names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for stage in STAGES:
        cells = [
            f"{summary['backends'][name]['timing'][stage]['mean_ms']:.3f} / {summary['backends'][name]['timing'][stage]['p95_ms']:.3f}"
            for name in names
        ]
        lines.append("| " + stage + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Python / C++ reuse mean pipeline latency ratio: **{summary['python_to_cpp_pipeline_speedup']:.3f}x**.",
        f"Python / C++ reuse mean request latency ratio: **{summary['python_to_cpp_request_speedup']:.3f}x**.",
    ]
    if "cpp_fresh" in names:
        lines += [
            f"C++ fresh / reuse mean pipeline latency ratio: **{summary['cpp_buffer_reuse_pipeline_speedup']:.3f}x**. Only tensor allocation/output ownership differs in the implementation; timing variation can also affect this ratio."
        ]
    lines += [
        "",
        "| Backend | Input tensor creation events / trial | Output tensor creation events / trial |",
        "|---|---|---|",
    ]
    for name in names:
        if "buffers_per_trial" in summary["backends"][name]:
            buffers = summary["backends"][name]["buffers_per_trial"]
            inputs = ", ".join(str(row["input_tensor_allocations"]) for row in buffers)
            outputs = ", ".join(
                str(row["output_tensor_allocations"]) for row in buffers
            )
            lines.append(f"| {name} | {inputs} | {outputs} |")
    lines += [
        "",
        "Counts include warm-up and describe application tensor creation events, not process-wide malloc calls or ONNX Runtime's internal arena allocations.",
    ]
    if summary["fixed_metrics"]:
        lines += [
            "",
            "Fixed-threshold detection metrics on this selected subset (IoU 0.5):",
            "",
            "| Backend | Precision (%) | Recall (%) | F1 (%) |",
            "|---|---:|---:|---:|",
        ]
        for name, metrics in summary["fixed_metrics"].items():
            lines.append(
                f"| {name} | {metrics['precision'] * 100:.2f} | {metrics['recall'] * 100:.2f} | {metrics['f1'] * 100:.2f} |"
            )
    lines += [
        "",
        "## Measurement scope",
        "",
        "- Python uses the existing Pillow/NumPy/YoloSession implementation. C++ uses stb_image plus Pillow-compatible fixed-point bilinear resampling, FP32 RGB/255 CHW, and same-class NMS.",
        "- All backends load the exact same ONNX Runtime shared library, graph optimization=all, intra-op threads as shown, inter-op=1; each trial uses a fresh process and warmed session. Backend execution order rotates between trials.",
        "- Pipeline includes preprocessing, model execution and coordinate restoration/NMS. Request also includes file reading/decoding. Neither includes startup, warm-up, prediction serialization, tensor dumps or metric evaluation.",
        "- FPS = measured image count / total measured stage seconds. Percentiles pool per-image measurements across trials. Per-trial means remain in summary.json to show variation; this is not a claim of statistical significance.",
        "- C++ reuse retains input/output Ort::Value objects and their buffers. C++ fresh creates application input storage each prediction and lets ORT allocate outputs. Both reuse preprocessing/NMS scratch capacity and cached resize plans. Internal ORT/image-decoder allocations remain; this is not an allocation-free runtime.",
        "- RSS is the process high-water mark including startup, libraries, decoding scratch and retained results. Python and C++ process overhead differs. It is not model-only memory.",
        "- PNG parity is verified for the measured images. JPEG decoders may differ; a failed parity gate invalidates a numerically equivalent speedup claim.",
        "- These are CPU measurements on the recorded host, not Jetson or Qualcomm-device measurements. Dataset-pass FPS is retained in trial artifacts but is not compared because tensor-dump/serialization overhead differs.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def compare(config):
    import onnxruntime as ort

    if (
        min(config["threads"], config["trials"]) < 1
        or min(config["warmup"], config["max_images"], config["dump_tensors"]) < 0
    ):
        raise ValueError("Invalid thread/image/trial/warm-up counts")
    if any(
        not np.isfinite(config[key]) or not 0 <= config[key] <= 1
        for key in ["confidence", "nms_iou"]
    ):
        raise ValueError("Confidence and NMS IoU must be in [0,1]")
    if config["dump_tensors"] < 1:
        raise ValueError("Compare at least one input/raw tensor: set dump_tensors >= 1")
    if any(
        not np.isfinite(config[key]) or config[key] < 0
        for key in ["raw_atol", "raw_rtol", "box_atol", "score_atol"]
    ):
        raise ValueError("Parity tolerances must be finite and nonnegative")
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    if output.is_relative_to(Path(config["images"])):
        raise ValueError("Output must be outside input images")
    binary = Path(config["binary"])
    build_info = read_json(binary.parent / "build_info.json")
    library = Path(ort.__file__).parent / "capi/libonnxruntime.so.1.31.0"
    if build_info["onnxruntime"] != ort.__version__ or build_info[
        "library_sha256"
    ] != sha256(library):
        raise ValueError(
            "Rebuild C++ against this Python environment using cpp/build.py"
        )
    if build_info["source_sha256"] != sha256(
        Path(config["project_root"]) / "cpp/src/main.cpp"
    ):
        raise ValueError("C++ source changed; rebuild using cpp/build.py")
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    config["requested_threads"] = config["threads"]
    config["threads"] = min(config["threads"], physical, available)
    source = read_json(config["annotations"]) if config.get("annotations") else None
    records = (
        source["images"]
        if source
        else [
            {"id": index, "file_name": path.relative_to(config["images"]).as_posix()}
            for index, path in enumerate(sorted(Path(config["images"]).rglob("*")), 1)
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ]
    )
    if config["max_images"] and len(records) > config["max_images"]:
        indices = sorted(
            random.Random(config["seed"]).sample(
                range(len(records)), config["max_images"]
            )
        )
        records = [records[index] for index in indices]
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Need nonempty inputs with unique image IDs")
    manifest = []
    for row in records:
        path = image_path(config["images"], row)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            raise ValueError(f"Unsupported C++ image format: {path}")
        manifest.append({**row, "sha256": sha256(path)})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", {"images": manifest})
    write_json(output / "config.json", config)
    names = (
        ["python", "cpp_fresh", "cpp_reuse"]
        if config.get("compare_fresh_buffers", True)
        else ["python", "cpp_reuse"]
    )
    print(
        f"Comparing {len(records)} images, {config['threads']} threads, {config['trials']} trials: {names}",
        flush=True,
    )
    runs = defaultdict(list)
    for trial in range(config["trials"]):
        order = names[trial % len(names) :] + names[: trial % len(names)]
        for name in order:
            folder = output / f"trial_{trial + 1:02d}" / name
            run_config = {
                **config,
                "output": str(folder),
                "manifest": str(output / "manifest.json"),
                "buffer_mode": "fresh" if name == "cpp_fresh" else "reuse",
                "repeats": 1,
                "dump_tensors": min(config["dump_tensors"], len(records))
                if trial == 0
                else 0,
            }
            config_file = output / "run_configs" / f"trial_{trial + 1:02d}_{name}.json"
            write_json(config_file, run_config)
            command = (
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--python-worker",
                    str(config_file),
                ]
                if name == "python"
                else [str(binary), "--config", str(config_file)]
            )
            subprocess.run(command, check=True)
            runs[name].append(folder)
    parity = {}
    reference = read_json(runs["python"][0] / "predictions.json")
    for name in names:
        checks = []
        for trial, folder in enumerate(runs[name]):
            rows = read_json(folder / "per_image.json")
            if [row["image_id"] for row in rows] != [row["id"] for row in records]:
                raise ValueError("Backends did not process identical images/order")
            checks.append(
                prediction_parity(
                    reference,
                    read_json(folder / "predictions.json"),
                    config["box_atol"],
                    config["score_atol"],
                )
            )
        tensors = tensor_parity(
            runs["python"][0],
            runs[name][0],
            min(config["dump_tensors"], len(records)),
            config,
        )
        parity[name] = {
            "detections": checks,
            "tensors": tensors,
            "passed": all(check["passed"] for check in checks) and tensors["passed"],
        }
    write_json(output / "parity.json", parity)
    backends, fixed = {}, {}
    all_rows = []
    for name in names:
        rows, summaries = [], []
        for trial, folder in enumerate(runs[name]):
            measured = read_json(folder / "per_image.json")
            rows.extend(measured)
            summaries.append(read_json(folder / "summary.json"))
            for row in measured:
                all_rows.append(
                    {
                        "backend": name,
                        "trial": trial + 1,
                        "image_id": row["image_id"],
                        "file_name": row["file_name"],
                        "detections": row["detections"],
                        **{f"{stage}_ms": row["timing_ms"][stage] for stage in STAGES},
                    }
                )
        result = {
            "timing": {
                stage: timing_statistics([row["timing_ms"][stage] for row in rows])
                for stage in STAGES
            },
            "peak_process_rss_mib": max(
                row["peak_process_rss_mib"] for row in summaries
            ),
            "trial_pipeline_mean_ms": [
                row["timing"]["pipeline"]["mean_ms"] for row in summaries
            ],
            "trial_request_mean_ms": [
                row["timing"]["request"]["mean_ms"] for row in summaries
            ],
        }
        for stage in ["inference", "pipeline", "request"]:
            result[f"{stage}_fps"] = throughput_fps(
                row["timing_ms"][stage] for row in rows
            )
        if name.startswith("cpp"):
            result["buffers_per_trial"] = [row["buffers"] for row in summaries]
        backends[name] = result
        if source and "annotations" in source:
            fixed[name] = fixed_metrics(
                source, records, read_json(runs[name][0] / "predictions.json")
            )
    summary = {
        "images": len(records),
        "threads": config["threads"],
        "trials": config["trials"],
        "backends": backends,
        "fixed_metrics": fixed,
        "parity_passed": all(row["passed"] for row in parity.values()),
        "python_to_cpp_pipeline_speedup": backends["python"]["timing"]["pipeline"][
            "mean_ms"
        ]
        / backends["cpp_reuse"]["timing"]["pipeline"]["mean_ms"],
        "python_to_cpp_request_speedup": backends["python"]["timing"]["request"][
            "mean_ms"
        ]
        / backends["cpp_reuse"]["timing"]["request"]["mean_ms"],
    }
    if "cpp_fresh" in backends:
        summary["cpp_buffer_reuse_pipeline_speedup"] = (
            backends["cpp_fresh"]["timing"]["pipeline"]["mean_ms"]
            / backends["cpp_reuse"]["timing"]["pipeline"]["mean_ms"]
        )
    write_json(output / "summary.json", summary)
    write_json(
        output / "environment.json",
        {
            "platform": platform.platform(),
            "cpu": cpu_name(),
            "physical_cores": physical,
            "available_cpus": available,
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "model_sha256": sha256(config["model"]),
            "image_manifest_sha256": sha256(output / "manifest.json"),
            "binary_sha256": sha256(binary),
            "build": build_info,
        },
    )
    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_report(output, summary)
    print(
        f"Comparison report: {output / 'report.md'}; parity={'PASS' if summary['parity_passed'] else 'FAIL'}",
        flush=True,
    )
    if not summary["parity_passed"]:
        raise ValueError(
            "Parity failed; inspect parity.json before interpreting the speed comparison"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/compare_cpp_python.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-images", type=int, help="0 means the full image list")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--python-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.python_worker:
        python_worker(read_json(args.python_worker))
        return
    config = load_config(args.config)
    for name in ["max_images", "threads", "trials"]:
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.output:
        config["output"] = str(args.output.resolve())
    compare(config)


if __name__ == "__main__":
    main()
