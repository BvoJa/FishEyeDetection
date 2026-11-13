"""Isolated-process CPU latency/memory benchmarks and COCO accuracy evaluation."""

import argparse
import csv
import glob
import multiprocessing
import os
import platform
import random
import time
import traceback
from pathlib import Path

from PIL import Image

from efficient_detection.shared.common import (
    coco_detection,
    dataset_config,
    read_json,
    resolve_image,
    sha256,
    write_json,
)
from efficient_detection.shared.evaluation import evaluate_predictions
from efficient_detection.shared.performance import (
    FPS_SCOPES,
    peak_memory,
    throughput_fps,
    timing_statistics,
)
from efficient_detection.yolo.runtime import YoloSession


def accuracy(model, root, split="val", threads=4):
    annotation = root / "annotations" / f"{split}.json"
    truth = read_json(annotation)
    if not any(not ann.get("iscrowd", 0) for ann in truth["annotations"]):
        raise ValueError(f"{split} needs labeled non-crowd objects for AP evaluation")
    session = YoloSession(model, threads)
    detections = []
    for image in truth["images"]:
        with Image.open(resolve_image(root / "images", image["file_name"])) as source:
            result = session.predict(
                source.convert("RGB"), confidence=0.001, multi_label=True
            )
        for label, box, score in zip(
            result["labels"], result["boxes"], result["scores"]
        ):
            detections.append(coco_detection(image["id"], label, box, score))
    return evaluate_predictions(annotation, detections)[0]


def measure(
    model,
    paths,
    threads=4,
    optimization="all",
    warmup=20,
    repeats=100,
    trials=3,
    profile=None,
):
    import psutil

    with peak_memory(psutil.Process()) as peak:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        started = time.perf_counter()
        session = YoloSession(model, threads, optimization)
        startup_ms = (time.perf_counter() - started) * 1000
        for index in range(warmup):
            session.predict(images[index % len(images)])
        timings = []
        for _ in range(trials):
            for index in range(repeats):
                timings.append(
                    session.predict(images[index % len(images)])["timing_ms"]
                )
    result = {
        "threads": threads,
        "graph_optimization": optimization,
        "startup_ms": startup_ms,
        "peak_process_rss_mb": peak[0] / 1024**2,
        "timed_iterations": len(timings),
        "cached_images": len(images),
    }
    for stage in timings[0]:
        stats = timing_statistics([row[stage] for row in timings])
        for statistic in ["mean_ms", "p50_ms", "p95_ms"]:
            result[f"{stage}_{statistic}"] = stats[statistic]
    for stage in ["inference", "pipeline"]:
        result[stage + "_fps"] = throughput_fps(row[stage] for row in timings)
    if profile:
        # Profiling runs outside timed iterations and memory measurements.
        diagnostic = YoloSession(model, threads, optimization, profile=profile)
        diagnostic.predict(images[0])
        result["profile_trace"] = diagnostic.session.end_profiling()
    return result


def worker(kind, payload, connection):
    try:
        if kind == "accuracy":
            result = accuracy(**payload)
        else:
            result = measure(**payload)
        connection.send({"result": result})
    except Exception:  # noqa: BLE001 -- propagate worker failures and their traceback to the parent.
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def isolated(kind, payload):
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(kind, payload, sending))
    process.start()
    sending.close()
    try:
        message = receiving.recv()
    except EOFError as error:
        process.join()
        raise RuntimeError(
            f"Benchmark worker exited without results (exit={process.exitcode})"
        ) from error
    finally:
        receiving.close()
    process.join()
    if "error" in message:
        raise RuntimeError(message["error"])
    return message["result"]


def select_candidate(rows, baseline, budget_points=2):
    baseline = str(Path(baseline).resolve())
    references = [row for row in rows if row["model"] == baseline]
    if not references:
        raise ValueError("--baseline must be included in --models")
    reference_ap = references[0]["AP"]
    for row in rows:
        row["accuracy_delta_ap_points"] = (row["AP"] - reference_ap) * 100
        row["accuracy_eligible"] = (
            row["AP"] >= reference_ap - budget_points / 100 - 1e-9
        )
    candidates = [row for row in rows if row["accuracy_eligible"]]
    return min(candidates, key=lambda row: row["pipeline_p95_ms"])


def plot_results(rows, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fastest = {}
    for row in rows:
        if (
            row["model"] not in fastest
            or row["pipeline_p95_ms"] < fastest[row["model"]]["pipeline_p95_ms"]
        ):
            fastest[row["model"]] = row
    fig, axis = plt.subplots(figsize=(9, 5))
    for row in fastest.values():
        axis.scatter(
            row["pipeline_p95_ms"],
            row["AP"] * 100,
            color="tab:green" if row["accuracy_eligible"] else "tab:red",
        )
        axis.annotate(
            Path(row["model"]).stem,
            (row["pipeline_p95_ms"], row["AP"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(
        xlabel="Pipeline p95 latency (ms; excludes image I/O)",
        ylabel="COCO AP50:95 (%)",
        title="YOLO11 CPU accuracy–latency tradeoff",
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--graph-optimizations", choices=["off", "all"], nargs="+", default=["all"]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-cached-images", type=int, default=16)
    parser.add_argument(
        "--accuracy-budget",
        type=float,
        default=2,
        help="Absolute AP points; 2 means 0.02 in AP units",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--selected",
        type=Path,
        help="On test only: evaluate a selection.json previously chosen on validation",
    )
    args = parser.parse_args()
    if (
        min(args.threads) < 1
        or args.warmup < 0
        or min(args.repeats, args.trials, args.max_cached_images) < 1
        or args.accuracy_budget < 0
    ):
        parser.error("Invalid iteration/thread counts or accuracy budget")
    if args.split == "test" and not args.selected:
        parser.error(
            "Choose a model on validation first, then pass --selected selection.json for held-out testing"
        )
    if args.split != "test" and args.selected:
        parser.error("--selected is only used for a held-out test evaluation")
    root, _, manifest = dataset_config(args.data)
    if args.split not in manifest["splits"]:
        parser.error(f"No {args.split} split in this dataset")
    expanded = [
        Path(match)
        for pattern in args.models
        for match in (sorted(glob.glob(str(pattern))) or [str(pattern)])
    ]
    models = list(dict.fromkeys(path.resolve() for path in expanded))
    if args.baseline.resolve() not in models:
        parser.error("Include the baseline in --models")
    import onnxruntime as ort
    import psutil

    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    threads = sorted({min(value, physical) for value in args.threads})
    if args.selected:
        selected = read_json(args.selected)
        model = Path(selected["model"]).resolve()
        if model not in models:
            parser.error("Include the previously selected model in --models")
        if sha256(model) != selected["onnx_sha256"]:
            parser.error("Selected ONNX file has changed since validation")
        models = [model]
        threads = [min(selected["threads"], physical)]
        args.graph_optimizations = [selected["graph_optimization"]]
    args.output.mkdir(parents=True, exist_ok=True)
    records = manifest["splits"][args.split]
    records = random.Random(0).sample(
        records, min(len(records), args.max_cached_images)
    )
    paths = [root / "images" / record["file_name"] for record in records]
    rows = []
    for model in models:
        print(f"Accuracy evaluation: {model.name}", flush=True)
        metrics = isolated(
            "accuracy",
            {
                "model": model,
                "root": root,
                "split": args.split,
                "threads": min(4, physical),
            },
        )
        metadata = read_json(model.with_suffix(".json"))
        for count in threads:
            for optimization in args.graph_optimizations:
                print(
                    f"Timing: {model.name}, threads={count}, graph={optimization}",
                    flush=True,
                )
                profile = None
                if args.profile:
                    profile = (
                        args.output
                        / "profiles"
                        / f"{model.stem}_{sha256(model)[:8]}_{count}_{optimization}"
                    )
                    profile.parent.mkdir(parents=True, exist_ok=True)
                measured = isolated(
                    "measure",
                    {
                        "model": model,
                        "paths": paths,
                        "threads": count,
                        "optimization": optimization,
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "trials": args.trials,
                        "profile": profile,
                    },
                )
                rows.append(
                    {
                        "model": str(model),
                        "onnx_sha256": sha256(model),
                        "image_size": metadata["image_size"],
                        "parameters": metadata["parameters"],
                        "model_size_mb": model.stat().st_size / 1024**2,
                        **metrics,
                        **measured,
                    }
                )
                print(
                    f"FPS: inference={measured['inference_fps']:.2f}, pipeline={measured['pipeline_fps']:.2f} (cached RGB inputs; excludes image I/O)",
                    flush=True,
                )
    if args.split == "val":
        best = select_candidate(rows, args.baseline, args.accuracy_budget)
        write_json(args.output / "selection.json", best)
        plot_results(rows, args.output / "accuracy_latency.png")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "results.json", rows)
    write_json(
        args.output / "environment.json",
        {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_cores": physical,
            "logical_cores": os.cpu_count(),
            "onnxruntime": ort.__version__,
            "data_manifest_sha256": sha256(root / "manifest.json"),
            "split": args.split,
            "accuracy_configuration": {
                "threads": min(4, physical),
                "graph_optimization": "all",
                "confidence": 0.001,
                "multi_label": True,
            },
            "timing_configuration": {
                "confidence": 0.25,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "trials": args.trials,
            },
            "timing_scope": "RGB preprocessing + inference + NMS; excludes image decoding/writing and session startup",
            "fps_scope": {
                key: value
                for key, value in FPS_SCOPES.items()
                if key in {"inference_fps", "pipeline_fps"}
            },
            "fps_aggregation": "Timed image count / sum of measured stage seconds after warm-up; cached RGB inputs, batch 1; per-image reciprocals are not averaged",
            "memory_scope": "Peak sampled process RSS including session startup, warm-up and cached RGB inputs; excludes accuracy/profiling passes",
        },
    )
    print(f"Benchmark report: {args.output}")


if __name__ == "__main__":
    main()
