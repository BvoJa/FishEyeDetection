"""Throughput, timing and host facts from recorded serial, batch-one CPU measurements."""

import math
import os
import platform
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

STAGES = (
    "preprocess",
    "inference",
    "postprocess",
    "pipeline",
    "image_decode",
    "request",
)
FPS_STAGES = ("inference", "pipeline", "request")

FPS_SCOPES = {
    "inference_fps": "ONNX model execution only",
    "pipeline_fps": "RGB preprocessing + ONNX inference + coordinate restoration/NMS",
    "request_fps": "Image reading/decoding + deployment pipeline",
}


def cpu_name():
    """Human-readable CPU model, with a portable fallback."""
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def usable_threads(requested):
    """Clamp requested threads to physical cores and CPU affinity."""
    import psutil

    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    return min(requested, physical, available), physical, available


@contextmanager
def peak_memory(process):
    """Sample process RSS in the background and yield a one-element peak list."""
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        yield peak
    finally:
        stop.set()
        watcher.join()
        peak[0] = max(peak[0], process.memory_info().rss)


def throughput_fps(latencies_ms):
    """Return N / total elapsed seconds, not the mean of per-image reciprocals."""
    values = tuple(float(value) for value in latencies_ms)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("FPS requires finite, positive latency measurements")
    return len(values) * 1000 / math.fsum(values)


def timing_statistics(values):
    """Summarize measured milliseconds without averaging reciprocal FPS values."""
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Timing statistics require finite, nonnegative measurements")
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }
