"""Throughput from recorded serial, batch-one CPU measurements."""

import math

import numpy as np

FPS_SCOPES = {
    "inference_fps": "ONNX model execution only",
    "pipeline_fps": "RGB preprocessing + ONNX inference + coordinate restoration/NMS",
    "request_fps": "Image reading/decoding + deployment pipeline",
}


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
