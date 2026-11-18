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


def detect_p_cores():
    """Detect Performance cores on hybrid architectures (e.g. Intel Alder Lake / Raptor Lake).

    Inspects sysfs max frequency, thread siblings, and core types. Returns a list of CPU IDs
    belonging to Performance cores, preferring the primary physical thread of each core.
    Returns None if the architecture is homogeneous or core detection is not supported.
    """
    sys_cpu = Path("/sys/devices/system/cpu")
    if not sys_cpu.is_dir():
        return None

    cpu_dirs = sorted(
        [d for d in sys_cpu.glob("cpu[0-9]*") if d.is_dir()],
        key=lambda d: int(d.name[3:]),
    )
    if not cpu_dirs:
        return None

    freqs = {}
    core_ids = {}
    for d in cpu_dirs:
        cpuid = int(d.name[3:])
        freq_file = d / "cpufreq/cpuinfo_max_freq"
        core_file = d / "topology/core_id"
        if freq_file.is_file() and core_file.is_file():
            try:
                freqs[cpuid] = int(freq_file.read_text().strip())
                core_ids[cpuid] = int(core_file.read_text().strip())
            except (ValueError, OSError):
                pass

    if len(freqs) != len(cpu_dirs):
        return None

    unique_freqs = set(freqs.values())
    if len(unique_freqs) <= 1:
        return None  # Homogeneous CPU

    # Hybrid CPU: P-cores have the higher max frequency
    max_freq = max(unique_freqs)
    p_cpus = [cpuid for cpuid, freq in freqs.items() if freq == max_freq]

    # Order P-cores preferring one physical thread per physical core first
    seen_cores = set()
    primary = []
    secondary = []
    for cpuid in p_cpus:
        cid = core_ids.get(cpuid, cpuid)
        if cid not in seen_cores:
            seen_cores.add(cid)
            primary.append(cpuid)
        else:
            secondary.append(cpuid)

    return primary + secondary


def auto_p_core_affinity(requested_threads=None):
    """Automatically bind current process to Performance cores if on a hybrid CPU."""
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        return None

    p_cores = detect_p_cores()
    if not p_cores:
        return None

    try:
        available = os.sched_getaffinity(0)
    except OSError:
        available = set(range(os.cpu_count() or 1))

    available_p = [c for c in p_cores if c in available]
    if not available_p:
        return None

    count = (
        min(requested_threads, len(available_p))
        if requested_threads
        else len(available_p)
    )
    target_affinity = set(available_p[:count])
    try:
        os.sched_setaffinity(0, target_affinity)
        return target_affinity
    except OSError:
        return None


def usable_threads(requested, auto_affinity=True):
    """Clamp requested threads to physical cores and CPU affinity, auto-selecting P-cores."""
    import psutil

    if auto_affinity:
        auto_p_core_affinity(requested)

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
