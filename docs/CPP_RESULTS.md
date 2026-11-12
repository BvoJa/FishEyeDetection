# Measured C++ / Python comparison

Measured on **2026-10-19**, native Linux x86-64, Intel Core i7-1360P, 4 ONNX threads,
FP32 YOLO11n at 640, batch 1. All backends use the **same ONNX Runtime 1.31.0 shared
library**. Python 3.11.14, NumPy 2.4.6, Pillow 12.3.0; C++ compiled in Release mode.

Inputs: **100 FishEye8K validation images**, seed 0, 20 warm-ups per process,
**3 trials** with rotated backend order. These results cover the selected subset,
not a new full-dataset evaluation.

| Backend | Mean pipeline (ms) | Pipeline p95 (ms) | Pipeline FPS | Request FPS | Peak process RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| Existing Python | 74.22 | 108.59 | 13.47 | 9.51 | 203.64 |
| C++ fresh tensors | 64.15 | 87.67 | 15.59 | 10.66 | 152.30 |
| C++ reusable tensors | 58.82 | 75.79 | 17.00 | 11.27 | 142.35 |

Pipeline includes letterbox/normalization, model execution and coordinate
restoration/NMS. Request also includes reading/decoding images. Startup, warm-up,
diagnostic dumps, metric computation and final output writing are excluded.

- **Correctness:** all **1,417 detections** match across all backends/trials, with
  zero coordinate/score difference. Input and raw-output tensors match exactly on
  four diagnostic images. Fixed-threshold F1 is **52.80%** for every backend.
- **Observed latency ratios:** Python/C++ reuse pipeline **1.262x**; request
  **1.185x**. C++ fresh/reuse pipeline **1.091x**.
- **Buffer ownership:** fresh mode creates 120 input/output tensors per trial
  including warm-up; reuse mode creates **one of each**. These are application
  tensor creation events, not total heap allocation counts.
- **Variation:** per-trial mean pipeline latency was 70.36/73.44/78.87 ms for Python,
  59.21/59.45/73.79 ms for C++ fresh and 61.86/57.23/57.36 ms for C++ reuse. The
  incremental buffer benefit needs more controlled measurements before attributing
  the entire ratio to allocation savings. ONNX internals and image decoding still
  allocate; C++ also changes preprocessing/NMS and the image decoder.

Model SHA-256: `0e1142f5487ac6d7463ef936e985176b317888906e57e359df2869622a00b2bf`.
Selected-image manifest SHA-256:
`e917d0ef52d9954331d02df81a6ced7f6a9402a515b59bb89088af40e577b08d`.

See [CPP.md](CPP.md) for build/run commands. The complete local artifacts are in
`reports/cpp_vs_python/`, including `report.md`, `parity.json`, `per_image.csv`,
`environment.json` and individual trial results. Reports and model binaries remain
outside Git; rerun the supplied preset with a fresh output directory to reproduce
the comparison on your machine.
