# C++ ONNX Runtime inference and Python comparison

Run from the `efficient_detection/` checkout. This is a Linux CPU, FP32, batch-1
runtime for the static YOLO11 ONNX exports already produced by this project.
Python training, export and inference remain available.
The measured comparison on this project's current model is recorded in
[CPP_RESULTS.md](CPP_RESULTS.md).

## Build once

Use the existing Python 3.11 virtual environment and a C++17 compiler:

```bash
source .venv/bin/activate
python -m pip install -r requirements/cpp.txt
python cpp/build.py
```

The first build downloads pinned headers and verifies their SHA-256 checksums.
It installs no system packages. CMake/Ninja are installed inside `.venv`; the build
helper finds them there. If no compiler is available, install `gcc-c++` on Fedora
or `build-essential` on Ubuntu. Subsequent builds with cached dependencies can use
`python cpp/build.py --offline`.

Output: `build/cpp/efficient_infer` and `build/cpp/build_info.json`. The binary
uses the same ONNX Runtime **1.31.0 shared library** as Python, ensuring that the
runtime version is held constant. The CMake preset is `cpp/CMakePresets.json`,
Release mode, without fast-math or fused floating-point contraction.

## Run C++ inference

```bash
./build/cpp/efficient_infer --config configs/cpp_cpu.json
```

`configs/cpp_cpu.json` processes the full image list from the existing FishEye8K
COCO JSON. Ground-truth annotations are not needed for inference: only the `images`
records are used. Omit `annotations` to enumerate a directory, or set `manifest`
to a JSON file containing an `images` list. Without a manifest, generated IDs follow
sorted filenames; they are not AICITY submission IDs.

Paths are resolved against `project_root`, which is relative to the configuration
file. The provided configs use `"project_root": ".."`, so model/dataset/output paths
resolve from the project root regardless of the shell's directory. Each run needs
an empty output directory; change `output` before repeating a run.

Important fields:

| Field | Meaning |
|---|---|
| `model`, `images`, `annotations`, `output` | Existing model/metadata pair, images, optional COCO image list and report directory |
| `threads` | ONNX intra-op threads; inter-op threads are fixed to 1 |
| `confidence`, `nms_iou`, `max_detections` | Same deployment filtering as Python; defaults 0.25 / 0.7 / 300 |
| `warmup`, `repeats` | Unmeasured warm-up predictions and measured passes |
| `max_images` | First N manifest images for standalone C++; 0 means all |
| `buffer_mode` | `reuse` keeps application input/output tensors; `fresh` allocates input storage each prediction and requests ORT-allocated outputs |
| `dump_tensors` | Save input/raw float32 tensors for the first N images, outside pipeline/request timing; default 0 for standalone runs |

Outputs: `summary.json` (stage mean/p50/p95/p99, FPS, process peak RSS, buffer
allocation counters), `per_image.json`, COCO-style `predictions.json` and the
effective `config.json`. Predictions are written once even when `repeats > 1`.
This C++ CLI writes JSON; it does not draw boxes or compute COCO AP itself.

## Compare Python and C++

```bash
python yolo/compare_cpp.py --config configs/compare_cpp_python.json
```

The default comparison uses **100 reproducibly selected images, seed 0, 4 threads,
20 warm-ups and 3 trials**. Each trial runs these backends in separate processes:

1. `python`: the existing Pillow/NumPy/`YoloSession` implementation.
2. `cpp_fresh`: the same C++ implementation with fresh input/ORT-allocated outputs.
3. `cpp_reuse`: preallocated input and application-owned output buffers.

Execution order rotates between trials. Input selection, model, confidence, NMS,
ONNX library and thread settings are identical. The comparison caps threads by
physical cores and process affinity. It retains image IDs and hashes in a manifest.
The standalone C++ config uses the requested thread count directly.

To compare all **2,712 images**, or choose a fresh report directory:

```bash
python yolo/compare_cpp.py --config configs/compare_cpp_python.json \
  --max-images 0 --output reports/cpp_vs_python_full
```

`--threads`, `--trials` and `--max-images` override the config. A positive image
limit uses a seeded sample rather than taking the first N images in the comparison.

Read `reports/cpp_vs_python/report.md` and these supporting artifacts:

| File | Contents |
|---|---|
| `summary.json` | Aggregate latency, FPS, peak RSS, per-trial means, speed ratios and fixed-threshold precision/recall/F1 on the selected subset |
| `parity.json` | Detection parity for every image/trial; input/raw tensor parity for the first 4 selected images |
| `per_image.csv` | Backend, trial, image ID, detection count and each stage's latency |
| `environment.json` | Hardware, versions, model/library/binary/source/dependency checksums |
| `manifest.json` | Exact selected image list, original IDs and image SHA-256 hashes |
| `trial_*/` | Per-backend predictions, stage timings, summaries and diagnostic tensors |

Parity requires exact input tensors, raw outputs within `atol=1e-4, rtol=1e-5`,
and the same number of same-class detections with box error <=0.01 pixel and score
error <=1e-5. Nonfinite values are rejected. Detection matching is one-to-one and
independent of output order, including equal-score/overlapping candidates. The
script writes diagnostics and exits with an error if parity fails.
The fixed-threshold F1 is at matching IoU 0.5; it is not the official AICITY score.

## Buffer ownership and measurement

The default C++ mode retains `std::vector<float>` input/output storage and wraps it
in `Ort::Value` tensors once. It uses the
[Session::Run overload accepting application-provided outputs](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_session.html).
The graph must have static input/output shapes. Resize plans are cached by source
dimensions, and preprocessing/NMS vectors retain capacity across images.

This reduces application tensor allocation and avoids allocating a new ONNX output
array for every image. Image decoding and ONNX Runtime internals can still allocate.
Buffer reuse is not a claim of zero allocation, zero-copy image loading or guaranteed
latency improvement. The `cpp_fresh` comparison measures its incremental effect.

Both runtimes use RGB, padding 114, Pillow-compatible bilinear letterboxing,
normalization by 255, CHW FP32, coordinate restoration and class-aware NMS. C++ uses
`stb_image` to decode images. Verified PNG inputs match Pillow; JPEG decoding may
differ and must pass the same parity gate before interpreting a speed comparison.

Pipeline timing covers preprocessing, inference and postprocessing. Request timing
adds file reading/decoding. Startup, warm-up, tensor dumps, metrics and final JSON
writing are excluded. FPS is image count divided by total measured stage seconds.
Dataset-pass FPS includes extra loop work and is retained in individual trial
artifacts, but is not compared as deployment FPS. Filesystem caches are not flushed.
RSS is the process high-water mark, including runtime libraries and retained results.

Compare the Python/C++ ratios and the fresh/reuse ratio separately. Faster NMS,
different image decoders and lower language/runtime overhead also contribute to a
Python/C++ difference. Three trials expose run-to-run variation; small differences
do not establish a statistically significant improvement.

## Checks

```bash
python -m pytest tests/test_cpp_runtime.py -q
```

These tests exercise the compiled executable: PNG input tensor parity for odd,
non-square, upsampled/downsampled and one-pixel inputs; same-class NMS/cross-class
overlap, invalid anchors, coordinate restoration, model checksum validation and
allocation counts for both buffer modes. Without the optional C++ build, compiled
runtime tests are skipped. Existing YOLO tests continue to run independently.

The existing `Dockerfile` packages the Python CPU runner. This C++ preset runs
natively; it is not included in that image. To relocate the executable, include
its ONNX shared library and license notices as described in
[cpp/THIRD_PARTY.md](../cpp/THIRD_PARTY.md).
