# Efficient Detection: YOLO training and CPU inference

Run from the current `efficient_detection/` directory. Code, requirements, notebooks,
and documentation can be accessed directly using their relative paths.
See [note.md](note.md) for the complete workflow.

For native C++ inference with preallocated ONNX input/output buffers and a measured
comparison against Python, follow [docs/CPP.md](docs/CPP.md). Runnable presets live
in `configs/cpp_cpu.json` and `configs/compare_cpp_python.json`.
See [docs/CPP_RESULTS.md](docs/CPP_RESULTS.md) for the measured 100-image comparison
and its correctness checks and measurement limits.
See [docs/AUDIT.md](docs/AUDIT.md) for the code/documentation audit and verified limits,
and [docs/DFINE.md](docs/DFINE.md) for the separate optional D-FINE training workflow.

For a CPU Docker image with the ONNX model and full FishEye8K evaluation, follow
[docs/DOCKER.md](docs/DOCKER.md). It produces COCO accuracy, per-image and per-class
statistics, precision/recall, F1 and explicitly defined detection accuracy,
latency percentiles and memory reports directly from the test JSON;
no prepared training dataset is required.

In a Kaggle cell, clone into the directory name you will use and change to it:

```python
%cd /kaggle/working
!git clone https://github.com/BvoJa/FishEyeDetection.git efficient_detection
%cd /kaggle/working/efficient_detection
%pip install -r requirements/training.txt
```

If you are already inside the checkout, start with the dependency installation.
It registers this flat source layout in editable mode, allowing direct script
execution while preserving shared imports.

```python
!python yolo/prepare_yolo.py --help
!python yolo/train_yolo.py --help
!python yolo/package_local.py --help
```

Import [notebooks/yolo_train.ipynb](notebooks/yolo_train.ipynb) or use its cells in
your Kaggle notebook. The notebook clones or updates this repository into
`/kaggle/working/efficient_detection` itself and then runs files such as
`yolo/train_yolo.py` directly. Enable GPU and Internet and edit the attached
dataset/checkpoint paths.

```text
efficient_detection/               # Current working directory / Git checkout
├── yolo/                          # Prepare, annotate, train, prune, export, infer, benchmark, package
├── dfine/                         # Annotation backend setup and original D-FINE workflow
├── shared/                        # Dataset readers and common utilities
├── cpp/                           # Native CPU runtime, CMake preset and build helper
├── configs/                       # C++ inference and Python/C++ comparison presets
├── notebooks/                     # yolo_train.ipynb and dfine_train.ipynb
├── requirements/                  # training.txt, inference.txt, benchmark.txt
├── docs/                          # Local deployment guides
├── tests/
├── pyproject.toml                 # Import registration for direct script execution
├── third_party/D-FINE/            # Downloaded only when needed; excluded from Git
└── note.md
```

Main capabilities:

- Prepare YOLO/COCO data with five-class mapping, camera splits, and duplicate checks.
- Generate optional pseudo labels using D-FINE-S/M/L/X, preserving human labels and held-out splits.
- Train/resume pretrained YOLO11n and retain its best validation checkpoint.
- Physically prune internal channels at 10%/20% targets and validate without recovery training.
- Export static FP32 ONNX at 512/640/960 with PyTorch parity checks.
- Run CPU inference with RGB letterboxing, coordinate restoration, NMS, and image/JSON outputs.
- Run native C++ inference with reusable tensors; compare Python/C++ fresh/reuse modes with input/output/detection parity checks.
- Benchmark AP, small-object AP, inference/pipeline FPS, stage p50/p95 latency, memory, model size and parameters.
- Report request FPS including image decoding and dataset-pass FPS including evaluation-loop overhead for full-dataset runs.
- Package the CPU runtime/model in Docker and evaluate every original COCO test image.
- Select the lowest pipeline p95 within 2 AP points of the baseline and package models for local use.

D-FINE source is needed only for annotation or original D-FINE training. The
notebook fetches a pinned revision when needed. For manual use:

```bash
python dfine/setup_backend.py
```

An existing D-FINE checkout can be supplied with `--repo-root` to annotation or
legacy commands. Model configurations are included in this project.

Defaults: YOLO11n, 640 pixels, 30 epochs, seed 0; FP32 ONNX at 512/640/960. Classes
are bus, bike, car, pedestrian, truck (0–4). There is no recovery training,
INT8 quantization, or knowledge-distillation stage in the YOLO workflow.

With the default Kaggle checkout location, outputs are stored next to the checkout:
`/kaggle/working/yolo_workflow/` and `/kaggle/working/yolo_local.zip`.
Extract the zip locally and follow `LOCAL.md`; it includes the CPU package layout
and does not require installing this training project.

```bash
python -m pip install pytest
python -m pytest tests -q
```

References: [D-FINE](https://github.com/Peterande/D-FINE),
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/),
[Torch-Pruning](https://github.com/VainF/Torch-Pruning),
[ONNX Runtime profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
