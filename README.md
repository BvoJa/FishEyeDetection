# Efficient Detection: YOLO optimization and CPU inference

Start with [note.md](note.md) for the complete execution guide, or import
[notebooks/yolo_train.ipynb](notebooks/yolo_train.ipynb) into Kaggle.

Main capabilities:

- Prepare five-class datasets from YOLO folders or COCO JSON, with camera splits and duplicate-image checks.
- Generate optional pseudo labels with D-FINE-S/M/L/X, preserving human labels and excluding held-out images/cameras.
- Train or resume a pretrained YOLO11n baseline using a reproducible seed.
- Physically prune internal channels at 10%/20% targets, report actual reductions, and validate without recovery training.
- Export static FP32 ONNX at 512/640/960 and verify predictions against PyTorch.
- Run local CPU inference with letterboxing, coordinate restoration, class-aware NMS, and annotated/JSON outputs.
- Compare CPU threads and ONNX Runtime graph optimization, with separate preprocessing/inference/postprocessing timings.
- Benchmark COCO AP, small-object AP, p50/p95 latency, memory, parameters and model size; select the fastest candidate within 2 AP points of the baseline.
- Package models, metadata, CPU-only code and optional validation images into a portable local bundle.

```text
efficient_detection/
├── yolo/                    # Main YOLO11 workflow and local CPU runtime
│   ├── generate_pseudo_labels.py
│   ├── prepare_yolo.py
│   ├── train_yolo.py
│   ├── prune_yolo.py
│   ├── export_yolo.py
│   ├── infer_yolo.py
│   ├── benchmark_yolo.py
│   ├── package_local.py     # Portable models + CPU code + validation data
│   ├── models.py            # YOLO checkpoint loading/saving
│   └── runtime.py           # Letterbox, ONNX Runtime, decoding and NMS
├── dfine/                   # D-FINE annotator and original D-FINE workflow
│   ├── annotator.py         # S/M/L/X presets and strict checkpoint loading
│   ├── config.py
│   ├── prepare_data.py
│   ├── copy_paste.py
│   ├── train.py
│   ├── export_onnx.py
│   └── infer.py
├── shared/
│   ├── common.py            # Class order, hashes, JSON, boxes and metrics
│   └── datasets.py          # YOLO/COCO readers and camera/random splitting
├── notebooks/
│   ├── yolo_train.ipynb   # Recommended YOLO workflow
│   └── dfine_train.ipynb    # Original D-FINE training workflow
├── docs/
│   ├── LOCAL.md             # YOLO local inference and benchmarking
│   └── DFINE_LOCAL.md       # Original D-FINE local inference
├── requirements/
│   ├── training.txt         # Kaggle training, pruning and export
│   ├── inference.txt        # Minimal local CPU inference dependencies
│   └── benchmark.txt        # Additional CPU benchmark dependencies
└── tests/
```

Use package entrypoints from the **repository root**, so imports resolve consistently:

```bash
python -m pip install -r efficient_detection/requirements/training.txt
python -m efficient_detection.yolo.prepare_yolo --help
python -m efficient_detection.yolo.train_yolo --help
python -m efficient_detection.yolo.package_local --help
```

Run the YOLO stages in this order:

1. Prepare human-labeled splits with `yolo.prepare_yolo`.
2. Optionally annotate extra training images with `yolo.generate_pseudo_labels`, then merge using `yolo.prepare_yolo --add-pseudo`.
3. Train the YOLO11n baseline with `yolo.train_yolo`.
4. Prune each candidate from that baseline using `yolo.prune_yolo`.
5. Export each model/resolution with `yolo.export_yolo` and build the zip using `yolo.package_local`.
6. Extract the zip locally, follow `LOCAL.md`, and run `yolo.infer_yolo` or `yolo.benchmark_yolo` through the same `efficient_detection` package.

D-FINE-S is the default annotator; S/M/L/X checkpoints must match their selected
architecture. Classes are bus, bike, car, pedestrian, truck (IDs 0–4). Human labels
take precedence; held-out image hashes and cameras are excluded from pseudo labels.

Defaults: pretrained YOLO11n, 640 pixels, 30 epochs, seed 0; pruning targets 10% and
20% of eligible internal channels; static FP32 exports at 512/640/960. There is no
recovery fine-tuning, INT8 quantization, or knowledge-distillation training. Pruning
reports actual channel/parameter reductions and validates each modified checkpoint.

The CPU benchmark sweeps threads 1/2/4, capped by physical cores. It reports COCO
AP, small-object AP, model size, parameters, peak process RSS, and stage p50/p95
latency. Selection uses the lowest pipeline p95 within 2 AP points of the original
baseline. The original model remains eligible.

Local bundles contain the package files needed for CPU inference/benchmarking,
ONNX files and metadata, requirements, instructions, and optional independent
validation image copies. They do not require PyTorch, Ultralytics, or D-FINE weights.

```bash
python -m pip install pytest
python -m pytest efficient_detection/tests -q
```

References: [D-FINE](https://github.com/Peterande/D-FINE),
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/),
[Torch-Pruning](https://github.com/VainF/Torch-Pruning),
[ONNX Runtime profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
