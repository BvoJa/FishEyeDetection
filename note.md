# YOLO11n training, pruning, and local inference

Train on Kaggle, export FP32 ONNX models, and compare them on your local CPU.

**Workflow:** prepare human labels → optional D-FINE annotation → YOLO11n training → structured pruning → ONNX export → CPU inference and benchmarking.

Pruning is evaluated directly: there is **no recovery fine-tuning, INT8 quantization, or knowledge-distillation training**.

## **1. Which files to run**

Run commands from the current `efficient_detection/` checkout. Files such as `yolo/train_yolo.py`, `dfine/setup_backend.py`, and `requirements/training.txt` are relative to this directory.

| File | Purpose | Environment |
|---|---|---|
| `notebooks/yolo_train.ipynb` | Run the full workflow and create the downloadable local bundle | Kaggle GPU |
| `yolo/prepare_yolo.py` | Prepare YOLO/COCO inputs and merge pseudo labels into training | Kaggle |
| `yolo/generate_pseudo_labels.py` | Generate hard annotations using D-FINE-S/M/L/X | Kaggle GPU |
| `yolo/train_yolo.py` | Train or resume the original YOLO11n baseline | Kaggle GPU |
| `yolo/prune_yolo.py` | Remove channels, reload the checkpoint, and validate without training | Kaggle |
| `yolo/export_yolo.py` | Export and verify static FP32 ONNX models | Kaggle |
| `yolo/package_local.py` | Package models, CPU code, and optional validation images | Kaggle |
| `yolo/infer_yolo.py` | Predict on an image or image directory | Local CPU |
| `yolo/benchmark_yolo.py` | Compare accuracy, latency, memory, resolution, and thread settings | Local CPU |
| `yolo/evaluate_yolo.py` | Evaluate a fixed model on every COCO-listed image; write AP/F1/FPS and `report.md` | Local CPU / Docker |
| `cpp/build.py` | Build the optional native ONNX Runtime executable | Local Linux |
| `cpp/src/main.cpp` | Native letterbox, normalization, inference, coordinate restoration and NMS | Local Linux CPU |
| `yolo/compare_cpp.py` | Compare Python/C++ fresh/reused buffers with parity checks and `report.md` | Local Linux CPU |

**Simplest route:** import `notebooks/yolo_train.ipynb` into Kaggle. Run it after your own clone/cd cell has changed to `efficient_detection/`. The configuration reads `Path.cwd()`; edit the dataset paths and run the cells in order. Download `yolo_local.zip`, then follow `LOCAL.md` inside it. The notebook packages all nine model variants and, by default, copies validation images for local benchmarking.

The separate D-FINE training workflow lives in `dfine/` and `notebooks/dfine_train.ipynb`. Shared dataset loading and class conventions live in `shared/`; local instructions and dependencies live in `docs/` and `requirements/`. Use the YOLO files listed above for this project.
See [docs/DFINE.md](docs/DFINE.md) for the optional D-FINE training/copy-paste/export entrypoints, and [docs/AUDIT.md](docs/AUDIT.md) for verified behavior and remaining measurement limits.

## **2. Set up Kaggle and prepare human labels**

Attach your dataset and any pretrained checkpoints. Enable GPU and Internet. Clone the current GitHub project and change to its root before running commands. In a Kaggle cell:

```python
%cd /kaggle/working
!git clone https://github.com/BvoJa/FishEyeDetection.git efficient_detection
%cd /kaggle/working/efficient_detection
%pip install -r requirements/training.txt
```

If you are already inside `efficient_detection/`, run only the dependency installation. The requirements install this project in editable mode so commands such as `python yolo/train_yolo.py` can resolve common imports. No nested `efficient_detection/` source directory or copy step is needed. Dataset paths continue to point to your attached `/kaggle/input/...` files. To update an existing checkout, run `!git pull --ff-only` from its current directory.

Class order: **bus (0), bike (1), car (2), pedestrian (3), truck (4)**.

For YOLO image/label folders:

```bash
python yolo/prepare_yolo.py --format yolo \
  --images /kaggle/input/fisheye/images/train \
  --labels /kaggle/input/fisheye/labels/train \
  --val-images /kaggle/input/fisheye/images/val \
  --val-labels /kaggle/input/fisheye/labels/val \
  --output /kaggle/working/yolo_workflow/prepared
```

For COCO JSON annotations:

```bash
python yolo/prepare_yolo.py --format coco \
  --images /kaggle/input/fisheye/images \
  --annotations /kaggle/input/fisheye/train.json \
  --val-images /kaggle/input/fisheye/images \
  --val-annotations /kaggle/input/fisheye/val.json \
  --output /kaggle/working/yolo_workflow/prepared
```

Run only the command matching your input format, using an empty output directory. COCO category IDs are remapped by name; person becomes pedestrian, and bicycle/motorcycle become bike. YOLO IDs must already follow the project order. Empty label files are valid negative images; missing labels raise an error.

Supply validation/test image directories together with their matching labels or JSON;
orphan label/annotation arguments are rejected. COCO crowd boxes remain in evaluation
JSON and are excluded from YOLO training labels. Non-crowd `ignore=1` annotations
are rejected by preparation because the training converters cannot preserve ignored
regions; review those annotations explicitly rather than silently treating them as positives.

Byte-identical image files within one split are retained with their individual human labels. Preparation reports counts of additional copies per split in the console and `manifest.json` (`within_split_duplicates`). Identical files across train, validation, or test splits still raise an error listing both paths, before any output is written.

If no validation split exists, omit the `--val-*` arguments and use `--split camera` (default). Whole cameras remain separate. Use `--split random` only when filenames lack camera IDs. Optional `--test-images` and matching `--test-labels`/`--test-annotations` preserve a separate test split.

Outputs: `data.yaml`, YOLO image/label folders, COCO evaluation JSON under `annotations/`, and `manifest.json`. Image paths in evaluation JSON are relative to the prepared `images/` directory. Default preparation uses symlinks to save Kaggle space; use `--mode copy` for independent image files.

### FishEye8K preset: flap1812/fisheye8k

The [Kaggle dataset](https://www.kaggle.com/datasets/flap1812/fisheye8k) contains
`Fisheye8K/train/{images,labels,annotations,train.json}` (5,288 images) and
`Fisheye8K/test/{images,labels,annotations,test.json}` (2,712 images).
YOLO TXT IDs already match bus, bike, car, pedestrian, truck (0–4). The existing
YOLO reader supports these five-column normalized labels without conversion rules.

The notebook now defaults to `/kaggle/input/fisheye8k/Fisheye8K`; change `DATA_ROOT`
if your Kaggle mount path differs. Keep `UNLABELED_IMAGES=None` for the labeled baseline.
For direct execution from the current project directory:

```bash
python yolo/prepare_yolo.py --format yolo \
  --images /kaggle/input/fisheye8k/Fisheye8K/train/images \
  --labels /kaggle/input/fisheye8k/Fisheye8K/train/labels \
  --val-images /kaggle/input/fisheye8k/Fisheye8K/test/images \
  --val-labels /kaggle/input/fisheye8k/Fisheye8K/test/labels \
  --output /kaggle/working/yolo_workflow/prepared

python yolo/train_yolo.py \
  --data /kaggle/working/yolo_workflow/prepared/data.yaml \
  --model yolo11n.pt --image-size 640 --epochs 30 \
  --batch-size 32 --seed 0 --device 0,1 \
  --output /kaggle/working/yolo_workflow/runs/baseline
```

This follows [AICITY Track 4](https://www.aicitychallenge.org/2025-track4/), which
uses the original FishEye8K test split as validation. For an untouched FishEye8K
test set instead, omit the `--val-*` flags, split the original training set by camera,
and supply `--test-images .../test/images --test-labels .../test/labels`.
The CLI supports this; the notebook preset uses the AICITY validation convention.

## **3. Optional D-FINE annotation**

Prepare human splits first so annotation can exclude labeled images and held-out cameras. Choose **dfine-s**, **dfine-m**, **dfine-l**, or **dfine-x**; S is the default. Download the matching checkpoint from the [D-FINE project](https://github.com/Peterande/D-FINE), or attach your own matching checkpoint.

The notebook sets up the pinned D-FINE backend when annotation is enabled. For manual CLI use, run setup first:

```bash
python dfine/setup_backend.py
python yolo/generate_pseudo_labels.py \
  --annotation-model dfine-s \
  --checkpoint /kaggle/input/dfine-weights/dfine_s.pth \
  --images /kaggle/input/additional-training/images \
  --data /kaggle/working/yolo_workflow/prepared/data.yaml \
  --output /kaggle/working/yolo_workflow/pseudo \
  --confidence 0.5 --device cuda

python yolo/prepare_yolo.py \
  --data /kaggle/working/yolo_workflow/prepared/data.yaml \
  --add-pseudo /kaggle/working/yolo_workflow/pseudo
```

Replace the example checkpoint filename with your actual file. COCO checkpoints use 80 classes; add `--checkpoint-classes 5` for a custom checkpoint using the five project classes. Architecture mismatches fail rather than partially loading weights.

Outputs: five-column YOLO labels and `annotations.json` containing scores, source-image hashes, and generator settings. Images with no accepted detections are skipped. Merging preserves human labels and validation/test splits, and repeated merges do not duplicate images. Skip this section when you do not need additional annotations.
Custom five-class annotation checkpoints must use the project class order. When
class-name metadata exists it is checked; otherwise class semantics must be
confirmed from the checkpoint's training setup, since weights alone do not identify class names.

## **4. Train YOLO11n once**

```bash
python yolo/train_yolo.py \
  --data /kaggle/working/yolo_workflow/prepared/data.yaml \
  --model yolo11n.pt --image-size 640 --epochs 30 \
  --batch-size 32 --seed 0 --device 0,1 \
  --output /kaggle/working/yolo_workflow/runs/baseline
```

Use an attached pretrained `.pt` path instead of `yolo11n.pt` for offline training. The best validation checkpoint is saved as `runs/baseline/weights/best.pt`. Interrupted baseline runs can resume using `--resume /path/to/last.pt`; pruned checkpoints are rejected by the training script.

## **5. Apply structured pruning without recovery training**

```bash
python yolo/prune_yolo.py \
  --checkpoint /kaggle/working/yolo_workflow/runs/baseline/weights/best.pt \
  --data /kaggle/working/yolo_workflow/prepared/data.yaml \
  --prune-ratio 0.1 --image-size 640 --device 0 \
  --output /kaggle/working/yolo_workflow/runs/pruned10

python yolo/prune_yolo.py \
  --checkpoint /kaggle/working/yolo_workflow/runs/baseline/weights/best.pt \
  --data /kaggle/working/yolo_workflow/prepared/data.yaml \
  --prune-ratio 0.2 --image-size 640 --device 0 \
  --output /kaggle/working/yolo_workflow/runs/pruned20
```

Each candidate starts from the original baseline. Dependency-aware pruning physically removes low-magnitude internal bottleneck channels while preserving attention and detection interfaces. Retained channels are rounded to multiples of eight. The requested percentage applies to eligible channels, not total model parameters; actual reductions are recorded in `pruning.json`.

Each output contains `pruned.pt`, validation results, and a report confirming **zero optimizer steps**. There is no recovery stage. Accuracy loss is measured directly, and a pruned model need not become the deployment model.

## **6. Export FP32 ONNX models**

```bash
python yolo/export_yolo.py \
  --checkpoint /kaggle/working/yolo_workflow/runs/baseline/weights/best.pt \
  --image-size 640 \
  --output /kaggle/working/yolo_workflow/models/baseline_640.onnx
```

Repeat for `runs/pruned10/pruned.pt` and `runs/pruned20/pruned.pt`, and for sizes **512, 640, and 960**, naming outputs `pruned10_512.onnx`, etc. The notebook performs all nine exports automatically.

The exporter checks the ONNX graph and compares raw predictions with PyTorch. Add `--test-image /path/to/validation/image.png` to verify a real image too; the notebook does this automatically. Keep each `.onnx` file together with its matching `.json` metadata. Exports are static batch-1 FP32, with RGB letterboxing and external class-aware NMS.

To create the same portable bundle from the command line:

```bash
python yolo/package_local.py \
  --models /kaggle/working/yolo_workflow/models \
  --data /kaggle/working/yolo_workflow/prepared/data.yaml \
  --output /kaggle/working/yolo_local.zip
```

`--data` includes independent validation image copies. Omit it for an inference-only bundle. The notebook calls this packaging module automatically.

## **7. Run and benchmark on the local CPU**

Extract the notebook's `yolo_local.zip`. Use Python 3.11 in a virtual environment, then run these commands from the extracted folder:

```bash
python -m pip install -r requirements/inference.txt
python -m efficient_detection.yolo.infer_yolo --model models/baseline_640.onnx \
  --input /path/to/images --output predictions
```

Outputs: annotated images, `predictions.json`, `images.json`, and stage timing statistics in `summary.json`. Console/JSON include inference FPS (model only), pipeline FPS (preprocessing + inference + NMS), request FPS (also image decoding) and dataset FPS (the image loop, including optional image writing). FPS is image count divided by total measured seconds, excluding startup and warm-up. Image decoding and file writing are excluded from pipeline latency. PyTorch, D-FINE weights, CUDA, and TensorRT are not required locally.

For accuracy and performance comparisons:

```bash
python -m pip install -r requirements/benchmark.txt
python -m efficient_detection.yolo.benchmark_yolo --models "models/*.onnx" \
  --baseline models/baseline_640.onnx \
  --data validation/data.yaml --output reports \
  --threads 1 2 4 --profile
```

The quoted model wildcard works on all supported platforms. Add `--graph-optimizations off all` to compare graph optimization settings. Thread counts are capped by physical core count. Defaults: 20 warm-up iterations, 100 measured iterations per trial, and 3 trials. Accuracy is evaluated once per model using a fixed configuration; latency/memory checks use fresh worker processes and profiling runs separately.

Reports include AP50:95, small-object AP, parameters, model size, peak process RSS, inference/pipeline FPS, p50/p95 latency for preprocessing/inference/NMS, `results.csv`, `accuracy_latency.png`, and `selection.json`. Benchmark timing uses cached RGB inputs and excludes image I/O. Peak RSS includes session startup and cached RGB inputs, not just weights. The selected model has the lowest pipeline p95 latency within **2 absolute AP points** of the original baseline. The original model remains a candidate.

Use validation for selection. If you prepared a separate test split, run the benchmark with `--split test --selected reports/selection.json`; supply the full prepared dataset locally. That evaluates the existing selection without choosing again on test results.

## **8. Docker deployment and full original FishEye8K evaluation**

See [docs/DOCKER.md](docs/DOCKER.md) to build the CPU Docker image, bundle the ONNX/JSON
model pair, and mount `datasets/Fisheye8K/test` read-only. The entrypoint
`yolo/evaluate_yolo.py` evaluates all images listed in `test.json` without requiring
`prepare_yolo.py` or a training split. It writes COCO AP, precision/recall, F1,
Detection accuracy (TP/(TP+FP+FN)), inference/pipeline/request/dataset FPS, stage latency, memory,
per-image/per-class CSVs and a Markdown report. This is evaluation of one fixed
model; it does not select or train model candidates on the test split.

## **9. Optional C++ runtime and Python comparison**

From the current project directory, build the native CPU runtime and run the
comparison preset:

```bash
python -m pip install -r requirements/cpp.txt
python cpp/build.py
python yolo/compare_cpp.py --config configs/compare_cpp_python.json
```

The preset compares the current Python runtime, C++ with fresh buffers and C++
with reusable input/output tensors on the same 100 sampled images, 4 threads and
3 trials. It verifies detection parity on all selected images and input/raw
tensor parity on 4 images. Read `reports/cpp_vs_python/report.md` for stage latency,
FPS, RSS and speed ratios. For all images, add `--max-images 0` and a fresh `--output`.

Standalone C++ inference: `./build/cpp/efficient_infer --config configs/cpp_cpu.json`.
This preset processes the full COCO image list and writes JSON predictions/timings.
See [docs/CPP.md](docs/CPP.md) for configuration, parity tolerances and timing scope.

## **10. Verification**

```bash
python -m pip install pytest
python -m pytest tests -q
```

Checks cover input formats, annotation model presets, class mapping, split exclusion, pseudo-label merging, physical pruning, checkpoint reload, rejection of pruned training, letterbox coordinate restoration, AP calculation, and PyTorch–ONNX parity. Small synthetic CPU runs verify the command workflow; real Kaggle training and real-data accuracy must be measured using your dataset.
