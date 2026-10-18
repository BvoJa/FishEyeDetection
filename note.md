# YOLO11n training, pruning, and local inference

Train on Kaggle, export FP32 ONNX models, and compare them on your local CPU.

**Workflow:** prepare human labels → optional D-FINE annotation → YOLO11n training → structured pruning → ONNX export → CPU inference and benchmarking.

Pruning is evaluated directly: there is **no recovery fine-tuning, INT8 quantization, or knowledge-distillation training**.

## **1. Which files to run**

All files below are implemented under `efficient_detection/`. Run module commands (`python -m ...`) from the repository root; notebook subprocesses set this directory automatically.

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

**Simplest route:** import `efficient_detection/notebooks/yolo_train.ipynb` into Kaggle, edit the first configuration cell, and run the cells in order. Download `yolo_local.zip`, then follow `LOCAL.md` inside it. The notebook packages all nine model variants and, by default, copies validation images for local benchmarking.

The separate D-FINE training workflow lives in `efficient_detection/dfine/` and `notebooks/dfine_train.ipynb`. Shared dataset loading and class conventions live in `shared/`; local instructions and dependencies live in `docs/` and `requirements/`. Use the YOLO files listed above for this project.

## **2. Set up Kaggle and prepare human labels**

Attach this repository, your dataset, and any pretrained checkpoints. Enable a GPU accelerator. Internet is needed to install packages and to download YOLO weights when an attached checkpoint is not supplied.

```bash
python -m pip install -r efficient_detection/requirements/training.txt
```

Class order: **bus (0), bike (1), car (2), pedestrian (3), truck (4)**.

For YOLO image/label folders:

```bash
python -m efficient_detection.yolo.prepare_yolo --format yolo \
  --images /kaggle/input/fisheye/images/train \
  --labels /kaggle/input/fisheye/labels/train \
  --val-images /kaggle/input/fisheye/images/val \
  --val-labels /kaggle/input/fisheye/labels/val \
  --output /kaggle/working/prepared
```

For COCO JSON annotations:

```bash
python -m efficient_detection.yolo.prepare_yolo --format coco \
  --images /kaggle/input/fisheye/images \
  --annotations /kaggle/input/fisheye/train.json \
  --val-images /kaggle/input/fisheye/images \
  --val-annotations /kaggle/input/fisheye/val.json \
  --output /kaggle/working/prepared
```

Run only the command matching your input format, using an empty output directory. COCO category IDs are remapped by name; person becomes pedestrian, and bicycle/motorcycle become bike. YOLO IDs must already follow the project order. Empty label files are valid negative images; missing labels raise an error.

If no validation split exists, omit the `--val-*` arguments and use `--split camera` (default). Whole cameras remain separate. Use `--split random` only when filenames lack camera IDs. Optional `--test-images` and matching `--test-labels`/`--test-annotations` preserve a separate test split.

Outputs: `data.yaml`, YOLO image/label folders, COCO evaluation JSON under `annotations/`, and `manifest.json`. Image paths in evaluation JSON are relative to the prepared `images/` directory. Default preparation uses symlinks to save Kaggle space; use `--mode copy` for independent image files.

## **3. Optional D-FINE annotation**

Prepare human splits first so annotation can exclude labeled images and held-out cameras. Choose **dfine-s**, **dfine-m**, **dfine-l**, or **dfine-x**; S is the default. Download the matching checkpoint from the [D-FINE project](https://github.com/Peterande/D-FINE), or attach your own matching checkpoint.

```bash
python -m efficient_detection.yolo.generate_pseudo_labels \
  --annotation-model dfine-s \
  --checkpoint /kaggle/input/dfine-weights/dfine_s.pth \
  --images /kaggle/input/additional-training/images \
  --data /kaggle/working/prepared/data.yaml \
  --output /kaggle/working/pseudo \
  --confidence 0.5 --device cuda

python -m efficient_detection.yolo.prepare_yolo \
  --data /kaggle/working/prepared/data.yaml \
  --add-pseudo /kaggle/working/pseudo
```

Replace the example checkpoint filename with your actual file. COCO checkpoints use 80 classes; add `--checkpoint-classes 5` for a custom checkpoint using the five project classes. Architecture mismatches fail rather than partially loading weights.

Outputs: five-column YOLO labels and `annotations.json` containing scores, source-image hashes, and generator settings. Images with no accepted detections are skipped. Merging preserves human labels and validation/test splits, and repeated merges do not duplicate images. Skip this section when you do not need additional annotations.

## **4. Train YOLO11n once**

```bash
python -m efficient_detection.yolo.train_yolo \
  --data /kaggle/working/prepared/data.yaml \
  --model yolo11n.pt --image-size 640 --epochs 30 \
  --batch-size 2 --seed 0 --device 0 \
  --output /kaggle/working/runs/baseline
```

Use an attached pretrained `.pt` path instead of `yolo11n.pt` for offline training. The best validation checkpoint is saved as `runs/baseline/weights/best.pt`. Interrupted baseline runs can resume using `--resume /path/to/last.pt`; pruned checkpoints are rejected by the training script.

## **5. Apply structured pruning without recovery training**

```bash
python -m efficient_detection.yolo.prune_yolo \
  --checkpoint /kaggle/working/runs/baseline/weights/best.pt \
  --data /kaggle/working/prepared/data.yaml \
  --prune-ratio 0.1 --image-size 640 --device 0 \
  --output /kaggle/working/runs/pruned10

python -m efficient_detection.yolo.prune_yolo \
  --checkpoint /kaggle/working/runs/baseline/weights/best.pt \
  --data /kaggle/working/prepared/data.yaml \
  --prune-ratio 0.2 --image-size 640 --device 0 \
  --output /kaggle/working/runs/pruned20
```

Each candidate starts from the original baseline. Dependency-aware pruning physically removes low-magnitude internal bottleneck channels while preserving attention and detection interfaces. Retained channels are rounded to multiples of eight. The requested percentage applies to eligible channels, not total model parameters; actual reductions are recorded in `pruning.json`.

Each output contains `pruned.pt`, validation results, and a report confirming **zero optimizer steps**. There is no recovery stage. Accuracy loss is measured directly, and a pruned model need not become the deployment model.

## **6. Export FP32 ONNX models**

```bash
python -m efficient_detection.yolo.export_yolo \
  --checkpoint /kaggle/working/runs/baseline/weights/best.pt \
  --image-size 640 \
  --output /kaggle/working/models/baseline_640.onnx
```

Repeat for `runs/pruned10/pruned.pt` and `runs/pruned20/pruned.pt`, and for sizes **512, 640, and 960**, naming outputs `pruned10_512.onnx`, etc. The notebook performs all nine exports automatically.

The exporter checks the ONNX graph and compares raw predictions with PyTorch. Add `--test-image /path/to/validation/image.png` to verify a real image too; the notebook does this automatically. Keep each `.onnx` file together with its matching `.json` metadata. Exports are static batch-1 FP32, with RGB letterboxing and external class-aware NMS.

To create the same portable bundle from the command line:

```bash
python -m efficient_detection.yolo.package_local \
  --models /kaggle/working/models \
  --data /kaggle/working/prepared/data.yaml \
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

Outputs: annotated images, `predictions.json`, `images.json`, and stage timing statistics in `summary.json`. Image decoding and file writing are excluded from the reported pipeline latency. PyTorch, D-FINE weights, CUDA, and TensorRT are not required locally.

For accuracy and performance comparisons:

```bash
python -m pip install -r requirements/benchmark.txt
python -m efficient_detection.yolo.benchmark_yolo --models "models/*.onnx" \
  --baseline models/baseline_640.onnx \
  --data validation/data.yaml --output reports \
  --threads 1 2 4 --profile
```

The quoted model wildcard works on all supported platforms. Add `--graph-optimizations off all` to compare graph optimization settings. Thread counts are capped by physical core count. Defaults: 20 warm-up iterations, 100 measured iterations per trial, and 3 trials. Accuracy is evaluated once per model using a fixed configuration; latency/memory checks use fresh worker processes and profiling runs separately.

Reports include AP50:95, small-object AP, parameters, model size, peak process RSS, p50/p95 latency for preprocessing/inference/NMS, `results.csv`, `accuracy_latency.png`, and `selection.json`. Peak RSS includes session startup and cached RGB inputs, not just weights. The selected model has the lowest end-to-end p95 latency within **2 absolute AP points** of the original baseline. The original model remains a candidate.

Use validation for selection. If you prepared a separate test split, run the benchmark with `--split test --selected reports/selection.json`; supply the full prepared dataset locally. That evaluates the existing selection without choosing again on test results.

## **8. Verification**

```bash
python -m pip install pytest
python -m pytest efficient_detection/tests -q
```

Checks cover input formats, annotation model presets, class mapping, split exclusion, pseudo-label merging, physical pruning, checkpoint reload, rejection of pruned training, letterbox coordinate restoration, AP calculation, and PyTorch–ONNX parity. Small synthetic CPU runs verify the command workflow; real Kaggle training and real-data accuracy must be measured using your dataset.
