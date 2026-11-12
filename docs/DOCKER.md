# CPU Docker deployment and full FishEye8K evaluation

Run from the `efficient_detection/` checkout. The image packages the CPU runtime,
the FP32 ONNX model and its matching JSON metadata. It does not need PyTorch,
Ultralytics, CUDA or D-FINE. The dataset remains on the host and is mounted read-only.

The implementation follows this sequence: export and verify the trained model;
build a versioned CPU image; run every image listed in the original COCO JSON;
compute accuracy and deployment statistics; retain reports outside the container.
Docker makes the deployment repeatable. Any acceleration must be demonstrated by
measurements with the same model, data, thresholds and CPU settings.

## Export and build

If the matching model pair is not present:

```bash
source .venv/bin/activate
python yolo/export_yolo.py --checkpoint best.pt --image-size 640 \
  --test-image datasets/Fisheye8K/test/images/camera1_A_0.png \
  --output models/baseline_640.onnx
```

Build with the project directory as context:

```bash
docker build --tag efficient-detection:cpu .
```

`Dockerfile` builds the Python package in a separate stage, installs the exact
runtime versions in `requirements/evaluation.lock.txt`, and runs as a non-root user.
`.dockerignore` excludes datasets, checkpoints, virtual environments and reports.
Only the selected ONNX/JSON pair is copied into the final image. To package another
export, supply both `--build-arg MODEL=models/other.onnx` and
`--build-arg MODEL_METADATA=models/other.json`.

## Run all 2,712 test images

Create a fresh output directory, then run:

```bash
mkdir -p reports/fisheye8k_docker
docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd)/datasets/Fisheye8K/test,target=/data,readonly" \
  --mount "type=bind,source=$(pwd)/reports/fisheye8k_docker,target=/output" \
  efficient-detection:cpu
```

These Linux bind-mount commands preserve output ownership. On Fedora with SELinux,
use the corresponding volume syntax with relabeling if bind-mount access is denied:

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$(pwd)/datasets/Fisheye8K/test:/data:ro,Z" \
  -v "$(pwd)/reports/fisheye8k_docker:/output:Z" \
  efficient-detection:cpu
```

The default command uses `/app/models/model.onnx`, 4 CPU threads, batch 1, 20 warm-up
iterations, confidence 0.25 and NMS IoU 0.7. Threads are capped by physical cores
and process CPU affinity. Every COCO image is then measured once. The command
rejects a nonempty output directory so different runs cannot be mixed.

To override options, provide the complete evaluation arguments after the image name:

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$(pwd)/datasets/Fisheye8K/test:/data:ro,Z" \
  -v "$(pwd)/reports/fisheye8k_docker_threads2:/output:Z" \
  efficient-detection:cpu \
  --model /app/models/model.onnx --images /data/images \
  --annotations /data/test.json --output /output --threads 2
```

Create the chosen host output directory first. Add `--save-images` to save all
annotated images; image encoding/writing is excluded from pipeline latency.

## Reports and interpretation

The output directory contains:

| File | Contents |
|---|---|
| `report.md` | Readable accuracy, latency, throughput, memory and class tables |
| `summary.json` | Model/dataset checksums, counts, COCO AP, FPS, timing percentiles and RSS |
| `per_image.csv` | Original image IDs, detection/TP/FP/FN counts, fixed-threshold metrics and stage timings |
| `per_class.csv` | Ground-truth/detection counts, AP/AP50, precision, recall, F1 and Detection accuracy by class |
| `predictions.json` | COCO predictions at the low AP threshold, preserving original image IDs |
| `deployment_predictions.json` | Single-label predictions at the deployment confidence, used for fixed-threshold metrics |
| `environment.json` | Hardware, Python/library versions and native/container execution mode |

Accuracy includes AP50:95, AP50, AP75, small/medium/large-object AP and per-class AP.
Fixed-threshold metrics additionally include precision, recall, F1 micro/macro,
TP/FP/FN and **Detection accuracy**, globally and per class (per-image metrics are
also in `per_image.csv`). Defaults are confidence 0.25 and matching IoU 0.5,
configurable with `--confidence` and `--match-iou`. Matching IoU is distinct from
the suppression threshold `--nms-iou`.

Detections are matched in descending score order to the highest-IoU unused ground
truth of the same class. Wrong classes, duplicate detections and localization
errors contribute false positives; unmatched normal ground truths contribute
false negatives. Ignored/crowd annotations are excluded; crowd overlap uses
intersection divided by detection area and can ignore multiple predictions.

- Precision = TP / (TP + FP).
- Recall = TP / (TP + FN).
- F1 micro = 2TP / (2TP + FP + FN), pooling counts across images/classes.
- F1 macro averages class F1 over classes with nonignored ground truth or predictions.
- Detection accuracy = TP / (TP + FP + FN), an explicitly defined Jaccard-style
  detection score. This is **not classification accuracy**; true negatives are
  undefined for the detection task. Zero denominators produce zero.

These fixed-threshold scores are separate from COCO AP; changing `--match-iou`
does not change AP. Metric matching is outside the deployment latency measurement.
For each image, one model forward pass supplies two decoders: confidence 0.25,
single-label decoding for deployment timing/counts; confidence 0.001, multi-label
decoding for COCO AP. Extra AP decoding and COCO evaluation are outside the reported
deployment latency. Standard COCO evaluation uses maxDets=100.

Pipeline latency covers preprocessing, ONNX inference and coordinate restoration/NMS.
Request latency additionally includes image reading/decoding. Both exclude startup,
warm-up, report writing and additional AP decoding. Dataset-pass throughput includes
the additional AP processing; it should not be confused with pipeline FPS.

Console output, `summary.json` and `report.md` include these separate FPS statistics:

| JSON field | Included work |
|---|---|
| `inference_fps` | ONNX model execution only |
| `pipeline_fps` | RGB preprocessing, inference, coordinate restoration and NMS |
| `request_fps` | Image reading/decoding and the deployment pipeline |
| `dataset_fps` | Wall-clock image loop, including prediction formatting, fixed-threshold matching, extra AP decoding, progress logging and optional image saving |

For serial batch-1 processing, stage FPS is `N / sum(stage_seconds)`, equivalent
to `1000 / mean_stage_ms`. Per-image FPS values are not averaged. Dataset FPS is
`N / dataset_pass_seconds`; the existing `dataset_images_per_second` field is kept
with the same value. Dataset timing excludes startup, warm-up, final COCO evaluation
and final report/JSON/CSV writing. These are measurements on the current CPU, not
Jetson measurements. Rebuild the image after updating the code to obtain these fields.

RSS includes Python, annotations and retained predictions. Both inference-pass peak
and the peak including COCO evaluation are reported; these are not model-only memory.
The original FishEye8K test split is used as validation in the project's AICITY
preset. If your checkpoint used that preset, label the resulting accuracy as
validation performance, not untouched held-out test performance. Do not select
new pruning/resolution configurations using a genuinely held-out test set.

These F1 metrics use the documented fixed matching IoU (default 0.5). They are not
the official AICITY leaderboard F1. Small-object AP uses standard COCO area ranges,
not the competition's modified size ranges. The Docker image runs the Python CPU
evaluator; native C++ runs separately as described in [CPP.md](CPP.md).

## Native comparison and reproducibility

The same code runs without Docker:

```bash
source .venv/bin/activate
python -m pip install -r requirements/evaluation.lock.txt -e .
python yolo/evaluate_yolo.py --model models/baseline_640.onnx \
  --images datasets/Fisheye8K/test/images \
  --annotations datasets/Fisheye8K/test/test.json \
  --output reports/fisheye8k_native --threads 4
```

Use identical model/dataset hashes and runtime versions for comparisons. Compare
accuracy and latency separately; do not claim a Docker speedup from measurements
that use different images or thresholds. Record the built image ID/base image
digest alongside the reports using `docker image inspect efficient-detection:cpu`.
Rebuild after changing runtime code or the bundled model. Runtime execution requires
no network access. Pushing/publishing the image is a separate step.
