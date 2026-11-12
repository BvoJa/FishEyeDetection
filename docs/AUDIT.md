# Code and documentation audit — 2026-10-19

The YOLO/ONNX workflow and native C++ runtime contain working implementations,
not placeholder entrypoints. This audit found and corrected functional bugs,
reduced duplicated evaluation code, and clarified documentation claims. Passing
checks establish the behavior below; they do not establish model quality on new
data or a speedup on Qualcomm hardware.

## Findings corrected

| Area | Previous behavior | Correction |
|---|---|---|
| Python/C++ detection parity | NaN could pass; reordered or ambiguously overlapping detections could fail despite a valid correspondence | Reject nonfinite/invalid detections; require a complete one-to-one match within box and score tolerances |
| Raw tensor parity | Matching infinities could pass; NaN error diagnostics could fail JSON serialization | Require finite tensors and emit valid failure diagnostics |
| Dataset preparation | Validation/test labels without their image directory were silently unused | Reject incomplete split arguments before writing prepared data |
| Legacy D-FINE splits | Identical image content could cross train/validation boundaries | Share the YOLO content-hash split guard; retain duplicates within a split |
| COCO conversion | Unmapped categories could conceal invalid image references; noncrowd `ignore=1` became ordinary positives | Validate every image reference; explicitly reject unsupported noncrowd ignored regions |
| YOLO training inputs | A crowd-only split passed the annotation-presence check despite producing no usable YOLO labels | Require at least one noncrowd, nonignored training box in both train and validation |
| D-FINE checkpoint loading | An `ema: null` checkpoint raised an error | Fall back to model weights; export metadata reflects the weights actually used |
| D-FINE annotation mapping | Five-class checkpoint metadata could disagree with the project's class order | Reject incompatible class metadata; absent metadata still requires manual semantic verification |
| D-FINE training installation | Upstream torchvision COCO loading required an undeclared dependency | Add `pycocotools` to training requirements and exercise real train/validation dataloaders |
| Test imports | The parent repository's legacy `pycocotools/` could shadow the installed compiled package | Use pytest's importlib import mode |

Crowd annotations remain available for COCO evaluation and are omitted from YOLO
training labels. Rejecting noncrowd `ignore=1` is intentional: this converter does
not implement an ignored-region training loss. No labels should silently acquire
different training semantics.

## Implementation and boilerplate

- Dataset conversion, pseudo-label filtering, physical channel pruning, ONNX
  parity, letterbox restoration, NMS, metrics and native buffer reuse have
  executable logic and regression coverage.
- COCO AP evaluation now lives in `shared/evaluation.py`; latency statistics and
  throughput calculations live in `shared/performance.py`. Model selection,
  full-dataset evaluation, inference and comparison reuse these implementations.
  The local bundle includes the shared dependencies.
- The Python and C++ preprocessing/postprocessing implementations are intentionally
  separate and checked against each other. Combining them would defeat the
  independent runtime comparison.
- CLI argument handling and report writers remain substantial. They implement
  validation, reproducible inputs and artifacts rather than empty scaffolding.
  No project-owned TODO/NotImplemented placeholder was found in the audited code.
- Optional D-FINE training is retained separately from YOLO. Its own resize/model
  contract must not be used with the YOLO C++ runner.

## Verification performed

| Check | Result and scope |
|---|---|
| Automated suite | **91 passed**, including compiled C++ tests; two upstream PyTorch ONNX-export deprecation warnings |
| Static checks | Ruff lint/format and Git whitespace checks pass |
| Dataset/pruning/export tests | YOLO and COCO conversion, pseudo-label split exclusion/merge, 10%/20% physical pruning, unchanged retained weights, reload, no recovery training, ONNX CPU parity and AP-budget selection |
| YOLO training | One real CPU epoch on four synthetic images, with two training and two validation images; checkpoint and workflow metadata saved |
| Optional D-FINE | One CPU training epoch on the same tiny fixture, ONNX export parity, and ONNX CPU image inference succeed after adding the missing dependency |
| C++ preprocessing | Nine deterministic and 24 seeded randomized PNG cases, including RGB/RGBA/grayscale and unusual dimensions; input tensors match Pillow exactly |
| Live integration | Rebuilt `efficient-detection:cpu`; three real images produce the same predictions and COCO metrics as the earlier Docker smoke. A fresh three-image Python/C++ fresh/C++ reuse comparison passes detection and tensor parity |
| Existing runtime results | Rechecked saved detections from 100 images × three trials × Python/C++ fresh/C++ reuse, plus four tensor samples per backend; all parity checks pass |
| Existing full-dataset results | Recomputed COCO AP from saved predictions for all 2,712 FishEye8K images; AP **0.2278604753**, AP50 **0.3941472736**, small-object AP **0.0671825939** match the previous report |

The last two checks reuse saved predictions; they are not a new full-dataset
inference or latency measurement. Audit artifacts are under
`reports/audit_20261019/`, which is excluded from Git. The original `best.pt`
still matches the checkpoint hash recorded by `models/baseline_640.json`.

Reproduce the main automated checks from the project root after installing the
training/test dependencies and, optionally, building C++:

```bash
python -m pytest tests -q
ruff check shared yolo dfine tests cpp/build.py
ruff format --check shared yolo dfine tests cpp/build.py
git diff --check
```

Tests needing the optional pinned D-FINE checkout or compiled C++ executable skip
when those components are absent. The 91-test audit run had both available.

## Documentation coverage

| Guide | Covered behavior |
|---|---|
| [README](../README.md), [note](../note.md) | Project scope, dataset formats, annotation, YOLO train/prune/export, experiment selection and command entrypoints |
| [LOCAL](LOCAL.md) | Python inference, benchmarking and extracted local bundle |
| [DOCKER](DOCKER.md) | Build/run, original COCO test JSON, AP, fixed-threshold precision/recall/F1, latency/FPS and report artifacts |
| [CPP](CPP.md), [CPP_RESULTS](CPP_RESULTS.md) | Native runtime, buffer ownership, reproducible comparison, correctness gates, measured results and interpretation limits |
| [DFINE](DFINE.md), [DFINE_LOCAL](DFINE_LOCAL.md) | Optional legacy preparation, copy-paste, training/resume, export and CPU inference |

Corrections distinguish image-list use of `infer_yolo.py --annotations` from actual
accuracy evaluation, and pipeline latency from request latency including image
decoding. The Docker image runs Python; C++ runs natively. C++ standalone inference
writes predictions/timings; the comparison script adds fixed-threshold metrics.
The existing F1 uses matching IoU 0.5 and is not the official AICITY score.

## Remaining validation limits

- Full Kaggle GPU training and the full real-data pruning × resolution × thread
  experiment matrix were not rerun. Pruning without recovery may violate the
  accuracy budget; selection must retain the baseline in that case.
- S/M/L/X annotation presets and strict checkpoint loading have coverage; the
  annotation quality of all released checkpoint variants was not measured here.
- Exact PNG parity does not establish exact JPEG decoder parity. New formats and
  models must pass the comparison gate before interpreting runtime speedups.
- Existing CPU results are from this Intel host. No Qualcomm/NPU deployment,
  asynchronous C++ pipeline or energy measurement is implemented or validated.
- Standard COCO AP and the project's fixed-threshold metrics are not a complete
  reproduction of AICITY's modified evaluation protocol.
