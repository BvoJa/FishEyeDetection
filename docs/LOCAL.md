# YOLO11 local CPU inference and benchmarking

Extract `yolo_local.zip`. It contains `models/`, the CPU runner and benchmark scripts, requirements, and (by default) a `validation/` dataset with independent image files. Keep every ONNX file together with its same-named JSON metadata.

Use Python 3.11 in a virtual environment:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. From the extracted folder:

```bash
python -m pip install -r requirements/inference.txt
python -m efficient_detection.yolo.infer_yolo --model models/baseline_640.onnx --input /path/to/images --output predictions
```

`--input` accepts one image or a recursive directory. Add `--no-images` for JSON output only, `--confidence 0.25` to change the confidence threshold, or `--threads 4` to set CPU threads. Results include annotated images, COCO predictions, an image-ID manifest, and preprocessing/inference/NMS timing statistics.

For accuracy evaluation, pass normalized COCO annotations using `--annotations validation/annotations/val.json` and `--input validation/images`. Only the listed images are processed and their IDs are preserved. Without annotations, numeric IDs follow sorted input filenames.

Install the additional CPU benchmark dependencies and compare the exported models:

```bash
python -m pip install -r requirements/benchmark.txt
python -m efficient_detection.yolo.benchmark_yolo --models "models/*.onnx" --baseline models/baseline_640.onnx --data validation/data.yaml --output reports --threads 1 2 4 --profile
```

The wildcard is expanded by the script, including on Windows. Defaults are 20 warm-ups, 100 iterations per trial and 3 trials. Add `--graph-optimizations off all` to compare graph settings. Physical core count limits the thread sweep. Profiling traces are collected separately from timed runs.

Read `reports/results.csv` and `accuracy_latency.png` to compare model size, parameters, AP, small-object AP, peak process RSS, and stage p50/p95 latency. Memory measurements include session startup and cached RGB inputs. Accuracy uses a fixed configuration once per model; timing excludes image decoding and file writing. Confidence is 0.001 for AP evaluation and 0.25 for deployment latency measurement.

`reports/selection.json` identifies the fastest end-to-end p95 candidate within 2 absolute AP points of the baseline. AP is stored on a 0–1 scale, so the default budget is 0.02. The unpruned baseline is retained when it is the best eligible candidate. Use the selected model, `--threads`, and `--graph-optimization` setting with `infer_yolo.py`.

For a separate held-out test set, provide its full prepared dataset, use `--split test --selected reports/selection.json`, and include the baseline and selected model in `--models`. Test results evaluate the prior choice rather than selecting another model.

Inference needs NumPy, Pillow, and ONNX Runtime. Benchmarking adds CPU-only evaluation/profiling packages. PyTorch, Ultralytics, CUDA, D-FINE weights, and TensorRT are not needed on the local machine. Models use FP32, RGB letterboxing, coordinate restoration, and per-class NMS. Pruned models have not undergone recovery training.
