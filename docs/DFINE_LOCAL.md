# Legacy D-FINE local CPU inference

This guide belongs to `dfine_train.ipynb` and its `fisheye_local.zip` bundle.
For YOLO exports, follow `LOCAL.md` instead.

Extract the bundle, activate a Python 3.11 virtual environment, and run from its directory:

```bash
python -m pip install -r requirements/inference.txt
python -m efficient_detection.dfine.infer --model fisheye.onnx --input /path/to/images --output results
```

Keep `fisheye.onnx` together with `fisheye.json`. The bundle uses CPU ONNX Runtime
and does not require PyTorch or a GPU. `--input` accepts an image or directory;
use `python -m efficient_detection.dfine.infer --help` for confidence, thread, and evaluation options.
