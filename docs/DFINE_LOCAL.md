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
use `python -m efficient_detection.dfine.infer --help` for confidence, thread and image-ID selection options.

`--annotations` selects COCO-listed images and preserves their IDs; it does not
calculate AP/F1. The legacy `inference_fps` measures ONNX execution after one warm-up.
Its `pipeline_seconds` covers the image loop including preprocessing, image I/O,
postprocessing, optional image writing and that initial warm-up. Use the YOLO
evaluation/comparison tools for the separately defined pipeline/request FPS metrics.
