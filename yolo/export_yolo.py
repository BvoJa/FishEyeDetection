"""Export original/pruned YOLO11 to static FP32 ONNX and check numerical parity."""

import argparse
import inspect
from pathlib import Path

import numpy as np
from PIL import Image

from efficient_detection.shared.common import CLASSES, sha256, write_json
from efficient_detection.yolo.models import load_yolo
from efficient_detection.yolo.runtime import letterbox


def verify(reference, actual):
    if (
        reference.shape != actual.shape
        or reference.shape[1] != 9
        or not np.isfinite(actual).all()
    ):
        raise AssertionError("Unexpected YOLO output shape or non-finite predictions")
    np.testing.assert_allclose(reference[:, :4], actual[:, :4], rtol=1e-4, atol=0.05)
    np.testing.assert_allclose(reference[:, 4:], actual[:, 4:], rtol=1e-3, atol=2e-5)


def export_model(checkpoint, output, image_size=640, test_image=None, threads=4):
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics.nn.modules import Detect

    if image_size < 256 or image_size % 32 or threads < 1:
        raise ValueError("Use size >=256 divisible by 32, and positive threads")
    torch.set_num_threads(threads)
    yolo = load_yolo(checkpoint)
    parameter_count = sum(parameter.numel() for parameter in yolo.model.parameters())
    detector = yolo.model
    detector.fuse(verbose=False)
    for module in detector.modules():
        if isinstance(module, Detect):
            module.export = True
            module.format = "onnx"
            module.dynamic = False
    detector.eval()

    class DeploymentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = detector

        def forward(self, images):
            output = self.model(images)
            return output[0] if isinstance(output, tuple) else output

    model = DeploymentModel().cpu().float().eval()
    torch.manual_seed(0)
    dummy = torch.rand(1, 3, image_size, image_size)
    options = {
        "opset_version": 17,
        "input_names": ["images"],
        "output_names": ["predictions"],
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, (dummy,), str(output), **options)
    onnx.checker.check_model(onnx.load(str(output)))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(output), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    samples = [dummy.numpy()]
    if test_image:
        with Image.open(test_image) as image:
            samples.append(letterbox(image, image_size)[0])
    for pixels in samples:
        with torch.inference_mode():
            reference = model(torch.from_numpy(pixels)).numpy()
        verify(reference, session.run(None, {"images": pixels})[0])
    metadata = {
        "model_family": "yolo11",
        "classes": CLASSES,
        "precision": "float32",
        "image_size": image_size,
        "batch_size": 1,
        "color": "RGB",
        "scale": "divide_by_255",
        "resize": "letterbox",
        "padding_value": 114,
        "output": "cxcywh_input_pixels_and_class_probabilities",
        "parameters": parameter_count,
        "checkpoint": Path(checkpoint).name,
        "checkpoint_sha256": sha256(checkpoint),
        "onnx_sha256": sha256(output),
        "pruning": (yolo.ckpt or {}).get("pruning"),
        "verified_samples": len(samples),
        "opset": 17,
    }
    write_json(output.with_suffix(".json"), metadata)
    print(f"Verified FP32 ONNX export: {output}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--test-image", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    export_model(
        args.checkpoint, args.output, args.image_size, args.test_image, args.threads
    )
