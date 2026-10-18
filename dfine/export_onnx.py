"""Export an EMA checkpoint and verify ONNX Runtime against PyTorch on CPU."""

import argparse
import inspect
import tempfile
from pathlib import Path

import numpy as np

from efficient_detection.dfine.config import DEFAULT_REPO, bootstrap, save_config
from efficient_detection.shared.common import CLASSES, write_json


def compare_outputs(reference, actual):
    """TopK ties may reorder predictions: compare scores and same-class box sets."""
    ref_labels, ref_boxes, ref_scores = reference
    labels, boxes, scores = actual
    for expected, result in zip(reference, actual):
        if expected.shape != result.shape or not np.isfinite(result).all():
            raise AssertionError("ONNX output shape/non-finite value mismatch")
    np.testing.assert_allclose(scores, ref_scores, rtol=1e-3, atol=2e-4)
    # Compare only detections away from ambiguous low-confidence ties.
    for batch in range(len(ref_labels)):
        for index in np.flatnonzero(ref_scores[batch] > 0.05):
            candidates = np.flatnonzero(
                (labels[batch] == ref_labels[batch, index])
                & (np.abs(scores[batch] - ref_scores[batch, index]) < 2e-3)
            )
            if not len(candidates):
                raise AssertionError("ONNX class/score differs from PyTorch")
            distances = np.max(
                np.abs(boxes[batch, candidates] - ref_boxes[batch, index]), axis=1
            )
            if distances.min() > 0.5:
                raise AssertionError(
                    "ONNX bounding box differs from PyTorch by more than 0.5 pixels"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--image-size",
        type=int,
        help="Optional deployment resolution; defaults to training resolution",
    )
    parser.add_argument("--test-image", type=Path, help="Also verify with a real image")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    import onnx
    import onnxruntime as ort
    import torch
    from PIL import Image

    from efficient_detection.dfine.train import checkpoint_weights

    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise ValueError(
            "Use a best.pth or last.pth produced by this folder's train.py"
        )
    cfg_data = checkpoint["config"]
    image_size = args.image_size or cfg_data["eval_spatial_size"][0]
    if image_size < 256 or image_size % 32:
        parser.error("--image-size must be a multiple of 32 and at least 256")
    cfg_data["eval_spatial_size"] = [image_size, image_size]
    cfg_data["HGNetv2"]["pretrained"] = False
    YAMLConfig = bootstrap(args.repo_root)
    with tempfile.TemporaryDirectory() as directory:
        config_path = Path(directory) / "export.yml"
        save_config(config_path, cfg_data)
        cfg = YAMLConfig(str(config_path))
        base = cfg.model.cpu().eval()
        base.load_state_dict(checkpoint_weights(checkpoint), strict=True)

        class DeploymentModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.detector = base.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                return self.postprocessor(self.detector(images), orig_target_sizes)

        model = DeploymentModel().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    dummy = torch.rand(1, 3, image_size, image_size)
    sizes = torch.tensor(
        [[1280, 720]], dtype=torch.int64
    )  # Width, height, not height, width.
    options = {
        "opset_version": 17,
        "input_names": ["images", "orig_target_sizes"],
        "output_names": ["labels", "boxes", "scores"],
        "do_constant_folding": True,
    }
    # The repository uses tracing-dependent deploy paths; select its legacy exporter explicitly.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(model, (dummy, sizes), str(args.output), **options)
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(
        str(args.output),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    samples = [(dummy, sizes)]
    if args.test_image:
        with Image.open(args.test_image) as image:
            image = image.convert("RGB")
            original_size = torch.tensor([image.size], dtype=torch.int64)
            pixels = (
                np.asarray(
                    image.resize((image_size, image_size), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                / 255.0
            )
            tensor = torch.from_numpy(
                np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
            )
        samples.append((tensor, original_size))
    for images, original_size in samples:
        with torch.inference_mode():
            reference = [value.numpy() for value in model(images, original_size)]
        actual = session.run(
            ["labels", "boxes", "scores"],
            {"images": images.numpy(), "orig_target_sizes": original_size.numpy()},
        )
        compare_outputs(reference, actual)
    metadata = {
        "classes": checkpoint.get("classes", CLASSES),
        "image_size": image_size,
        "color": "RGB",
        "scale": "divide_by_255",
        "resize": "stretch",
        "size_order": "width_height",
        "boxes": "xyxy_original_pixels",
        "batch_size": 1,
        "precision": "float32",
        "opset": 17,
        "weights": "ema" if "ema" in checkpoint else "model",
        "verified_samples": len(samples),
        "best_validation_ap": checkpoint.get("best_ap"),
    }
    write_json(args.output.with_suffix(".json"), metadata)
    print(
        f"Exported and verified {args.output}; copy it and {args.output.with_suffix('.json').name} to your local machine"
    )


if __name__ == "__main__":
    main()
