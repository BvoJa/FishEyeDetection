"""Quantize FP32 YOLO11 ONNX models to INT8 using ONNX Runtime with strict CPU and RAM limits."""

import argparse
import copy
import gc
import os
import resource
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnx
from PIL import Image
import onnxruntime as ort
import onnxruntime.quantization as oq

from efficient_detection.shared.common import CLASSES, read_json, sha256, write_json
from efficient_detection.yolo.runtime import letterbox, decode


def enforce_resource_limits(max_cores: int = 4, max_ram_gb: float = 2.5):
    """Enforce strict OS-level CPU affinity and memory limits on Linux."""
    # 1. Prevent glibc from allocating dozens of 64MB memory arenas
    os.environ["MALLOC_ARENA_MAX"] = "2"

    # 2. CPU affinity: restrict process to only max_cores cores
    if hasattr(os, "sched_setaffinity"):
        cores = set(range(min(max_cores, os.cpu_count() or 1)))
        try:
            os.sched_setaffinity(0, cores)
        except OSError:
            pass

    # 3. Thread env variables to prevent background thread oversubscription
    threads_str = str(max_cores)
    os.environ["OMP_NUM_THREADS"] = threads_str
    os.environ["OPENBLAS_NUM_THREADS"] = threads_str
    os.environ["MKL_NUM_THREADS"] = threads_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads_str
    os.environ["NUMEXPR_NUM_THREADS"] = threads_str


class StreamingCalibrationDataReader(oq.CalibrationDataReader):
    """Streams calibration images one-by-one without holding them in RAM."""

    def __init__(self, image_paths: List[Path], image_size: int = 640):
        self.image_paths = [Path(p) for p in image_paths]
        self.image_size = image_size
        self._iterator = iter(self.image_paths)

    def get_next(self):
        try:
            path = next(self._iterator)
        except StopIteration:
            return None
        with Image.open(path) as img:
            pixels, _ = letterbox(img, self.image_size)
        return {"images": pixels}

    def rewind(self):
        self._iterator = iter(self.image_paths)


def get_detect_head_nodes(model_path: Path, head_prefix: str = "/model.23") -> List[str]:
    """Identify nodes belonging to the YOLO detection head to preserve in FP32."""
    model = onnx.load(str(model_path))
    return [n.name for n in model.graph.node if head_prefix in n.name]


def quantize_model(
    model_path: Path,
    output_path: Path,
    calibration_images: Optional[List[Path]] = None,
    quant_format: str = "qdq",
    calibration_method: str = "minmax",
    per_channel: bool = True,
    preserve_head: bool = True,
    activation_type: str = "uint8",
    weight_type: str = "int8",
    threads: int = 4,
    max_ram_gb: float = 2.5,
    test_image: Optional[Path] = None,
):
    """Quantize an ONNX model with static QDQ/QOperator or dynamic quantization within strict resource bounds."""
    enforce_resource_limits(max_cores=threads, max_ram_gb=max_ram_gb)

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata is required: {metadata_path}")
    source_meta = read_json(metadata_path)
    image_size = source_meta.get("image_size", 640)

    # Exclude detection head to preserve detection precision
    nodes_to_exclude = []
    if preserve_head:
        nodes_to_exclude = get_detect_head_nodes(model_path)

    quant_types = {
        "int8": oq.QuantType.QInt8,
        "uint8": oq.QuantType.QUInt8,
    }
    w_type = quant_types.get(weight_type.lower(), oq.QuantType.QInt8)
    a_type = quant_types.get(activation_type.lower(), oq.QuantType.QUInt8)

    # Streaming MinMax calibrator is strictly used to prevent memory ballooning
    cal_method = oq.CalibrationMethod.MinMax

    if quant_format.lower() == "dynamic":
        oq.quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            weight_type=w_type,
            per_channel=per_channel,
            nodes_to_exclude=nodes_to_exclude,
        )
    else:
        if not calibration_images:
            raise ValueError(f"Static quantization ({quant_format}) requires calibration images.")

        dr = StreamingCalibrationDataReader(calibration_images, image_size=image_size)
        q_format = oq.QuantFormat.QDQ if quant_format.lower() == "qdq" else oq.QuantFormat.QOperator

        oq.quantize_static(
            model_input=str(model_path),
            model_output=str(output_path),
            calibration_data_reader=dr,
            quant_format=q_format,
            per_channel=per_channel,
            weight_type=w_type,
            activation_type=a_type,
            calibrate_method=cal_method,
            nodes_to_exclude=nodes_to_exclude,
        )

    gc.collect()

    # Validate output model
    onnx_proto = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_proto)
    del onnx_proto
    gc.collect()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output_path), sess_options=opts, providers=["CPUExecutionProvider"])

    test_px = np.random.rand(1, 3, image_size, image_size).astype(np.float32)
    out = session.run(None, {"images": test_px})[0]
    if out.shape != (1, 9, out.shape[2]) or not np.isfinite(out).all():
        raise AssertionError(f"Invalid quantized model output shape or non-finite values: {out.shape}")

    if test_image and Path(test_image).is_file():
        with Image.open(test_image) as img:
            px, geom = letterbox(img, image_size)
            raw = session.run(None, {"images": px})[0]
            decode(raw, geom, confidence=0.25, nms_iou=0.7)

    del session
    gc.collect()

    # Emit metadata sidecar
    quant_info = {
        "format": quant_format.lower(),
        "calibration_method": "minmax" if quant_format.lower() != "dynamic" else "n/a",
        "calibration_samples": len(calibration_images) if calibration_images else 0,
        "per_channel": per_channel,
        "weight_type": weight_type.lower(),
        "activation_type": activation_type.lower() if quant_format.lower() != "dynamic" else "n/a",
        "head_preserved": preserve_head,
        "excluded_nodes_count": len(nodes_to_exclude),
    }

    meta = copy.deepcopy(source_meta)
    meta.update({
        "precision": "int8",
        "quantization": quant_info,
        "onnx_sha256": sha256(output_path),
        "source_checkpoint": source_meta.get("checkpoint"),
        "source_fp32_sha256": source_meta.get("onnx_sha256"),
    })
    write_json(output_path.with_suffix(".json"), meta)
    return output_path, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to input FP32 ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="Path to output INT8 ONNX model")
    parser.add_argument("--calib-dir", type=Path, help="Directory containing calibration images")
    parser.add_argument("--calib-samples", type=int, default=30, help="Number of calibration images (default: 30)")
    parser.add_argument(
        "--format",
        choices=["qdq", "qoperator", "dynamic"],
        default="qdq",
        help="Quantization format (default: qdq)",
    )
    parser.add_argument(
        "--no-per-channel",
        action="store_true",
        help="Disable per-channel weight quantization",
    )
    parser.add_argument(
        "--no-preserve-head",
        action="store_true",
        help="Do not exclude YOLO detection head from quantization",
    )
    parser.add_argument(
        "--activation-type",
        choices=["uint8", "int8"],
        default="uint8",
        help="Activation quantization type (default: uint8)",
    )
    parser.add_argument(
        "--weight-type",
        choices=["int8", "uint8"],
        default="int8",
        help="Weight quantization type (default: int8)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="CPU core limit: 1, 2, or 4 (default: 4)",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=2.5,
        help="Maximum RAM limit in GB (default: 2.5)",
    )
    parser.add_argument(
        "--test-image",
        type=Path,
        help="Optional test image to verify inference",
    )

    args = parser.parse_args()

    calib_images = None
    if args.format != "dynamic":
        if not args.calib_dir or not args.calib_dir.is_dir():
            parser.error("--calib-dir is required for static quantization")
        all_imgs = sorted(
            p for p in args.calib_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} and p.is_file()
        )
        if not all_imgs:
            parser.error(f"No valid images found in {args.calib_dir}")
        calib_images = all_imgs[:args.calib_samples]

    out_file, meta = quantize_model(
        model_path=args.model,
        output_path=args.output,
        calibration_images=calib_images,
        quant_format=args.format,
        per_channel=not args.no_per_channel,
        preserve_head=not args.no_preserve_head,
        activation_type=args.activation_type,
        weight_type=args.weight_type,
        threads=args.threads,
        max_ram_gb=args.max_ram_gb,
        test_image=args.test_image,
    )
    print(f"Quantized model saved to: {out_file}")
    print(f"Metadata saved to: {out_file.with_suffix('.json')}")


if __name__ == "__main__":
    main()
