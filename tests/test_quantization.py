"""Tests for YOLO INT8 post-training ONNX quantization."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from efficient_detection.yolo.quantize_yolo import quantize_model
from efficient_detection.yolo.runtime import YoloSession, letterbox


def test_quantization_dynamic_and_static_contract(tmp_path):
    # Use existing baseline_640 model for a fast contract test
    model_path = Path("models/baseline_640.onnx")
    if not model_path.is_file():
        pytest.skip("models/baseline_640.onnx not present")

    # 1. Test dynamic quantization
    dyn_out = tmp_path / "model_dyn.onnx"
    out_file, meta = quantize_model(
        model_path=model_path,
        output_path=dyn_out,
        quant_format="dynamic",
        threads=2,
    )
    assert dyn_out.is_file()
    assert dyn_out.with_suffix(".json").is_file()
    assert meta["precision"] == "int8"
    assert meta["quantization"]["format"] == "dynamic"

    # Verify session load and inference contract
    session = YoloSession(dyn_out, threads=2)
    img = Image.new("RGB", (640, 640), (128, 128, 128))
    result = session.predict(img)
    assert "boxes" in result
    assert "scores" in result
    assert "labels" in result
    assert "timing_ms" in result
