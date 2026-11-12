import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from efficient_detection.shared.common import CLASSES, read_json, sha256, write_json
from efficient_detection.yolo.compare_cpp import prediction_parity
from efficient_detection.yolo.runtime import YoloSession, letterbox
from PIL import Image


@pytest.fixture
def cpp_binary():
    path = Path(
        os.environ.get("EFFICIENT_DETECTION_CPP_BINARY", "build/cpp/efficient_infer")
    ).resolve()
    if not path.is_file():
        pytest.skip("Build optional C++ runtime with python cpp/build.py first")
    return path


@pytest.fixture
def cpp_inputs(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    images = tmp_path / "images"
    images.mkdir()
    records = []
    rng = np.random.default_rng(7)
    # Down/up sampling, odd padding, ties-to-even dimensions and degenerate axes.
    for index, (width, height) in enumerate(
        [
            (120, 80),
            (259, 441),
            (641, 257),
            (320, 320),
            (1, 1),
            (2, 3),
            (1080, 1080),
            (512, 101),
            (101, 512),
        ]
    ):
        name = f"image_{index}.png"
        Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(
            images / name
        )
        records.append(
            {"id": index * 17 + 1, "file_name": name, "width": width, "height": height}
        )
    write_json(tmp_path / "manifest.json", {"images": records})
    raw = np.zeros((1, 9, 6), dtype=np.float32)
    raw[0, :4, :] = np.array([128, 128, 160, 120], dtype=np.float32)[:, None]
    raw[0, 4, 0:2] = [0.95, 0.9]  # Duplicate same-class box is suppressed.
    raw[0, 5, 2] = 0.95  # Equal score, overlapping box in another class remains.
    raw[0, 0, 3] = np.nan  # Nonfinite anchor is ignored.
    raw[0, 4, 3:5] = 0.99
    raw[0, 2, 4] = -1  # Invalid geometry is ignored.
    raw[0, 4, 5] = 0.1
    graph = helper.make_graph(
        [],
        "parity_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 256, 256])],
        [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [1, 9, 6])],
        [numpy_helper.from_array(raw, "predictions")],
    )
    model = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
        ),
        model,
    )
    write_json(
        model.with_suffix(".json"),
        {
            "model_family": "yolo11",
            "classes": CLASSES,
            "precision": "float32",
            "image_size": 256,
            "onnx_sha256": sha256(model),
        },
    )
    config = {
        "project_root": str(tmp_path),
        "model": str(model),
        "images": str(images),
        "manifest": str(tmp_path / "manifest.json"),
        "threads": 1,
        "warmup": 2,
        "repeats": 2,
        "confidence": 0.25,
        "nms_iou": 0.7,
        "dump_tensors": len(records),
    }
    return config, records


@pytest.mark.parametrize("mode", ["reuse", "fresh"])
def test_cpp_preprocess_detection_parity_and_tensor_allocations(
    tmp_path, cpp_binary, cpp_inputs, mode
):
    config, records = cpp_inputs
    output = tmp_path / mode
    config.update(output=str(output), buffer_mode=mode)
    path = tmp_path / f"{mode}.json"
    write_json(path, config)
    subprocess.run(
        [str(cpp_binary), "--config", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    session = YoloSession(config["model"], threads=1)
    predictions = []
    for index, record in enumerate(records):
        with Image.open(Path(config["images"]) / record["file_name"]) as source:
            image = source.convert("RGB")
        pixels, _ = letterbox(image, 256)
        actual = np.fromfile(
            output / "tensors" / f"{index}_input.f32", dtype=np.float32
        ).reshape(pixels.shape)
        np.testing.assert_array_equal(actual, pixels, err_msg=record["file_name"])
        result = session.predict(image, return_raw=True)
        raw = np.fromfile(
            output / "tensors" / f"{index}_raw.f32", dtype=np.float32
        ).reshape(result["raw"].shape)
        np.testing.assert_array_equal(raw, result["raw"])
        for box, score, label in zip(
            result["boxes"], result["scores"], result["labels"]
        ):
            x1, y1, x2, y2 = map(float, box)
            predictions.append(
                {
                    "image_id": record["id"],
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    assert prediction_parity(predictions, read_json(output / "predictions.json"))[
        "passed"
    ]
    summary = read_json(output / "summary.json")
    allocations = 1 if mode == "reuse" else 2 + len(records) * 2
    assert summary["buffers"]["input_tensor_allocations"] == allocations
    assert summary["buffers"]["output_tensor_allocations"] == allocations
    assert summary["timed_iterations"] == 2 * len(records)
    assert summary["inference_fps"] >= summary["pipeline_fps"] >= summary["request_fps"]


def test_cpp_rejects_mismatched_model_metadata(tmp_path, cpp_binary, cpp_inputs):
    config, _ = cpp_inputs
    model = Path(config["model"])
    metadata = read_json(model.with_suffix(".json"))
    metadata["onnx_sha256"] = "0" * 64
    write_json(model.with_suffix(".json"), metadata)
    output = tmp_path / "invalid"
    config["output"] = str(output)
    write_json(tmp_path / "invalid.json", config)
    result = subprocess.run(
        [str(cpp_binary), "--config", str(tmp_path / "invalid.json")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "checksum" in result.stderr
    assert not output.exists()


def test_parity_gate_detects_class_and_score_changes():
    reference = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9}
    ]
    assert not prediction_parity(reference, [{**reference[0], "category_id": 1}])[
        "passed"
    ]
    assert not prediction_parity(reference, [{**reference[0], "score": 0.8}])["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_parity_rejects_nonfinite_detections(value):
    reference = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9}
    ]
    for candidate in [
        [{**reference[0], "score": value}],
        [{**reference[0], "bbox": [value, 0, 10, 10]}],
    ]:
        assert not prediction_parity(reference, candidate)["passed"]


def test_parity_matches_reordered_detections_with_identical_boxes():
    reference = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.5},
    ]
    assert prediction_parity(reference, reference[::-1])["passed"]


def test_parity_finds_a_complete_matching_for_ambiguous_boxes():
    # The first reference has two valid partners; the second has only the first.
    reference = [
        {"image_id": 1, "category_id": 0, "bbox": [0.009, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
    ]
    candidate = [
        {**reference[0], "bbox": [0.005, 0, 10, 10]},
        {**reference[0], "bbox": [0.018, 0, 10, 10]},
    ]
    assert prediction_parity(reference, candidate, box_atol=0.01)["passed"]


def test_tensor_parity_reports_nonfinite_values_without_invalid_json(tmp_path):
    from efficient_detection.yolo.compare_cpp import tensor_parity

    for name in ["python", "cpp"]:
        folder = tmp_path / name / "tensors"
        folder.mkdir(parents=True)
        for kind in ["input", "raw"]:
            np.array([float("inf")], dtype=np.float32).tofile(folder / f"0_{kind}.f32")
    result = tensor_parity(
        tmp_path / "python", tmp_path / "cpp", 1, {"raw_atol": 1e-4, "raw_rtol": 1e-5}
    )
    assert not result["passed"]
    write_json(tmp_path / "parity.json", result)
