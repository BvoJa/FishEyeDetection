import csv
import subprocess
import sys

import numpy as np
import pytest
from efficient_detection.shared.common import CLASSES, read_json, sha256, write_json
from efficient_detection.shared.performance import throughput_fps
from efficient_detection.yolo.evaluate_yolo import run
from efficient_detection.yolo.runtime import YoloSession, letterbox
from PIL import Image


@pytest.fixture
def evaluation_input(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    images = tmp_path / "images"
    images.mkdir()
    records = []
    for index, identity in enumerate([21167, 15]):
        name = f"camera1_A_{index}.png"
        Image.new("RGB", (120, 80), (50, 80 + index, 100)).save(images / name)
        records.append({"id": identity, "file_name": name, "width": 120, "height": 80})
    annotation = tmp_path / "test.json"
    write_json(
        annotation,
        {
            "images": records,
            "categories": [
                {"id": index, "name": name.title()}
                for index, name in enumerate(CLASSES)
            ],
            "annotations": [
                {
                    "id": index,
                    "image_id": row["id"],
                    "category_id": 0,
                    "bbox": [30, 20, 60, 40],
                    "area": 2400,
                    "iscrowd": 0,
                }
                for index, row in enumerate(records, 1)
            ],
        },
    )
    _, geometry = letterbox(Image.new("RGB", (120, 80)), 256)
    raw = np.zeros((1, 9, 1), dtype=np.float32)
    raw[0, :4, 0] = [
        60 * geometry["ratio"],
        40 * geometry["ratio"] + geometry["top"],
        60 * geometry["ratio"],
        40 * geometry["ratio"],
    ]
    raw[0, 4, 0] = 0.99
    graph = helper.make_graph(
        [],
        "constant_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 256, 256])],
        [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [1, 9, 1])],
        [numpy_helper.from_array(raw, "predictions")],
    )
    model = tmp_path / "constant.onnx"
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
            "parameters": 0,
            "onnx_sha256": sha256(model),
        },
    )
    return model, images, annotation


def test_full_evaluation_reuses_forward_preserves_ids_and_reports(
    tmp_path, evaluation_input, monkeypatch
):
    model, images, annotation = evaluation_input
    calls = []
    original = YoloSession.predict

    def counted(self, *args, **kwargs):
        calls.append(kwargs.get("return_raw", False))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(YoloSession, "predict", counted)
    output = tmp_path / "report"
    summary = run(
        model, images, annotation, output, threads=1, warmup=3, save_images=True
    )
    assert calls == [False, False, False, True, True]
    assert (
        summary["images"]
        == summary["expected_images"]
        == summary["timed_iterations"]
        == 2
    )
    assert summary["accuracy"]["AP"] > 0.99
    fixed = summary["detection_metrics"]
    assert fixed["match_iou"] == 0.5
    assert fixed["micro"]["tp"] == 2
    assert fixed["micro"]["fp"] == fixed["micro"]["fn"] == 0
    assert fixed["micro"]["f1"] == fixed["micro"]["detection_accuracy"] == 1
    assert fixed["macro_f1"] == 1
    assert summary["detections_by_class"]["bus"] == summary["detections"] == 2
    assert summary["peak_process_rss_mib"] >= summary["peak_inference_rss_mib"] > 0
    assert (
        summary["timing"]["request"]["mean_ms"]
        >= summary["timing"]["pipeline"]["mean_ms"]
    )
    assert {row["image_id"] for row in read_json(output / "predictions.json")} == {
        21167,
        15,
    }
    with (output / "per_image.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["image_id"]) for row in rows] == [21167, 15]
    for stage in ["inference", "pipeline", "request"]:
        expected = len(rows) * 1000 / sum(float(row[f"{stage}_ms"]) for row in rows)
        assert summary[f"{stage}_fps"] == pytest.approx(expected)
    assert summary["inference_fps"] >= summary["pipeline_fps"] >= summary["request_fps"]
    assert summary["dataset_fps"] == summary["dataset_images_per_second"]
    assert summary["dataset_fps"] == pytest.approx(
        len(rows) / summary["dataset_pass_seconds"]
    )
    assert len(list((output / "images").glob("*.png"))) == 2
    assert len(read_json(output / "deployment_predictions.json")) == 2
    assert all(
        float(row["f1"]) == float(row["detection_accuracy"]) == 1 for row in rows
    )
    assert "0.001" in (output / "report.md").read_text()
    assert "FPS (images/s)" in (output / "report.md").read_text()
    assert "torch" not in read_json(output / "environment.json")["versions"]


def test_throughput_uses_total_time_instead_of_averaging_reciprocals():
    # Two frames taking 10 + 30 ms give 50 FPS, not mean(100, 33.33) FPS.
    assert throughput_fps([10, 30]) == 50
    assert throughput_fps(value for value in [10, 30]) == 50


def test_inference_fps_excludes_warmup_and_reports_image_io(
    tmp_path, evaluation_input, monkeypatch, capsys
):
    from efficient_detection.yolo.infer_yolo import run as infer

    model, images, annotation = evaluation_input
    calls = []
    original = YoloSession.predict

    def counted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(result["timing_ms"])
        return result

    monkeypatch.setattr(YoloSession, "predict", counted)
    output = tmp_path / "inference"
    infer(model, images, output, threads=1, annotations=annotation, save_images=False)
    summary = read_json(output / "summary.json")
    assert len(calls) == summary["warmup"] + summary["images"] == 5
    for stage in ["inference", "pipeline"]:
        expected = 2 * 1000 / sum(row[stage] for row in calls[3:])
        assert summary[f"{stage}_fps"] == pytest.approx(expected)
    assert summary["request_fps"] <= summary["pipeline_fps"]
    assert summary["dataset_fps"] <= summary["request_fps"]
    assert summary["dataset_fps"] == pytest.approx(2 / summary["dataset_pass_seconds"])
    assert "FPS: inference=" in capsys.readouterr().out


def test_evaluation_handles_no_predictions(tmp_path, evaluation_input):
    model, images, annotation = evaluation_input
    output = tmp_path / "empty_predictions"
    summary = run(
        model,
        images,
        annotation,
        output,
        threads=1,
        warmup=0,
        confidence=1,
        ap_confidence=0.001,
    )
    assert summary["detections"] == 0
    assert summary["images_without_detections"] == 2
    assert summary["detection_metrics"]["micro"]["fn"] == 2
    assert summary["detection_metrics"]["micro"]["f1"] == 0
    # AP uses its own low threshold, so deployment filtering cannot change AP.
    assert summary["accuracy"]["AP"] > 0.99
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper

    graph = onnx.load(model)
    raw = numpy_helper.to_array(graph.graph.initializer[0]).copy()
    raw[:, 4:] = 0
    graph.graph.initializer[0].CopyFrom(numpy_helper.from_array(raw, "predictions"))
    onnx.save(graph, model)
    metadata = read_json(model.with_suffix(".json"))
    metadata["onnx_sha256"] = sha256(model)
    write_json(model.with_suffix(".json"), metadata)
    summary = run(
        model, images, annotation, tmp_path / "no_predictions", threads=1, warmup=0
    )
    assert summary["accuracy"]["AP"] == 0
    assert summary["detections"] == 0


@pytest.mark.parametrize(
    "problem", ["missing_image", "category_mapping", "path_escape"]
)
def test_evaluation_rejects_invalid_inputs_before_output(
    tmp_path, evaluation_input, problem
):
    model, images, annotation = evaluation_input
    data = read_json(annotation)
    if problem == "missing_image":
        (images / data["images"][0]["file_name"]).unlink()
    elif problem == "category_mapping":
        data["categories"][0]["name"] = "car"
    else:
        data["images"][0]["file_name"] = "../outside.png"
    write_json(annotation, data)
    output = tmp_path / "invalid"
    with pytest.raises((ValueError, FileNotFoundError)):
        run(model, images, annotation, output, threads=1, warmup=0)
    assert not output.exists()


def test_evaluation_import_is_torch_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import efficient_detection.yolo.evaluate_yolo; assert 'torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
