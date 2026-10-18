import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from efficient_detection.dfine.config import DEFAULT_REPO, build_config
from efficient_detection.dfine.copy_paste import augment
from efficient_detection.dfine.infer import postprocess, preprocess
from efficient_detection.dfine.prepare_data import write_split
from efficient_detection.shared.common import (
    CLASSES,
    box_iou,
    class_nms,
    detection_metrics,
    validate_coco,
    write_json,
)
from efficient_detection.shared.datasets import load_coco, load_yolo, split_records
from PIL import Image


def make_yolo(tmp_path):
    images, labels = tmp_path / "raw/images", tmp_path / "raw/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    samples = [
        ("camera1_A_1.png", "red", "0 0.1 0.1 0.1 0.1"),
        ("camera1_A_2.png", "green", "2 0.9 0.9 0.1 0.1"),
        ("camera2_A_1.png", "blue", "4 0.5 0.5 0.1 0.1"),
    ]
    for name, color, annotation in samples:
        Image.new("RGB", (120, 80), color).save(images / name)
        (labels / Path(name).with_suffix(".txt")).write_text(annotation + "\n")
    return images, labels


def test_yolo_camera_split_and_coco_roundtrip(tmp_path):
    images, labels = make_yolo(tmp_path)
    records = load_yolo([images], [labels])
    train, val = split_records(records, val_cameras=[2])
    assert len(train) == 2 and len(val) == 1
    output = tmp_path / "prepared"
    write_split(train, output, "train", "copy")
    write_split(val, output, "val", "copy")
    data = validate_coco(output / "train.json", output / "images")
    assert data["annotations"][0]["bbox"] == pytest.approx([6, 4, 12, 8])
    assert {cat["name"] for cat in data["categories"]} == set(CLASSES)
    assert all("camera1" in image["file_name"] for image in data["images"])


def test_yolo_pseudo_confidence_and_negative_images(tmp_path):
    images, labels = make_yolo(tmp_path)
    (labels / "camera1_A_1.txt").write_text(
        "0 0.5 0.5 0.2 0.2 0.4\n2 0.5 0.5 0.2 0.2 0.9\n"
    )
    (labels / "camera1_A_2.txt").write_text("")
    records = load_yolo([images], [labels], confidence=0.5)
    assert [box["category_id"] for box in records[0]["boxes"]] == [2]
    assert not records[1]["boxes"]
    (labels / "camera1_A_2.txt").unlink()
    with pytest.raises(FileNotFoundError):
        load_yolo([images], [labels])


def test_coco_sparse_ids_remap_and_clipping(tmp_path):
    images, _ = make_yolo(tmp_path)
    data = {
        "images": [
            {"id": 20, "file_name": "camera1_A_1.png", "width": 120, "height": 80}
        ],
        "categories": [{"id": 91, "name": "person"}, {"id": 17, "name": "bus"}],
        "annotations": [
            {"id": 1, "image_id": 20, "category_id": 91, "bbox": [-2, 10, 12, 20]}
        ],
    }
    annotation = tmp_path / "original.json"
    write_json(annotation, data)
    records = load_coco(annotation, images)
    assert records[0]["boxes"][0]["category_id"] == 3
    assert records[0]["boxes"][0]["bbox"] == [0, 10, 10, 20]
    records = load_coco(annotation, images, category_map={"91": 4})
    assert records[0]["boxes"][0]["category_id"] == 4


def test_split_requires_camera_names_and_is_reproducible(tmp_path):
    records = [{"path": Path(f"image{i}.png")} for i in range(10)]
    with pytest.raises(ValueError, match="requires camera"):
        split_records(records)
    assert split_records(records, "random", seed=7) == split_records(
        records, "random", seed=7
    )
    train, val = split_records(records, "random")
    assert {r["path"] for r in train}.isdisjoint({r["path"] for r in val})


def test_copy_paste_cannot_use_validation_donors(tmp_path):
    images, labels = make_yolo(tmp_path)
    train, val = split_records(load_yolo([images], [labels]), val_cameras=[2])
    output = tmp_path / "prepared"
    write_split(train, output, "train", "copy")
    write_split(val, output, "val", "copy")
    original_validation = (output / "val.json").read_bytes()
    result = augment(output, max_objects=1)
    assert len(result["images"]) > len(train)
    assert all(ann["category_id"] != 4 for ann in result["annotations"])
    assert (output / "val.json").read_bytes() == original_validation
    validate_coco(output / "train_augmented.json", output / "images")


def test_preprocess_preserves_rgb_and_width_height():
    image = Image.new("RGB", (120, 80), (255, 0, 0))
    pixels, size = preprocess(image, 256)
    assert pixels.shape == (1, 3, 256, 256)
    assert pixels.dtype == np.float32 and pixels.flags.c_contiguous
    assert np.all(pixels[:, 0] == 1) and np.all(pixels[:, 2] == 0)
    assert size.tolist() == [[120, 80]]


def test_iou_nms_and_metrics_use_one_to_one_class_matches():
    boxes = np.asarray(
        [[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], dtype=np.float32
    )
    scores, labels = np.asarray([0.9, 0.8, 0.7]), np.asarray([0, 0, 1])
    assert box_iou(boxes[0], boxes[1:2])[0] == pytest.approx(81 / 119)
    assert class_nms(boxes, scores, labels).tolist() == [0, 2]
    metrics = detection_metrics(
        [{"boxes": boxes[:1], "labels": [0]}],
        [{"boxes": boxes, "labels": labels, "scores": scores}],
    )
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(0.5)


def test_postprocess_clips_boxes_and_makes_heuristics_optional():
    labels = np.asarray([[0, 1, 9]])
    boxes = np.asarray([[[-2, -3, 20, 20], [5, 5, 5, 20], [0, 0, 20, 20]]])
    scores = np.asarray([[0.4, 0.99, 0.99]])
    assert not len(postprocess(labels, boxes, scores, 120, 80)[0])
    result = postprocess(
        labels, boxes, scores, 120, 80, heuristics=True, filename="camera1_N_1.png"
    )
    assert result[0].tolist() == [0]
    assert result[1].tolist() == [[0, 0, 20, 20]]
    assert result[2][0] == pytest.approx(0.52)


def test_config_has_one_flat_training_definition_and_explicit_stop(tmp_path):
    cfg = build_config(
        DEFAULT_REPO, tmp_path, tmp_path / "run", epochs=10, final_epochs=2
    )
    assert cfg["num_classes"] == 5 and cfg["HGNetv2"]["pretrained"] is False
    assert cfg["train_dataloader"]["total_batch_size"] == 2
    assert cfg["train_dataloader"]["collate_fn"]["stop_epoch"] == 8
    assert cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] == 8
    assert "__include__" not in cfg
    with pytest.raises(ValueError):
        build_config(DEFAULT_REPO, tmp_path, tmp_path / "run", image_size=300)


@pytest.mark.parametrize("format", ["yolo", "coco"])
def test_prepare_cli_supports_both_formats_and_training_dry_run(tmp_path, format):
    images, labels = make_yolo(tmp_path)
    args = [
        sys.executable,
        "-m",
        "efficient_detection.dfine.prepare_data",
        "--format",
        format,
        "--images",
        str(images),
        "--output",
        str(tmp_path / "prepared"),
        "--val-cameras",
        "2",
    ]
    if format == "yolo":
        args += ["--labels", str(labels)]
    else:
        data = {
            "images": [
                {"id": i + 1, "file_name": path.name, "width": 120, "height": 80}
                for i, path in enumerate(sorted(images.glob("*.png")))
            ],
            "categories": [{"id": 1, "name": "car"}],
            "annotations": [],
        }
        write_json(tmp_path / "coco.json", data)
        args += ["--annotations", str(tmp_path / "coco.json")]
    subprocess.run(args, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "efficient_detection.dfine.train",
            "--data",
            str(tmp_path / "prepared"),
            "--output",
            str(tmp_path / "run"),
            "--from-scratch",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "run/config.yml").is_file()


@pytest.mark.parametrize("filename", ["yolo_train.ipynb", "dfine_train.ipynb"])
def test_notebook_is_valid_python_and_contains_no_outputs(filename):
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "notebooks" / filename).read_text()
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<notebook cell>", "exec")
            assert cell["outputs"] == []


def test_onnx_runtime_outputs_original_coordinates(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from efficient_detection.dfine.infer import run
    from onnx import TensorProto, helper, numpy_helper

    # An actual ONNX graph scales normalized XYXY boxes by [width,height,width,height].
    constants = [
        numpy_helper.from_array(np.asarray([[0, 2]], dtype=np.int64), "labels"),
        numpy_helper.from_array(np.asarray([[0.9, 0.8]], dtype=np.float32), "scores"),
        numpy_helper.from_array(
            np.asarray(
                [[[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.1, 0.1]]], dtype=np.float32
            ),
            "normalized",
        ),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axes"),
    ]
    nodes = [
        helper.make_node(
            "Cast", ["orig_target_sizes"], ["float_sizes"], to=TensorProto.FLOAT
        ),
        helper.make_node("Concat", ["float_sizes", "float_sizes"], ["whwh"], axis=1),
        helper.make_node("Unsqueeze", ["whwh", "axes"], ["scale"]),
        helper.make_node("Mul", ["normalized", "scale"], ["boxes"]),
    ]
    graph = helper.make_graph(
        nodes,
        "coordinate_test",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, 256, 256]
            ),
            helper.make_tensor_value_info(
                "orig_target_sizes", TensorProto.INT64, [1, 2]
            ),
        ],
        [
            helper.make_tensor_value_info("labels", TensorProto.INT64, [1, 2]),
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 2, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2]),
        ],
        constants,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.save(model, tmp_path / "test.onnx")
    Image.new("RGB", (120, 80), "red").save(tmp_path / "image.png")
    predictions = run(
        tmp_path / "test.onnx", tmp_path / "image.png", tmp_path / "result"
    )
    assert predictions[0]["bbox"] == pytest.approx([30, 20, 60, 40])
    assert predictions[0]["category_id"] == 0
    assert (tmp_path / "result/images/image.png").is_file()
    # A shared images root can contain training files outside the validation JSON.
    write_json(
        tmp_path / "val.json", {"images": [{"id": 91, "file_name": "image.png"}]}
    )
    Image.new("RGB", (120, 80), "blue").save(tmp_path / "training.png")
    predictions = run(
        tmp_path / "test.onnx",
        tmp_path,
        tmp_path.parent / (tmp_path.name + "-evaluation"),
        annotations=tmp_path / "val.json",
    )
    assert {prediction["image_id"] for prediction in predictions} == {91}


def test_pretrained_coco_class_heads_follow_project_order():
    torch = pytest.importorskip("torch")
    from efficient_detection.dfine.train import initialize_weights

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.enc_score_head = torch.nn.Linear(4, 5)
            self.decoder.denoising_class_embed = torch.nn.Embedding(6, 4)

    model = Detector()
    scores = torch.arange(80, dtype=torch.float32)
    embeds = torch.arange(81, dtype=torch.float32)
    initialize_weights(
        model,
        {
            "model": {
                "module.decoder.enc_score_head.weight": scores[:, None].repeat(1, 4),
                "module.decoder.enc_score_head.bias": scores,
                "module.decoder.denoising_class_embed.weight": embeds[:, None].repeat(
                    1, 4
                ),
            }
        },
    )
    assert model.decoder.enc_score_head.bias.tolist() == [5, 3, 2, 0, 7]
    assert model.decoder.denoising_class_embed.weight[:, 0].tolist() == [
        5,
        3,
        2,
        0,
        7,
        80,
    ]
