import argparse
import copy
from pathlib import Path

import numpy as np
import pytest
from efficient_detection.dfine.annotator import annotation_config
from efficient_detection.shared.common import (
    CLASSES,
    dataset_config,
    read_json,
    sha256,
    write_json,
)
from efficient_detection.yolo.generate_pseudo_labels import candidates, map_detections
from efficient_detection.yolo.prepare_yolo import merge_pseudo, prepare
from efficient_detection.yolo.runtime import decode, letterbox
from PIL import Image


def gold_dataset_args(tmp_path, format="yolo"):
    images, labels = tmp_path / "raw/images", tmp_path / "raw/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for camera in [1, 2]:
        for frame in [1, 2]:
            name = f"camera{camera}_A_{frame}"
            Image.new("RGB", (120, 80), (camera * 50, frame * 50, 10)).save(
                images / f"{name}.png"
            )
            (labels / f"{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    annotation = tmp_path / "original.json"
    write_json(
        annotation,
        {
            "images": [
                {"id": i, "file_name": path.name, "width": 120, "height": 80}
                for i, path in enumerate(sorted(images.glob("*")), 1)
            ],
            "categories": [{"id": 6, "name": "bus"}],
            "annotations": [
                {"id": i, "image_id": i, "category_id": 6, "bbox": [30, 20, 60, 40]}
                for i in range(1, 5)
            ],
        },
    )
    args = argparse.Namespace(
        format=format,
        images=[images],
        labels=[labels],
        annotations=annotation,
        val_images=None,
        val_labels=None,
        val_annotations=None,
        test_images=None,
        test_labels=None,
        test_annotations=None,
        category_map=None,
        split="camera",
        val_fraction=0.2,
        seed=42,
        val_cameras=[2],
        mode="copy",
        output=tmp_path / "prepared",
    )
    return args


def gold_dataset(tmp_path, format="yolo"):
    args = gold_dataset_args(tmp_path, format)
    prepare(args)
    return args.output


@pytest.mark.parametrize("format", ["yolo", "coco"])
def test_yolo_preparation_and_portable_dataset(tmp_path, format):
    root = gold_dataset(tmp_path, format)
    _, cfg, manifest = dataset_config(root / "data.yaml")
    assert cfg["names"] == CLASSES
    assert len(manifest["splits"]["train"]) == len(manifest["splits"]["val"]) == 2
    assert all(row["camera_scene"][0] == 1 for row in manifest["splits"]["train"])
    labels = list((root / "labels").rglob("*.txt"))
    assert all(len(path.read_text().split()) == 5 for path in labels)
    moved = root.with_name("moved")
    root.rename(moved)
    assert dataset_config(moved / "data.yaml")[0] == moved


@pytest.mark.parametrize("format", ["yolo", "coco"])
@pytest.mark.parametrize("split,camera", [("train", 1), ("val", 2)])
def test_preparation_preserves_same_split_duplicates_and_labels(
    tmp_path, format, split, camera, capsys
):
    args = gold_dataset_args(tmp_path, format)
    first = args.images[0] / f"camera{camera}_A_1.png"
    duplicate = args.images[0] / f"camera{camera}_A_2.png"
    duplicate.write_bytes(first.read_bytes())
    # Identical images can carry different human annotations; retain both.
    (args.labels[0] / duplicate.with_suffix(".txt").name).write_text(
        "0 0.25 0.25 0.2 0.2\n"
    )
    annotation = read_json(args.annotations)
    annotation["annotations"][camera * 2 - 1]["bbox"] = [18, 12, 24, 16]
    write_json(args.annotations, annotation)

    manifest = prepare(args)

    rows = manifest["splits"][split]
    assert len(rows) == 2
    assert rows[0]["sha256"] == rows[1]["sha256"]
    assert rows[0]["boxes"][0]["bbox"] == [30, 20, 60, 40]
    assert rows[1]["boxes"][0]["bbox"] == pytest.approx([18, 12, 24, 16])
    assert len(list((args.output / "labels" / split).rglob("*.txt"))) == 2
    exported = read_json(args.output / "annotations" / f"{split}.json")
    assert len(exported["images"]) == len(exported["annotations"]) == 2
    counts = {"train": 0, "val": 0}
    counts[split] = 1
    assert read_json(args.output / "manifest.json")["within_split_duplicates"] == counts
    assert f"{split}=1" in capsys.readouterr().out


@pytest.mark.parametrize("format", ["yolo", "coco"])
@pytest.mark.parametrize(
    "first_split,other_split", [("train", "val"), ("train", "test"), ("val", "test")]
)
def test_preparation_rejects_cross_split_duplicates_before_writing(
    tmp_path, format, first_split, other_split
):
    args = gold_dataset_args(tmp_path, format)
    camera = 1 if first_split == "train" else 2
    first = args.images[0] / f"camera{camera}_A_1.png"
    if other_split == "val":
        duplicate = args.images[0] / "camera2_A_1.png"
    else:
        args.test_images = tmp_path / "test/images"
        args.test_labels = tmp_path / "test/labels"
        args.test_annotations = tmp_path / "test.json"
        args.test_images.mkdir(parents=True)
        args.test_labels.mkdir(parents=True)
        duplicate = args.test_images / "camera3_A_1.png"
        (args.test_labels / "camera3_A_1.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        annotation = read_json(args.annotations)
        annotation["images"] = [
            {"id": 1, "file_name": duplicate.name, "width": 120, "height": 80}
        ]
        annotation["annotations"] = [annotation["annotations"][0]]
        write_json(args.test_annotations, annotation)
    duplicate.write_bytes(first.read_bytes())

    with pytest.raises(
        ValueError, match="Duplicate image content across dataset splits"
    ) as error:
        prepare(args)

    assert f"{first} ({first_split})" in str(error.value)
    assert f"{duplicate} ({other_split})" in str(error.value)
    assert not args.output.exists()


def test_annotation_class_mapping():
    boxes = np.tile([0, 0, 20, 10], (7, 1))
    rows = map_detections([5, 3, 1, 2, 0, 7, 60], boxes, np.ones(7) * 0.9, 120, 80)
    assert [row["category_id"] for row in rows] == [0, 1, 1, 2, 3, 4]
    assert not map_detections([5], boxes[:1], [0.4], 120, 80)


def test_pseudo_exclusion_and_idempotent_merge(tmp_path):
    root = gold_dataset(tmp_path)
    _, _, original = dataset_config(root / "data.yaml")
    unlabeled = tmp_path / "unlabeled"
    unlabeled.mkdir()
    Image.new("RGB", (120, 80), "purple").save(unlabeled / "camera1_A_9.png")
    Image.new("RGB", (120, 80), "yellow").save(unlabeled / "camera2_A_9.png")
    duplicate = unlabeled / "camera9_A_9.png"
    duplicate.write_bytes(Path(original["splits"]["train"][0]["source"]).read_bytes())
    selected = list(candidates(unlabeled, original))
    assert [path.name for path, _ in selected] == ["camera1_A_9.png"]
    before_val = (root / "annotations/val.json").read_bytes()
    before_human = {
        path: path.read_bytes() for path in (root / "labels/train").rglob("*.txt")
    }
    pseudo = tmp_path / "pseudo"
    records = [
        {
            "source": str(path),
            "file_name": path.name,
            "sha256": sha256(path),
            "width": 120,
            "height": 80,
            "boxes": [{"category_id": 2, "bbox": [5, 5, 10, 10], "score": 0.9}],
        }
        for path in sorted(unlabeled.glob("*"))
    ]
    write_json(
        pseudo / "annotations.json",
        {"classes": CLASSES, "generator": {"model": "dfine-s"}, "images": records},
    )
    merged = merge_pseudo(root / "data.yaml", pseudo, "copy")
    assert len(merged["splits"]["train"]) == 3
    assert (root / "annotations/val.json").read_bytes() == before_val
    assert all(path.read_bytes() == value for path, value in before_human.items())
    assert len(merge_pseudo(root / "data.yaml", pseudo, "copy")["splits"]["train"]) == 3


@pytest.mark.parametrize(
    "variant,backbone,layers",
    [
        ("dfine-s", "B0", 3),
        ("dfine-m", "B2", 4),
        ("dfine-l", "B4", 6),
        ("dfine-x", "B5", 6),
    ],
)
def test_annotation_architecture_presets(variant, backbone, layers):
    cfg = annotation_config(variant)
    assert cfg["HGNetv2"]["name"] == backbone
    assert cfg["HGNetv2"]["pretrained"] is False
    assert cfg["DFINETransformer"]["num_layers"] == layers
    if variant == "dfine-x":
        assert cfg["HybridEncoder"]["hidden_dim"] == 384
        assert cfg["DFINETransformer"]["feat_channels"] == [384] * 3


def test_dfine_s_strict_checkpoint_reload_after_bootstrap(tmp_path):
    torch = pytest.importorskip("torch")
    from efficient_detection.dfine.annotator import load_annotator
    from efficient_detection.dfine.config import DEFAULT_REPO, bootstrap, save_config

    torch.set_num_threads(4)
    YAMLConfig = bootstrap(DEFAULT_REPO)
    config = tmp_path / "annotation.yml"
    save_config(config, annotation_config("dfine-s", 5, 256))
    original = YAMLConfig(str(config)).model.eval()
    checkpoint = tmp_path / "dfine_s.pt"
    torch.save({"ema": {"module": original.state_dict()}}, checkpoint)
    restored, postprocessor = load_annotator(checkpoint, "dfine-s", 5, 256, "cpu")
    with torch.inference_mode():
        prediction = postprocessor(
            restored(torch.zeros(1, 3, 256, 256)), torch.tensor([[120, 80]])
        )[0]
    assert prediction["boxes"].shape == (300, 4)
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    with pytest.raises(ValueError, match="Checkpoint must match"):
        load_annotator(checkpoint, "dfine-s", 80, 256, "cpu")


def test_letterbox_rgb_and_original_coordinate_restoration():
    pixels, geometry = letterbox(Image.new("RGB", (120, 80), "red"), 256)
    assert pixels.shape == (1, 3, 256, 256) and pixels.dtype == np.float32
    assert pixels[0, 0, 100, 100] == 1 and pixels[0, 2, 100, 100] == 0
    assert pixels[0, 0, 0, 0] == pytest.approx(114 / 255)
    raw = np.zeros((1, 9, 2), dtype=np.float32)
    x1, y1, x2, y2 = np.asarray([30, 20, 90, 60], dtype=np.float32) * geometry["ratio"]
    y1 += geometry["top"]
    y2 += geometry["top"]
    raw[0, :4, :] = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])[
        :, None
    ]
    raw[0, 4] = [0.9, 0.8]
    result = decode(raw, geometry)
    assert result["boxes"].shape == (1, 4)
    np.testing.assert_allclose(result["boxes"][0], [30, 20, 90, 60], atol=1e-5)


@pytest.fixture(scope="module")
def yolo_checkpoint(tmp_path_factory):
    torch = pytest.importorskip("torch")
    ultralytics = pytest.importorskip("ultralytics")
    import yaml

    torch.set_num_threads(4)
    root = tmp_path_factory.mktemp("yolo_model")
    cfg = yaml.safe_load(
        (Path(ultralytics.__file__).parent / "cfg/models/11/yolo11.yaml").read_text()
    )
    cfg.update(nc=5, scale="n")
    path = root / "yolo11n.yaml"
    path.write_text(yaml.safe_dump(cfg))
    yolo = ultralytics.YOLO(str(path))
    yolo.model.names = dict(enumerate(CLASSES))
    yolo.model.args = {"task": "detect", "imgsz": 256, "data": ""}
    yolo.save(root / "baseline.pt")
    return root / "baseline.pt"


@pytest.mark.parametrize("ratio", [0.1, 0.2])
def test_physical_pruning_preserves_weights_and_reload(
    tmp_path, yolo_checkpoint, ratio
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_pruning")
    from efficient_detection.yolo.models import load_yolo, save_pruned_model
    from efficient_detection.yolo.prune_yolo import prune_model

    original = load_yolo(yolo_checkpoint)
    before = copy.deepcopy(original.model.state_dict())
    model, report = prune_model(original.model, ratio, 256)
    assert report["parameters_after"] < report["parameters_before"]
    assert report["optimizer_steps"] == 0 and not report["recovery_training"]
    row = report["layers"][0]
    key = row["module"] + ".cv1.conv.weight"
    retained = [i for i in range(row["before"]) if i not in row["removed_indices"]]
    assert torch.equal(model.state_dict()[key], before[key][retained])
    # Detection-head tensors are preserved exactly, without optimizer updates.
    head = f"model.{len(model.model) - 1}."
    for key, value in before.items():
        if key.startswith(head):
            assert torch.equal(model.state_dict()[key], value)
    target = tmp_path / "pruned.pt"
    save_pruned_model(model, original, target, report)
    restored = load_yolo(target)
    sample = torch.rand(1, 3, 256, 256)
    with torch.inference_mode():
        torch.testing.assert_close(model(sample)[0], restored.model(sample)[0])


def test_yolo_onnx_export_and_cpu_inference(tmp_path, yolo_checkpoint):
    import os
    import subprocess
    import sys
    import zipfile

    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from efficient_detection.yolo.export_yolo import export_model
    from efficient_detection.yolo.infer_yolo import run
    from efficient_detection.yolo.package_local import build_bundle

    image = tmp_path / "image.png"
    Image.new("RGB", (120, 80), "red").save(image)
    output = tmp_path / "model.onnx"
    metadata = export_model(yolo_checkpoint, output, 256, image)
    assert metadata["verified_samples"] == 2 and metadata["precision"] == "float32"
    run(output, image, tmp_path / "result", save_images=False)
    assert read_json(tmp_path / "result/summary.json")["images"] == 1
    prepared = gold_dataset(tmp_path / "dataset")
    archive = build_bundle(tmp_path, tmp_path / "local.zip", prepared / "data.yaml")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(extracted)
    assert (extracted / "requirements/benchmark.txt").is_file()
    assert (extracted / "efficient_detection/yolo/runtime.py").is_file()
    assert not (extracted / "efficient_detection/yolo/models.py").exists()
    local, _, manifest = dataset_config(extracted / "validation/data.yaml")
    local_image = local / "images" / manifest["splits"]["val"][0]["file_name"]
    assert local_image.is_file() and not local_image.is_symlink()
    # Exercise the extracted package without relying on repository imports or torch.
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from efficient_detection.yolo.infer_yolo import run; "
                "assert 'torch' not in sys.modules; "
                "run('models/model.onnx', sys.argv[1], 'predictions', save_images=False); "
                "assert 'torch' not in sys.modules"
            ),
            str(local_image),
        ],
        cwd=extracted,
        env={**os.environ, "PYTHONPATH": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    assert read_json(extracted / "predictions/summary.json")["images"] == 1


def test_accuracy_budget_uses_ap_points_and_retains_baseline():
    from efficient_detection.yolo.benchmark_yolo import select_candidate

    baseline = str(Path("baseline.onnx").resolve())
    rows = [
        {"model": baseline, "AP": 0.4, "pipeline_p95_ms": 50},
        {"model": "bad", "AP": 0.37, "pipeline_p95_ms": 10},
        {"model": "good", "AP": 0.39, "pipeline_p95_ms": 30},
    ]
    assert select_candidate(rows, baseline)["model"] == "good"
    assert select_candidate(rows[:2], baseline)["model"] == baseline


def test_cpu_coco_accuracy_on_perfect_predictions(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("faster_coco_eval")
    from efficient_detection.yolo.benchmark_yolo import accuracy, measure
    from onnx import TensorProto, helper, numpy_helper

    root = gold_dataset(tmp_path)
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
        "perfect_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 256, 256])],
        [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [1, 9, 1])],
        [numpy_helper.from_array(raw, "predictions")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    path = tmp_path / "perfect.onnx"
    onnx.save(model, path)
    write_json(
        path.with_suffix(".json"),
        {
            "model_family": "yolo11",
            "classes": CLASSES,
            "precision": "float32",
            "image_size": 256,
        },
    )
    assert accuracy(path, root, threads=1)["AP"] > 0.99
    image = (
        root
        / "images"
        / read_json(root / "manifest.json")["splits"]["val"][0]["file_name"]
    )
    metrics = measure(path, [image], threads=1, warmup=0, repeats=2, trials=1)
    assert metrics["timed_iterations"] == 2
    assert metrics["peak_process_rss_mb"] > 0


def test_pruned_checkpoints_cannot_enter_training(tmp_path, yolo_checkpoint):
    import subprocess
    import sys

    pytest.importorskip("torch_pruning")
    from efficient_detection.yolo.models import load_yolo, save_pruned_model
    from efficient_detection.yolo.prune_yolo import prune_model

    root = gold_dataset(tmp_path)
    yolo = load_yolo(yolo_checkpoint)
    model, report = prune_model(yolo.model, 0.2, 256)
    checkpoint = tmp_path / "pruned.pt"
    save_pruned_model(model, yolo, checkpoint, report)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "efficient_detection.yolo.train_yolo",
            "--data",
            str(root / "data.yaml"),
            "--model",
            str(checkpoint),
            "--output",
            str(tmp_path / "run"),
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Recovery training is excluded" in result.stderr
