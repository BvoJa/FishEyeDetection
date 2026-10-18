"""Build a flat, portable configuration without changing the original configs."""

import copy
import sys
from pathlib import Path

import yaml

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND = DEFAULT_REPO / "third_party" / "D-FINE"


def backend_root(repo_root=DEFAULT_REPO):
    """Accept the project root or an explicitly supplied D-FINE checkout."""
    root = Path(repo_root).resolve()
    for candidate in [root / "third_party/D-FINE", root, root / "dfine"]:
        if (candidate / "src/core/yaml_config.py").is_file():
            return candidate
    raise FileNotFoundError(
        "D-FINE source is missing. From the efficient_detection directory run "
        "`python dfine/setup_backend.py`, or pass "
        "--repo-root /path/to/D-FINE to use an existing checkout."
    )


def base_config():
    return yaml.safe_load(Path(__file__).with_name("model.yml").read_text())


def bootstrap(repo_root):
    source = backend_root(repo_root)
    import torchvision

    # This repository calls the deprecated helper even in recent torchvision.
    if not hasattr(torchvision, "disable_beta_transforms_warning"):
        torchvision.disable_beta_transforms_warning = lambda: None
    sys.path.insert(0, str(source))
    from src.core import YAMLConfig

    return YAMLConfig


def build_config(
    repo_root,
    data,
    output,
    image_size=960,
    batch_size=2,
    epochs=30,
    final_epochs=5,
    workers=2,
    lr=0.00025,
    multiscale=True,
    train_annotations=None,
):
    if image_size < 256 or image_size % 32:
        raise ValueError("Image size must be a multiple of 32 and at least 256")
    if min(batch_size, epochs) < 1 or not 0 <= final_epochs <= epochs or workers < 0:
        raise ValueError("Invalid batch size, epoch count, final epochs or workers")
    data, output = Path(data).resolve(), Path(output).resolve()
    cfg = base_config()
    stop = epochs - final_epochs
    cfg.update(
        {
            "num_classes": 5,
            "remap_mscoco_category": False,
            "eval_spatial_size": [image_size, image_size],
            "epochs": epochs,
            "output_dir": str(output),
            "use_amp": True,
            "use_ema": True,
            "clip_max_norm": 0.1,
            "evaluator": {"type": "CocoEvaluator", "iou_types": ["bbox"]},
        }
    )
    cfg["HGNetv2"].update(
        {
            "name": "B4",
            "pretrained": False,
            "freeze_stem_only": True,
            "freeze_at": 0,
            "freeze_norm": True,
        }
    )
    cfg["optimizer"] = {
        "type": "AdamW",
        "lr": lr,
        "betas": [0.9, 0.999],
        "weight_decay": 0.000125,
        "params": [
            {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": lr / 20},
            {
                "params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$",
                "weight_decay": 0.0,
            },
        ],
    }
    train_ops = [
        {"type": "RandomPhotometricDistort", "p": 0.5},
        {"type": "RandomZoomOut", "fill": 0},
        {"type": "RandomIoUCrop", "p": 0.8},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "RandomHorizontalFlip"},
        {"type": "Resize", "size": [image_size, image_size]},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]
    for split in ["train", "val"]:
        training = split == "train"
        ops = (
            train_ops
            if training
            else [
                {"type": "Resize", "size": [image_size, image_size]},
                {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            ]
        )
        transforms = {"type": "Compose", "ops": copy.deepcopy(ops)}
        if training:
            transforms["policy"] = {
                "name": "stop_epoch",
                "epoch": stop,
                "ops": ["RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"],
            }
        annotation = (
            Path(train_annotations).resolve()
            if training and train_annotations
            else data / f"{split}.json"
        )
        cfg[f"{split}_dataloader"] = {
            "type": "DataLoader",
            "total_batch_size": batch_size,
            "num_workers": workers,
            "shuffle": training,
            "drop_last": False,
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(data / "images"),
                "ann_file": str(annotation),
                "return_masks": False,
                "transforms": transforms,
            },
            "collate_fn": {
                "type": "BatchImageCollateFunction",
                "base_size": image_size,
                "base_size_repeat": 3 if training and multiscale else None,
                "stop_epoch": stop,
            },
        }
    return cfg


def save_config(path, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
