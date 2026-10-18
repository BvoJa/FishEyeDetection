"""D-FINE-S/M/L/X annotation presets and strict checkpoint loading."""

import tempfile
from pathlib import Path

from efficient_detection.dfine.config import (
    DEFAULT_REPO,
    base_config,
    bootstrap,
    save_config,
)


def annotation_config(variant="dfine-s", classes=80, image_size=640, repo=DEFAULT_REPO):
    """Official S/M/L/X architecture settings; no backbone downloads on load."""
    if variant not in {"dfine-s", "dfine-m", "dfine-l", "dfine-x"}:
        raise ValueError(f"Unsupported annotation model: {variant}")
    if classes not in {5, 80} or image_size < 256 or image_size % 32:
        raise ValueError("Use 5/80 classes and an image size >=256 divisible by 32")
    cfg = base_config()
    cfg.update(
        num_classes=classes,
        remap_mscoco_category=False,
        eval_spatial_size=[image_size, image_size],
    )
    name = {"dfine-s": "B0", "dfine-m": "B2", "dfine-l": "B4", "dfine-x": "B5"}[variant]
    small = variant in {"dfine-s", "dfine-m"}
    cfg["HGNetv2"].update(
        name=name,
        pretrained=False,
        return_idx=[1, 2, 3],
        use_lab=small,
        freeze_at=-1 if small else 0,
        freeze_norm=not small,
        freeze_stem_only=True,
    )
    if variant == "dfine-s":
        cfg["HybridEncoder"].update(
            in_channels=[256, 512, 1024], depth_mult=0.34, expansion=0.5
        )
        cfg["DFINETransformer"]["num_layers"] = 3
    elif variant == "dfine-m":
        cfg["HybridEncoder"].update(in_channels=[384, 768, 1536], depth_mult=0.67)
        cfg["DFINETransformer"]["num_layers"] = 4
    elif variant == "dfine-x":
        cfg["HybridEncoder"].update(hidden_dim=384, dim_feedforward=2048)
        cfg["DFINETransformer"].update(feat_channels=[384, 384, 384], reg_scale=8)
    return cfg


def load_annotator(checkpoint, variant, classes, image_size, device, repo=DEFAULT_REPO):
    import torch

    YAMLConfig = bootstrap(repo)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "annotation.yml"
        save_config(path, annotation_config(variant, classes, image_size, repo))
        cfg = YAMLConfig(str(path))
        model = cfg.model
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # D-FINE's bootstrap also exposes dfine/train.py, so avoid importing a
        # helper through the ambiguous top-level name "train" in notebooks.
        ema = state.get("ema")
        weights = ema["module"] if ema else state.get("model", state)
        weights = {key.removeprefix("module."): value for key, value in weights.items()}
        try:
            model.load_state_dict(weights, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"Checkpoint must match {variant}, {classes} classes, and its official architecture: {error}"
            ) from error
        return model.to(device).eval(), cfg.postprocessor.to(device).eval()
