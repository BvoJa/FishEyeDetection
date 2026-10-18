"""Load YOLO11 checkpoints and preserve physically pruned model topology."""

import copy
from pathlib import Path

from efficient_detection.shared.common import CLASSES


def load_yolo(checkpoint):
    from ultralytics import YOLO

    yolo = YOLO(str(checkpoint), task="detect")
    names = yolo.names
    names = (
        [names[i] for i in range(len(names))]
        if isinstance(names, dict)
        else list(names)
    )
    if names != CLASSES:
        raise ValueError(
            f"Expected a trained five-class checkpoint with names {CLASSES}; got {names}"
        )
    if not any(type(module).__name__ == "C3k2" for module in yolo.model.modules()):
        raise ValueError("This workflow supports YOLO11 detection checkpoints")
    yolo.model.float().cpu().eval()
    return yolo


def save_pruned_model(module, original, path, report):
    """Preserve the modified topology, rather than reconstructing an unpruned YAML."""
    import torch
    from ultralytics import __version__

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    args = copy.deepcopy(getattr(original.model, "args", {}))
    torch.save(
        {
            "model": copy.deepcopy(module).cpu().float(),
            "ema": None,
            "optimizer": None,
            "train_args": args,
            "epoch": -1,
            "version": __version__,
            "pruning": report,
        },
        path,
    )
