"""Dependency-aware bottleneck channel removal, with zero recovery training."""

import argparse
import copy
from pathlib import Path

from efficient_detection.shared.common import dataset_config, sha256, write_json
from efficient_detection.yolo.models import load_yolo, save_pruned_model


def prune_model(module, ratio=0.1, image_size=640, round_to=8):
    import torch
    import torch_pruning as tp
    from ultralytics.nn.modules import Bottleneck

    if not 0 < ratio < 1 or round_to < 1:
        raise ValueError(
            "Pruning ratio must be in (0,1) and channel alignment positive"
        )
    model = copy.deepcopy(module).cpu().float().eval()
    model.requires_grad_(True)
    parameters_before = sum(parameter.numel() for parameter in model.parameters())
    eligible = [
        (name, block)
        for name, block in model.named_modules()
        if isinstance(block, Bottleneck)
        and block.cv1.conv.groups == 1
        and block.cv2.conv.groups == 1
    ]
    dummy = torch.zeros(1, 3, image_size, image_size)
    channel_count = removed_count = 0
    layers = []
    for name, block in eligible:
        conv = block.cv1.conv
        original = conv.out_channels
        channel_count += original
        keep = max(round_to, round(original * (1 - ratio) / round_to) * round_to)
        remove = max(0, original - keep)
        if not remove:
            continue
        importance = conv.weight.detach().square().flatten(1).sum(1)
        indices = importance.argsort()[:remove].tolist()
        with torch.enable_grad():
            dependency = tp.DependencyGraph().build_dependency(
                model,
                dummy,
                verbose=False,
                output_transform=lambda output: (
                    output[0] if isinstance(output, tuple) else output
                ),
            )
            group = dependency.get_pruning_group(
                conv, tp.prune_conv_out_channels, idxs=indices
            )
            if not dependency.check_pruning_group(group):
                raise RuntimeError(f"Unsafe pruning group at {name}")
            group.prune()
        if block.cv2.conv.in_channels != keep:
            raise AssertionError(f"Dependency pruning failed to update {name}.cv2")
        removed_count += remove
        layers.append(
            {
                "module": name,
                "before": original,
                "after": keep,
                "removed_indices": indices,
            }
        )
    if removed_count == 0:
        raise ValueError(
            "This ratio/alignment removed no eligible channels; choose another ratio or --round-to"
        )
    model.requires_grad_(False)
    with torch.inference_mode():
        output = model(dummy)
    output = output[0] if isinstance(output, tuple) else output
    if output.shape[1] != 9 or not torch.isfinite(output).all():
        raise AssertionError("Pruning changed the detection output contract")
    parameters_after = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "requested_ratio": ratio,
        "eligible_channels": channel_count,
        "removed_channels": removed_count,
        "actual_eligible_channel_reduction": removed_count / channel_count,
        "parameters_before": parameters_before,
        "parameters_after": parameters_after,
        "parameter_reduction": 1 - parameters_after / parameters_before,
        "round_to": round_to,
        "layers": layers,
        "optimizer_steps": 0,
        "recovery_training": False,
    }
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Gold validation split for immediate evaluation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prune-ratio", type=float, default=0.1)
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Validation device; structural pruning runs on CPU",
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.image_size < 256 or args.image_size % 32:
        parser.error("Image size must be >=256 and divisible by 32")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    dataset_config(args.data)
    import torch

    torch.set_num_threads(4)
    yolo = load_yolo(args.checkpoint)
    if yolo.ckpt and yolo.ckpt.get("pruning"):
        raise ValueError(
            "Prune each candidate from the original baseline, not an already-pruned model"
        )
    model, report = prune_model(
        yolo.model, args.prune_ratio, args.image_size, args.round_to
    )
    report["source_checkpoint"] = str(args.checkpoint.resolve())
    report["source_sha256"] = sha256(args.checkpoint)
    checkpoint = args.output / "pruned.pt"
    save_pruned_model(model, yolo, checkpoint, report)
    reloaded = load_yolo(checkpoint)
    sample = torch.rand(1, 3, args.image_size, args.image_size)
    with torch.inference_mode():
        torch.testing.assert_close(model(sample)[0], reloaded.model(sample)[0])
    # val() performs inference only; it never creates or steps a training optimizer.
    metrics = reloaded.val(
        data=str(args.data.resolve()),
        imgsz=args.image_size,
        batch=1,
        device=args.device,
        workers=args.workers,
        plots=False,
        project=str(args.output.resolve()),
        name="validation",
        exist_ok=True,
    )
    report["validation"] = {
        "AP": float(metrics.box.map),
        "AP50": float(metrics.box.map50),
        "AP75": float(metrics.box.map75),
    }
    write_json(args.output / "pruning.json", report)
    print(
        f"Saved and validated {checkpoint}; parameters reduced by {report['parameter_reduction']:.1%}; optimizer steps: 0"
    )


if __name__ == "__main__":
    main()
