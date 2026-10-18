"""Single-GPU/CPU training with COCO validation, EMA and resumable checkpoints."""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np

from efficient_detection.dfine.config import (
    DEFAULT_REPO,
    bootstrap,
    build_config,
    save_config,
)
from efficient_detection.shared.common import (
    CLASSES,
    detection_metrics,
    validate_coco,
    write_json,
)


def checkpoint_weights(checkpoint):
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint.get("model", checkpoint)
    return {key.removeprefix("module."): value for key, value in state.items()}


def initialize_weights(model, checkpoint):
    """Load exact weights or remap the COCO heads to this project's class order."""
    import torch

    state = checkpoint_weights(checkpoint)
    current = model.state_dict()
    selected, skipped, remapped = {}, [], []
    coco_indices = [
        5,
        3,
        2,
        0,
        7,
    ]  # bus, motorcycle, car, person, truck; zero-based COCO.
    for key, target in current.items():
        value = state.get(key)
        if value is None:
            skipped.append(key)
            continue
        if value.shape == target.shape:
            selected[key] = value
        elif (
            "score_head" in key
            and value.shape[0] == 80
            and target.shape[0] == 5
            and value.shape[1:] == target.shape[1:]
        ):
            selected[key] = value[coco_indices]
            remapped.append(key)
        elif (
            key == "decoder.denoising_class_embed.weight"
            and value.shape[0] == 81
            and target.shape[0] == 6
        ):
            selected[key] = torch.cat([value[coco_indices], value[-1:]], dim=0)
            remapped.append(key)
        else:
            skipped.append(key)
    coverage = sum(current[key].numel() for key in selected) / sum(
        value.numel() for value in current.values()
    )
    if coverage < 0.8:
        raise ValueError(
            f"Checkpoint matches only {coverage:.1%} of model values; use D-FINE-L weights"
        )
    model.load_state_dict(selected, strict=False)
    print(
        f"Initialized {coverage:.1%} of weights; remapped {len(remapped)} COCO heads; skipped {len(skipped)} tensors"
    )


def seed_worker(worker_id):
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def train_epoch(
    model, criterion, loader, optimizer, scaler, ema, device, epoch, epochs, accumulate
):
    import torch

    model.train()
    criterion.train()
    loader.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for step, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = [
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(images, targets=targets)
        with torch.autocast(device_type=device.type, enabled=False):
            losses = criterion(
                outputs,
                targets,
                epoch=epoch,
                step=step,
                global_step=epoch * len(loader) + step,
                epoch_step=len(loader),
            )
            loss = sum(losses.values())
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch + 1}, batch {step + 1}"
            )
        group_start = (step // accumulate) * accumulate
        group_size = min(accumulate, len(loader) - group_start)
        scaler.scale(loss / group_size).backward()
        if (step + 1) % accumulate == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        total += value
        if step % 25 == 0 or step + 1 == len(loader):
            print(
                f"Epoch {epoch + 1}/{epochs} batch {step + 1}/{len(loader)} loss={value:.4f}",
                flush=True,
            )
    return total / max(len(loader), 1)


def evaluate(model, postprocessor, loader, evaluator, device):
    import torch

    model.eval()
    evaluator.cleanup()
    truths, predictions = [], []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            results = postprocessor(model(images), sizes)
            batch = {}
            for target, result in zip(targets, results):
                result = {key: value.cpu() for key, value in result.items()}
                batch[int(target["image_id"].item())] = result
                ratio = target["orig_size"].repeat(2) / torch.tensor(
                    [images.shape[-1], images.shape[-2]] * 2
                )
                truths.append(
                    {
                        "boxes": (
                            target["boxes"].as_subclass(torch.Tensor) * ratio
                        ).numpy(),
                        "labels": target["labels"].numpy(),
                    }
                )
                predictions.append(
                    {key: value.numpy() for key, value in result.items()}
                )
            evaluator.update(batch)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.coco_eval["bbox"].stats
    metrics = dict(
        zip(
            ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"],
            map(float, stats[:6]),
        )
    )
    metrics.update(detection_metrics(truths, predictions))
    return metrics


def atomic_save(value, path):
    import torch

    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Prepared directory with images/, train.json and val.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Pretrained or pre-training checkpoint; starts a new training stage",
    )
    source.add_argument(
        "--resume",
        type=Path,
        help="Restore last.pth, optimizer, scheduler, EMA and completed epoch",
    )
    source.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--stage", choices=["pretrain", "finetune"], default="finetune")
    parser.add_argument(
        "--train-annotations",
        type=Path,
        help="For example, prepared/train_augmented.json",
    )
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--final-epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-multiscale", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and write config without importing PyTorch",
    )
    args = parser.parse_args()
    if args.accumulate < 1 or args.lr <= 0:
        parser.error("--accumulate and --lr must be positive")
    for path in [args.checkpoint, args.resume]:
        if path is not None and not path.is_file():
            parser.error(f"Checkpoint not found: {path}")
    if (args.output / "last.pth").exists() and args.resume is None:
        parser.error(
            "Output already contains a training run; choose another output or use --resume"
        )
    epochs = (
        args.epochs
        if args.epochs is not None
        else (2 if args.stage == "pretrain" else 30)
    )
    final_epochs = min(args.final_epochs, epochs) if args.stage != "pretrain" else 0
    config = build_config(
        args.repo_root,
        args.data,
        args.output,
        args.image_size,
        args.batch_size,
        epochs,
        final_epochs,
        args.workers,
        args.lr,
        not args.no_multiscale,
        args.train_annotations,
    )
    if args.dry_run:
        for split in ["train", "val"]:
            dataset = config[f"{split}_dataloader"]["dataset"]
            validate_coco(dataset["ann_file"], dataset["img_folder"])
        save_config(args.output / "config.yml", config)
        print(f"Data checked; configuration written to {args.output / 'config.yml'}")
        return
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Enable a Kaggle GPU accelerator, or use --device cpu for a small smoke test."
        )
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.set_num_threads(4)
    checkpoint = (
        torch.load(
            args.resume or args.checkpoint, map_location="cpu", weights_only=False
        )
        if (args.resume or args.checkpoint)
        else None
    )
    if args.resume:
        previous = checkpoint["config"]
        if args.epochs is not None and args.epochs != previous["epochs"]:
            raise ValueError(
                "Resume with the original epoch count; use --checkpoint to start a new stage with a different schedule"
            )
        # Restore model and optimization settings, while allowing a new Kaggle mount/output.
        for split in ["train", "val"]:
            if split == "train" and args.train_annotations is None:
                old_name = Path(
                    previous["train_dataloader"]["dataset"]["ann_file"]
                ).name
                config["train_dataloader"]["dataset"]["ann_file"] = str(
                    args.data.resolve() / old_name
                )
            for field in ["ann_file", "img_folder"]:
                previous[f"{split}_dataloader"]["dataset"][field] = config[
                    f"{split}_dataloader"
                ]["dataset"][field]
            previous[f"{split}_dataloader"]["num_workers"] = args.workers
        previous["output_dir"] = str(args.output.resolve())
        config = previous
        args.stage = checkpoint.get("stage", args.stage)
    datasets = {}
    for split in ["train", "val"]:
        dataset = config[f"{split}_dataloader"]["dataset"]
        contents = validate_coco(dataset["ann_file"], dataset["img_folder"])
        if not any(not ann.get("iscrowd", 0) for ann in contents["annotations"]):
            raise ValueError(
                f"The {split} split needs at least one non-crowd labeled object"
            )
        datasets[split] = {
            (Path(dataset["img_folder"]) / image["file_name"]).resolve()
            for image in contents["images"]
        }
    if datasets["train"] & datasets["val"]:
        raise ValueError("The same image appears in both training and validation")
    config["use_amp"] = not args.no_amp and device.type == "cuda"
    save_config(args.output / "config.yml", config)
    write_json(args.output / "classes.json", CLASSES)
    YAMLConfig = bootstrap(args.repo_root)
    cfg = YAMLConfig(str(args.output / "config.yml"))
    model = cfg.model.to(device)
    if checkpoint and not args.resume:
        initialize_weights(model, checkpoint)
    criterion, postprocessor = cfg.criterion.to(device), cfg.postprocessor.to(device)
    optimizer = cfg.optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    from src.optim import ModelEMA

    ema = ModelEMA(model, decay=0.9999, warmups=1000)
    scaler = torch.amp.GradScaler("cuda", enabled=config["use_amp"])
    start, best_ap = 0, -1.0
    if args.resume:
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start = checkpoint["epoch"] + 1
        args.accumulate = checkpoint.get("accumulate", 1)
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.set_rng_state(checkpoint["torch_state"])
        if device.type == "cuda" and checkpoint.get("cuda_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_state"])
        previous_best = args.resume.parent / "best.pth"
        if previous_best.is_file():
            destination = args.output / "best.pth"
            if previous_best.resolve() != destination.resolve():
                shutil.copy2(previous_best, destination)
            best_ap = checkpoint["best_ap"]
        print(f"Resuming at epoch {start + 1}")
    train_loader, val_loader = cfg.train_dataloader, cfg.val_dataloader
    train_loader.worker_init_fn = seed_worker
    val_loader.worker_init_fn = seed_worker
    # Bound multiscale growth by the selected resolution to control Kaggle VRAM.
    if train_loader.collate_fn.scales is not None:
        train_loader.collate_fn.scales = [
            size
            for size in train_loader.collate_fn.scales
            if size <= config["eval_spatial_size"][0]
        ]
    evaluator = cfg.evaluator
    for epoch in range(start, config["epochs"]):
        started = time.perf_counter()
        loss = train_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            ema,
            device,
            epoch,
            config["epochs"],
            args.accumulate,
        )
        metrics = evaluate(ema.module, postprocessor, val_loader, evaluator, device)
        scheduler.step()
        improved = metrics["AP"] > best_ap
        best_ap = max(best_ap, metrics["AP"])
        state = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "config": config,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_ap": best_ap,
            "accumulate": args.accumulate,
            "classes": CLASSES,
            "stage": args.stage,
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(),
            "cuda_state": torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None,
        }
        atomic_save(state, args.output / "last.pth")
        if improved:
            atomic_save(state, args.output / "best.pth")
        record = {
            "epoch": epoch + 1,
            "loss": loss,
            **metrics,
            "seconds": time.perf_counter() - started,
        }
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
    print(f"Finished. Best validation AP: {best_ap:.4f}; output: {args.output}")


if __name__ == "__main__":
    main()
