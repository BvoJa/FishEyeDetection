"""Train or resume the YOLO11n baseline. Pruned checkpoints cannot be retrained."""

import argparse
from pathlib import Path

from efficient_detection.shared.common import dataset_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if (
        args.image_size < 256
        or args.image_size % 32
        or min(args.epochs, args.batch_size) < 1
    ):
        parser.error("Use a size >=256 divisible by 32 and positive epochs/batch size")
    root, _, manifest = dataset_config(args.data)
    for split, records in manifest["splits"].items():
        for record in records:
            if not (root / "images" / record["file_name"]).is_file():
                raise FileNotFoundError(root / "images" / record["file_name"])
        if split in {"train", "val"} and not any(row["boxes"] for row in records):
            raise ValueError(f"{split} needs at least one labeled object")
    if (args.output / "weights/last.pt").exists() and not args.resume:
        parser.error(
            "Output already contains training weights; use --resume or a new output"
        )
    from ultralytics import YOLO

    yolo = YOLO(str(args.resume or args.model), task="detect")
    if yolo.ckpt and yolo.ckpt.get("pruning"):
        raise ValueError(
            "Recovery training is excluded: use the original unpruned baseline checkpoint"
        )
    if not any(type(module).__name__ == "C3k2" for module in yolo.model.modules()):
        raise ValueError("Use a YOLO11 detection model")
    yolo.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        workers=args.workers,
        project=str(args.output.resolve().parent),
        name=args.output.name,
        exist_ok=True,
        plots=False,
        amp=args.amp,
        resume=str(args.resume) if args.resume else False,
    )
    output = Path(yolo.trainer.save_dir)
    write_json(
        output / "workflow.json",
        {
            "model_family": "yolo11",
            "data": str(args.data.resolve()),
            "seed": args.seed,
            "epochs": args.epochs,
            "image_size": args.image_size,
            "best_checkpoint": str(output / "weights/best.pt"),
            "pruned": False,
        },
    )
    print(f"Baseline checkpoint: {output / 'weights/best.pt'}")


if __name__ == "__main__":
    main()
