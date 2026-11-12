# Optional D-FINE workflow

The main project trains and deploys YOLO11n. D-FINE-S/M/L/X pseudo-label generation
is documented in [note.md](../note.md). This separate workflow retains D-FINE-L
training and local ONNX inference; it is not required to run YOLO or native C++.

Run from the current project checkout after installing `requirements/training.txt`
in a PyTorch/torchvision environment. Kaggle orchestration is in
`notebooks/dfine_train.ipynb`.
Training requirements include both `faster-coco-eval` for evaluation and
`pycocotools` for the upstream dataset's torchvision loader.

| Entrypoint | Behavior and outputs |
|---|---|
| `dfine/setup_backend.py` | Fetch the pinned upstream source into `third_party/D-FINE`; validate its revision on reuse |
| `dfine/prepare_data.py` | Convert YOLO/COCO inputs into `images/`, `train.json`, `val.json` and `classes.json`; camera or random split |
| `dfine/copy_paste.py` | Copy rectangular objects between same-camera/scene training images of matching dimensions; write augmented images and `train_augmented.json`; validation remains unchanged |
| `dfine/train.py` | Train D-FINE-L with EMA, optional CUDA AMP, multiscale transforms, gradient accumulation, COCO validation and resume state |
| `dfine/export_onnx.py` | Export EMA weights when available, otherwise model weights, to static FP32 ONNX with postprocessing; verify random/optional real-image parity |
| `dfine/infer.py` | CPU inference with stretch resize, optional score heuristics and class-aware NMS; write predictions, images and timing JSON |

Example after preparing data with `dfine/prepare_data.py`:

```bash
python dfine/setup_backend.py
python dfine/copy_paste.py --dataset prepared_dfine

python dfine/train.py --data prepared_dfine --output runs/dfine_pretrain \
  --checkpoint /path/to/dfine_l_obj2coco_e25.pth --stage pretrain --epochs 2 \
  --train-annotations prepared_dfine/train_augmented.json

python dfine/train.py --data prepared_dfine --output runs/dfine_finetune \
  --checkpoint runs/dfine_pretrain/best.pth --stage finetune --epochs 30

python dfine/export_onnx.py --checkpoint runs/dfine_finetune/best.pth \
  --output models/dfine.onnx --test-image /path/to/validation/image.png
```

Use `--resume /path/to/last.pth` instead of `--checkpoint` to restore model,
optimizer, scheduler, scaler, EMA and random-generator state. Keep `best.pth`
beside the resume checkpoint. `--dry-run` validates inputs and writes `config.yml`
without importing PyTorch; `--device cpu --no-amp --no-multiscale` supports small
CPU checks. Checkpoints and per-epoch metrics are written to `best.pth`, `last.pth`
and `metrics.jsonl`.

Pretraining/copy-paste are optional. These original-model training stages are
separate from the YOLO path, which rejects recovery training of pruned models.
Copy-paste is rectangular box augmentation, not instance-mask compositing; it is
not integrated into YOLO training by this project.

For CPU inference and the notebook's `fisheye_local.zip`, see
[DFINE_LOCAL.md](DFINE_LOCAL.md). D-FINE exports have two inputs (`images`,
`orig_target_sizes`) and three outputs (`labels`, `boxes`, `scores`); YOLO exports
use one input and one raw output. Keep the respective runtimes/model formats paired.
