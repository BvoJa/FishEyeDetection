"""Generate INT8 models across dtypes (U8S8, S8S8, U8U8) with strict RAM limits."""

from pathlib import Path
from efficient_detection.yolo.quantize_yolo import quantize_model

base_model = Path("models/baseline_640.onnx")
calib_dir = Path("datasets/Fisheye8K/test/images")
calib_imgs = sorted(list(calib_dir.glob("*.png")) + list(calib_dir.glob("*.jpg")))[:30]

dtype_configs = [
    ("u8s8", "uint8", "int8"),
    ("s8s8", "int8", "int8"),
    ("u8u8", "uint8", "uint8"),
]

for name, act_type, wt_type in dtype_configs:
    out_path = Path(f"models/quant_qdq_640_{name}.onnx")
    print(f"Generating {out_path} ({name.upper()}: act={act_type}, wt={wt_type})...")
    quantize_model(
        model_path=base_model,
        output_path=out_path,
        calibration_images=calib_imgs,
        quant_format="qdq",
        activation_type=act_type,
        weight_type=wt_type,
        threads=2,
        preserve_head=True,
    )
    print(f"Generated {out_path}")
