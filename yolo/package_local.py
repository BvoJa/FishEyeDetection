"""Build a portable CPU-only YOLO inference and benchmark zip."""

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml

from efficient_detection.shared.common import (
    CLASSES,
    dataset_config,
    read_json,
    sha256,
    write_json,
)


def build_bundle(models, output, data=None):
    models, output = Path(models).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("Bundle output must end in .zip")
    exports = sorted(models.glob("*.onnx"))
    if not exports:
        raise ValueError(f"No ONNX models found in {models}")
    for model in exports:
        metadata = read_json(model.with_suffix(".json"))
        if (
            metadata.get("model_family") != "yolo11"
            or metadata.get("classes") != CLASSES
        ):
            raise ValueError(f"Expected this workflow's YOLO export: {model}")
        if metadata.get("onnx_sha256") != sha256(model):
            raise ValueError(f"ONNX checksum does not match its metadata: {model}")
    output.parent.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parents[1]
    source = package
    with tempfile.TemporaryDirectory(
        prefix=".yolo-bundle-", dir=output.parent
    ) as temporary:
        root = Path(temporary) / "bundle"
        for relative in [
            "__init__.py",
            "yolo/__init__.py",
            "yolo/runtime.py",
            "yolo/infer_yolo.py",
            "yolo/benchmark_yolo.py",
            "shared/__init__.py",
            "shared/common.py",
        ]:
            target = root / "efficient_detection" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package / relative, target)
        for name in ["inference.txt", "benchmark.txt"]:
            target = root / "requirements" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "requirements" / name, target)
        shutil.copy2(source / "docs/LOCAL.md", root / "LOCAL.md")
        (root / "models").mkdir()
        for model in exports:
            for path in [model, model.with_suffix(".json")]:
                shutil.copy2(path, root / "models" / path.name)
        if data:
            prepared, _, manifest = dataset_config(data)
            records = manifest["splits"]["val"]
            if not records:
                raise ValueError("Validation split is empty")
            validation = root / "validation"
            for row in records:
                relative = Path(row["file_name"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Validation image paths must be relative")
                target = validation / "images" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prepared / "images" / relative, target)
            (validation / "annotations").mkdir(parents=True)
            shutil.copy2(
                prepared / "annotations/val.json", validation / "annotations/val.json"
            )
            write_json(
                validation / "manifest.json",
                {"classes": manifest["classes"], "splits": {"val": records}},
            )
            (validation / "data.yaml").write_text(
                yaml.safe_dump({"val": "images/val", "names": manifest["classes"]})
            )
        archive = Path(
            shutil.make_archive(str(Path(temporary) / "archive"), "zip", root)
        )
        archive.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        help="Prepared data.yaml; include independent validation image copies",
    )
    args = parser.parse_args()
    print(f"Local CPU bundle: {build_bundle(args.models, args.output, args.data)}")


if __name__ == "__main__":
    main()
