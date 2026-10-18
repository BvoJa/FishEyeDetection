import ast
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from efficient_detection.dfine.config import backend_root

PROJECT = Path(__file__).resolve().parents[1]


def run_cell(source, namespace):
    exec(compile(source, "<project notebook cell>", "exec"), namespace)  # noqa: S102 -- trusted cells from this repository.


@pytest.fixture
def standalone_clone(tmp_path):
    destination = tmp_path / "FishEyeDetection"
    shutil.copytree(
        PROJECT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            "third_party",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "*.egg-info",
            "build",
            "dist",
        ),
    )
    return destination


def test_modules_and_scripts_work_from_flat_editable_checkout(
    tmp_path, standalone_clone
):
    wheel_dir = tmp_path / "editable_wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_editable; import sys; build_editable(sys.argv[1])",
            str(wheel_dir),
        ],
        cwd=standalone_clone,
        check=True,
        capture_output=True,
        text=True,
    )
    site_dir = tmp_path / "installed_editable"
    with zipfile.ZipFile(next(wheel_dir.glob("*.whl"))) as archive:
        archive.extractall(site_dir)
    # Process only this checkout's editable .pth, while sharing test dependencies.
    # -S avoids loading the development checkout's own editable registration.
    dependencies = [
        path
        for path in sys.path
        if Path(path).name in {"site-packages", "dist-packages"}
    ]
    initialization = f"import sys, site; site.addsitedir({str(site_dir)!r}); sys.path.extend({dependencies!r}); "
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            initialization
            + (
                "from pathlib import Path; "
                "import efficient_detection; "
                "from efficient_detection.dfine.config import DEFAULT_REPO; "
                "from efficient_detection.dfine.annotator import annotation_config; "
                "assert DEFAULT_REPO == Path.cwd(); "
                "assert Path(efficient_detection.__file__).resolve().parent == Path.cwd(); "
                "assert annotation_config('dfine-x')['HGNetv2']['name'] == 'B5'; "
                "print('standalone flat checkout imports passed'); "
                "import runpy; sys.argv=['yolo/train_yolo.py', '--help']; "
                "runpy.run_path('yolo/train_yolo.py', run_name='__main__')"
            ),
        ],
        cwd=standalone_clone,
        env={**os.environ, "PYTHONPATH": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "standalone flat checkout imports passed" in result.stdout
    assert "--epochs" in result.stdout


@pytest.mark.parametrize(
    "filename,folder,script",
    [
        ("yolo_train.ipynb", "yolo", "train_yolo.py"),
        ("dfine_train.ipynb", "dfine", "train.py"),
    ],
)
def test_notebook_uses_current_directory_and_direct_files(
    tmp_path, monkeypatch, standalone_clone, filename, folder, script
):
    notebook = json.loads((PROJECT / "notebooks" / filename).read_text())
    monkeypatch.chdir(standalone_clone)
    namespace = {}
    run_cell("".join(notebook["cells"][1]["source"]), namespace)
    assert namespace["REPO"] == standalone_clone
    assert namespace["WORK"].parent == standalone_clone.parent
    dependency_cell = ast.parse("".join(notebook["cells"][3]["source"]))
    runner = next(
        node
        for node in dependency_cell.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = []

    def record(arguments, **options):
        calls.append((arguments, options))

    monkeypatch.setattr(subprocess, "run", record)
    namespace.update(sys=sys, subprocess=subprocess)
    run_cell(ast.unparse(runner), namespace)
    namespace["run"](script, "--help")
    arguments, options = calls[0]
    assert arguments == [
        sys.executable,
        "-u",
        str(standalone_clone / folder / script),
        "--help",
    ]
    assert options == {"cwd": standalone_clone, "check": True}


@pytest.mark.parametrize("relative", [".", "third_party/D-FINE", "dfine"])
def test_backend_can_be_resolved_from_supported_roots(tmp_path, relative):
    expected = tmp_path / relative
    target = expected / "src/core/yaml_config.py"
    target.parent.mkdir(parents=True)
    target.write_text("# source marker\n")
    assert backend_root(tmp_path) == expected.resolve()


def test_missing_backend_explains_setup_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="dfine/setup_backend"):
        backend_root(tmp_path)


def test_legacy_bundle_uses_clone_root_assets(tmp_path, standalone_clone):
    notebook = json.loads((PROJECT / "notebooks/dfine_train.ipynb").read_text())
    work = tmp_path / "dfine_workflow"
    work.mkdir()
    model = work / "fisheye.onnx"
    model.write_bytes(b"test graph packaging")
    model.with_suffix(".json").write_text("{}")
    namespace = {
        "REPO": standalone_clone,
        "WORK": work,
        "ONNX": model,
    }
    run_cell("".join(notebook["cells"][-1]["source"]), namespace)
    with zipfile.ZipFile(tmp_path / "fisheye_local.zip") as archive:
        files = archive.namelist()
    assert "efficient_detection/dfine/infer.py" in files
    assert "requirements/inference.txt" in files
    assert "DFINE_LOCAL.md" in files
