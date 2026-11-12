"""Build the Linux CPU runner with the same ONNX Runtime library as Python."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Require cached headers")
    args = parser.parse_args()
    import onnxruntime

    if platform.system() != "Linux" or onnxruntime.__version__ != "1.31.0":
        raise SystemExit("This preset requires Linux and onnxruntime==1.31.0")
    source = Path(__file__).resolve().parent
    root = source.parent
    deps = root / "third_party/cpp"
    lock = json.loads((source / "dependencies.lock.json").read_text())
    for item in lock:
        target = deps / item["path"]
        if not target.exists() or sha256(target) != item["sha256"]:
            if args.offline:
                raise SystemExit(f"Missing or invalid cached dependency: {target}")
            print(f"Downloading {item['path']}", flush=True)
            data = urllib.request.urlopen(item["url"], timeout=60).read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise SystemExit(f"Dependency checksum mismatch: {item['url']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    libraries = Path(onnxruntime.__file__).parent / "capi"
    library = libraries / "libonnxruntime.so.1.31.0"
    if not library.exists():
        raise SystemExit(f"CPU shared library missing: {library}")
    (deps / "lib").mkdir(exist_ok=True)
    for name, target in {
        "libonnxruntime.so": library,
        "libonnxruntime.so.1": library,
        "libonnxruntime.so.1.31.0": library,
        "libonnxruntime_providers_shared.so": libraries
        / "libonnxruntime_providers_shared.so",
    }.items():
        link = deps / "lib" / name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise SystemExit(f"Refusing to overwrite a regular library: {link}")
        link.symlink_to(target.resolve())

    def executable(name):
        local = Path(sys.executable).parent / name
        found = str(local) if local.exists() else shutil.which(name)
        if not found:
            raise SystemExit("Install requirements/cpp.txt and a C++17 compiler first")
        return found

    cmake = executable("cmake")
    subprocess.run(
        [cmake, "--preset", "release", f"-DCMAKE_MAKE_PROGRAM={executable('ninja')}"],
        cwd=source,
        check=True,
    )
    subprocess.run([cmake, "--build", "--preset", "release"], cwd=source, check=True)
    info = {
        "onnxruntime": onnxruntime.__version__,
        "library": str(library.resolve()),
        "library_sha256": sha256(library),
        "source_sha256": sha256(source / "src/main.cpp"),
        "dependencies_lock_sha256": sha256(source / "dependencies.lock.json"),
        "cmake": subprocess.check_output([cmake, "--version"], text=True).splitlines()[
            0
        ],
    }
    (root / "build/cpp/build_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"C++ executable: {root / 'build/cpp/efficient_infer'}")


if __name__ == "__main__":
    main()
