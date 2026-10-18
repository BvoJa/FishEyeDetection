"""Fetch the pinned D-FINE source used for annotation and legacy training."""

import argparse
import subprocess
from pathlib import Path

from efficient_detection.dfine.config import DEFAULT_BACKEND

BACKEND_URL = "https://github.com/Peterande/D-FINE.git"
BACKEND_REVISION = "956d1709314c2c6a4df6f34de232054578a7449f"


def setup_backend(destination=DEFAULT_BACKEND):
    destination = Path(destination).resolve()
    if destination.exists():
        if not (destination / ".git").exists():
            raise FileExistsError(f"Choose an empty backend directory: {destination}")
        if (destination / "src/core/yaml_config.py").is_file():
            revision = subprocess.check_output(
                ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != BACKEND_REVISION:
                raise ValueError(
                    f"Backend revision differs from {BACKEND_REVISION}; choose a new --output directory"
                )
            return destination
        # Resume an interrupted fetch only in an otherwise empty checkout of our remote.
        remote = subprocess.check_output(
            ["git", "-C", str(destination), "remote", "get-url", "origin"], text=True
        ).strip()
        if remote != BACKEND_URL or any(
            path.name != ".git" for path in destination.iterdir()
        ):
            raise FileExistsError(f"Choose an empty backend directory: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(destination)], check=True)
        subprocess.run(
            ["git", "-C", str(destination), "remote", "add", "origin", BACKEND_URL],
            check=True,
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--depth",
            "1",
            "origin",
            BACKEND_REVISION,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )
    if not (destination / "src/core/yaml_config.py").is_file():
        raise FileNotFoundError(f"Missing D-FINE source in {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BACKEND)
    args = parser.parse_args()
    print(f"D-FINE backend: {setup_backend(args.output)} ({BACKEND_REVISION})")


if __name__ == "__main__":
    main()
