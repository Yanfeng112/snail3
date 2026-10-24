"""Record SHA-256 hashes and actual execution environment for a repository release."""

import argparse
import hashlib
import importlib.metadata
import platform
from pathlib import Path
from python.common import write_json


def main(root, output):
    files = []
    root = Path(root).resolve()
    output = Path(output).resolve()
    for p in sorted(root.rglob("*")):
        if (
            p.is_file()
            and p.resolve() != output
            and "__pycache__" not in p.parts
            and ".git" not in p.parts
        ):
            h = hashlib.sha256()
            with p.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            files.append(
                dict(
                    path=str(p.relative_to(root)),
                    bytes=p.stat().st_size,
                    sha256=h.hexdigest(),
                )
            )
    packages = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "rasterio",
        "pyproj",
        "tensorflow",
        "keras",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json(
        output,
        dict(
            python=platform.python_version(),
            platform=platform.platform(),
            packages=packages,
            files=files,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="outputs/provenance.json")
    args = parser.parse_args()
    main(args.root, args.output)
