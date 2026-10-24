"""Small auditable checkpoint helpers; checkpoints are bound to exact input bytes."""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import importlib.metadata
import platform


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(configuration: dict, inputs: list[str | Path]) -> str:
    records = [
        {"path": str(Path(p).resolve()), "sha256": sha256_file(p)}
        for p in sorted(set(map(str, inputs)))
    ]
    environment = {"python": platform.python_version(), "platform": platform.platform()}
    for package in [
        "numpy",
        "scipy",
        "pandas",
        "rasterio",
        "pyproj",
        "tensorflow",
        "keras",
        "scikit-learn",
    ]:
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None
    payload = json.dumps(
        {"configuration": configuration, "inputs": records, "environment": environment},
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, value: dict) -> None:
    """Replace a JSON file only after the entire new file has reached the filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, expected_fingerprint):
    p = Path(path)
    if not p.exists():
        return None
    checkpoint = json.loads(p.read_text(encoding="utf-8"))
    if checkpoint.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Inputs or settings changed; do not reuse stale checkpoint: {p}"
        )
    if checkpoint.get("status") != "complete":
        return None
    for output in checkpoint.get("outputs", []):
        if (
            not Path(output["path"]).is_file()
            or sha256_file(output["path"]) != output["sha256"]
        ):
            raise ValueError(
                f"Checkpoint output is missing or changed: {output['path']}"
            )
    return checkpoint


@contextmanager
def stage_status(path, run_fingerprint, stage):
    """Write a complete/failed record; failures are never silently averaged away."""
    started = time.time()
    record = {"fingerprint": run_fingerprint, "stage": stage, "status": "running"}
    atomic_json(path, record)
    try:
        yield record
    except Exception as error:
        record.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.time() - started,
        )
        atomic_json(path, record)
        raise
    else:
        record.update(status="complete", elapsed_seconds=time.time() - started)
        atomic_json(path, record)
