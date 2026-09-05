"""Atomic experiment artifact persistence."""

from __future__ import annotations

from dataclasses import asdict
import csv
import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .config import ExperimentConfig
from .engine import SimulationResult


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str | Path, document: Any, *, compressed: bool | None = None) -> Path:
    target = Path(path)
    compressed = target.suffix == ".gz" if compressed is None else compressed
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if compressed:
        payload = gzip.compress(payload, compresslevel=6, mtime=0)
    _atomic_bytes(target, payload)
    return target


def read_json(path: str | Path) -> Any:
    source = Path(path)
    payload = source.read_bytes()
    if source.suffix == ".gz":
        payload = gzip.decompress(payload)
    return json.loads(payload.decode("utf-8"))


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    return write_json(path, config.to_dict(), compressed=False)


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(read_json(path))


def save_result(result: SimulationResult, path: str | Path) -> Path:
    return write_json(path, asdict(result))


def load_result(path: str | Path) -> dict[str, Any]:
    return read_json(path)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    if not rows:
        return write_json(target.with_suffix(target.suffix + ".empty.json"), [], compressed=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target

