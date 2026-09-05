"""Code and upstream-operator provenance capture."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import subprocess

from micropsi_core.nodenet.stepoperators import DoernerianEmotionalModulators

from . import __version__
from .config import ExperimentConfig


def _git(repo: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_provenance(config: ExperimentConfig) -> dict:
    operator_path = Path(inspect.getsourcefile(DoernerianEmotionalModulators) or "").resolve()
    repo = Path(__file__).resolve().parents[1]
    payload = operator_path.read_bytes()
    relative_operator = str(operator_path.relative_to(repo)) if operator_path.is_relative_to(repo) else str(operator_path)
    operator_sha256 = hashlib.sha256(payload).hexdigest()
    operator_last_commit = _git(repo, "log", "-1", "--format=%H", "--", relative_operator)
    if relative_operator != config.sources.operator_path:
        raise RuntimeError(
            f"Loaded operator path {relative_operator!r} does not match the pinned path "
            f"{config.sources.operator_path!r}"
        )
    if operator_sha256 != config.sources.operator_sha256:
        raise RuntimeError(
            "The MicroPsi2 operator file differs from the pinned implementation hash; "
            "create a documented source revision instead of continuing silently"
        )
    if operator_last_commit != config.sources.operator_last_commit:
        raise RuntimeError(
            "The MicroPsi2 operator commit differs from the pinned source revision"
        )
    paper_path = repo / config.sources.paper_filename
    paper_sha256 = hashlib.sha256(paper_path.read_bytes()).hexdigest() if paper_path.exists() else None
    if paper_sha256 is not None and paper_sha256 != config.sources.paper_sha256:
        raise RuntimeError("The supplied Cai et al. reference PDF differs from the registered source hash")
    package_root = Path(__file__).resolve().parent
    package_hasher = hashlib.sha256()
    package_files = []
    for path in sorted(package_root.glob("*.py")):
        relative = str(path.relative_to(repo))
        package_files.append(relative)
        package_hasher.update(relative.encode("utf-8") + b"\0")
        package_hasher.update(path.read_bytes())
    return {
        "protocol_version": config.protocol_version,
        "package_version": __version__,
        "configuration_digest": config.digest(),
        "repository_head": _git(repo, "rev-parse", "HEAD"),
        "repository_dirty": bool(_git(repo, "status", "--short")),
        "operator_path": relative_operator,
        "operator_sha256": operator_sha256,
        "operator_last_commit": operator_last_commit,
        "paper_path": config.sources.paper_filename,
        "paper_sha256": paper_sha256,
        "expected_paper_sha256": config.sources.paper_sha256,
        "implementation_tree_sha256": package_hasher.hexdigest(),
        "implementation_files": package_files,
        "adapter_version": config.emotion.adapter_version,
        "classifier_version": config.emotion.classifier_version,
        "operator_class": f"{DoernerianEmotionalModulators.__module__}.{DoernerianEmotionalModulators.__name__}",
        "age_influence_term_status": (
            "pinned operator hard-codes youthful_exuberance_term=1; "
            "base_age_influence_on_competence is explicitly supplied as 0"
        ),
    }
