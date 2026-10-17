"""Reproducibility metadata for paper-grade experiment artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .ml_model import (
    create_logistic_pipeline,
    create_random_forest_pipeline,
    create_xgboost_pipeline,
)


PACKAGE_NAMES = ("numpy", "pandas", "matplotlib", "scikit-learn", "xgboost", "joblib")


def sha256_file(path: str | Path) -> str:
    """Return a stable SHA-256 digest for the exact source data file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    """Collect exact versions of the numerical runtime dependencies."""
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_commit(workspace: str | Path) -> str | None:
    """Return the repository commit when Git metadata is available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def _simple_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    simple: dict[str, Any] = {}
    for name, value in parameters.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            simple[name] = value
    return simple


def model_parameters() -> dict[str, dict[str, Any]]:
    """Return the estimator parameters frozen by this research protocol."""
    pipelines = {
        "logistic_regression": create_logistic_pipeline(),
        "random_forest": create_random_forest_pipeline(),
        "xgboost": create_xgboost_pipeline(),
    }
    return {
        name: _simple_parameters(pipeline.named_steps[name].get_params(deep=False))
        for name, pipeline in pipelines.items()
    }


def runtime_metadata(workspace: str | Path) -> dict[str, Any]:
    """Describe the exact interpreter, platform, packages, and source revision."""
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "git_commit": git_commit(workspace),
    }
