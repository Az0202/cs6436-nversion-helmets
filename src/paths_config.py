"""
paths_config.py

Centralized path definitions for the project.

- Detects project root (by looking for pyproject.toml / .git / setup.cfg)
- Provides standard directories (data/, runs/, logs/, etc.)
- Allows overrides via environment variables
- Includes helpers to ensure directories exist
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _find_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk upward from `start` (or this file) to find a marker of the repo root.
    """
    start = start or Path(__file__).resolve()
    if start.is_file():
        start = start.parent

    markers = ("pyproject.toml", "setup.cfg", "requirements.txt", ".git")
    cur = start
    while True:
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            # Fallback: directory containing this file
            return Path(__file__).resolve().parent
        cur = cur.parent


def _env_path(var: str, default: Path) -> Path:
    """
    Read an env var as a path, else default.
    """
    val = os.getenv(var)
    return Path(val).expanduser().resolve() if val else default


@dataclass(frozen=True)
class Paths:
    # Root
    ROOT: Path

    # Common project dirs
    CONFIGS: Path
    DATA: Path
    RAW_DATA: Path
    PROCESSED_DATA: Path

    RUNS: Path
    CHECKPOINTS: Path
    LOGS: Path
    FIGURES: Path
    CACHE: Path
    OUTPUTS: Path

    def ensure(self) -> "Paths":
        """
        Create directories that should exist. Returns self.
        """
        for p in (
            self.CONFIGS,
            self.DATA,
            self.RAW_DATA,
            self.PROCESSED_DATA,
            self.RUNS,
            self.CHECKPOINTS,
            self.LOGS,
            self.FIGURES,
            self.CACHE,
            self.OUTPUTS,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(project_root: Optional[Path] = None) -> Paths:
    """
    Build the Paths object. Allows override by passing project_root or env vars.

    Env vars:
      PROJECT_ROOT
      PROJECT_DATA_DIR
      PROJECT_RUNS_DIR
      PROJECT_LOGS_DIR
      PROJECT_CACHE_DIR
      PROJECT_OUTPUTS_DIR
    """
    root = _env_path("PROJECT_ROOT", project_root or _find_project_root())

    # Defaults under root
    configs = root / "configs"

    data = _env_path("PROJECT_DATA_DIR", root / "data")
    raw = data / "raw"
    processed = data / "processed"

    runs = _env_path("PROJECT_RUNS_DIR", root / "runs")
    checkpoints = runs / "checkpoints"

    logs = _env_path("PROJECT_LOGS_DIR", root / "logs")
    figures = root / "figures"

    cache = _env_path("PROJECT_CACHE_DIR", root / ".cache")
    outputs = _env_path("PROJECT_OUTPUTS_DIR", root / "outputs")

    return Paths(
        ROOT=root,
        CONFIGS=configs,
        DATA=data,
        RAW_DATA=raw,
        PROCESSED_DATA=processed,
        RUNS=runs,
        CHECKPOINTS=checkpoints,
        LOGS=logs,
        FIGURES=figures,
        CACHE=cache,
        OUTPUTS=outputs,
    )


# Convenience singleton (common pattern)
PATHS = get_paths().ensure()