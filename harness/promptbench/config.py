"""Provider and model configuration loaded from `models.toml`.

The file lives beside `pyproject.toml` by default; `PROMPTBENCH_CONFIG` points elsewhere.
Values are read once at import so every module sees the same frozen configuration, and
`prepare()` copies the file text into each run's inputs for provenance.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models.toml"


def config_path() -> Path:
    return Path(os.environ.get("PROMPTBENCH_CONFIG", DEFAULT_PATH)).resolve()


def load(path: Path | None = None) -> dict[str, Any]:
    target = path or config_path()
    with target.open("rb") as handle:
        data = tomllib.load(handle)
    for section, keys in (
        ("hosts", ("ollama", "openrouter")),
        ("local", ("tags",)),
        ("hosted", ("models", "endpoints")),
        ("calibration", ("conditions", "reference")),
        ("ablation", ("executors",)),
    ):
        if section not in data or any(k not in data[section] for k in keys):
            raise ValueError(f"models.toml: section [{section}] must define {keys}")
    if set(data["hosted"]["endpoints"]) != set(data["hosted"]["models"]):
        raise ValueError("models.toml: every hosted model needs exactly one pinned endpoint")
    return data


def config_text(path: Path | None = None) -> str:
    return (path or config_path()).read_text()


CONFIG = load()
OLLAMA: str = str(CONFIG["hosts"]["ollama"]).rstrip("/")
OPENROUTER: str = str(CONFIG["hosts"]["openrouter"]).rstrip("/")
LOCAL_MODELS: tuple[str, ...] = tuple(CONFIG["local"]["tags"])
HOSTED_MODELS: tuple[str, ...] = tuple(CONFIG["hosted"]["models"])
PREFERRED_ENDPOINTS: dict[str, str] = dict(CONFIG["hosted"]["endpoints"])
CALIBRATION_CONDITIONS: tuple[str, ...] = tuple(CONFIG["calibration"]["conditions"])
CALIBRATION_REFERENCE: str = str(CONFIG["calibration"]["reference"])
ABLATION_EXECUTORS: tuple[str, ...] = tuple(CONFIG["ablation"]["executors"])
