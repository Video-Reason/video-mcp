from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> None:
    """
    Minimal .env loader (no external deps).
    - Ignores blank lines and comments starting with '#'
    - Supports KEY=VALUE with optional single/double quotes
    - Does NOT override existing environment variables
    """
    p = Path(path)
    if not p.exists():
        return

    text = p.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        k = key.strip()
        if not k:
            continue
        if k in os.environ and os.environ[k] != "":
            continue

        v = value.strip()
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        os.environ[k] = v


def ensure_hf_cache_dirs() -> None:
    """
    Keep HF caches inside the repo by default.
    Uses HF_HOME if set; otherwise defaults to ./hf_home.
    """
    hf_home = Path(os.environ.get("HF_HOME") or "hf_home")
    (hf_home / "hub").mkdir(parents=True, exist_ok=True)
    (hf_home / "datasets").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))

