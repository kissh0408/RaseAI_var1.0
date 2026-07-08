"""Manifest helpers for layer artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    out_dir: Path,
    *,
    inputs: dict[str, str],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write manifest.json alongside generated artifacts."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "config": config,
    }
    if extra:
        payload.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
