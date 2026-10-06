"""Shared disk cache helper (moved out of footprint.py so ranges.py and
network_observation.py can use the same TTL'd JSON-on-disk cache without importing
footprint.py itself - PART A item 2)."""

from __future__ import annotations

import json
import time
from pathlib import Path


def _cache_get(path: Path, ttl_s: float | None) -> dict | list | None:
    if not path.exists():
        return None
    if ttl_s is not None and time.time() - path.stat().st_mtime > ttl_s:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_put(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
