"""
Atomic + corruption-tolerant JSON helpers.

Every persisted JSON file in this project (thread state, preferences,
expectations, general behavior guide, runtime config, chat history) should go
through these helpers so that:

* **writes are atomic** — data is written to ``<path>.tmp`` and then
  ``os.replace``d into place, so a crash mid-write can never leave a
  half-written (corrupt) file behind.
* **corrupt reads never silently lose data** — a file that fails to parse (or
  fails a caller-supplied shape check) is *quarantined* (moved aside to
  ``<path>.corrupt-<utc-timestamp>``) instead of being overwritten/ignored.

Note: atomic writes do NOT prevent lost updates from concurrent writers — they
only guarantee a reader always sees a complete file. Multi-process write
coordination (locking) is out of scope here.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("json_io")


def write_json_atomic(path: str | Path, data: Any, *, indent: int = 2) -> None:
    """Write ``data`` as JSON to ``path`` atomically (tmp file + os.replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read JSON from ``path``; return ``default`` when the file is missing."""
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def quarantine(path: str | Path) -> str | None:
    """Move a corrupt file aside. Returns the backup path (or None on failure)."""
    p = Path(path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = p.with_name(f"{p.name}.corrupt-{ts}")
    try:
        os.replace(p, backup)
        return str(backup)
    except OSError:
        logger.exception("Could not quarantine corrupt file %s", p)
        return None


def load_or_quarantine(
    path: str | Path,
    default: Any = None,
    *,
    validate: Callable[[Any], bool] | None = None,
) -> tuple[Any, str | None]:
    """Tolerant load.

    Returns ``(data, backup_path)``:
      * file missing          → ``(default, None)``
      * unparseable / fails ``validate`` → file quarantined, ``(default, backup)``
      * valid                 → ``(data, None)``
    """
    p = Path(path)
    if not p.exists():
        return default, None
    try:
        data = read_json(p)
        if validate is not None and not validate(data):
            raise ValueError("shape validation failed")
        return data, None
    except Exception:
        backup = quarantine(p)
        logger.error(
            "Corrupt JSON at %s — quarantined to %s (using default)", p, backup
        )
        return default, backup
