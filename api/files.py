"""Web-API file upload support.

Bridges the self-contained ``file_input`` subsystem (FileUploadStore → background
parse) into the FastAPI server so the web UI can upload files, watch their
parse status (pending → done/error), and later reference them by upload_id when
sending a chat message.

Storage is disk-backed under ``uploads/`` (project root); parsed parts live in
the in-memory ``FileUploadStore`` for the process lifetime.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import UploadFile

from config import PROJECT_DIR
from file_input.upload_store import FileUploadStore

logger = logging.getLogger("api.files")

UPLOAD_DIR = PROJECT_DIR / "uploads"

_store: FileUploadStore | None = None
# upload_id -> original client filename (store keeps the on-disk token name)
_names: dict[str, str] = {}
# upload_id -> on-disk path (upload_id != on-disk token, so track it explicitly)
_paths: dict[str, Path] = {}


def get_store() -> FileUploadStore:
    """Return the process-wide upload store (lazily created)."""
    global _store
    if _store is None:
        _store = FileUploadStore()
    return _store


async def save_upload(upload: UploadFile) -> dict:
    """Persist an uploaded file to disk and start background parsing.

    Returns ``{upload_id, name, size}`` immediately (never blocks).
    """
    original_name = Path(upload.filename or "upload").name
    token = uuid.uuid4().hex[:12]

    # Save inside a per-upload directory but KEEP the original filename, so the
    # processed parts (and thus the manifest / human message) show the real name.
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest_dir = UPLOAD_DIR / token
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / original_name
    content = await upload.read()
    dest.write_bytes(content)

    upload_id = get_store().add_file(dest)
    _names[upload_id] = original_name
    _paths[upload_id] = dest
    logger.info(
        "UPLOAD → saved %s (%d bytes) as %s [%s]", original_name, len(content), dest.name, upload_id
    )
    return {"upload_id": upload_id, "name": original_name, "size": len(content)}


def original_name(upload_id: str) -> str:
    """Return the client filename for an upload_id (falls back to upload_id)."""
    return _names.get(upload_id, upload_id)


def status_of(upload_ids: list[str]) -> dict:
    """Return per-upload status records keyed for the UI: pending/done/error."""
    store = get_store()
    status = store.get_status(upload_ids)
    records: list[dict] = []
    for uid in upload_ids:
        name = original_name(uid)
        if uid in status.error:
            # Store reports name + reason; use our original filename.
            reason = status.error[uid][1]
            records.append(
                {"upload_id": uid, "name": name, "status": "error", "reason": reason}
            )
        elif uid in status.pending:
            records.append(
                {"upload_id": uid, "name": name, "status": "pending", "reason": ""}
            )
        else:
            records.append(
                {"upload_id": uid, "name": name, "status": "done", "reason": ""}
            )
    return {"uploads": records, "ready": status.ready}


def discard(upload_ids: list[str]) -> None:
    """Drop uploads: remove store entries + on-disk files + name records."""
    store = get_store()
    store.remove(upload_ids)
    for uid in upload_ids:
        _names.pop(uid, None)
        path = _paths.pop(uid, None)
        if path is not None:
            try:
                path.unlink()
                path.parent.rmdir()  # per-upload dir (only if now empty)
            except OSError:
                pass
