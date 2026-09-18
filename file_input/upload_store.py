"""
Background file-upload store.

Lets the UI/agent accept file uploads instantly and parse them in background
threads (docling on big PDFs can take seconds to minutes). On send, the
caller checks readiness via get_status() — sending is GATED, never blocking:
if anything is still pending or errored, the user gets a notice instead.

In-memory and process-local: entries live until remove() is called.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from file_input.processing import process_file
from file_input.types import ProcessedFile

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_ERROR = "error"


@dataclass
class _Entry:
    upload_id: str
    path: Path
    status: str = STATUS_PENDING
    processed: ProcessedFile | None = None
    error_reason: str = ""


@dataclass
class StoreStatus:
    """Snapshot of requested uploads. ready == True iff all are 'done'."""
    ready: bool = True
    done: dict[str, str] = field(default_factory=dict)      # upload_id -> filename
    pending: dict[str, str] = field(default_factory=dict)   # upload_id -> filename
    error: dict[str, tuple[str, str]] = field(default_factory=dict)  # id -> (filename, reason)

    def notice(self) -> str:
        """Human-readable reason why sending is not allowed (ready → '')."""
        if self.ready:
            return ""
        lines = ["Cannot send message — files not ready:"]
        for uid, name in self.pending.items():
            lines.append(f"  ⏳ {name} is still processing")
        for uid, (name, reason) in self.error.items():
            lines.append(f"  ❌ {name} failed: {reason}")
        return "\n".join(lines)


class FileUploadStore:
    """Thread-safe store of in-flight / completed file processing jobs."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    # ── upload time ─────────────────────────────────────────────────
    def add_file(self, path: str | Path) -> str:
        """Register a file and start processing it in a background thread.

        Returns the upload_id immediately (never blocks).
        """
        p = Path(path)
        upload_id = uuid.uuid4().hex[:12]
        entry = _Entry(upload_id=upload_id, path=p)
        with self._lock:
            self._entries[upload_id] = entry

        worker = threading.Thread(
            target=self._process_entry, args=(entry,), daemon=True, name=f"file-parse-{upload_id}"
        )
        worker.start()
        return upload_id

    def _process_entry(self, entry: _Entry) -> None:
        try:
            pf = process_file(entry.path)
            with self._lock:
                entry.processed = pf
                if pf.has_error():
                    entry.status = STATUS_ERROR
                    entry.error_reason = "; ".join(e.reason for e in pf.errors())
                else:
                    entry.status = STATUS_DONE
        except Exception as e:  # processing must never crash the app
            with self._lock:
                entry.status = STATUS_ERROR
                entry.error_reason = f"processing raised: {e}"

    # ── send time (non-blocking checks) ─────────────────────────────
    def get_status(self, upload_ids: list[str]) -> StoreStatus:
        """Instant snapshot: ready only if ALL requested ids are done."""
        status = StoreStatus()
        with self._lock:
            for uid in upload_ids:
                entry = self._entries.get(uid)
                if entry is None:
                    status.error[uid] = ("<unknown upload_id>", "no such upload")
                elif entry.status == STATUS_PENDING:
                    status.pending[uid] = entry.path.name
                elif entry.status == STATUS_ERROR:
                    status.error[uid] = (entry.path.name, entry.error_reason)
                else:
                    status.done[uid] = entry.path.name
        status.ready = not status.pending and not status.error
        return status

    def get_processed(self, upload_ids: list[str]) -> list[ProcessedFile]:
        """Return processed files in the requested order.

        Only call after get_status(...).ready is True — pending entries
        raise RuntimeError.
        """
        out: list[ProcessedFile] = []
        with self._lock:
            for uid in upload_ids:
                entry = self._entries.get(uid)
                if entry is None:
                    raise RuntimeError(f"no such upload_id: {uid}")
                if entry.status != STATUS_DONE or entry.processed is None:
                    raise RuntimeError(
                        f"upload {uid} ({entry.path.name}) is not ready "
                        f"(status={entry.status})")
                out.append(entry.processed)
        return out

    # ── cleanup ─────────────────────────────────────────────────────
    def remove(self, upload_ids: list[str]) -> None:
        """Delete cache entries for removed/discarded attachments."""
        with self._lock:
            for uid in upload_ids:
                self._entries.pop(uid, None)
