"""
File router: classifies a file by its extension into a FileKind.

Categories:
  - IMAGE:   .jpg .jpeg .png .gif .webp          → image bytes kept for vision LLM
  - TABLE:   .xlsx .xls                          → docling (table export)
  - COMPLEX: .pdf .docx .pptx                    → docling (split into parts)
  - TEXT:    everything else readable as text   → read content directly
Unknown extensions are sniffed: valid UTF-8 → TEXT, else COMPLEX (docling best-effort).
"""
from __future__ import annotations

from pathlib import Path

from file_input.types import FileKind

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
TABLE_EXTENSIONS = {".xlsx", ".xls"}
COMPLEX_EXTENSIONS = {".pdf", ".docx", ".pptx"}


def classify_file(path: str | Path) -> FileKind:
    """Classify a file path into a FileKind based on its extension."""
    suffix = Path(path).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return FileKind.IMAGE
    if suffix in TABLE_EXTENSIONS:
        return FileKind.TABLE
    if suffix in COMPLEX_EXTENSIONS:
        return FileKind.COMPLEX

    # Unknown extension: sniff content — valid UTF-8 means it's a text file
    try:
        Path(path).read_text(encoding="utf-8")
        return FileKind.TEXT
    except (UnicodeDecodeError, OSError):
        # Not UTF-8 (binary) → hand to docling as a best-effort complex file;
        # if the path doesn't exist / can't be read at all, still route to TEXT
        # so the reader produces a clear error part.
        if not Path(path).exists():
            return FileKind.TEXT
        return FileKind.COMPLEX
