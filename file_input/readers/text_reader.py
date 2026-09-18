"""
Plain-text file reader: reads the full content of a text file into a TextPart.
"""
from __future__ import annotations

from pathlib import Path

from file_input.config import MAX_TEXT_FILE_BYTES
from file_input.types import FileKind, FilePartError, ProcessedFile, TextPart


def read_text_file(path: str | Path) -> ProcessedFile:
    """Read a text file's full content. UTF-8 first, latin-1 fallback."""
    p = Path(path)
    name = p.name

    if not p.is_file():
        return ProcessedFile(name, FileKind.TEXT,
                             [FilePartError(name, f"file not found: {p}")])
    if p.stat().st_size > MAX_TEXT_FILE_BYTES:
        return ProcessedFile(name, FileKind.TEXT, [FilePartError(
            name, f"text file too large ({p.stat().st_size} bytes > "
                  f"{MAX_TEXT_FILE_BYTES})")])

    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = p.read_text(encoding="latin-1")
        except (UnicodeDecodeError, OSError) as e:
            return ProcessedFile(name, FileKind.TEXT,
                                 [FilePartError(name, f"unreadable as text: {e}")])
    except OSError as e:
        return ProcessedFile(name, FileKind.TEXT,
                             [FilePartError(name, f"read failed: {e}")])

    return ProcessedFile(name, FileKind.TEXT, [TextPart(name, content)])
