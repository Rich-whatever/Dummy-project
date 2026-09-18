"""
File → processed-parts routing, shared by the public API and the upload store.

Lives in its own module because ``file_input/__init__.py`` imports
``upload_store``; defining this there would create an import cycle.
"""

from __future__ import annotations

from pathlib import Path

from file_input.readers.docling_reader import read_with_docling
from file_input.readers.image_reader import read_image_file
from file_input.readers.text_reader import read_text_file
from file_input.router import classify_file
from file_input.types import FileKind, ProcessedFile


def process_file(path: str | Path) -> ProcessedFile:
    """Route one file to the right reader and return its processed parts."""
    p = Path(path)
    kind = classify_file(p)
    if kind is FileKind.TEXT:
        return read_text_file(p)
    if kind is FileKind.IMAGE:
        return read_image_file(p)
    # TABLE and COMPLEX both go through docling
    return read_with_docling(p, kind)


def process_files(paths: list[str | Path]) -> list[ProcessedFile]:
    """Process a batch of files; errors become FilePartError entries, never raise."""
    return [process_file(p) for p in paths]
