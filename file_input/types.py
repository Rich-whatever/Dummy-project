"""
Data structures for the file_input package.

Every processed file becomes an ordered list of FilePart objects. Parts carry
their content in an LLM-consumable form:
  - TextPart / TablePart: plain strings (markdown for tables)
  - ImagePart: raw image bytes + mime type (kept as an image — never converted)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FileKind(Enum):
    """How the router classified a file."""
    TEXT = "text"          # plain-text file → read content directly
    IMAGE = "image"        # JPEG/PNG/GIF/WebP → kept as bytes, sent to a vision LLM
    TABLE = "table"        # tabular file (xlsx) → docling
    COMPLEX = "complex"    # pdf/docx/... → docling, split into text/table/image parts


@dataclass
class FilePartError:
    """A file (or a piece of one) that could not be processed."""
    source_name: str
    reason: str


@dataclass
class TextPart:
    """A chunk of plain text extracted from a file."""
    source_name: str
    content: str


@dataclass
class TablePart:
    """A table rendered as markdown (docling export) — directly LLM-consumable."""
    source_name: str
    content: str


@dataclass
class ImagePart:
    """An image kept as raw bytes (never OCR'd / never force-converted to text)."""
    source_name: str
    data: bytes
    mime: str = "image/png"


# Union of everything that can appear in a processed file
FilePart = TextPart | TablePart | ImagePart | FilePartError


@dataclass
class ProcessedFile:
    """Result of processing one file: its kind plus its ordered parts."""
    source_name: str
    kind: FileKind
    parts: list[FilePart] = field(default_factory=list)

    def has_image(self) -> bool:
        return any(isinstance(p, ImagePart) for p in self.parts)

    def has_error(self) -> bool:
        return any(isinstance(p, FilePartError) for p in self.parts)

    def errors(self) -> list[FilePartError]:
        return [p for p in self.parts if isinstance(p, FilePartError)]
