"""
Image file reader: loads image bytes and keeps them as an ImagePart.

No API call happens here — the image is preserved untouched. Whether it is
sent inline (base64) or via the DeepSeek Files API is decided later by the
builder / vision layer.
"""
from __future__ import annotations

from pathlib import Path

from file_input.config import MAX_IMAGE_BYTES
from file_input.types import FileKind, FilePartError, ImagePart, ProcessedFile

# Magic-byte signatures; DeepSeek detects the real format from file content,
# so we validate the same way instead of trusting the extension.
_MAGIC = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]

# WebP: RIFF....WEBP
def _detect_mime(data: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def read_image_file(path: str | Path) -> ProcessedFile:
    """Load an image file's bytes into an ImagePart (validated by magic bytes)."""
    p = Path(path)
    name = p.name

    if not p.is_file():
        return ProcessedFile(name, FileKind.IMAGE,
                             [FilePartError(name, f"file not found: {p}")])

    data = p.read_bytes()

    if len(data) > MAX_IMAGE_BYTES:
        mb = len(data) / (1024 * 1024)
        limit_mb = MAX_IMAGE_BYTES // (1024 * 1024)
        return ProcessedFile(name, FileKind.IMAGE, [FilePartError(
            name, f"image too large: {mb:.1f} MB (limit {limit_mb} MB)")])

    mime = _detect_mime(data)
    if mime is None:
        return ProcessedFile(name, FileKind.IMAGE, [FilePartError(
            name, "not a supported image (JPEG/PNG/GIF/WebP) by content")])

    return ProcessedFile(name, FileKind.IMAGE, [ImagePart(name, data, mime)])
