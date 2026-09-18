"""Attachment manifest: a metadata-only summary of uploaded files.

This is the string that non-main-agent subsystems (router, preference
retrieve/injection, LTM query, memory) receive alongside the user's typed text
(concatenated into ``state.user_message``). It intentionally carries only the
file NAME and KIND — never file content — so:
  - token cost stays ~zero regardless of file size,
  - memory stores what the classifier saw (consistent),
  - file bodies remain exclusive to the main agent's HumanMessage.

    [attached file: report.pdf (complex)]
    [attached image: screenshot.png (image)]
"""
from __future__ import annotations

from file_input.types import ProcessedFile

_FILE_LABEL = "attached file"
_IMAGE_LABEL = "attached image"


def manifest_line(pf: ProcessedFile) -> str:
    """One manifest line for a processed file (name + kind)."""
    label = _IMAGE_LABEL if pf.has_image() else _FILE_LABEL
    return f"[{label}: {pf.source_name} ({pf.kind.value})]"


def render_manifest(processed: list[ProcessedFile]) -> str:
    """Render the manifest block for the given processed files ('' if none)."""
    if not processed:
        return ""
    return "\n".join(manifest_line(pf) for pf in processed)


def abbreviate(user_message: str, manifest: str) -> str:
    """Combine the typed user message with the file manifest (metadata only)."""
    text = (user_message or "").strip()
    if not manifest:
        return text
    if not text:
        return manifest
    return f"{text}\n\n{manifest}"
