"""LLM input builder: assembles ProcessedFile parts into content blocks.

Output is a list of chat content blocks (``list[dict]``) — the caller
(the graph's ``human_message_node``) wraps them into a ``HumanMessage``:
  - any ImagePart present → text blocks + image blocks interleaved
    (base64 inline or Files API file_id — images are NEVER converted to text)
  - no images → a single text block of labeled text/table sections

Composition order (one combined human input): the uploaded-file block comes
FIRST, labelled with the FILES_UPLOADED_HEADER; the user's typed message is
appended AFTER it (labelled "User Message"). Images are interleaved at their
position within the file block, each preceded by a caption text block
("=== IMAGE n: source ===") so the model knows which image is which.
"""
from __future__ import annotations

from file_input.config import MAX_IMAGES_PER_MESSAGE
from file_input.types import (
    FilePartError,
    ImagePart,
    ProcessedFile,
    TablePart,
    TextPart,
)
from file_input.vision.client import build_image_block

FILES_UPLOADED_HEADER = (
    "File(s) Uploaded (These are files that the user uploaded along with "
    "the user's message):"
)
USER_MESSAGE_LABEL = "User Message"


def _render_text_sections(processed: list[ProcessedFile]) -> str:
    """Render all text/table parts as labeled sections in one string."""
    sections: list[str] = []
    for pf in processed:
        file_chunks: list[str] = [f"=== FILE: {pf.source_name} ({pf.kind.value}) ==="]
        for part in pf.parts:
            if isinstance(part, TextPart):
                file_chunks.append(part.content)
            elif isinstance(part, TablePart):
                file_chunks.append(f"--- Table ---\n{part.content}")
            elif isinstance(part, FilePartError):
                file_chunks.append(f"--- Unreadable: {part.reason} ---")
        if len(file_chunks) > 1:  # at least one content chunk beyond the header
            sections.append("\n".join(file_chunks))
    return "\n\n".join(sections)


def _collect_images(processed: list[ProcessedFile]) -> list[ImagePart]:
    return [part for pf in processed for part in pf.parts if isinstance(part, ImagePart)]


def _render_multimodal_blocks(
    processed: list[ProcessedFile], image_method: str, max_images: int
) -> tuple[list[dict], list[str]]:
    """Build content blocks: per-file text sections with image blocks
    interleaved at their position, each preceded by a caption block.

    Only the first ``max_images`` images are emitted as real image blocks; the
    rest become name-only text annotations and are returned as ``omitted`` so
    the caller can add an explanatory notice.
    """
    blocks: list[dict] = []
    omitted: list[str] = []
    image_index = 0

    for pf in processed:
        file_header = f"=== FILE: {pf.source_name} ({pf.kind.value}) ==="
        file_text: list[str] = [file_header]

        def _flush() -> None:
            if len(file_text) > 1:
                blocks.append({"type": "text", "text": "\n".join(file_text)})

        for part in pf.parts:
            if isinstance(part, TextPart):
                file_text.append(part.content)
            elif isinstance(part, TablePart):
                file_text.append(f"--- Table ---\n{part.content}")
            elif isinstance(part, FilePartError):
                file_text.append(f"--- Unreadable: {part.reason} ---")
            elif isinstance(part, ImagePart):
                _flush()
                file_text = [f"=== FILE: {pf.source_name} ({pf.kind.value}) (continued) ==="]
                image_index += 1
                if image_index > max_images:
                    omitted.append(part.source_name)
                    blocks.append({"type": "text",
                                   "text": f"=== IMAGE {image_index}: {part.source_name} "
                                           f"[OMITTED — image limit reached] ==="})
                else:
                    blocks.append({"type": "text",
                                   "text": f"=== IMAGE {image_index}: {part.source_name} ==="})
                    blocks.append(build_image_block(part, method=image_method))

        _flush()

    return blocks, omitted


def _append_user_message(
    blocks: list[dict], user_message: str | None
) -> list[dict]:
    """Append the typed user message as a trailing text block (if provided)."""
    if user_message:
        blocks.append({"type": "text", "text": f"\n{USER_MESSAGE_LABEL}\n{user_message}"})
    return blocks


def build_content_blocks(
    processed: list[ProcessedFile],
    *,
    image_method: str = "base64",
    user_message: str | None = None,
    max_images: int = MAX_IMAGES_PER_MESSAGE,
) -> list[dict]:
    """Build the LLM content blocks from processed files + typed user message.

    Always returns a content-block list (forward-compatible with vision):
    - images present → file text sections + interleaved image blocks (only the
      first ``max_images`` are sent; the rest are name-only annotations + notice)
    - no images      → a single text block (file header + sections)
    File content comes first; the user message, when given, is appended after
    as a labelled text block.
    """
    images = _collect_images(processed)

    if images:
        blocks, omitted = _render_multimodal_blocks(
            processed, image_method, max_images
        )
        blocks.insert(0, {"type": "text", "text": FILES_UPLOADED_HEADER})
        if omitted:
            blocks.append({"type": "text", "text": (
                f"Note: {len(omitted)} additional image(s) were omitted because the "
                f"per-message image limit ({max_images}) was reached: "
                f"{', '.join(omitted)}. If you need to see any of these, tell the "
                f"user to upload just those image(s)."
            )})
        return _append_user_message(blocks, user_message)

    text = FILES_UPLOADED_HEADER + "\n\n" + (
        _render_text_sections(processed) or "(no content in files)"
    )
    return _append_user_message([{"type": "text", "text": text}], user_message)

