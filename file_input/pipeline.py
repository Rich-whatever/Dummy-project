"""
Pipeline entry point: the one function called when the user hits "send".

Gated flow:
    upload time : store.add_file(path)  → upload_id, parsing starts in background
    send time   : files_to_llm_input(store, ids, user_message=...)
                    ├─ not ready → PipelineResult(ok=False, notice="Cannot send: ...")
                    └─ ready     → fast assembly (parts → content blocks)
"""
from __future__ import annotations

from dataclasses import dataclass

from file_input.builder import build_content_blocks
from file_input.manifest import render_manifest
from file_input.types import ProcessedFile
from file_input.upload_store import FileUploadStore


@dataclass
class PipelineResult:
    ok: bool
    content_blocks: list[dict] | None = None    # LLM content blocks (only when ok)
    manifest: str = ""                          # metadata-only manifest
    notice: str | None = None                   # reason why sending was blocked


def files_to_llm_input(
    store: FileUploadStore,
    upload_ids: list[str],
    *,
    user_message: str | None = None,
    image_method: str = "base64",
) -> PipelineResult:
    """One-call gated pipeline: uploaded files + user text → content blocks.

    Checks the store first: if ANY requested file is still pending or errored,
    returns ok=False with a human-readable notice and does NOT build anything.
    Otherwise returns the chat content blocks (file content, with the user
    message appended after) plus the metadata-only manifest for the
    non-main-agent subsystems. The graph wraps the blocks into a HumanMessage.
    """
    status = store.get_status(upload_ids)
    if not status.ready:
        return PipelineResult(ok=False, notice=status.notice())

    processed = store.get_processed(upload_ids)
    content_blocks = build_content_blocks(
        processed, image_method=image_method, user_message=user_message
    )
    manifest = render_manifest(processed)

    return PipelineResult(ok=True, content_blocks=content_blocks, manifest=manifest)


def summarize_processed(processed: list[ProcessedFile]) -> str:
    """Debug/log summary: kinds, part counts and errors per file."""
    lines = []
    for pf in processed:
        counts: dict[str, int] = {}
        for p in pf.parts:
            counts[type(p).__name__] = counts.get(type(p).__name__, 0) + 1
        lines.append(f"{pf.source_name} [{pf.kind.value}] "
                     f"{', '.join(f'{k}x{v}' for k, v in counts.items()) or 'empty'}")
    return "\n".join(lines)
