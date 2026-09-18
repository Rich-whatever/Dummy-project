"""
Standalone file-ingestion subsystem: routes files by kind, extracts ordered
parts (text / table / image), and assembles them into chat content blocks.

Deliberately self-contained — no imports from the main agent system.

Public API:
    store = FileUploadStore(); id = store.add_file(path)   # upload → background parse
    files_to_llm_input(store, ids, user_message=...)       # send → PipelineResult
    build_content_blocks(processed)                        # low-level parts → content blocks
"""
from __future__ import annotations

from file_input.builder import build_content_blocks
from file_input.manifest import abbreviate, manifest_line, render_manifest
from file_input.pipeline import PipelineResult, files_to_llm_input, summarize_processed
from file_input.processing import process_file, process_files
from file_input.router import classify_file
from file_input.types import (
    FileKind,
    FilePart,
    FilePartError,
    ImagePart,
    ProcessedFile,
    TablePart,
    TextPart,
)
from file_input.upload_store import FileUploadStore, StoreStatus

__all__ = [
    "classify_file",
    "process_file",
    "process_files",
    "build_content_blocks",
    "render_manifest",
    "manifest_line",
    "abbreviate",
    "files_to_llm_input",
    "PipelineResult",
    "summarize_processed",
    "FileUploadStore",
    "StoreStatus",
    "FileKind",
    "FilePart",
    "FilePartError",
    "TextPart",
    "TablePart",
    "ImagePart",
    "ProcessedFile",
]
