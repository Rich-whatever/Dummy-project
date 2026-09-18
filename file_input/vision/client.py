"""
DeepSeek image-attachment helpers.

Images are never converted to text by this package. This module only builds
OpenAI-compatible content blocks that carry the raw image into a request:
  - "base64":    inline data URL (stateless, default; ≤32 MiB per image)
  - "files_api": upload once via POST /files, reference by file_id
                 (≤64 MiB, reusable across requests; caller manages lifecycle)
"""
from __future__ import annotations

import base64
import os
from functools import lru_cache

from openai import OpenAI

from file_input.config import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_BASE_URL,
)
from file_input.types import ImagePart


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Shared OpenAI-compatible client pointed at the DeepSeek API."""
    api_key = os.environ.get(DEEPSEEK_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Environment variable {DEEPSEEK_API_KEY_ENV} is not set")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _b64_block(part: ImagePart) -> dict:
    b64 = base64.b64encode(part.data).decode("utf-8")
    return {"type": "image_url",
            "image_url": {"url": f"data:{part.mime};base64,{b64}"}}


def _files_api_block(part: ImagePart) -> dict:
    """Upload the image (purpose=user_data, permanent) and reference its file_id."""
    client = get_client()
    from io import BytesIO

    buf = BytesIO(part.data)
    buf.name = part.source_name
    uploaded = client.files.create(file=buf, purpose="user_data")
    return {"type": "file", "file_id": uploaded.id}


def build_image_block(part: ImagePart, method: str = "base64") -> dict:
    """Convert an ImagePart into an OpenAI-compatible chat content block.

    method: "base64" | "files_api"
    """
    if method == "base64":
        return _b64_block(part)
    if method == "files_api":
        return _files_api_block(part)
    raise ValueError(f"unknown image send method: {method!r}")


def delete_file(file_id: str) -> None:
    """Delete a previously uploaded Files API image (lifecycle cleanup)."""
    get_client().files.delete(file_id)
