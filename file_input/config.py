"""
Configuration for the file_input package.
"""
from __future__ import annotations

# ── DeepSeek API (used only for the Files API image-attachment option) ──
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"

# How images are attached to LLM requests: "base64" (inline data URL, default)
# or "files_api" (upload to DeepSeek Files API, reference by file_id).
# Images are never converted to text anywhere in this package.
IMAGE_SEND_METHOD = "base64"

# ── Image guards ────────────────────────────────────────────────────
# Per-image raw-byte cap (base64 inflates ~33% on the wire). An image over this
# becomes a FilePartError → the upload shows ✗ with a reason and sending is blocked.
MAX_IMAGE_BYTES = 24 * 1024 * 1024          # 32 MB per image
# Max images actually sent inline per message. Images beyond this are replaced
# by name-only text annotations plus a notice telling the LLM to ask the user.
MAX_IMAGES_PER_MESSAGE = 5

# ── Guards ──────────────────────────────────────────────────────────
MAX_TEXT_FILE_BYTES = 1_000_000   # skip text files larger than ~1 MB
