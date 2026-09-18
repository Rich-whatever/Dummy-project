"""
Shared text helpers used by more than one agent sub-system.

Deliberately dependency-free (stdlib only) so any layer can import it — both the
worker and the pipeline graph need identical log-line truncation.
"""

from __future__ import annotations

# Default truncation cap for log lines carrying long content (tool outputs,
# worker reports, agent notes, tool-call arguments).
DEFAULT_LOG_LIMIT = 2000


def clip(text: object, limit: int = DEFAULT_LOG_LIMIT) -> str:
    """Shorten long text for log lines, marking the omitted length."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated: {len(text)} chars, showing {limit}]"
