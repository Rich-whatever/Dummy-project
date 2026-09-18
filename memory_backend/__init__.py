"""
Memory backend — thread management and short-term / long-term memory logic.

Sub-modules:
    models         — dataclasses for ConversationRound, Thread, ThreadBar
    compression    — rolling summarization policy               (Phase 3)
    eviction       — LRU thread eviction with valuation         (Phase 3)
"""

from .compression import rolling_summarize
from .eviction import EvictionResult, evict_lru_thread
from .models import ConversationRound, Thread, ThreadBar

__all__ = [
    "ConversationRound",
    "Thread",
    "ThreadBar",
    "rolling_summarize",
    "evict_lru_thread",
    "EvictionResult",
]
