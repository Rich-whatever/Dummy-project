"""
LRU Thread Eviction — memory management policy.

Trigger: When the Intelligent Router identifies a NEW_THREAD but the
         ThreadBar already has MAX_ACTIVE_THREADS active threads (see config).

Process:
    1. Select the thread with the oldest ``last_active`` timestamp (LRU).
    2. **Valuation Step**:
       - If the thread has <= 2 rounds → dump it (delete entirely).
       - If the thread has >= 3 rounds → quick LLM check:
           "Is this thread valuable info or noise?"
         * Valuable: LLM summarises the entire thread into a summary block,
           pushed to LTM vector store.
         * Noise: Delete the thread entirely.
    3. The new thread is initialised in the vacated slot.

Note: The actual creation of the new thread is the caller's responsibility
      (the Router/Graph). This module handles eviction determination and
      summarisation of evicted threads.
"""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from config import BRIDGE_SUMMARY_WORD_LIMIT
from memory_backend.models import Thread, ThreadBar, format_rounds
from vector_store.client import MemoryVectorStore
from vector_store.operations import upsert_summary
from workflow.timing import timed_invoke

# ── Prompt Templates ───────────────────────────────────────────────

_VALUATION_PROMPT = (
    "Determine whether this thread contains **valuable information** that "
    "should be saved to long-term memory, or whether it is **noise** "
    "(casual chat, off-topic, unimportant). Just ask yourself, is"
    "this conversation worth remembering for weeks, months long, and likely recalled in the future\n\n"
    "Conversation:\n{conversation_text}\n\n"
    "Reply with exactly one word: VALUABLE or NOISE"
)

_SUMMARIZE_EVICTED_PROMPT = (
    "Summarise the following conversation into a single paragraph of "
    "no more than {word_limit} words."
    "Imagine you are trying to remember "
    "this conversation months later. Summarise what you "
    "would most likely remember about it, Your summary should focusing "
    "on the main subject, the most notable information, ideas, events,"
    " or conclusions. Do not recount conversation evenly in agent said, user said"
    "style, that is not information dense and not helpful"
    "Write in past tense.\n\n"
    "Conversation:\n{conversation_text}"
)


# ── Helpers ────────────────────────────────────────────────────────

# Round formatting is shared: see ``memory_backend.models.format_rounds``.


# ── Core Logic ─────────────────────────────────────────────────────

class EvictionResult:
    """
    Result of an eviction decision.

    Attributes:
        evicted_thread_id : str — the thread_id that was removed
        was_summarized    : bool — whether the evicted thread was summarised to LTM
        summary_text      : str — the LTM summary text ("" if not summarized)
    """
    def __init__(
        self,
        evicted_thread_id: str,
        was_summarized: bool = False,
        summary_text: str = "",
    ) -> None:
        self.evicted_thread_id = evicted_thread_id
        self.was_summarized = was_summarized
        self.summary_text = summary_text

    def __repr__(self) -> str:
        return (
            f"EvictionResult(id={self.evicted_thread_id[:8]}…, "
            f"summarized={self.was_summarized})"
        )


def _is_valuable(thread: Thread, llm: BaseChatModel) -> bool:
    """
    Ask the LLM whether the thread contains valuable information.

    Returns True if VALUABLE, False for NOISE.
    """
    conversation_text = format_rounds(thread.conversation_rounds)
    prompt = _VALUATION_PROMPT.format(conversation_text=conversation_text)
    response = timed_invoke("EvictionValuation", llm.invoke, prompt)
    answer = response.content.strip().upper()
    return "VALUABLE" in answer


def _summarize_and_store(
    thread: Thread,
    vector_store: MemoryVectorStore,
    llm: BaseChatModel,
) -> str:
    """
    Summarise the entire thread and push to LTM vector store.

    The summary uses the thread's current ``summary_counter + 1`` as its
    block number, so it integrates cleanly with any existing LTM entries
    for this thread.

    Returns the generated summary text.
    """
    from datetime import datetime, timezone

    conversation_text = format_rounds(thread.conversation_rounds)
    prompt = _SUMMARIZE_EVICTED_PROMPT.format(
        word_limit=BRIDGE_SUMMARY_WORD_LIMIT,
        conversation_text=conversation_text,
    )
    response = timed_invoke("EvictionSummary", llm.invoke, prompt)
    summary_text = response.content.strip()


    timestamp = datetime.now(timezone.utc).isoformat()
    new_block_number = thread.summary_counter + 1

    upsert_summary(
        store=vector_store,
        thread_id=thread.thread_id,
        block_number=new_block_number,
        summary_text=summary_text,
        timestamp=timestamp,
    )

    return summary_text


def evict_lru_thread(
    thread_bar: ThreadBar,
    vector_store: MemoryVectorStore,
    llm: BaseChatModel,
) -> Optional[EvictionResult]:
    """
    Evict the LRU thread from the thread bar using the valuation policy.

    This function:
      1. Finds the LRU thread.
      2. Applies valuation logic (<=2 rounds → dump; >=3 rounds → LLM check).
      3. Removes the thread from the bar.
      4. If valuable, summarises and stores to LTM.
      5. Returns an EvictionResult describing what happened.

    Parameters
    ----------
    thread_bar : ThreadBar
        The active threads container (mutated in-place).
    vector_store : MemoryVectorStore
        The LTM vector store.
    llm : BaseChatModel
        The LLM instance for valuation and summarisation.

    Returns
    -------
    EvictionResult or None
        ``None`` if there's nothing to evict (bar empty).
    """
    if not thread_bar.threads:
        return None

    # Step 1: Select LRU thread
    lru = thread_bar.get_lru_thread()
    if lru is None:
        return None

    evicted_id = lru.thread_id
    was_summarized = False

    # Step 2: Valuation
    if lru.round_count <= 2:
        # Dump — delete entirely
        pass
    else:
        # 3+ rounds → ask LLM
        if _is_valuable(lru, llm):
            summary_text = _summarize_and_store(lru, vector_store, llm)
            was_summarized = True
        # If noise, just delete (do nothing)

    # Step 3: Remove the thread from the bar
    thread_bar.remove_thread(evicted_id)

    return EvictionResult(
        evicted_thread_id=evicted_id,
        was_summarized=was_summarized,
        summary_text=summary_text if was_summarized else "",
    )
