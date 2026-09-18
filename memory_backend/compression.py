"""
Rolling Summarization — memory management policy.

Trigger: When a thread reaches MAX_MESSAGES_PER_THREAD (8) rounds.

Process:
    1. Extract the earliest 6 messages from the thread.
    2. LLM produces a ~150-word summary of those 6 rounds.
       (Note: the bridge_summary is NOT re-summarised here — only raw messages.)
    3. The new summary is upserted to the LTM vector store with
       thread_id and the *next* summary_counter value.
    4. The new summary replaces the thread's bridge_summary.
    5. thread.summary_counter is incremented.
    6. The earliest 6 rounds are removed from the thread (keeping the latest 2).
    7. The LLM briefly updates the thread's topic_description to reflect
       any topic shifts.
"""

from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel

from config import BRIDGE_SUMMARY_WORD_LIMIT, MAX_MESSAGES_PER_THREAD
from memory_backend.models import Thread, format_rounds
from memory_backend.topic import generate_topic
from vector_store.client import MemoryVectorStore
from vector_store.operations import upsert_summary
from workflow.timing import timed_invoke

# ── Prompt Templates ───────────────────────────────────────────────

_SUMMARIZE_PROMPT = (
    "Summarise the following conversation into a single paragraph of "
    "no more than {word_limit} words. Capture the key questions, answers, "
    "and decisions. Write in past tense.\n\n"
    "Conversation:\n{conversation_text}"
)



# ── Core Functions ─────────────────────────────────────────────────

# Round formatting is shared: see ``memory_backend.models.format_rounds``.


def _call_summarizer(conversation_text: str, llm: BaseChatModel) -> str:
    """
    Call the LLM to produce a ~150-word summary of the given text.
    """
    prompt = _SUMMARIZE_PROMPT.format(
        word_limit=BRIDGE_SUMMARY_WORD_LIMIT,
        conversation_text=conversation_text,
    )
    response = timed_invoke("RollingSummarize", llm.invoke, prompt)
    return response.content.strip()




# ── Main Entry Point ───────────────────────────────────────────────

def rolling_summarize(
    thread: Thread,
    vector_store: MemoryVectorStore,
    llm: BaseChatModel,
) -> str:
    """
    Perform rolling summarization on a thread.

    This function mutates *thread* in-place:
      - The earliest 6 rounds are removed (keeping the last 2).
      - bridge_summary is replaced with the new summary.
      - summary_counter is incremented.
      - topic_description is updated.

    The new summary is also upserted into the LTM vector store.

    Parameters
    ----------
    thread : Thread
        The thread to compress (must be full — 8 rounds).
    vector_store : MemoryVectorStore
        The LTM vector store to upsert into.
    llm : BaseChatModel
        The LLM instance for summarisation and topic updates.

    Returns
    -------
    str
        The generated summary text.
    """
    rounds_to_compress = MAX_MESSAGES_PER_THREAD - 2  # keep 2 → compress 6

    # Step 1: extract oldest rounds
    conversation_text = format_rounds(thread.conversation_rounds[:rounds_to_compress])

    # Step 2: LLM summarisation
    summary_text = _call_summarizer(conversation_text, llm)

    # Step 3: upsert to LTM
    timestamp = datetime.now(timezone.utc).isoformat()
    new_block_number = thread.summary_counter + 1
    upsert_summary(
        store=vector_store,
        thread_id=thread.thread_id,
        block_number=new_block_number,
        summary_text=summary_text,
        timestamp=timestamp,
    )

    # Step 4: increment summary counter
    thread.summary_counter = new_block_number

    # Step 5: replace bridge_summary
    thread.bridge_summary = summary_text

    # Step 6: remove oldest rounds (keep the latest 2)
    thread.conversation_rounds = thread.conversation_rounds[-2:]

    # Step 7: update topic description from raw conversation (not summary)
    new_topic = generate_topic(conversation_text, llm)
    if new_topic:
        thread.topic_description = new_topic

    return summary_text
