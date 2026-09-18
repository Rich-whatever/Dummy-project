"""
Topic Description Generation — shared utility.

Provides a single function ``generate_topic()`` that uses an LLM to produce
a short (≤20-word) topic description from raw conversation text.

This function is used in two places:
    1. ``memory_backend/compression.py`` — after rolling summarisation.
    2. ``workflow/graph.py`` — when creating a new thread.

The conversation text should be formatted as::

    [USER] What is Python?
    [ASSISTANT] Python is a programming language.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from workflow.timing import timed_invoke

# ── Prompt Template ────────────────────────────────────────────────

_TOPIC_GENERATION_PROMPT = (
    "Based on the following conversation, produce a short topic description "
    "of no more than 20 words. Respond with ONLY the topic phrase.\n\n"
    "Conversation:\n{conversation_text}\n\n"
    "Topic:"
)


# ── Public API ─────────────────────────────────────────────────────

def generate_topic(conversation_text: str, llm: BaseChatModel) -> str:
    """
    Generate a short (≤20-word) topic description from conversation text.

    Parameters
    ----------
    conversation_text : str
        Raw conversation text formatted as ``[USER] ... [ASSISTANT] ...``
        lines.  This should be the **actual conversation content**, not a
        summary or bridge_summary.
    llm : BaseChatModel
        The LLM instance to use for generation.

    Returns
    -------
    str
        A topic description of at most 20 words.
    """
    prompt = _TOPIC_GENERATION_PROMPT.format(
        conversation_text=conversation_text,
    )
    response = timed_invoke("TopicGeneration", llm.invoke, prompt)
    topic = response.content.strip()

    # Safety clamp: if the LLM goes over, truncate to 20 words
    words = topic.split()
    if len(words) > 20:
        topic = " ".join(words[:20])

    return topic
