"""
Track 1 — Intelligent Router.

Uses the LLM to decide whether the user message continues an existing
thread or starts a new discussion, and optionally identifies an
additional thread whose context is needed.

The router does NOT append the message; it only identifies the target
thread.  Appending happens later (in the pipeline) once the agent
response is available, so a ``ConversationRound`` is formed with both parts.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from memory_backend.models import ThreadBar, format_rounds

from .llm import get_llm
from .timing import timed_invoke

# ── Constants ──────────────────────────────────────────────────────

NEW_THREAD = "NEW_THREAD"

RouterResult = namedtuple("RouterResult", ["thread_id", "thread_for_context"])
"""
Combined output of the intelligent router.

``thread_id``
    Either an existing ``thread_id`` or ``NEW_THREAD``.

``thread_for_context``
    An optional list of additional thread IDs whose conversation context
    should be fetched and included in the assembly prompt.
    Empty when not needed.
"""

SYSTEM_PROMPT = """
Your task is to decide whether the user's new message continues an existing thread or starts a new one.

You are given:
1. The user's new message.
2. Existing threads, each with:
   - thread_id
   - topic_description
   - last 2 conversation rounds

Judgement:
1. check on context relevance between thread and user message. If message continues the thread, they should have strong relevancy
2. Ask yourself, does having memory of the thread conversation helps agent better response to user's message?
If not, then user message doesn't continue the thread
3. If none of the thread is a good fit for continuaation, return NEW_THREAD

Note: topic_description might not be as accurate, Rely primarily on the conversation rounds




Cross-thread context:
If more than one thread seems relevant, choose the best-fit(you believe user likely refer to its context)
- Add "| context: <secondary_thread_id>" only when another thread provides useful context that the main thread alone cannot provide.
- Do not add it for simple topic similarity or normal topic changes.

Return exactly one line:
<thread_id or NEW_THREAD>
or
<thread_id or NEW_THREAD> | context: <secondary_thread_id>
"""


# ── Router Implementation ──────────────────────────────────────────

def _build_router_input_message(
    user_message: str,
    thread_bar: ThreadBar,
) -> str:
    """
    Format the router prompt body: describe each active thread with
    its topic, last_active, and last 2 messages.
    """
    lines = [f"New user message: \"{user_message}\"", ""]

    lines.append("Active threads:")
    for thread in thread_bar:
        last_two = thread.get_last_n_rounds(2)
        msgs_text = "; ".join(
            f"[USER] {r.user_message} \n [Agent response] {r.agent_response}"
            for r in last_two
        ) if last_two else "(no messages yet)"
        lines.append(
            f"  - thread_id: {thread.thread_id}\n"
            f"    topic: {thread.topic_description or '(untitled)'}\n"
            f"    last_2_messages: {msgs_text}"
        )

    return "\n".join(lines)


def route_message(
    user_message: str,
    thread_bar: ThreadBar,
    llm: Optional[BaseChatModel] = None,
) -> RouterResult:
    """
    Determine thread routing for the user message.

    Parameters
    ----------
    user_message : str
        The incoming user message.
    thread_bar : ThreadBar
        The currently active threads.
    llm : BaseChatModel, optional
        An LLM instance.  If ``None``, the module-level singleton is used.

    Returns
    -------
    RouterResult
        A namedtuple with:
        - ``thread_id``: either an existing ``thread_id`` or ``NEW_THREAD``.
        - ``thread_for_context``: list of additional thread IDs whose context
          should be fetched (empty list when not needed).
    """
    # Shortcut: empty bar → NEW_THREAD, skip LLM call
    if thread_bar.count == 0:
        return RouterResult(
            thread_id=NEW_THREAD,
            thread_for_context=[],
        )

    if llm is None:
        llm = get_llm()

    input_text = _build_router_input_message(user_message, thread_bar)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=input_text),
    ]
    response = timed_invoke("Router", llm.invoke, messages)
    raw = response.content.strip().strip('"').strip("'")

    # Parse output format: "<thread_id>" or "<thread_id> | context: <secondary_id>"
    thread_for_context: list[str] = []

    if " | context: " in raw:
        parts = raw.split(" | context: ", 1)
        thread_id_part = parts[0].strip()
        context_part = parts[1].strip()
        # Validate the context thread ID exists in the bar
        if thread_bar.has_thread(context_part):
            thread_for_context = [context_part]
        # If the context thread doesn't exist, silently drop it
    elif "||" in raw:
        # Fallback: old format "<thread_id> || <chat|execute>" — ignore exec part
        parts = raw.split("||", 1)
        thread_id_part = parts[0].strip()
    else:
        thread_id_part = raw

    # Validate thread_id
    if thread_bar.has_thread(thread_id_part):
        resolved_thread_id = thread_id_part
    else:
        resolved_thread_id = NEW_THREAD

    return RouterResult(
        thread_id=resolved_thread_id,
        thread_for_context=thread_for_context,
    )


# ── Pre-Router: continuation check on the most recent thread ────────

class ContinuationCheck(BaseModel):
    """Pre-router output: does the message continue the most recent thread?"""
    continues_recent_conversation: bool = False


CONTINUATION_PROMPT = """\
You receive the user's new message and the last 2 rounds of the most recent
conversation, aka what user and bot just last talking about.

Decide ONE thing: does this new message CONTINUE that most recent conversation?

- true  → the message is a follow-up, reply, or natural continuation of those
          recent messages (same topic, follow-up question, those "it", "that",
          "also", "what about...", etc.)
- false → the message starts a different topic, switches to something else,
          or refers to some OTHER (older) conversation rather than this one.

Return only the boolean.
"""


def is_continuation_of_recent(
    user_message: str,
    thread_bar: ThreadBar,
    llm: Optional[BaseChatModel] = None,
) -> bool:
    """
    Pre-router (Layer 0): cheap boolean check on whether ``user_message``
    continues the most recently active thread.

    Returns False when there is no recent thread to continue (empty bar or
    a thread with no rounds) — the caller should then use the main router.
    On LLM failure, also returns False (conservative: fall through to the
    main router).
    """
    if thread_bar.count == 0:
        return False
    recent = thread_bar.get_most_recent_thread()
    if recent is None or recent.round_count == 0:
        return False

    if llm is None:
        from .llm import get_llm_no_thinking
        llm = get_llm_no_thinking()

    try:
        structured_llm = llm.with_structured_output(
            ContinuationCheck, method="function_calling"
        )
        messages = [
            SystemMessage(content=CONTINUATION_PROMPT),
            HumanMessage(content=(
                f"=== Recent conversation (last 2 rounds) ===\n"
                f"{format_rounds(recent.get_last_n_rounds(2)) or '(no messages yet)'}\n\n"
                f"=== User's new message ===\n{user_message}"
            )),
        ]
        result: ContinuationCheck = timed_invoke(
            "PreRouter", structured_llm.invoke, messages
        )
        return bool(result.continues_recent_conversation)
    except Exception:
        return False
