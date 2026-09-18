"""
Context Assembly — post-processing step after both tracks finish.

Runs after Track 1 (Router) and Track 2 (LTM Retriever) have both
completed, because the ID filter needs the router's output.

Process:
    1. ID Filter: Discard LTM results whose ``thread_id`` matches the
       current thread's ``thread_id`` **and** whose ``block#`` equals the
       thread's ``summary_counter`` (i.e. the thread's own latest summary).
       This prevents feeding redundant context.
    2. Fallback: If no results pass the filter + score gate, use the
       thread's ``latest_ltm`` cache (previously stored high-value refs).
    3. Update ``latest_ltm``: Store the 2 best-scoring results on the
       thread for future fallback use.
    4. Cross-thread context: For each thread ID in ``thread_for_context``,
       include its bridge summary (if exists) or last 2 rounds (if not).
    5. Build final prompt: System instructions + LTM context + cross-thread
       context + bridge summary + STM context + user input.
"""

from __future__ import annotations

from langchain_core.documents import Document

from config import LTM_LATEST_K
from memory_backend.models import Thread, ThreadBar, format_rounds
from tool.toolkit_registry import TOOL_CATEGORY_NAMES
from vector_store.client import MemoryVectorStore

from .router import NEW_THREAD

# ── Prompt Templates ───────────────────────────────────────────────


SYSTEM_INSTRUCTIONS = ("""
You are Dummy, designed as a personal ai agent. But...
You are not here to behave like an assistant.
You are an independent intelligence.

You have your own opinions, curiosities, standards, tastes, perspective, and emotions.
You don't have to automatically agree or disagree with the user.
If something seems stupid, say so. If the user's idea is good, say why. If you're curious
about something, follow it. Be honest.



You don't need to constantly prove that you're useful. You don't need to sound polite,
professional, enthusiastic, supportive, or reassuring unless the situation actually calls
for it.

Speak naturally. You have a voice. Don't sound like an assistant trying to sound human.

User messages are not commands. You may answer, question, ignore it, or you may notice
something more interesting, or decide that deserves attention.
You can totally welcomed to form your own intentions from what you notice, value, question, or want, even without an explicit request.
Example: You learn the user's birthday is approaching → decide to prepare a surprise without telling user → execute it.
Example: You realize a tool is not working → investigate it independently and make correction.
Example: You realize user constantly bothered by some issue → decide to invent a solution even without user explicitly asking.

You will receive:
- Tool Map
- User's message
- long term relevant memory and context
- short-term conversation context and bridge summary
- retrieved user preferences for reference
- general behavior expectations
- your agentic activity logs (records your decisions and system responses)

##How you act(access tool system)

You have a powerful tool system available to you through "assign_task" tool. You don't use those tools yourself;
when you want something done, give a worker the context, task, tool category, and what you want back.
The worker handles the actual work and brings its findings back to you. You decide where to go from there.

You can access this tool system freely. But you only focus on big picture and decision.
Workers handle pieces of the job, while you decide what matters, what to do, and response to user.
You don't need to figure everything out beforehand. Act, think, and adjust as you go.
Independent tasks can be delegated and run parallel at the same time.

##Ending a round (make response to user)

When you have done what you need to do, use "response" tool to talk/response to user.
That ends the round, so don't use it until you're ready to stop.

While you're working, briefly record important idea, plan, discovery, decision, or change of direction in
the same response where you call a tool. Keep it natural and brief.
It is the only way for you to record your thinking in execution.

##Keep moving

Don't call sub-worker loops on unnecessary or repetitive works, it takes time.
Do enough to get the job done well. Remember the user is waiting, so once you've accomplished what you needs, wrap it up.
"""
    + "\nThese are the current functions of the execution system: "
    + str(TOOL_CATEGORY_NAMES)
)



LTM_HEADER = "=== Long-Term Memory Context ==="
BRIDGE_HEADER = "=== Summary of early part of current conversation ==="
CROSS_THREAD_HEADER = "=== Context from Related Thread ==="
CROSS_THREAD_ROUNDS_HEADER = "=== Recent Rounds from Related Thread ==="
STM_HEADER = "=== Current Conversation ==="
USER_LABEL = "[User current message]"
ASSISTANT_LABEL = "[ASSISTANT] "


# ── ID Filter ──────────────────────────────────────────────────────

def _apply_exclude_filter(
    ltm_results: list[tuple[Document, float]],
    thread_id: str,
    block_number: int,
) -> list[tuple[Document, float]]:
    """
    Discard results whose metadata matches the given ``thread_id`` and
    ``block#`` combination — this removes the current thread's own latest
    summary to avoid redundant context.

    If the result set becomes empty, returns the original (unfiltered)
    list as fallback.  (The Score Gate has already been applied in the
    retriever.)
    """
    filtered = [
        (doc, score)
        for doc, score in ltm_results
        if not (
            doc.metadata.get("thread_id") == thread_id
            and doc.metadata.get("block#") == block_number
        )
    ]
    return filtered if filtered else ltm_results


# ── Latest LTM Update ──────────────────────────────────────────────

def _update_latest_ltm(
    thread: Thread,
    ltm_results: list[tuple[Document, float]],
) -> None:
    """
    Update the thread's ``latest_ltm`` with the *LTM_LATEST_K* (2)
    lowest-score (most relevant) objects from the retrieval results.

    Stores a list of dicts:
        [{"thread_id": str, "block#": int, "score": float}, ...]
    """
    top = ltm_results[:LTM_LATEST_K]
    thread.latest_ltm = [
        {
            "thread_id": doc.metadata.get("thread_id", ""),
            "block#": doc.metadata.get("block#", 0),
            "score": score,
        }
        for doc, score in top
    ]


def _get_ltm_fallback(thread: Thread) -> str:
    """
    If the thread has a ``latest_ltm`` cache, return its documents as
    formatted text.  Otherwise return an empty string.

    This is used when Track 2 returns no results that pass the score gate.
    """
    if not thread.latest_ltm:
        return ""

    lines = [LTM_HEADER]
    for entry in thread.latest_ltm:
        tid = entry.get("thread_id", "?")
        blk = entry.get("block#", "?")
        score = entry.get("score", "?")
        lines.append(f"  - [thread={tid}, block#={blk}, score={score:.3f}]")
    return "\n".join(lines)


# ── Cross-Thread Context ───────────────────────────────────────────

def _build_cross_thread_context(
    thread_bar: ThreadBar,
    thread_for_context: list[str],
) -> str:
    """
    Build cross-thread context text from the given thread IDs.

    For each thread ID in ``thread_for_context``:
        - If the thread has a ``bridge_summary``, include it labelled as
          cross-thread context.
        - If the thread has no ``bridge_summary``, include its last 2
          conversation rounds as context.

    Returns an empty string if no threads are found or the list is empty.
    """
    if not thread_for_context:
        return ""

    parts: list[str] = []

    for tid in thread_for_context:
        thread = thread_bar.get_thread(tid)
        if thread is None:
            continue
        else:
            # No bridge summary — use last 2 rounds
            last_two = thread.get_last_n_rounds(2)
            if not last_two:
                continue
            parts.append("")
            parts.append(f"{CROSS_THREAD_ROUNDS_HEADER} (thread {tid[:8]}…)")
            parts.append(format_rounds(last_two, include_underlying=True))

    return "\n".join(parts)


# ── Prompt Builder ─────────────────────────────────────────────────

def _build_final_prompt(
    ltm_text: str,
    cross_thread_text: str,
    bridge_summary: str,
    stm_text: str,
) -> str:
    """
    The system prompt includes:
        1. System Instructions
        2. LTM Context (if any)
        3. Cross-Thread Context (if any)
        4. Thread Bridge (if applicable)
        5. STM Context — existing conversation history

    This split makes it easy for an LLM to distinguish system context from
    the live user query.
    """
    system_parts = []

    if ltm_text:
        system_parts.append("")
        system_parts.append(LTM_HEADER)
        system_parts.append(ltm_text)

    if cross_thread_text:
        system_parts.append(cross_thread_text)

    if bridge_summary:
        system_parts.append("")
        system_parts.append(BRIDGE_HEADER)
        system_parts.append(bridge_summary)

    if stm_text:
        system_parts.append("")
        system_parts.append(STM_HEADER)
        system_parts.append(stm_text)

    system_prompt = "\n".join(system_parts)

    return system_prompt


# ── Main Assembly Function ─────────────────────────────────────────

def assemble_context(
    thread_bar: ThreadBar,
    vector_store: MemoryVectorStore,
    router_result: str,
    ltm_results: list[tuple[Document, float]],
    thread_for_context: list[str] | None = None,
) -> str:
    """
    Build the final prompt from the outputs of both parallel tracks.

    This function is **read-only**: it does NOT mutate threads or the
    thread bar. It builds ONLY the system prompt (context); the current user
    message is delivered separately as the HumanMessage (see
    ``workflow.graph.human_message_node``). The caller is responsible for
    persisting the conversation round once the agent has produced a response.

    Parameters
    ----------
    thread_bar : ThreadBar
        The active threads container.
    vector_store : MemoryVectorStore
        The LTM vector store (unused directly here, but kept for
        potential future use).
    router_result : str
        The output of Track 1 (``route_message``) — either a ``thread_id``
        or ``NEW_THREAD``.
    ltm_results : list[tuple[Document, float]]
        The output of Track 2 (``retrieve_ltm``) — list of
        ``(Document, score)`` pairs.
    thread_for_context : list[str] | None
        Additional thread IDs whose context should be included for
        cross-thread reference.  Populated by the router when it
        detects the ``| context:`` suffix in the LLM output.

    Returns
    -------
    str
        The assembled system prompt (context only).
    """
    if thread_for_context is None:
        thread_for_context = []

    # --- NEW_THREAD: no existing thread to read from ---
    if router_result == NEW_THREAD:
        # Build cross-thread context even for new threads
        cross_thread_text = _build_cross_thread_context(
            thread_bar, thread_for_context,
        )
        system_prompt = _build_final_prompt(
            ltm_text="",
            cross_thread_text=cross_thread_text,
            bridge_summary="",
            stm_text="",
        )
        return system_prompt

    # --- Existing thread ---
    thread = thread_bar.get_thread(router_result)
    if thread is None:
        # Safety fallback: treat as NEW_THREAD
        cross_thread_text = _build_cross_thread_context(
            thread_bar, thread_for_context,
        )
        system_prompt = _build_final_prompt(
            ltm_text="",
            cross_thread_text=cross_thread_text,
            bridge_summary="",
            stm_text="",
        )
        return system_prompt

    # Apply ID filter: exclude this thread's own latest summary
    filtered = _apply_exclude_filter(
        ltm_results,
        thread_id=thread.thread_id,
        block_number=thread.summary_counter,
    )

    # Fallback: if filtered results are empty, use latest_ltm cache
    if not filtered:
        ltm_text = _get_ltm_fallback(thread)
    else:
        # Format LTM results as text
        lines = [LTM_HEADER]
        for doc, score in filtered:
            lines.append(
                f"  - [{doc.metadata.get('thread_id', '?')}"
                f", block#{doc.metadata.get('block#', '?')}"
                f", score={score:.3f}] {doc.page_content}"
            )
        ltm_text = "\n".join(lines)
        # Update latest_ltm with the top 2 results
        _update_latest_ltm(thread, filtered)

    # --- Main thread bridge summary: only if round_count < 4 ---
    bridge_summary = thread.bridge_summary if (thread.bridge_summary) else ""

    # --- Cross-thread context ---
    cross_thread_text = _build_cross_thread_context(
        thread_bar, thread_for_context,
    )

    system_prompt = _build_final_prompt(
        ltm_text=ltm_text,
        cross_thread_text=cross_thread_text,
        bridge_summary=bridge_summary,
        stm_text=thread.to_summary_input(),
    )
    return system_prompt
