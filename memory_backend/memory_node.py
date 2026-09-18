"""
LangGraph nodes for the memory pipeline (router + LTM + memory update).

Contains:
    - ``router_node``           — Track 1: Intelligent Router
    - ``ltm_node``              — Track 2: LTM Retriever
    - ``update_memory_node``    — post-generation memory persistence & eviction

These nodes are designed to be imported and wired into a LangGraph
``StateGraph`` (see ``workflow/graph.py``).
"""

from __future__ import annotations

import asyncio
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from memory_backend.compression import rolling_summarize
from memory_backend.eviction import evict_lru_thread
from memory_backend.models import Thread
from memory_backend.topic import generate_topic
from preference_system.state_injection import run_state_injection
from workflow.ltm_retriever import retrieve_ltm
from workflow.router import (
    NEW_THREAD,
    RouterResult,
    is_continuation_of_recent,
    route_message,
)
from workflow.state import PipelineState
from workflow.timing import timed_ainvoke

# ── Logger ───────────────────────────────────────────────────────────
logger = logging.getLogger("memory_pipeline")
# Where the PreferenceStore is persisted (one source of truth so the save and
# its log line can never drift apart).
PREFERENCES_PATH = "preferences.json"


def _log_state_injection(
    trigger: str, result: dict[str, object] | None, path: str
) -> None:
    """Log one state-injection outcome: did it store, what, and where."""
    if result is None:
        logger.info(
            "STATE-INJ → (%s) NOT STORED — summary not state-worthy or the "
            "classifier/updater LLM call failed",
            trigger,
        )
        return

    action = result.get("action")
    name = result.get("state_name")
    if action == "created":
        evicted = result.get("evicted_state")
        logger.info(
            "STATE-INJ → (%s) STORED new state '%s' in %s — description=%r  "
            "initial_event=%r%s",
            trigger, name, path,
            result.get("description"), result.get("event"),
            f"  (evicted oldest state '{evicted}')" if evicted else "",
        )
    elif action == "updated":
        logger.info(
            "STATE-INJ → (%s) STORED update to state '%s' in %s — "
            "description_changed=%s  new_description=%r  new_event=%r",
            trigger, name, path,
            result.get("description_changed"),
            result.get("new_description"), result.get("new_event"),
        )
    else:  # unchanged
        logger.info(
            "STATE-INJ → (%s) NOT STORED — state '%s' needs no changes",
            trigger, name,
        )


# ── retrieve_with_last_round probe ───────────────────────────────────
# Decides whether the preference-retrieve should include the target thread's
# last round. Runs concurrently with the router; if the router finishes first,
# the probe result is discarded and the conservative default (True) is kept.


class RetrieveNeedsLastRound(BaseModel):
    """Whether the user message needs the last round to be understood."""

    needs_last_round: bool = False


RETRIEVE_LAST_ROUND_PROBE_PROMPT = """\
You receive the user's current message.

Decide whether the message is self-explanatory on its own, or whether it
references the previous round of conversation and can only be understood with
that context.

- needs_last_round = true   if the message is ambiguous / refers to prior talk
- needs_last_round = false  if the message stands alone

Return valid JSON only.
"""


async def _probe_retrieve_needs_last_round(user_message: str) -> bool:
    """Small-LLM probe: does the retrieve need the last round? Returns bool."""
    from workflow.llm import get_small_llm

    llm = get_small_llm()
    structured_llm = llm.with_structured_output(
        RetrieveNeedsLastRound, method="function_calling"
    )
    result: RetrieveNeedsLastRound = await timed_ainvoke(
        "RetrieveNeedsLastRound",
        structured_llm.ainvoke,
        [
            SystemMessage(content=RETRIEVE_LAST_ROUND_PROBE_PROMPT),
            HumanMessage(content=f"=== User Message ===\n{user_message}"),
        ],
    )
    logger.debug(
        "RETRIEVE PROBE → needs_last_round=%s  msg=\"%s\"",
        result.needs_last_round,

        user_message[:60],
    )
    return bool(result.needs_last_round)



async def arouter_node(state: PipelineState) -> dict:
    """
    Async Track 1 node: pre-router + main router + retrieve probe.

    Flow:
      1. The small retrieve-needs-last-round probe and the pre-router
         (continuation check, no-thinking LLM) both start at node start.
      2. Pre-router true  → quick path: route to the most recent thread,
         skip the main router (thread_for_context stays empty).
      3. Pre-router false → run the main router; the retrieve probe result
         is used if ready, else cancelled (default True kept).

    Returns ``router_result``, ``thread_for_context``, ``execution_path``,
    and ``retrieve_with_last_round``.
    """
    user_message = state.user_message
    node_start = time.perf_counter()

    # ── 1. len heuristic first (instant, no LLM) ─────────────────────
    if len(user_message) > 40:
        retrieve_with_last_round = False
        probe_task = None
    else:
        retrieve_with_last_round = True  # conservative default
        probe_task = asyncio.create_task(
            _probe_retrieve_needs_last_round(user_message)
        )

    # ── 2. Pre-router races with the retrieve probe ──────────────────
    from workflow.llm import get_llm_no_thinking
    no_think_llm = get_llm_no_thinking()

    pre_task = asyncio.create_task(
        asyncio.to_thread(
            is_continuation_of_recent,
            user_message,
            state.thread_bar,
            no_think_llm,
        )
    )
    continuation = await pre_task

    # ── 3a. Quick path: continuation → most recent thread, no main router
    if continuation:
        recent = state.thread_bar.get_most_recent_thread()
        result_id = recent.thread_id if recent else NEW_THREAD

        if probe_task is not None:
            try:
                # NOTE: await the task itself — .result() is not a coroutine and
                # raises InvalidStateError if the task hasn't finished yet.
                retrieve_with_last_round = await probe_task
            except Exception:
                logger.exception("ROUTER → retrieve probe failed; keeping default")

        elapsed = time.perf_counter() - node_start
        logger.info(
            "ROUTER → PRE-ROUTER continuation=True → latest thread=%s…  msg=\"%s\"  | took %.2fs",
            result_id[:8],
            user_message[:60],
            elapsed,
        )
        return {
            "router_result": result_id,
            "thread_for_context": [],
            "execution_path": "",
            "retrieve_with_last_round": retrieve_with_last_round,
        }

    # ── 3b. Not a continuation → main router (blocking call in thread) ─
    router_task = asyncio.create_task(
        asyncio.to_thread(
            route_message,
            user_message,
            state.thread_bar,
            no_think_llm,
        )
    )
    result: RouterResult = await router_task

    # ── 4. After main router is done: use probe if ready, else halt it ─
    if probe_task is not None:
        if probe_task.done():
            try:
                retrieve_with_last_round = probe_task.result()
            except Exception:
                logger.exception("ROUTER → retrieve probe failed; keeping default")
        else:
            # can't afford to wait — keep len default; reap the cancelled task
            # so we don't leave an un-awaited CancelledError behind
            probe_task.cancel()
            try:
                await probe_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("ROUTER → retrieve probe failed during cancel; keeping default")

    # ── logging ─────────────────────────────────────────────────────
    elapsed = time.perf_counter() - node_start
    if result.thread_id == NEW_THREAD:
        logger.info(
            "ROUTER → PRE-ROUTER continuation=False → NEW_THREAD  msg=\"%s\"  | took %.2fs",
            user_message[:60],
            elapsed,
        )
    else:
        thread = state.thread_bar.get_thread(result.thread_id)
        topic = thread.topic_description if thread else "?"
        rounds = thread.round_count if thread else 0
        ctx = f"  context={result.thread_for_context}" if result.thread_for_context else ""
        logger.info(
            "ROUTER → PRE-ROUTER continuation=False → thread=%s…  rounds=%d  topic=\"%s\"%s  msg=\"%s\"  | took %.2fs",
            result.thread_id[:8],
            rounds,
            topic[:40],
            ctx,
            user_message[:60],
            elapsed,
        )

    return {
        "router_result": result.thread_id,
        "thread_for_context": result.thread_for_context,
        "execution_path": "",
        "retrieve_with_last_round": retrieve_with_last_round,
    }


# ── Nodes ──────────────────────────────────────────────────────────


def router_node(state: PipelineState) -> dict:
    """
    LangGraph node for Track 1 (sync variant).

    Pre-router first: if the message continues the most recent thread,
    route there and skip the main router. Otherwise fall through to the
    full intelligent router.
    """
    # Fall back to the live singleton when no LLM was injected (mirrors the
    # async router node / agent node).
    if state.llm is None:
        from workflow.llm import get_llm

        llm = get_llm()
    else:
        llm = state.llm

    if is_continuation_of_recent(
        user_message=state.user_message,
        thread_bar=state.thread_bar,
        llm=llm,
    ):
        recent = state.thread_bar.get_most_recent_thread()
        result_id = recent.thread_id if recent else NEW_THREAD
        logger.info(
            "ROUTER → PRE-ROUTER continuation=True → latest thread=%s…  msg=\"%s\"",
            result_id[:8],
            state.user_message[:60],
        )
        return {
            "router_result": result_id,
            "thread_for_context": [],
            "execution_path": "",
        }

    result: RouterResult = route_message(
        user_message=state.user_message,
        thread_bar=state.thread_bar,
        llm=llm,
    )

    # ── logging ─────────────────────────────────────────────────────
    if result.thread_id == NEW_THREAD:
        logger.info(
            "ROUTER → NEW_THREAD  msg=\"%s\"",
            state.user_message[:60],
        )
    else:
        thread = state.thread_bar.get_thread(result.thread_id)
        topic = thread.topic_description if thread else "?"
        rounds = thread.round_count if thread else 0
        ctx = f"  context={result.thread_for_context}" if result.thread_for_context else ""
        logger.info(
            "ROUTER → CONTINUE thread=%s…  rounds=%d  topic=\"%s\"%s  msg=\"%s\"",
            result.thread_id[:8],
            rounds,
            topic[:40],
            ctx,
            state.user_message[:60],
        )

    return {
        "router_result": result.thread_id,
        "thread_for_context": result.thread_for_context,
        "execution_path": "",
    }


def ltm_node(state: PipelineState) -> dict:
    """
    LangGraph node for Track 2 (LTM Retriever).

    Queries the vector store (score-gated, no ID filter) and returns
    the raw LTM results.
    """
    t0 = time.perf_counter()
    results = retrieve_ltm(
        user_message=state.user_message,
        vector_store=state.vector_store,
    )
    elapsed = time.perf_counter() - t0

    # ── logging ─────────────────────────────────────────────────────
    n_raw = len(results)
    if n_raw == 0:
        logger.info(
            "LTM     → query returned 0 results  msg=\"%s\"  | took %.2fs",
            state.user_message[:60],
            elapsed,
        )
    else:
        scores = ", ".join(f"{s:.3f}" for _, s in results)
        details = "; ".join(
            f"[{d.metadata.get('thread_id','?')[:8]}… "
            f"sum#{d.metadata.get('block#','?')} "
            f"score={s:.3f}]"
            for d, s in results
        )
        logger.info(
            "LTM     → query returned %d results  scores=[%s]  %s  | took %.2fs",
            n_raw,
            scores,
            details,
            elapsed,
        )

    return {"ltm_results": results}


def update_memory_node(state: PipelineState) -> dict:
    """
    LangGraph node for Memory Update.

    Handles three post-generation responsibilities:

    1. **Persist the round**: Appends the user message and agent response
       to the appropriate thread (or creates a new thread for NEW_THREAD).

    2. **Rolling Summarisation**: If the thread is full after appending,
       calls ``rolling_summarize()`` to compress the oldest 6 rounds.

    3. **LRU Eviction**: If a NEW_THREAD was created and the bar is full,
       evicts the LRU thread before creating the new one.

    Note: If ``router_result == NEW_THREAD`` and the bar is full, eviction
    happens BEFORE the new thread is added.
    """

    evicted_info = ""
    target_thread: Thread | None = None

    # Resolve the LLM the same way the other nodes do: the graph may pass
    # ``llm=None`` (e.g. the web API) and expect each node to fall back to the
    # live singleton. Without this, the ``and state.llm`` gates below silently
    # skip eviction / topic generation / rolling summarisation.
    if state.llm is None:
        from workflow.llm import get_llm

        llm = get_llm()
    else:
        llm = state.llm

    if state.router_result == NEW_THREAD:
        # --- New Thread ---
        # Step A: Eviction check
        if state.thread_bar.is_full and llm:
            result = evict_lru_thread(
                thread_bar=state.thread_bar,
                vector_store=state.vector_store,
                llm=llm,
            )
            if result:
                evicted_info = (
                    f"evicted={result.evicted_thread_id[:8]}…"
                    f", summarized={result.was_summarized}"
                )
                logger.info(
                    "MEMORY  → LRU eviction — thread %s…  "
                    "summarized=%s  (bar was full, new thread incoming)",
                    result.evicted_thread_id[:8],
                    result.was_summarized,
                )
                if (
                    result.was_summarized
                    and result.summary_text
                    and state.preference_store is not None
                ):
                    state_injection_result = run_state_injection(
                        summary_text=result.summary_text,
                        preference_store=state.preference_store,
                    )
                    _log_state_injection(
                        "eviction summary", state_injection_result, PREFERENCES_PATH)
                    try:
                        state.preference_store.save(PREFERENCES_PATH)

                    except Exception:
                        logger.exception(
                            "MEMORY  → failed to save preference store (eviction)"
                        )

        # Step B: Create and add new thread
        new_thread = Thread()
        if llm and len(state.user_message) > 80:
            new_thread.topic_description = generate_topic(
                f"[USER] {state.user_message}",
                llm,
            )
        else:
            new_thread.topic_description = (
                f"New conversation about: {state.user_message[:50]}"
            )
        if not state.thread_bar.add_thread(new_thread):
            logger.error(
                "MEMORY  → could not add new thread (bar still full: %d) — "
                "the round will NOT be persisted",
                state.thread_bar.count,
            )
        target_thread = new_thread
        logger.info(
            "MEMORY  → created new thread %s…  topic=\"%s\"",
            target_thread.thread_id[:8],
            target_thread.topic_description[:50],
        )

    else:
        # --- Existing Thread ---
        target_thread = state.thread_bar.get_thread(state.router_result)
        round_count = target_thread.round_count if target_thread else 0
        logger.info(
            "MEMORY  → continuing thread %s…  (rounds before append: %d)",
            state.router_result[:8],
            round_count,
        )

    # Step C: Append the round (if we have a valid thread)
    if target_thread is not None:
        target_thread.append_round(
            user_message=state.user_message,
            agent_response=state.agent_response,
            underlying_content=state.underlying_content,
        )
        logger.info(
            "MEMORY  → appended round #%d to thread %s…",
            target_thread.round_count,
            target_thread.thread_id[:8],
        )

        # Step D: Rolling summarisation if thread is now full
        if target_thread.is_full and llm:
            logger.info(
                "MEMORY  → Rolling summarisation triggered on %s…  "
                "(6→2, sum#=%d)",
                target_thread.thread_id[:8],
                target_thread.summary_counter + 1,
            )
            summary_text = rolling_summarize(
                thread=target_thread,
                vector_store=state.vector_store,
                llm=llm,
            )
            logger.info(
                "MEMORY  → Rolling summarisation complete — %s…  "
                "new sum#=%d  bridge=\"%s\"",
                target_thread.thread_id[:8],
                target_thread.summary_counter,
                target_thread.bridge_summary[:60],
            )
            if state.preference_store is not None:
                state_injection_result = run_state_injection(
                    summary_text=summary_text,
                    preference_store=state.preference_store,
                )
                _log_state_injection(
                    "rolling summary", state_injection_result, PREFERENCES_PATH
                )
                try:
                    state.preference_store.save(PREFERENCES_PATH)

                except Exception:
                    logger.exception(
                        "MEMORY  → failed to save preference store (rolling)"
                    )

    return {"thread_was_evicted": evicted_info}
