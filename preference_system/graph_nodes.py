"""
LangGraph nodes that wire the preference system into the main agent pipeline.

- ``preference_retrieve_node``  — runs the preference retrieve branch after the
                                  router; stores ``retrieved_preference`` in state.
- ``preference_injection_node`` — fire-and-forget: runs the merged
                                  preference-injection pipeline in the
                                  background, then ends.

These nodes are imported by ``workflow/graph.py``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from memory_backend.models import format_rounds
from workflow.llm import get_llm_no_thinking
from workflow.router import NEW_THREAD
from workflow.state import PipelineState

from .expectation_injection import DEFAULT_BEHAVIOR_GUIDE_FILE
from .preference_injection import run_preference_injection
from .retrieve_branch import arun_retrieve_branch

logger = logging.getLogger("preference_system.graph_nodes")

# Fire-and-forget background injection bookkeeping (see preference_injection_node).
# Strong references keep the Tasks from being garbage-collected mid-flight.
_background_tasks: set[asyncio.Task] = set()
_injection_lock = asyncio.Lock()


# ── Helpers ─────────────────────────────────────────────────────────


def load_general_behavior_expectations(
    guide_file: str | Path = DEFAULT_BEHAVIOR_GUIDE_FILE,
) -> str:
    """
    Load the Always-trigger expectations from the behavior-guide JSON file.

    The file stores ``{"length": <int>, "guides": [<str>, ...]}``. This returns
    the guides joined as a single context block, or ``""`` when the file is
    missing / malformed / empty.
    """
    path = Path(guide_file)
    if not path.exists():
        return ""
    from json_io import load_or_quarantine

    data, backup = load_or_quarantine(
        path, default={}, validate=lambda d: isinstance(d, dict)
    )
    if backup:
        logger.error(
            "Behavior guide %s was corrupt — quarantined to %s", path, backup
        )
        return ""
    guides = data.get("guides") if isinstance(data, dict) else None
    if not guides:
        return ""
    return "\n".join(f"- {g}" for g in guides if str(g).strip())


# ── Node: Preference Retrieve ───────────────────────────────────────


def _format_last_round(last_round) -> str | None:
    """Format a ConversationRound (or list of one) into a context string."""
    if last_round is None:
        return None
    rounds = last_round if isinstance(last_round, list) else [last_round]
    if not rounds:
        return None
    return format_rounds(rounds) or None


async def preference_retrieve_node(state: PipelineState) -> dict:
    """
    Run the preference retrieve branch after the router.

    If ``state.retrieve_with_last_round`` is True and the routed thread exists
    and has a last round, that last round is passed into the retrieve as
    context-only. Stores the resulting context block in ``retrieved_preference``
    and the Always-trigger behavior rules in ``general_behavior_expectation``.
    """
    # Always-trigger behavior rules are loaded regardless of the store, since
    # they are global agent guidelines.
    general_behavior_expectation = load_general_behavior_expectations()

    if state.preference_store is None:
        logger.info("PREFERENCE RETRIEVE → no preference_store, skipped")
        return {
            "retrieved_preference": "",
            "general_behavior_expectation": general_behavior_expectation,
        }

    user_message = state.user_message

    last_round = None
    if (
        state.retrieve_with_last_round
        and state.router_result
        and state.router_result != NEW_THREAD
    ):
        thread = state.thread_bar.get_thread(state.router_result)
        if thread is not None and thread.round_count > 0:
            last_round = _format_last_round(thread.get_last_n_rounds(1))
            logger.info(
                "PREFERENCE RETRIEVE → including last round of thread %s…",
                state.router_result[:8],
            )

    t0 = time.perf_counter()
    try:
        result = await arun_retrieve_branch(
            llm=get_llm_no_thinking(),
            user_message=user_message,
            store=state.preference_store,
            last_round=last_round,
        )
    except Exception:
        logger.exception("PREFERENCE RETRIEVE → failed, returning empty context")
        return {
            "retrieved_preference": "",
            "general_behavior_expectation": general_behavior_expectation,
        }
    retrieve_elapsed = time.perf_counter() - t0

    context_block = result.to_context_block()
    logger.info(
        "PREFERENCE RETRIEVE → %s | took %.2fs",
        result,
        retrieve_elapsed,
    )
    return {
        "retrieved_preference": context_block,
        "general_behavior_expectation": general_behavior_expectation,
    }


# ── Node: Preference Injection (fire-and-forget) ────────────────────


async def preference_injection_node(state: PipelineState) -> dict:
    """
    Launch the merged preference-injection pipeline (fire-and-forget → END).

    The injection runs in a background task (see
    ``_run_preference_injection_background``) so the pipeline is NOT blocked:
    ``assembly_node`` / ``agent_node`` proceed while notes are extracted and
    persisted.  State injection is intentionally NOT run here yet.
    """
    if state.preference_store is None:
        logger.info("PREFERENCE INJECTION → no preference_store, skipped")
        return {}

    llm = get_llm_no_thinking()

    _spawn_injection_task(
        _run_preference_injection_background(
            llm=llm,
            user_message=state.user_message,
            store=state.preference_vector_store,
            preference_store=state.preference_store,
        )
    )
    return {}


def _spawn_injection_task(coro) -> None:
    """Spawn a fire-and-forget background task with a strong reference.

    Without the module-level ``_background_tasks`` registry the ``Task`` could
    be garbage-collected mid-flight and cancelled before it finishes.
    """
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_preference_injection_background(
    llm,
    user_message: str,
    store,
    preference_store,
) -> None:
    """
    Run the merged preference-injection pipeline in the background.

    The v2 pipeline does ONE merged extraction call (notes + expectations),
    prefilters note duplicates, then ONE admission-gate call that validates
    both streams and assigns note categories.

    Serialized by ``_injection_lock``: since the response no longer waits for
    injection, a second user message may arrive while a previous injection is
    still running — the lock keeps overlapping injections from racing on the
    shared ``PreferenceStore`` / ``preferences.json``.
    """
    async with _injection_lock:
        t0 = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                run_preference_injection,
                llm,
                user_message,
                store,
                preference_store,
            )
            elapsed = time.perf_counter() - t0

            if result is None:
                injection_summary = "failed"
            else:
                injection_summary = result.summary()

            logger.info(
                "PREFERENCE INJECTION → %s | took %.2fs",
                injection_summary,
                elapsed,
            )

            # Persist the PreferenceStore (states / note_categories) back to
            # JSON so injected notes/states survive restarts and don't become
            # Chroma orphans.
            try:
                preference_store.save("preferences.json")
            except Exception:
                logger.exception(
                    "PREFERENCE INJECTION → failed to save preference store"
                )
        except Exception:
            logger.exception("PREFERENCE INJECTION → background task failed")


async def wait_for_background_injections() -> None:
    """Await any still-running fire-and-forget preference injections.

    The REPL calls this after the agent response is printed so the latency
    table and next input are not shown until the background note/expectation
    injection — and its ``preference_store.save(...)`` — has completed.
    """
    while _background_tasks:
        await asyncio.gather(*list(_background_tasks), return_exceptions=True)
