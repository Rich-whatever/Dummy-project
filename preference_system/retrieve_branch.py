"""
Retrieve Branch — three-tier LLM retrieval for user preferences.

Flow::

    Three async Map Retrievers (one per map section)
      ├── StateMapRetriever         → selected state names
      ├── ExpectationMapRetriever   → selected expectation indices
      └── NoteCategoryMapRetriever  → selected note category names
      │
      ├── "state_name" → StateRetriever (LLM)
      │                    ├── include_state=True/False
      │                    └── include_history=True/False
      ├── "category_name" → NoteRetriever (LLM)
      │                     └── selected_indices: list[int]
      │
      └── Selected names that don't match anything on the map
           are silently dropped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from workflow.timing import timed_ainvoke

from .models import (
    ExpectationMapDecision,
    NoteCategoryMapDecision,
    NoteRetrievalResult,
    PreferenceStore,
    StateMapDecision,
    StateRetrievalResult,
    load_expectations,
)

logger = logging.getLogger("preference_system.retrieve_branch")


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine from a synchronous context.

    Creates a fresh event loop unless one is already running. If called from
    inside a running event loop, raises a clear error directing the caller to
    use the ``arun_*`` async variant instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "Synchronous wrapper called from a running event loop — "
        "use the arun_* async variant instead."
    )


def _note_content(note: Any) -> str:
    """Extract the display text of a note (dict or dataclass)."""
    if isinstance(note, dict):
        return str(note.get("content", ""))
    return str(getattr(note, "content", note))


def _user_block_with_context(
    user_message: str,
    last_round: str | None = None,
) -> str:
    """
    Build the ``=== User Message ===`` block, optionally with the last round of
    conversation appended as extra context.

    ``last_round`` is ONLY additional context for understanding — retrieval
    still targets ``user_message`` itself.
    """
    block = f"=== User Message ===\n{user_message}"
    if last_round:
        block += (
            f"\n\n=== Last Round of Conversation (context only) ===\n{last_round}"
        )
    return block


@dataclass
class TimingEntry:
    """Timing record for one LLM call inside the retrieve branch."""

    name: str          # e.g. "state_map", "state_retriever:coding"
    started_at: str    # ISO-8601 UTC — when the LLM call began
    completed_at: str  # ISO-8601 UTC — when it finished
    duration_ms: float # wall-clock duration of the call
    output: str        # short summary of what the call returned


def _summarize_result(result: Any) -> str:
    """Render a structured-output result as a short human-readable string."""
    if isinstance(result, StateMapDecision):
        return (
            f"state_names={result.state_names}" if result.state_names else "nothing selected"
        )
    if isinstance(result, NoteCategoryMapDecision):
        return (
            f"category_names={result.category_names}"
            if result.category_names
            else "nothing selected"
        )
    if isinstance(result, ExpectationMapDecision):
        return (
            f"expectation_indices={result.expectation_indices}"
            if result.expectation_indices
            else "nothing selected"
        )
    if isinstance(result, StateRetrievalResult):
        return (
            f"include_state={result.include_state}, "
            f"include_history={result.include_history}"
        )
    if isinstance(result, NoteRetrievalResult):
        return f"selected_indices={result.selected_indices}"
    return str(result)


async def _timed_await(
    name: str,
    coro: Any,
    timing: list[TimingEntry] | None,
) -> Any:
    """Await ``coro`` and record a :class:`TimingEntry` when ``timing`` is given.

    Used by the orchestrator to trace every LLM call: when it started, when it
    finished, how long it took, and what it selected.
    """
    t0 = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    result = await coro
    if timing is not None:
        timing.append(
            TimingEntry(
                name=name,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                output=_summarize_result(result),
            )
        )
    return result


# ─────────────────────────────────────────────────────────────────────
# 1. Map Retrievers — three independent async LLM calls, one per section.
#    Each sees ONLY its own map section so length differences between
#    sections cannot skew the selection. Prompts raise the selection bar
#    to avoid over-selecting.
# ─────────────────────────────────────────────────────────────────────

STATE_MAP_SYSTEM_PROMPT = """\
You will receive:
  - The user's message
  - A list of known state names, and their short descriptions
    Each state is an ongoing or recent situation in the user's life

Your job is to decide whether user is currently talking about, acting on, or needing help
with any state(ongoing situation)?
Include a state in state_names ONLY if the state context is needed for better response to user message

An empty list is fine when nothing applies.
"""


async def arun_state_map_retriever(
    llm: BaseChatModel,
    user_message: str,
    state_map_str: str,
    last_round: str | None = None,
) -> StateMapDecision:
    """Layer 1a: LLM decides which state names are relevant (state map only)."""
    structured_llm = llm.with_structured_output(StateMapDecision, method="function_calling")
    result: StateMapDecision = await timed_ainvoke(
        "StateMapRetriever",
        structured_llm.ainvoke,
        [
            SystemMessage(content=STATE_MAP_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"{_user_block_with_context(user_message, last_round)}\n\n"
                    f"=== State Map ===\n{state_map_str}"
                )
            ),
        ],
    )
    logger.debug(
        "StateMapRetriever: selected %d states: %s",
        len(result.state_names),
        result.state_names,
    )
    return result


EXPECTATION_MAP_SYSTEM_PROMPT = """\
You receive:
  - The user's message
  - A list of indexed expectations ("trigger | action")

Select the expectations that should guide the assistant's response.

For each expectation, check:
1. Does the message strongly indicate situation as described by the trigger?
2. If so, does the action meaningfully apply or help guide assistant's response?

Do not infer, stretch, or loosely relate a trigger to the message. Only select
an expectation when both the situation and action clearly fit.

When the fit is uncertain, do not select it. An empty list is fine.

Return only expectation_indices.
"""


async def arun_expectation_map_retriever(
    llm: BaseChatModel,
    user_message: str,
    expectation_map_str: str,
    last_round: str | None = None,
) -> ExpectationMapDecision:
    """Layer 1b: LLM decides which expectation indices apply (expectation map only)."""
    structured_llm = llm.with_structured_output(ExpectationMapDecision, method="function_calling")
    result: ExpectationMapDecision = await timed_ainvoke(
        "ExpectationMapRetriever",
        structured_llm.ainvoke,
        [
            SystemMessage(content=EXPECTATION_MAP_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"{_user_block_with_context(user_message, last_round)}\n\n"
                    f"=== Expectation Map ===\n{expectation_map_str}"
                )
            ),
        ],
    )
    logger.debug(
        "ExpectationMapRetriever: selected %d expectations: %s",
        len(result.expectation_indices),
        result.expectation_indices,
    )
    return result


NOTE_CATEGORY_MAP_SYSTEM_PROMPT = """\
You will receive:
  - The user's message
  - A list of known note categories. Each line is formatted as:

        Category Name: short description of what it contains

Your job is to decide which categories may contain information that would
materially improve the assistant's response to this message.

Select a category when it is reasonably possible that useful user information
is stored there and would matter for this response.

IMPORTANT:
- Output the EXACT category name only, never include the description, a number, an index, or any label.
  Example correct output for the category `Hobbies & Interests: ...` is just:
  `Hobbies & Interests`.
- An empty list is fine when nothing is likely to help.
"""


async def arun_note_category_map_retriever(
    llm: BaseChatModel,
    user_message: str,
    category_map_str: str,
    last_round: str | None = None,
) -> NoteCategoryMapDecision:
    """Layer 1c: LLM decides which note-category names are relevant (category map only)."""
    structured_llm = llm.with_structured_output(NoteCategoryMapDecision, method="function_calling")
    result: NoteCategoryMapDecision = await timed_ainvoke(
        "NoteCategoryMapRetriever",
        structured_llm.ainvoke,
        [
            SystemMessage(content=NOTE_CATEGORY_MAP_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"{_user_block_with_context(user_message, last_round)}\n\n"
                    f"=== Note Category Map ===\n{category_map_str}"
                )
            ),
        ],
    )
    logger.debug(
        "NoteCategoryMapRetriever: selected %d categories: %s",
        len(result.category_names),
        result.category_names,
    )
    return result


# ─────────────────────────────────────────────────────────────────────
# 2. State Retriever
# ─────────────────────────────────────────────────────────────────────

STATE_RETRIEVER_SYSTEM_PROMPT = """\
You are a state-relevance evaluator for an AI assistant.

You will receive:
  - The user's current message
  - A state (including state_name, description, event history)

The state is a record of active user project, goal, or any life situation
that evolves through events over time.
Your job is to decide whether this state is truly relevant to the
user's current message and should be provided as context to the
assistant for formulating its response.

**Rules:**
- ``include_state``: set to True if the state's description or general
  context helps the assistant understand the user's situation.
- ``include_history``: set to True if description alone is too shallow to response to user message
(assisstant needs full knowledge of state development to react)
- If the state is completely irrelevant to the current message, set
  BOTH to False.
"""


async def arun_state_retriever(
    llm: BaseChatModel,
    user_message: str,
    state_name: str,
    state_description: str,
    state_event_list: list[str],
    last_round: str | None = None,
) -> StateRetrievalResult:
    """
    Layer 2a (async): LLM decides whether to surface a state and/or its history.
    """
    lines = [
        _user_block_with_context(user_message, last_round),
        "",
        "=== State ===",
        f"  Name:        {state_name}",
        f"  Description: {state_description}",
        f"  Events ({len(state_event_list)}):",
    ]
    for i, event in enumerate(state_event_list, 1):
        lines.append(f"    {i}. {event}")

    structured_llm = llm.with_structured_output(StateRetrievalResult, method="function_calling")
    result: StateRetrievalResult = await timed_ainvoke(
        "StateRetriever",
        structured_llm.ainvoke,
        [
            SystemMessage(content=STATE_RETRIEVER_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(lines)),
        ],
    )
    logger.debug(
        "StateRetriever('%s'): include_state=%s, include_history=%s",
        state_name,
        result.include_state,
        result.include_history,
    )
    return result


# ─────────────────────────────────────────────────────────────────────
# 3. Note Retriever
# ─────────────────────────────────────────────────────────────────────

NOTE_RETRIEVER_SYSTEM_PROMPT = """\

You will receive:
  - The user's current message
  - A list of user preference, idea, or information.

Your job is to decide which notes should be surfaced to the assistant for this
message.

For each note, ask: "Would knowing this about the user improve or appropriately
change the response to this specific message?" Select it only if yes.

General or objective questions do not need personal information unless the user is asking
for a personalized answer.

Return the selected note indices only. An empty list is fine.

**Rules:**
- Return the 0-based index of each relevant note.
- If no notes are relevant, return an empty list.
- Be selective — only include notes that are clearly pertinent.
"""


async def arun_note_retriever(
    llm: BaseChatModel,
    category_name: str,
    user_message: str,
    notes: list[Any],
    last_round: str | None = None,
) -> NoteRetrievalResult:
    """
    Layer 2b (async): LLM decides which notes from a category are relevant.

    Parameters
    ----------
    notes : list[Any]
        Each entry may be a ``Note`` dataclass (with a ``content`` field) or
        a dict with a ``content`` key.
    """
    lines = [
        _user_block_with_context(user_message, last_round),
        "",
    ]
    for i, note in enumerate(notes):
        lines.append(f"\n  [{i}] {_note_content(note)}")

    structured_llm = llm.with_structured_output(NoteRetrievalResult, method="function_calling")
    result: NoteRetrievalResult = await timed_ainvoke(
        "NoteRetriever",
        structured_llm.ainvoke,
        [
            SystemMessage(content=NOTE_RETRIEVER_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(lines)),
        ],
    )
    logger.debug(
        "NoteRetriever('%s'): selected %d notes (indices=%s)",
        category_name,
        len(result.selected_indices),
        result.selected_indices,
    )
    return result

# ─────────────────────────────────────────────────────────────────────
# 4. Orchestrator
# ─────────────────────────────────────────────────────────────────────

RETRIEVE_BRANCH_FINAL_PROMPT = """\
The following preference data was retrieved and may assist in responding
to the user. Use it if relevant, otherwise ignore it.

"""


class RetrievedStateContext:
    """A state that passed the retrieval filter."""

    def __init__(
        self,
        state_name: str,
        description: str,
        include_state: bool,
        include_history: bool,
        event_list: list[str] | None = None,
    ) -> None:
        self.state_name = state_name
        self.description = description
        self.include_state = include_state
        self.include_history = include_history
        self.event_list = event_list or []

    def to_context_str(self) -> str:
        """Build a text block for the final system prompt injection."""
        lines = [f"[State: {self.state_name}]"]
        if self.include_state:
            lines.append(f"  Description: {self.description}")
        if self.include_history and self.event_list:
            lines.append("  Events:")
            for evt in self.event_list:
                lines.append(f"    - {evt}")
        return "\n".join(lines)


class RetrieveBranchResult:
    """
    Final result of the retrieve branch, ready to be injected into the
    assistant's system prompt or response context.
    """

    def __init__(self) -> None:
        self.selected_states: list[RetrievedStateContext] = []
        self.selected_notes: list[str] = []
        self.selected_expectation: list[str] = []

    def to_context_block(self) -> str:
        """Build a single string containing all retrieved preference data.

        Each section explains what the data represents so the consuming agent
        can judge relevance rather than treating raw labels as meaningful.
        """
        if not self.selected_states and not self.selected_notes and not self.selected_expectation:
            return ""
        parts = [RETRIEVE_BRANCH_FINAL_PROMPT]
        if self.selected_states:
            parts.append(
                "User's ongoing situations, projects, and progress"
            )
            parts.append("=== Retrieved States ===")
            for s in self.selected_states:
                parts.append(s.to_context_str())
            parts.append("")
        if self.selected_notes:
            parts.append(
                "Recorded facts about the user"
            )
            parts.append("=== Retrieved Notes ===")
            for n in self.selected_notes:
                parts.append(f"[Note {n}]")
            parts.append("")
        if self.selected_expectation:
            parts.append(
                "User expectation on agent's behavior)."
            )
            parts.append("=== Retrieved Expectations ===")
            for exp in self.selected_expectation:
                parts.append(f"[Expectation] {exp}")
            parts.append("")
        return "\n".join(parts)

    def __bool__(self) -> bool:
        return bool(
            self.selected_states
            or self.selected_notes
            or self.selected_expectation
        )

    def __repr__(self) -> str:
        return (
            f"RetrieveBranchResult(states={len(self.selected_states)}, "
            f"notes={len(self.selected_notes)}, "
            f"expectations={len(self.selected_expectation)})"
        )


async def arun_retrieve_branch(
    llm: BaseChatModel,
    user_message: str,
    store: PreferenceStore,
    timing: list[TimingEntry] | None = None,
    last_round: str | None = None,
) -> RetrieveBranchResult:
    """
    Run the full retrieve branch pipeline (async).

    1. Three async Map Retrievers run concurrently (one per map section):
       state map, expectation map, note category map.
    2. Selected expectation indices are resolved to their content.
    3. Selected state names → StateRetriever; selected category names →
       NoteRetriever.
    4. Results are collected into a ``RetrieveBranchResult``.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM instance to use for all retrieval calls.
    user_message : str
        The user's current message.
    store : PreferenceStore
        The preference data store.
    timing : list[TimingEntry] | None
        Optional list that gets appended a :class:`TimingEntry` per LLM call
        (plus one ``"total"`` entry). When ``None``, no timing is collected.
    last_round : str | None
        Optional last round of the target conversation, appended to every LLM
        prompt as context-only (retrieval still targets ``user_message``).

    Returns
    -------
    RetrieveBranchResult
        The collected relevant states, notes, and expectations.
    """
    result = RetrieveBranchResult()
    total_t0 = time.perf_counter()
    total_started = datetime.now(timezone.utc).isoformat()

    # ── Step 1: Concurrent map retrieval (one LLM call per section) ──
    state_map = store.get_state()
    expectation_map = store.get_expectations()
    category_map = store.get_categories()

    state_decision, expectation_decision, category_decision = await asyncio.gather(
        _timed_await(
            "state_map",
            arun_state_map_retriever(llm, user_message, state_map, last_round),
            timing,
        ),
        _timed_await(
            "expectation_map",
            arun_expectation_map_retriever(llm, user_message, expectation_map, last_round),
            timing,
        ),
        _timed_await(
            "category_map",
            arun_note_category_map_retriever(llm, user_message, category_map, last_round),
            timing,
        ),
    )

    # ── Step 2: Resolve selected expectation indices to content ──────
    expectations = load_expectations(store.expectations_file)
    for idx in expectation_decision.expectation_indices:
        if 0 <= idx < len(expectations):
            content = str(expectations[idx].get("expectation_content", "")).strip()
            if content:
                result.selected_expectation.append(content)
            else:
                logger.warning(
                    "Expectation index %d has empty expectation_content — skipping.",
                    idx,
                )
        else:
            logger.warning(
                "ExpectationMapRetriever returned out-of-range index %d "
                "(max=%d)",
                idx,
                len(expectations) - 1,
            )

    # ── Step 3: Selected state names → StateRetriever (concurrent) ────
    state_tasks: list[tuple[object, Any]] = []
    for name in state_decision.state_names:
        if name in store.states:
            state = store.states[name]
            state_tasks.append(
                (
                    state,
                    _timed_await(
                        f"state_retriever:{state.state_name}",
                        arun_state_retriever(
                            llm=llm,
                            user_message=user_message,
                            state_name=state.state_name,
                            state_description=state.description,
                            state_event_list=state.event_list,
                            last_round=last_round,
                        ),
                        timing,
                    ),
                )
            )
        else:
            logger.info(
                "StateMapRetriever selected unknown state '%s' — ignoring.", name
            )

    # ── Step 4: Selected category names → NoteRetriever (concurrent) ──
    note_tasks: list[tuple[object, Any]] = []
    for name in category_decision.category_names:
        if name in store.note_categories:
            category = store.note_categories[name]
            if not category:
                logger.info("Note category '%s' is empty — skipping.", name)
                continue
            note_tasks.append(
                (
                    category,
                    _timed_await(
                        f"note_retriever:{category.category_name}",
                        arun_note_retriever(
                            llm=llm,
                            category_name=category.category_name,
                            user_message=user_message,
                            notes=category.notes,
                            last_round=last_round,
                        ),
                        timing,
                    ),
                )
            )
        else:
            logger.info(
                "NoteCategoryMapRetriever selected unknown category '%s' — ignoring.",
                name,
            )

    # ── Step 5: Run ALL stage-2 retrievers concurrently ───────────────
    # States and note categories are independent of each other, so they all
    # go into one ``asyncio.gather`` — wall-clock latency is bounded by the
    # slowest single call instead of growing linearly with the amount of
    # retrieved information.
    all_tasks = [coro for _, coro in state_tasks] + [coro for _, coro in note_tasks]
    if all_tasks:
        all_results = await asyncio.gather(*all_tasks)

        n_states = len(state_tasks)
        for i, state_result in enumerate(all_results[:n_states]):
            state = state_tasks[i][0]
            if state_result.include_state or state_result.include_history:
                result.selected_states.append(
                    RetrievedStateContext(
                        state_name=state.state_name,
                        description=state.description,
                        include_state=state_result.include_state,
                        include_history=state_result.include_history,
                        event_list=state.event_list,
                    )
                )

        for i, note_result in enumerate(all_results[n_states:]):
            category = note_tasks[i][0]
            for idx in note_result.selected_indices:
                if 0 <= idx < len(category.notes):
                    note = category.notes[idx]
                    result.selected_notes.append(note.content)
                else:
                    logger.warning(
                        "NoteRetriever returned out-of-range index %d for "
                        "category '%s' (max=%d)",
                        idx,
                        category.category_name,
                        len(category.notes),
                    )

    if timing is not None:
        timing.append(
            TimingEntry(
                name="total",
                started_at=total_started,
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=(time.perf_counter() - total_t0) * 1000.0,
                output=(
                    f"{len(result.selected_states)} states, "
                    f"{len(result.selected_notes)} notes, "
                    f"{len(result.selected_expectation)} expectations"
                ),
            )
        )

    logger.info(
        "RetrieveBranch: %d states, %d notes, %d expectations selected",
        len(result.selected_states),
        len(result.selected_notes),
        len(result.selected_expectation),
    )
    return result


def run_retrieve_branch(
    llm: BaseChatModel,
    user_message: str,
    store: PreferenceStore,
    last_round: str | None = None,
) -> RetrieveBranchResult:
    """
    Synchronous wrapper around :func:`arun_retrieve_branch`.

    The three map-retrieval LLM calls run concurrently via ``asyncio.gather``,
    so wall-clock time is bounded by the slowest single call rather than the
    sum of all three calls.
    """
    return _run_sync(arun_retrieve_branch(llm, user_message, store, last_round=last_round))


def run_retrieve_branch_detailed(
    llm: BaseChatModel,
    user_message: str,
    store: PreferenceStore,
    last_round: str | None = None,
) -> tuple[RetrieveBranchResult, list[TimingEntry]]:
    """
    Synchronous detailed variant of :func:`run_retrieve_branch`.

    Runs the same pipeline but also returns a list of :class:`TimingEntry`
    records — one per LLM call, including ``started_at``, ``completed_at``,
    ``duration_ms``, and a summary of what each call selected.
    """
    timing: list[TimingEntry] = []
    result = _run_sync(
        arun_retrieve_branch(llm, user_message, store, timing=timing, last_round=last_round)
    )
    return result, timing
