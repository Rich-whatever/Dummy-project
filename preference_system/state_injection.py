"""
State Injection — feed LTM summaries from the memory pipeline into the
preference system's state store.

A *state* is something recent about the user that is worth recording and likely
to keep updating in the future (e.g. "planning travel to Japan", "designing an
agent system"). Two LLM stages:

    1. Classifier — does the summary belong to an existing state, or should it
       start a new state? (Low-information / unrelated / non-user discussion is
       skipped.)
    2. Updater — given the (existing or new) state and the summary, decide
       whether to update the description and/or append an event.

Flow::

    run_state_injection(llm, summary_text, preference_store)
        │
        ├── classifier (structured LLM)
        │       ├── not state-worthy            → no-op, return None
        │       ├── new_state=True              → create state + initial event
        │       └── belong to existing state    → updater (structured LLM)
        │                                            ├── description/event changes
        │                                            └── apply + refresh active_time
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from workflow.timing import timed_invoke

from .models import (
    PreferenceStore,
    StateClassificationResult,
    StateUpdateResult,
)

logger = logging.getLogger("preference_system.state_injection")

# Maximum number of states kept in the PreferenceStore. When a new state would
# push the total above this, the state with the earliest active_time is evicted.
MAX_STATES = 10

# ─────────────────────────────────────────────────────────────────────
# LLM prompts
# ─────────────────────────────────────────────────────────────────────

STATE_CLASSIFIER_SYSTEM_PROMPT = """\
You are a state-classification analyst for a personal AI assistant.

You receive:
- A ~150-word conversation summary
- The current state map

A state is a recent, ongoing aspect of the user's life, work, goals, or
projects that is likely to develop further.
Examples: "planning travel to Japan", "designing an agent system".

Decide:

1. Existing state match:
   -> new_state=false, state_name=the exact existing state name.

2. New state:
   -> new_state=true, state_name=a short state name.

3. Not state-worthy:
   -> new_state=false, state_name="".

Not state-worthy:
- generic discussion or unrelated topics
- one-off tasks without ongoing relevance
- temporary situations with no future development
- content that does not describe the user's ongoing activities, goals, or projects

Use judgement. Only create a state when there is a clear ongoing topic that may
change or accumulate over time.

Return ONLY a valid json format.
"""

STATE_UPDATE_SYSTEM_PROMPT = """
You are a state-maintenance analyst for a personal AI assistant.

You receive:
  - A ~150-word conversation summary
  - The current state, either existing (description + events) or new (name only)

Update the state with meaningful new information.

Output:

1. description — revised high-level state description (≤60 words).
   Describe what the state is and its current situation, not its history.
   Return "" if the current description remains accurate.

2. event — short event to append to history (≤20 words).
   Return "" if the summary contains no meaningful state update.

Rules:

  - New state: always create initial description and event.
  - Existing state: only record changes that affect the state's progress,
    direction, or future. Ignore normal discussion, explanations, temporary
    issues, and details without lasting value.
  - Description is a high-level representation of the state's overall situation. (Like the definition of the state)
    It shouldn't be updated on every progress or changes, but only be update if the description no longer apply to the state's general
    situation. Update description should be rewrite or minor adjustment, but not appending, description should be ≤60 words
  - Return ONLY valid JSON with description and event.
"""

# ─────────────────────────────────────────────────────────────────────
# Stage 1: Classifier
# ─────────────────────────────────────────────────────────────────────


def run_state_classifier(
    llm: BaseChatModel,
    summary_text: str,
    preference_store: PreferenceStore,
) -> StateClassificationResult | None:
    """
    Ask the LLM whether ``summary_text`` belongs to an existing state, starts a
    new state, or is not state-worthy at all.

    Returns a ``StateClassificationResult``, or ``None`` if the LLM call fails.
    """
    structured_llm = llm.with_structured_output(
        StateClassificationResult, method="function_calling"
    )
    state_map = preference_store.get_state()

    try:
        result: StateClassificationResult = timed_invoke(
            "StateClassificationResult",
            structured_llm.invoke,
            [
                SystemMessage(content=STATE_CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"{state_map}\n\n"
                        f"=== Conversation Summary ===\n{summary_text}"
                    )
                ),
            ],
        )
    except Exception:
        logger.exception("State classifier LLM call failed")
        return None

    logger.debug(
        "State classifier: state_name=%r, new_state=%s",
        result.state_name,
        result.new_state,
    )
    return result


# ─────────────────────────────────────────────────────────────────────
# Stage 2: Updater
# ─────────────────────────────────────────────────────────────────────


def run_state_update(
    llm: BaseChatModel,
    summary_text: str,
    state_name: str,
    is_new_state: bool,
    description: str = "",
    event_list: list[str] | None = None,
) -> StateUpdateResult | None:
    """
    Ask the LLM how to update (or initialise) a state given ``summary_text``.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM instance to call.
    summary_text : str
        The ~150-word conversation summary.
    state_name : str
        The state's name.
    is_new_state : bool
        True when this is a freshly created state (write initial values).
    description : str
        The existing state description (ignored for new states).
    event_list : list[str] | None
        The existing state event list (ignored for new states).

    Returns
    -------
    StateUpdateResult | None
        The updater's decision, or ``None`` if the LLM call fails.
    """
    structured_llm = llm.with_structured_output(StateUpdateResult, method="function_calling")

    if is_new_state:
        state_block = (
            "=== State (new) ===\n"
            f"  Name: {state_name}\n"
            "  (no description or events yet — created from this summary)"
        )
    else:
        lines = [
            "=== State ===",
            f"  Name:        {state_name}",
            f"  Description: {description or '(none)'}",
            f"  Events ({len(event_list or [])}):",
        ]
        for i, event in enumerate(event_list or [], 1):
            lines.append(f"    {i}. {event}")
        state_block = "\n".join(lines)

    try:
        result: StateUpdateResult = timed_invoke(
            "StateUpdateResult",
            structured_llm.invoke,
            [
                SystemMessage(content=STATE_UPDATE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"{state_block}\n\n"
                        f"=== Conversation Summary ===\n{summary_text}"
                    )
                ),
            ],
        )
    except Exception:
        logger.exception("State updater LLM call failed for '%s'", state_name)
        return None

    logger.debug(
        "State updater('%s'): description_len=%d, event=%r",
        state_name,
        len(result.description or ""),
        result.event,
    )
    return result


# ─────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────


def run_state_injection(
    summary_text: str,
    preference_store: PreferenceStore,
) -> dict[str, object] | None:
    """
    Feed a conversation summary into the state store.

    Returns a summary dict describing what happened, or ``None`` when
    nothing was state-worthy / the chain failed defensively.
    """
    from workflow.llm import get_llm_no_thinking
    llm = get_llm_no_thinking()
    summary_text = (summary_text or "").strip()
    if not summary_text:
        logger.debug("State injection skipped — empty summary")
        return None

    classification = run_state_classifier(llm, summary_text, preference_store)
    if classification is None:
        logger.error("State classifier failed — no state injected")
        return None

    state_name = (classification.state_name or "").strip()
    new_state = bool(classification.new_state)

    if not state_name:
        logger.info("State injection: summary not state-worthy — no-op")
        return None

    # ── New state ────────────────────────────────────────────────────
    if new_state:
        if state_name in preference_store.states:
            logger.error(
                "State classifier marked new_state=true but state '%s' already "
                "exists — no injection",
                state_name,
            )
            return None

        update = run_state_update(
            llm, summary_text, state_name, is_new_state=True
        )
        if update is None:
            logger.error("State updater failed for new state '%s'", state_name)
            return None

        description = (update.description or "").strip() or (
            "State created from a conversation summary."
        )
        event = (update.event or "").strip()

        preference_store.add_state(state_name, description)
        if event:
            preference_store.add_event(state_name, event)

        # Enforce the state cap: evict the oldest state if we went over.
        evicted_state: str | None = None
        if len(preference_store.states) > MAX_STATES:
            oldest = min(
                preference_store.states.values(), key=lambda s: s.active_time
            )
            evicted_state = oldest.state_name
            preference_store.remove_state(evicted_state)
            logger.warning(
                "State cap reached (%d states) — evicted oldest state '%s'",
                MAX_STATES,
                evicted_state,
            )

        logger.info(
            "State injection: created new state '%s' -> description=%r, "
            "initial_event=%r%s",
            state_name,
            description,
            event,
            f", evicted='{evicted_state}'" if evicted_state else "",
        )
        return {
            "action": "created",
            "state_name": state_name,
            "evicted_state": evicted_state,
            "description": description,
            "event": event,
        }

    # ── Existing state ──────────────────────────────────────────────
    state = preference_store.states.get(state_name)
    if state is None:
        logger.error(
            "State classifier selected '%s' but it does not exist — no injection",
            state_name,
        )
        return None

    update = run_state_update(
        llm,
        summary_text,
        state_name,
        is_new_state=False,
        description=state.description,
        event_list=state.event_list,
    )
    if update is None:
        logger.error("State updater failed for existing state '%s'", state_name)
        return None

    new_description = (update.description or "").strip()
    new_event = (update.event or "").strip()
    old_description = state.description

    changed = False
    if new_description and new_description != state.description:
        state.description = new_description
        changed = True
    if new_event:
        state.add_event(new_event)
        changed = True

    if changed:
        state.active_time = datetime.now(timezone.utc).isoformat()
        if new_description and new_description != old_description:
            logger.info(
                "State injection: updated state '%s' -> new_description=%r "
                "(was %r)",
                state_name,
                new_description,
                old_description,
            )
        else:
            logger.info(
                "State injection: updated state '%s' -> appended_event=%r",
                state_name,
                new_event,
            )
        return {
            "action": "updated",
            "state_name": state_name,
            "description_changed": bool(new_description and new_description != old_description),
            "new_description": new_description,
            "new_event": new_event,
        }

    logger.info("State injection: no changes needed for state '%s'", state_name)
    return {"action": "unchanged", "state_name": state_name}
