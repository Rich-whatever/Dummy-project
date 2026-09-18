"""
Note injection — the merged note pipeline (extract → prefilter → gate → persist).

``run_preference_extraction`` produces FLAT, uncategorized note candidates plus
expectation candidates in ONE call; ``prefilter_note_candidates`` drops
duplicates/orphans against the vector store; ``run_admission_gate`` then
validates notes AND expectations and assigns categories; finally
``persist_categorized_notes`` writes the survivors to the PreferenceStore and
the PreferenceVectorStore using the SAME note_id.

Ordering guarantee:
    The PreferenceStore is updated BEFORE the vector store. A vector-write
    failure rolls the PreferenceStore back, so nothing half-written is left
    behind.

When a category grows to ``NOTE_MAINTENANCE_THRESHOLD`` (20) notes, maintenance
is triggered: an LLM reviews the category's notes and may (a) delete obsolete
notes, and/or (b) merge overlapping notes into new consolidated notes.
``backfill_category_descriptions`` fills missing category descriptions.

Flow — see ``preference_injection.run_preference_injection``::

    extract (ONE LLM) -> prefilter duplicates -> admission gate (ONE LLM)
        -> persist_categorized_notes (PreferenceStore-first, same note_id,
           rollback, maintenance)
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from config import PREFERENCE_NOTE_SIMILARITY_THRESHOLD
from preference_system.models import (
    AdmissionResult,
    MergedExtractionResult,
    NoteCategory,
    PreferenceStore,
)
from preference_system.vector_store import PreferenceVectorStore
from workflow.timing import timed_invoke

logger = logging.getLogger("preference_system.note_injection")

# Maintenance runs once a category holds at least this many notes.
NOTE_MAINTENANCE_THRESHOLD = 20


PREFERENCE_EXTRACTION_SYSTEM_PROMPT = """\
You task is to extract lasting information worth remembering from the user's message.

Extract TWO kinds of information:

1. NOTES — long lasting facts that are true about the USER:
- lasting interests, preferences, values, opinions, priorities
- recurring habits or ways of doing things
- stable likes, dislikes, or tendencies
- preferred tools, workflows, or approaches
- Don't record any temporary state about user
  since they will likely be invalid in the future.

Some counter-examples (No notes record cases):
    - I'm applying for clubs and project team (no record since it is temporary)
    - I just started using tool X (current workflow; not necessarily a stable preference)
    - I decided to use X for this project (project-specific decision, not necessarily a general preference)
    - I need to finish X before the end of this semester (time-bound task / temporary goal)

2. EXPECTATIONS — things the USER wants the AGENT to do or avoid in future interactions:
- lasting rules about how the agent should respond, think, communicate, or behave
- preferences about the agent's style, decision-making, or workflow
- Ignore one-time/temporary expectation, focus on general or lasting expectation
- Only record when user explicitly, clearly stated behavior guide on agent. Do not imply expectation. Do not record every niche demand.
  Expecation have high bar for record, be selective on record or not.
- general / lasting expectation tends to have indications (like from now on, next time it happens...), but it's not definite.
  It is certainly possible for worth-recording expectation not having these indication words, you have to use your judgement considering guidelines mentioned here
- trigger says when the rule applies; use "Always" if it always applies. Action says what the agent should do. Keep both very short.

Some Counter-examples (no expectation record cases):
    - "Use your knowledge storage when analzying this file." (task-specific instruction, not a general agent preference)
    - "When you're writing this email, make it sound casual." / "Just give me the answer, no explanation." (one-time instruction, unless explicitly established as a general rule)


For Both Fields:
Only extract information that is likely to matter months from now.
Ignore one-time requests, temporary situations, facts that don't affect future
interactions, and details that are obvious from the current conversation.

Do not infer or generalize. Only record what the user clearly established.

Same information should only appear in either NOTES or EXPECTATIONS, not recorded in both.
If it describes the user, put it in NOTES. If it tells the agent how to behave,
put it in EXPECTATIONS.

Return:
{
  "notes": ["..."],
  "expectations": [
    {"expectation_content": "...", "trigger": "...", "action": "..."}
  ]
}

Return empty lists when nothing worth to record(it is a common case). Do not invent anything just to fill them.
"""

ADMISSION_GATE_SYSTEM_PROMPT = """\
You decide which candidate memories are worth keeping and where they belong.

You receive:
- candidate NOTES about the user
- candidate EXPECTATIONS for agent behavior
- existing NOTE CATEGORIES
- optionally, nearby existing notes for comparison

## NOTES

For each candidate, ADMIT or REJECT it.

ADMIT only if the note:
- is a lasting (stay true for long time) fact or preference about the user
- is likely to remain useful months from now

REJECT if it is temporary, one-time, vague, trivial, not about the
user, or already covered by an existing note. If it tells the agent how to
behave, reject it because it belongs in EXPECTATIONS.

For admitted notes, assign the best existing category.
Create a new category only if none fits. New categories
should be broad and reusable, with a short description of what kind of
information they contain.

## EXPECTATIONS

For each candidate, ACCEPT or REJECT it.

ACCEPT only explicit, lasting rules about how the agent should behave.
Reject one-time requests, temporary instructions, implied preferences, or
anything not clearly stated by the user.

Omit rejected candidates entirely — they are dropped silently and must not be
reported.

Return exactly these three fields, using 0-based candidate indices:
1. accepted_expectation_indices: indices of the accepted expectations.
2. notes: each accepted note's index mapped to the category name it belongs to.
3. created_category: for every NEW category used in ``notes``, a short
   description (<15 words) of the kind of information it holds.
"""

NOTE_MAINTENANCE_SYSTEM_PROMPT = """\
You are a preference-note maintenance analyst. You receive the numbered list of
all notes stored in ONE category. Each line is:

    <index>. <note content>

The notes are listed in SAVING ORDER: a lower index was saved earlier (older),
a higher index was saved later (newer / more recent knowledge about the user).

Your task includes following maintenance operations:

1. Delete obsolete notes (deleted_notes)
Delete notes that are no longer reliable user knowledge, including:
- Notes contradicted by newer information.
- Temporary/event-based notes that have expired.

When resolving contradictions:
- Consider permanence and context, not only recency.
- Temporary restrictions should not overwrite permanent preferences.
  Example: "I like seafood" should remain if "I don't want seafood this week" appears.
- If a newer note represents a permanent change, remove the old note.
  Example: "I like seafood" -> "I no longer eat seafood".

2. Merge redundant notes (merged_notes + merged_content)
Merge notes only when they represent the same or closely related information.
- merged_notes contains all source note indices.
- merged_content contains the new consolidated note.
- Preserve important details and avoid merging unrelated information.

Rules:
- Use valid 0-based indices only.
- Do not use the same index in both deleted_notes and merged_notes.
- Do not delete or merge all notes in a category.
- If no maintenance is needed, return empty results.
"""

# ─────────────────────────────────────────────────────────────────────
# Structured-output models
# ─────────────────────────────────────────────────────────────────────


class NoteMaintenanceResult(BaseModel):
    """LLM output for a category maintenance pass."""

    deleted_notes: list[int]
    merged_notes: list[int]
    merged_content: list[str]


class CategoryDescriptionResult(BaseModel):
    """LLM output for the category-description backfill pass.

    ``descriptions`` maps category name → a very short (<15 words) description
    of what the category does or should contain.
    """

    descriptions: dict[str, str]


# ─────────────────────────────────────────────────────────────────────
# Category-description backfill (one-shot for categories missing a description)
# ─────────────────────────────────────────────────────────────────────

CATEGORY_DESCRIPTION_SYSTEM_PROMPT = """\
You are a preference-taxonomy analyst. You receive note categories of a personal
memory system. Each category may come with some of its stored notes as context.

Your task: for EVERY category listed, write a very short description (under 15
words) of what that category does or should contain.

Rules:
- Descriptions must be broad and reusable, matching the category's scope —
  not a summary of any single note.
- Describe the TYPE of information the category holds (e.g. "What the user
  likes and dislikes eating and drinking.").
- Use the EXACT category name as the key in your output.
- Return one description per listed category, nothing else.
"""


def backfill_category_descriptions(
    llm: BaseChatModel | None,
    preference_store: PreferenceStore,
) -> int:
    """Fill in ``category_description`` for every category missing one.

    Uses a single structured LLM call over all description-less categories,
    with each category's existing notes as context. Saves nothing itself —
    the caller persists the store. Returns the number of categories filled.

    Never raises: any failure logs a warning and returns 0.
    """
    missing = [
        cat
        for cat in preference_store.note_categories.values()
        if not (cat.category_description or "").strip()
    ]
    if not missing:
        return 0

    if llm is None:
        try:
            from workflow.llm import get_llm_no_thinking

            llm = get_llm_no_thinking()
        except Exception:
            logger.warning(
                "Cannot build no-thinking LLM — skipping description backfill"
            )
            return 0

    sections: list[str] = []
    for cat in missing:
        notes_text = (
            "\n".join(f"  - {n.content}" for n in cat.notes[:10])
            if cat.notes
            else "  (no notes yet)"
        )
        sections.append(f"Category: {cat.category_name}\nNotes:\n{notes_text}")

    try:
        structured_llm = llm.with_structured_output(
            CategoryDescriptionResult, method="function_calling"
        )
        result: CategoryDescriptionResult = timed_invoke(
            "CategoryDescriptionResult",
            structured_llm.invoke,
            [
                SystemMessage(content=CATEGORY_DESCRIPTION_SYSTEM_PROMPT),
                HumanMessage(content="\n\n".join(sections)),
            ],
        )
    except Exception:
        logger.warning(
            "Category-description backfill LLM call failed", exc_info=True
        )
        return 0

    filled = 0
    for name, description in (getattr(result, "descriptions", None) or {}).items():
        category = preference_store.note_categories.get(str(name).strip())
        if category is None:
            continue
        description = str(description).strip()
        if not description:
            continue
        category.category_description = description
        filled += 1

    if filled:
        logger.info("Backfilled descriptions for %d note categories", filled)
    return filled


# ─────────────────────────────────────────────────────────────────────
# Merged extraction + admission gate (note/expectation v2)
# ─────────────────────────────────────────────────────────────────────


def run_preference_extraction(
    llm: BaseChatModel | None,
    user_message: str,
) -> MergedExtractionResult | None:
    """
    ONE structured LLM call that extracts BOTH note and expectation candidates.

    Notes are returned flat and uncategorized; expectations carry their
    ``trigger``/``action`` routing hints. Each candidate targets exactly one
    bucket. Returns ``None`` if the LLM call fails.
    """
    if llm is None:
        from workflow.llm import get_llm_no_thinking

        llm = get_llm_no_thinking()
    structured_llm = llm.with_structured_output(
        MergedExtractionResult, method="function_calling"
    )
    try:
        result: MergedExtractionResult = timed_invoke(
            "MergedExtractionResult",
            structured_llm.invoke,
            [
                SystemMessage(content=PREFERENCE_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=f"=== User Message ===\n{user_message}"),
            ],
        )
    except Exception:
        logger.exception("Merged preference extraction LLM call failed")
        return None

    logger.debug(
        "Preference extraction: %d note(s), %d expectation(s)",
        len(getattr(result, "notes", None) or []),
        len(getattr(result, "expectations", None) or []),
    )
    return result


def run_admission_gate(
    llm: BaseChatModel | None,
    candidate_notes: list[str],
    candidate_expectations: list,
    categories_text: str,
    neighbor_context: str = "",
) -> AdmissionResult | None:
    """
    ONE structured LLM call: strict validity gate + category assignment.

    Judges the candidate notes (admit/reject + category) and the candidate
    expectations (accept/reject). Notes are referenced by index; accepted
    expectations are referenced by index so their text is never rewritten.
    Returns ``None`` if the LLM call fails.
    """
    if llm is None:
        from workflow.llm import get_llm_no_thinking

        llm = get_llm_no_thinking()

    note_lines = [f"{i}. {n}" for i, n in enumerate(candidate_notes)] or ["(none)"]
    expectation_lines = [
        (
            f"{i}. {getattr(e, 'expectation_content', '')}"
            f" | trigger={getattr(e, 'trigger', '')}"
            f" | action={getattr(e, 'action', '')}"
        )
        for i, e in enumerate(candidate_expectations)
    ] or ["(none)"]

    parts = [
        "=== Candidate Notes ===",
        "\n".join(note_lines),
        "",
        "=== Candidate Expectations ===",
        "\n".join(expectation_lines),
        "",
        "=== Note Category Map ===",
        categories_text or "(none)",
    ]
    if neighbor_context:
        parts += [
            "",
            "=== Nearest Existing Notes (context only) ===",
            neighbor_context,
        ]

    structured_llm = llm.with_structured_output(
        AdmissionResult, method="function_calling"
    )
    try:
        result: AdmissionResult = timed_invoke(
            "AdmissionResult",
            structured_llm.invoke,
            [
                SystemMessage(content=ADMISSION_GATE_SYSTEM_PROMPT),
                HumanMessage(content="\n".join(parts)),
            ],
        )
    except Exception:
        logger.exception("Admission gate LLM call failed")
        return None
    return result


# ─────────────────────────────────────────────────────────────────────
# Common helpers (injection + maintenance)
# ─────────────────────────────────────────────────────────────────────


def _append_note_to_category(
    preference_store: PreferenceStore,
    category_name: str,
    note_content: str,
    note_id: str,
) -> bool:
    """Append a persisted note to an existing category. Returns True on success."""
    category = preference_store.note_categories.get(category_name)
    if category is None:
        logger.error(
            "Cannot append note %s — category '%s' does not exist",
            note_id,
            category_name,
        )
        return False
    category.add_note(note_content, note_id)
    logger.debug(
        "Appended note %s to existing category '%s'", note_id, category_name
    )
    return True


def _create_category_with_note(
    preference_store: PreferenceStore,
    category_name: str,
    note_content: str,
    note_id: str,
    category_description: str = "",
) -> bool:
    """Create a new category and append a note to it. Returns True on success."""
    if category_name in preference_store.note_categories:
        logger.error(
            "Cannot create category '%s' — it already exists", category_name
        )
        return False
    category = NoteCategory(
        category_name=category_name,
        category_description=category_description,
    )
    category.add_note(note_content, note_id)
    preference_store.note_categories[category_name] = category
    logger.debug(
        "Created new category '%s' and appended note %s",
        category_name,
        note_id,
    )
    return True


def _vector_note_identity(doc) -> tuple[str | None, str | None]:
    """Extract ``(note_id, category_name)`` from a vector-query Document.

    ``note_id`` is the sync key between the vector store and the
    PreferenceStore.  ``category_name`` records which category the note was
    stored under — used to locate it inside the PreferenceStore.
    """
    metadata = getattr(doc, "metadata", None) or {}
    note_id = metadata.get("note_id")
    if not note_id:
        note_id = getattr(doc, "id", None)
    category_name = metadata.get("category_name")
    return note_id, category_name


def _note_exists_in_category(
    preference_store: PreferenceStore,
    category_name: str | None,
    note_id: str,
) -> bool:
    """Return True if ``note_id`` exists inside its recorded category only."""
    category = (
        preference_store.note_categories.get(category_name)
        if category_name
        else None
    )
    if category is None:
        return False
    return any(note.id == note_id for note in category.notes)


def _delete_orphan_vector_note(store: PreferenceVectorStore, note_id: str) -> None:
    """Delete an orphaned vector note (best-effort)."""
    try:
        store.collection.delete(ids=[note_id])
        logger.info("[ORPHAN] Deleted orphan vector note %s", note_id)
    except Exception:
        logger.exception("[ORPHAN] Failed to delete orphan vector note %s", note_id)


def _resolve_category_name(
    preference_store: PreferenceStore,
    raw_name: str,
) -> str:
    """Match a gate-supplied category name to an existing one.

    Comparison is case/whitespace-insensitive so the gate can't accidentally
    create a near-duplicate category (e.g. "Food Preference" vs "food
    preference"). Returns the existing category's stored spelling when one
    matches, otherwise the trimmed raw name (i.e. a genuinely new category).
    """
    name = str(raw_name).strip()
    if not name:
        return name
    for existing in preference_store.note_categories:
        if existing.strip().lower() == name.lower():
            return existing
    return name


def _format_neighbor_context(
    position: int,
    note: str,
    results: list | None,
) -> str:
    """Render the nearest existing notes for one candidate (gate context only)."""
    lines = [f'Note {position}: "{note}"']
    if not results:
        lines.append("    (no similar existing notes)")
        return "\n".join(lines)
    for doc, score in results:
        meta = getattr(doc, "metadata", None) or {}
        category = meta.get("category_name", "?")
        content = meta.get("note_content") or getattr(doc, "page_content", "")
        lines.append(f"    - [category={category}, score={score:.3f}] {content}")
    return "\n".join(lines)


def prefilter_note_candidates(
    notes: list[str],
    store: PreferenceVectorStore,
    preference_store: PreferenceStore,
) -> tuple[list[int], str]:
    """
    Drop duplicate/orphaned note candidates BEFORE the admission gate.

    For each candidate, query the note vector store:
      - best match within ``PREFERENCE_NOTE_SIMILARITY_THRESHOLD`` AND present
        in the PreferenceStore → duplicate → hard-dropped (no update/merge);
      - a matched vector note missing from its recorded category → orphan,
        deleted, and the candidate is kept;
      - otherwise the candidate is kept and its nearest existing notes are
        collected as context for the gate.
    Within-batch duplicates are also dropped.

    Returns
    -------
    (kept_indices, neighbor_context)
        ``kept_indices`` index into ``notes`` (order preserved).
        ``neighbor_context`` is gate-ready text numbered 1..N to match the
        kept-note order the caller hands to the gate.
    """
    kept_indices: list[int] = []
    neighbor_blocks: list[str] = []
    seen: set[str] = set()

    for idx, raw in enumerate(notes):
        note = str(raw).strip()
        if not note:
            continue
        key = note.lower()
        if key in seen:
            logger.info("[OK] Skipping duplicate note within batch: %r", note)
            continue
        seen.add(key)

        results = store.query_similar_note(note)
        if results:
            best_doc, best_score = results[0]
            if best_score <= PREFERENCE_NOTE_SIMILARITY_THRESHOLD:
                matched_id, matched_category = _vector_note_identity(best_doc)
                if matched_id and _note_exists_in_category(
                    preference_store, matched_category, matched_id
                ):
                    logger.info(
                        "[OK] Skipping duplicate note (score=%.3f <= %.3f): %r",
                        best_score,
                        PREFERENCE_NOTE_SIMILARITY_THRESHOLD,
                        note,
                    )
                    continue
                if matched_id:
                    _delete_orphan_vector_note(store, matched_id)
                logger.warning(
                    "[ORPHAN] Vector note %s (category=%r, score=%.3f) has no "
                    "matching PreferenceStore note — deleting orphan and "
                    "keeping the candidate",
                    matched_id,
                    matched_category,
                    best_score,
                )

        kept_indices.append(idx)
        neighbor_blocks.append(
            _format_neighbor_context(len(kept_indices), note, results)
        )

    return kept_indices, "\n".join(neighbor_blocks)


# ─────────────────────────────────────────────────────────────────────
# Maintenance (prune + merge notes inside a single category)
# ─────────────────────────────────────────────────────────────────────


def run_note_maintenance(
    llm: BaseChatModel,
    category: NoteCategory,
    store: PreferenceVectorStore,
) -> dict[str, int] | None:
    """
    Review and prune/merge a category's notes once it grows large enough.

    Trigger: ``len(category.notes) >= NOTE_MAINTENANCE_THRESHOLD``. Below that,
    the function returns ``None`` without calling the LLM.

    When triggered:
      1. Send the category's notes (0-based indexed, saving order) to the LLM.
      2. Apply ``deleted_notes`` / ``merged_notes`` indices: remove all flagged
         notes from BOTH the ``PreferenceStore`` category and the vector store.
      3. Append every ``merged_content`` entry as a new note to BOTH stores.

    Returns a summary dict like ``{"deleted": n, "merged": m, "appended": k}``,
    ``{}`` if the LLM flagged nothing, or ``None`` if maintenance was skipped or
    the LLM call failed.
    """
    note_count = len(category.notes)
    if note_count < NOTE_MAINTENANCE_THRESHOLD:
        logger.debug(
            "Maintenance skipped for category '%s' (%d < %d) — under threshold",
            category.category_name,
            note_count,
            NOTE_MAINTENANCE_THRESHOLD,
        )
        return None

    indexed_lines = [
        f"{i}. {note.content}" for i, note in enumerate(category.notes)
    ]
    if llm is None:
        from workflow.llm import get_llm
        llm = get_llm()
    structured_llm = llm.with_structured_output(NoteMaintenanceResult, method="json_mode")
    try:
        result: NoteMaintenanceResult = timed_invoke(
            "NoteMaintenanceResult",
            structured_llm.invoke,
            [
                SystemMessage(content=NOTE_MAINTENANCE_SYSTEM_PROMPT),
                HumanMessage(content="\n".join(indexed_lines)),
            ],
        )
        deleted_notes = list(result.deleted_notes)
        merged_notes = list(result.merged_notes)
        merged_content = list(result.merged_content)
    except Exception:
        logger.exception(
            "Note maintenance LLM call failed for category '%s'",
            category.category_name,
        )
        return None

    # ── Validate indices ───────────────────────────────────────────
    valid_range = range(note_count)
    seen: set[int] = set()
    removed_indices: list[int] = []

    for idx in deleted_notes + merged_notes:
        if idx not in valid_range:
            logger.warning(
                "Maintenance('%s'): dropping out-of-range index %d (max=%d)",
                category.category_name,
                idx,
                note_count - 1,
            )
            continue
        if idx in seen:
            logger.warning(
                "Maintenance('%s'): dropping duplicate index %d",
                category.category_name,
                idx,
            )
            continue
        seen.add(idx)
        removed_indices.append(idx)

    removed_ids = [category.notes[i].id for i in removed_indices]

    # ── Remove flagged notes from the PreferenceStore ──────────────
    for idx in sorted(removed_indices, reverse=True):
        category.remove_note(idx)

    # ── Remove flagged notes from the vector store ─────────────────
    if removed_ids:
        try:
            store.collection.delete(ids=removed_ids)
        except Exception:
            logger.exception(
                "Failed to delete %d note(s) from vector store while "
                "maintaining category '%s'",
                len(removed_ids),
                category.category_name,
            )

    # ── Append merged content to both stores ───────────────────────
    appended = 0
    for text in merged_content:
        text = text.strip()
        if not text:
            logger.warning(
                "Maintenance('%s'): skipping empty merged_content entry",
                category.category_name,
            )
            continue
        new_note_id = store.upsert_note(text, category.category_name)
        category.add_note(text, new_note_id)
        appended += 1

    summary = {
        "deleted": len(removed_indices),
        "merged": len(merged_notes),
        "appended": appended,
    }
    logger.info(
        "Maintenance for category '%s' complete: %s",
        category.category_name,
        summary,
    )
    return summary


# ─────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────


def persist_categorized_notes(
    llm: BaseChatModel,
    categorized_notes: list[tuple[str, str, str]],
    store: PreferenceVectorStore,
    preference_store: PreferenceStore,
) -> list[tuple[str, str, str]]:
    """
    Write already-adjudicated notes to both stores (the persistence half).

    Each entry is ``(note_content, category_name, category_description)``.
    The PreferenceStore is written FIRST, then the vector store with the SAME
    ``note_id``; a vector-write failure rolls the PreferenceStore back. Category
    names are matched case/whitespace-insensitively against existing categories
    before create-vs-append is decided. Maintenance is triggered once per
    category that actually gained notes.

    Callers are responsible for de-duplication before calling this (see
    :func:`prefilter_note_candidates`).

    Returns ``[(note_id, note_content, category_name), ...]`` for stored notes.
    """
    if preference_store is None:
        raise ValueError(
            "persist_categorized_notes requires a PreferenceStore."
        )

    stored: list[tuple[str, str, str]] = []

    for note_content, raw_category, category_description in categorized_notes:
        note_content = str(note_content).strip()
        category_name = _resolve_category_name(preference_store, raw_category)
        if not note_content or not category_name:
            continue

        note_id = uuid.uuid4().hex
        category_was_new = category_name not in preference_store.note_categories
        if category_was_new:
            ok = _create_category_with_note(
                preference_store,
                category_name,
                note_content,
                note_id,
                category_description=str(category_description or "").strip(),
            )
        else:
            ok = _append_note_to_category(
                preference_store, category_name, note_content, note_id
            )

        if not ok:
            logger.error(
                "[ERROR] Failed to add note to PreferenceStore under "
                "category '%s' — note NOT stored",
                category_name,
            )
            continue

        category = preference_store.note_categories.get(category_name)

        # ── Persist to vector store with the SAME note_id ───────
        try:
            store.upsert_note(note_content, category_name, note_id=note_id)
        except Exception:
            logger.exception(
                "[ERROR] Vector-store write failed for note %s — rolling "
                "back PreferenceStore",
                note_id,
            )
            if category is not None:
                for idx, note in enumerate(category.notes):
                    if note.id == note_id:
                        category.remove_note(idx)
                        break
                if category_was_new and not category.notes:
                    preference_store.note_categories.pop(category_name, None)
            continue

        logger.info(
            "[OK] Stored note %r -> category '%s' (id=%s, category_was_new=%s)",
            note_content,
            category_name,
            note_id,
            category_was_new,
        )
        stored.append((note_id, note_content, category_name))

    # ── Maintenance (only for categories that gained notes) ─────
    touched_categories = {cat_name for _, _, cat_name in stored}
    for category_name in touched_categories:
        category = preference_store.note_categories.get(category_name)
        if category is not None:
            run_note_maintenance(llm, category, store)

    return stored
