"""
Preference Injection v2 — merged extraction + admission gate.

Replaces the two independent extraction calls (notes, expectations) with a
single pipeline:

    1. ONE merged extraction call producing flat note candidates plus
       expectation candidates, with mutual-exclusion routing (an item is
       either a note or an expectation, never both).
    2. A vector prefilter for notes: exact/near duplicates and orphans are
       handled BEFORE the gate (hard drop — no update/merge).
    3. ONE admission-gate call that validates the notes AND the expectations
       and assigns each admitted note a category (reusing exact existing
       names; creating new ones with a short description).
    4. Deterministic persistence: notes via ``persist_categorized_notes``,
       expectations via ``persist_expectation_candidates`` (validate +
       Always→behavior-guide routing + write).

Rejected candidates are NOT persisted anywhere (not the vector store, not a
category): the gate simply omits them and they are dropped silently. Their
counts are derived (candidates − accepted) for telemetry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from .expectation_injection import (
    DEFAULT_EXPECTATIONS_FILE,
    persist_expectation_candidates,
)
from .models import PreferenceStore
from .note_injection import (
    persist_categorized_notes,
    prefilter_note_candidates,
    run_admission_gate,
    run_preference_extraction,
)
from .vector_store import PreferenceVectorStore

logger = logging.getLogger("preference_system.preference_injection")


@dataclass
class PreferenceInjectionResult:
    """Per-message telemetry for the merged preference-injection pipeline."""

    notes_extracted: int = 0
    notes_duplicate: int = 0
    notes_stored: int = 0
    notes_rejected: int = 0
    expectations_extracted: int = 0
    expectations_stored: int = 0
    expectations_rejected: int = 0
    # What actually got injected (for logging): (category, note_content) / content.
    stored_notes: list[tuple[str, str]] = field(default_factory=list)
    stored_expectations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"notes: extracted={self.notes_extracted} "
            f"duplicate={self.notes_duplicate} stored={self.notes_stored} "
            f"rejected={self.notes_rejected} | "
            f"expectations: extracted={self.expectations_extracted} "
            f"stored={self.expectations_stored} "
            f"rejected={self.expectations_rejected}"
        )


def _clean_notes(raw_notes) -> list[str]:
    """Drop empty/whitespace-only note candidates, preserving order."""
    out: list[str] = []
    for raw in raw_notes or []:
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def run_preference_injection(
    llm: BaseChatModel | None,
    user_message: str,
    store: PreferenceVectorStore,
    preference_store: PreferenceStore,
    expectations_file: str | Path = DEFAULT_EXPECTATIONS_FILE,
) -> PreferenceInjectionResult | None:
    """
    Merged preference-injection pipeline for a single user message.

    Returns ``None`` if the extraction or admission-gate LLM call fails (halt,
    nothing persisted); otherwise a :class:`PreferenceInjectionResult` with
    per-stream counts. Never raises (LLM failures are logged and returned).
    """
    if llm is None:
        from workflow.llm import get_llm_no_thinking

        llm = get_llm_no_thinking()

    extraction = run_preference_extraction(llm, user_message)
    if extraction is None:
        return None

    notes = _clean_notes(getattr(extraction, "notes", None))
    expectations = list(getattr(extraction, "expectations", None) or [])

    result = PreferenceInjectionResult(
        notes_extracted=len(notes),
        expectations_extracted=len(expectations),
    )
    if not notes and not expectations:
        logger.info("[OK] Nothing to record from this message")
        return result

    # ── 1. Prefilter notes (hard-drop duplicates/orphans) ───────────
    neighbor_context = ""
    kept_notes: list[str] = []
    if notes:
        kept_indices, neighbor_context = prefilter_note_candidates(
            notes, store, preference_store
        )
        result.notes_duplicate = len(notes) - len(kept_indices)
        kept_notes = [notes[i] for i in kept_indices]

    if not kept_notes and not expectations:
        logger.info("[OK] All note candidates were duplicates — nothing to gate")
        return result

    # ── 2. Admission gate (validity for notes AND expectations) ─────
    admitted = run_admission_gate(
        llm,
        kept_notes,
        expectations,
        preference_store.get_categories(),
        neighbor_context,
    )
    if admitted is None:
        logger.error("ADMISSION GATE failed — nothing persisted")
        return None

    # ── 3. Persist admitted notes ───────────────────────────────────
    # The gate returns {note_index: category_name} plus descriptions for new
    # categories. Resolve each index back to its candidate text and flatten to
    # the shared writer's (note, category, description) tuples.
    created_category = getattr(admitted, "created_category", None) or {}
    categorized: list[tuple[str, str, str]] = []
    for note_index, category_name in (
        getattr(admitted, "notes", None) or {}
    ).items():
        try:
            note_index = int(note_index)
        except (TypeError, ValueError):
            continue
        if not (0 <= note_index < len(kept_notes)):
            logger.warning(
                "Admission gate returned out-of-range note index %r", note_index
            )
            continue
        category_name = str(category_name).strip()
        if not category_name:
            continue
        categorized.append(
            (
                kept_notes[note_index],
                category_name,
                str(created_category.get(category_name, "")).strip(),
            )
        )
    stored_notes = persist_categorized_notes(llm, categorized, store, preference_store)
    result.notes_stored = len(stored_notes)
    result.stored_notes = [(category, note) for _, note, category in stored_notes]

    # ── 4. Persist accepted expectations ────────────────────────────
    accepted_expectations = []
    for idx in getattr(admitted, "accepted_expectation_indices", None) or []:
        if isinstance(idx, int) and 0 <= idx < len(expectations):
            accepted_expectations.append(expectations[idx])
        else:
            logger.warning(
                "Admission gate returned out-of-range expectation index %r", idx
            )
    stored_expectations = persist_expectation_candidates(
        accepted_expectations, expectations_file
    )
    result.expectations_stored = len(stored_expectations)
    result.stored_expectations = list(stored_expectations)

    # ── 5. Telemetry (rejected candidates are dropped silently) ─────
    result.notes_rejected = max(0, len(kept_notes) - len(categorized))
    result.expectations_rejected = max(
        0, len(expectations) - len(accepted_expectations)
    )

    logger.info("Preference injection: %s", result.summary())
    for category, note in result.stored_notes:
        logger.info("  + note [%s]: %s", category, note)
    for expectation in result.stored_expectations:
        logger.info("  + expectation: %s", expectation)
    return result

