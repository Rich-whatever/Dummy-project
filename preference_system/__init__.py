"""
Preference System — user preferences, states, and expectation memory.

Owns the ``PreferenceStore`` (states + note categories), the retrieve branch
(LLM selection of relevant states / notes / expectations), the merged injection
pipeline, and state injection. Also runnable via ``cli.py`` for debugging.
"""

from .expectation_injection import (
    append_expectation,
    persist_expectation_candidates,
)
from .models import (
    ExpectationExtractionResult,
    ExpectationMapDecision,
    Note,
    NoteCategory,
    NoteCategoryMapDecision,
    NoteRetrievalResult,
    PreferenceState,
    PreferenceStore,
    StateMapDecision,
    StateRetrievalResult,
)
from .note_injection import (
    persist_categorized_notes,
    prefilter_note_candidates,
    run_admission_gate,
    run_preference_extraction,
)
from .preference_injection import (
    PreferenceInjectionResult,
    run_preference_injection,
)
from .retrieve_branch import run_retrieve_branch

# Public surface of the package. Declared explicitly so the re-exports above are
# recognised as intentional (not unused imports) by linters and tooling.
__all__ = [
    # models
    "ExpectationExtractionResult",
    "ExpectationMapDecision",
    "Note",
    "NoteCategory",
    "NoteCategoryMapDecision",
    "NoteRetrievalResult",
    "PreferenceState",
    "PreferenceStore",
    "StateMapDecision",
    "StateRetrievalResult",
    # retrieve branch
    "run_retrieve_branch",
    # expectation injection
    "append_expectation",
    "persist_expectation_candidates",
    # merged preference pipeline (extraction → gate → persist)
    "persist_categorized_notes",
    "prefilter_note_candidates",
    "run_admission_gate",
    "run_preference_extraction",
    # merged preference-injection orchestrator
    "PreferenceInjectionResult",
    "run_preference_injection",
]
