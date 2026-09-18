"""
Data models for the preference system.

Defines the dataclasses and Pydantic structured-output models that govern
how the preference system stores, serializes, and retrieves user state and
note-category information.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger("preference_system.models")


# ═══════════════════════════════════════════════════════════════════
# Data store dataclasses
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PreferenceState:
    """A named "state" the user can be in, with a description and event history."""

    state_name: str
    description: str
    event_list: list[str] = field(default_factory=list)
    active_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_event(self, event: str) -> None:
        """Append an event string to this state's event list."""
        self.event_list.append(event)


@dataclass
class Note:
    """a note"""
    content: str
    id: str
    saved_time: str

@dataclass
class NoteCategory:
    """A named category that holds a list of Note entries."""

    category_name: str
    notes: list[Note] = field(default_factory=list)
    # Very short (<15 words) description of what the category contains.
    category_description: str = ""

    def add_note(self, content: str, id: str) -> None:
        """Create and append a Note to this category."""
        self.notes.append(Note(content, id, datetime.now(timezone.utc).isoformat()))

    def remove_note(self, index: int) -> bool:
        """Remove a note by 0-based index. Returns True on success."""
        if 0 <= index < len(self.notes):
            del self.notes[index]
            return True
        return False


def load_expectations(path: str | Path) -> list[dict]:
    """Load the expectations list from a JSON file.

    The file is expected to look like::

        {"expectations": [
            {"trigger": "...", "action": "...", "expectation_content": "..."},
            ...
        ]}

    Returns an empty list if the path is empty, the file is missing, the JSON is
    malformed, or ``expectations`` is not a list of dicts. Never raises. A
    malformed file is quarantined (``<path>.corrupt-<ts>``) so data is not lost.
    """
    if not path:
        return []
    from json_io import load_or_quarantine

    data, backup = load_or_quarantine(
        path, default={}, validate=lambda d: isinstance(d, dict)
    )
    if backup:
        logger.error(
            "Expectations file %s was corrupt — quarantined to %s", path, backup
        )
        return []
    if not isinstance(data, dict):
        return []
    expectations = data.get("expectations", [])
    if not isinstance(expectations, list):
        logger.warning("Expectations file %s: 'expectations' is not a list", path)
        return []
    return [e for e in expectations if isinstance(e, dict)]


@dataclass
class PreferenceStore:
    """Top-level container holding all states and note categories."""

    states: dict[str, PreferenceState] = field(default_factory=dict)
    note_categories: dict[str, NoteCategory] = field(default_factory=dict)
    expectations_file: str = "expectations.json"

    # ── State operations ────────────────────────────────────────────

    def add_state(self, name: str, description: str) -> None:
        """Add or overwrite a state."""
        if name in self.states:
            logger.warning("State '%s' already exists — overwriting.", name)
        self.states[name] = PreferenceState(state_name=name, description=description)

    def update_state(self, name: str, new_description: str) -> PreferenceState | None:
        """Update a state's description. Returns None if not found."""
        if name not in self.states:
            logger.warning("State '%s' not found — cannot update.", name)
            return None
        self.states[name].description = new_description
        return self.states[name]

    def remove_state(self, name: str) -> bool:
        """Remove a state by name. Returns True on success."""
        if name not in self.states:
            return False
        del self.states[name]
        return True

    def add_event(self, state_name: str, event: str) -> bool:
        """Add an event to a state. Returns True on success."""
        if state_name not in self.states:
            return False
        self.states[state_name].add_event(event)
        return True

    # ── Note-category operations ────────────────────────────────────

    def add_category(self, name: str, description: str = "") -> None:
        """Add or overwrite a note category."""
        if name in self.note_categories:
            logger.warning("Note category '%s' already exists — overwriting.", name)
        self.note_categories[name] = NoteCategory(
            category_name=name, category_description=description
        )

    def remove_category(self, name: str) -> bool:
        """Remove a note category by name. Returns True on success."""
        if name not in self.note_categories:
            return False
        del self.note_categories[name]
        return True

    def add_note(self, category_name: str, content: str, id: str = "") -> bool:
        """Add a note to a category. Returns True on success."""
        if category_name not in self.note_categories:
            return False
        note_id = id or uuid.uuid4().hex
        self.note_categories[category_name].add_note(content, note_id)
        return True

    def remove_note(self, category_name: str, index: int) -> bool:
        """Remove a note by 0-based index from a category. Returns True on success."""
        if category_name not in self.note_categories:
            return False
        return self.note_categories[category_name].remove_note(index)

    # ── Map / serialisation ─────────────────────────────────────────

    def get_preference_map(self) -> str:
        """Produce a human-readable summary of every state, note category, and
        expectation (behavior rule).

        This string gets split into per-section maps and fed to the LLM's
        map retrievers (state / expectation / note category) in the
        retrieve branch so the model knows which names/indices are available
        to select.
        """
        return (
            self.get_state()
            + "\n"
            + self.get_categories()
            + "\n"
            + self.get_expectations()
        )

    def get_state(self) -> str:
        lines: list[str] = ["=== State Names ==="]
        if not self.states:
            lines.append("  (none)")
        else:
            for name, state in self.states.items():
                lines.append(f"  {name}: {state.description}")
        return "\n".join(lines)

    def get_categories(self) -> str:
        lines: list[str] = ["=== Note Category Names ==="]
        if not self.note_categories:
            lines.append("  (none)")
        else:
            for name, category in self.note_categories.items():
                desc = (category.category_description or "").strip()
                lines.append(f"  {name}: {desc if desc else '(no description)'}")

        return "\n".join(lines)

    def get_expectations(self) -> str:
        """Return the indexed expectation/behavior-rule section of the map.

        The strings are read fresh from ``self.expectations_file`` on every call
        so the user can edit the JSON file directly between CLI rounds.
        """
        lines: list[str] = ["=== Expectations ==="]
        expectations = load_expectations(self.expectations_file)
        if not expectations:
            lines.append("  (none)")
            return "\n".join(lines)

        for i, exp in enumerate(expectations):
            trigger = str(exp.get("trigger", "")).strip()
            action = str(exp.get("action", "")).strip()
            lines.append(f"  {i}. {trigger} | {action}")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Persist the entire store to a JSON file (atomic write)."""
        from json_io import write_json_atomic

        write_json_atomic(path, asdict(self))

    @staticmethod
    def load(path: str | Path) -> PreferenceStore:
        """Load a store from a JSON file.

        Returns an empty store if the file does not exist. If the file exists
        but is corrupt / the wrong shape, it is **quarantined** (moved aside to
        ``<path>.corrupt-<timestamp>``) so the data is never silently lost, an
        error is logged loudly, and an empty store is returned so the pipeline
        keeps running.
        """
        from json_io import load_or_quarantine

        path = Path(path)
        if not path.exists():
            return PreferenceStore()

        data, backup = load_or_quarantine(
            path, default={}, validate=lambda d: isinstance(d, dict)
        )
        if backup:
            msg = (
                f"  [ERROR] preferences file {path} was corrupt — quarantined to "
                f"{backup}; starting from an empty store."
            )
            print(msg)
            logger.error(msg)
        if not isinstance(data, dict):
            data = {}

        store = PreferenceStore()
        store.expectations_file = data.get(
            "expectations_file", store.expectations_file
        )

        states = data.get("states", {})
        if isinstance(states, dict):
            for fallback_name, state_data in states.items():
                if not isinstance(state_data, dict):
                    continue
                state = PreferenceState(
                    state_name=state_data.get("state_name", fallback_name),
                    description=state_data.get("description", ""),
                    event_list=list(state_data.get("event_list", []) or []),
                    active_time=state_data.get("active_time", ""),
                )
                store.states[state.state_name] = state

        categories = data.get("note_categories", {})
        if isinstance(categories, dict):
            for fallback_name, cat_data in categories.items():
                if not isinstance(cat_data, dict):
                    continue
                notes: list[Note] = []
                for n in (cat_data.get("notes", []) or []):
                    if isinstance(n, str):
                        # Legacy format: a saved note was just the content string.
                        notes.append(Note(content=n, id="", saved_time=""))
                    elif isinstance(n, dict):
                        notes.append(
                            Note(
                                content=n.get("content", ""),
                                id=n.get("id", ""),
                                saved_time=n.get("saved_time", ""),
                            )
                        )
                cat = NoteCategory(
                    category_name=cat_data.get("category_name", fallback_name),
                    notes=notes,
                    category_description=cat_data.get("category_description", ""),
                )
                store.note_categories[cat.category_name] = cat

        return store


# ═══════════════════════════════════════════════════════════════════
# Pydantic structured-output models (used with LLM.with_structured_output)
# ═══════════════════════════════════════════════════════════════════


class StateMapDecision(BaseModel):
    """Relevant state names (ongoing user situations). One field per map so the
    LLM never has to decide between disjoint schemas."""

    state_names: list[str]


class NoteCategoryMapDecision(BaseModel):
    """Relevant note-category names. Only the EXACT category name is expected —
    never the description."""

    category_names: list[str]


class ExpectationMapDecision(BaseModel):
    """Relevant expectation indices (0-based into the expectations file)."""

    expectation_indices: list[int]


class StateRetrievalResult(BaseModel):
    """Whether to include the state description and/or event history."""

    include_state: bool = False
    include_history: bool = False


class NoteRetrievalResult(BaseModel):
    """Which note indices (0-based) the LLM thinks are relevant."""

    selected_indices: list[int]


class ExpectationExtractionResult(BaseModel):
    """whether the user's message reveals a stable behavioral expectation, and its fields.

    ``expectation_content`` holds the real, complete information. ``trigger``
    (when the rule applies) and ``action`` (brief behavior summary) are short
    reference strings — together they must stay under 15 words. ``trigger`` is
    exactly ``"Always"`` when the rule applies unconditionally. Empty strings
    mean the message carried no recordable expectation.
    """

    expectation_content: str
    trigger: str
    action: str


# ── Merged preference extraction + admission gate (note/expectation v2) ──


class MergedExtractionResult(BaseModel):
    """Output of the merged preference-extraction call (ONE LLM).

    ``notes`` are flat, UNCATEGORIZED candidate notes about the user.
    ``expectations`` are candidate behavioral rules (reusing the standalone
    expectation shape). Each candidate belongs to exactly one bucket.
    """

    notes: list[str]
    expectations: list[ExpectationExtractionResult]


class AdmissionResult(BaseModel):
    """Output of the combined admission gate (notes + expectations).

    ``notes`` maps each ACCEPTED note's 0-based index (into the candidate-notes
    list the gate received) to the category it was assigned — either the EXACT
    name of an existing category or a new broad name. ``created_category``
    supplies a short description (<15 words) for every NEW category used in
    ``notes``. Accepted expectations are referenced by 0-based index too.
    Rejected candidates are dropped silently — they are not listed here.
    """

    accepted_expectation_indices: list[int]
    notes: dict[int, str]
    created_category: dict[str, str] = {}


class StateClassificationResult(BaseModel):
    """whether a summary belongs to an existing state, or should start a new one.

    ``state_name`` is the matched existing state name, or a short proposed name
    for a brand-new state. Empty ``state_name`` + ``new_state=False`` means the
    summary is not state-worthy (nothing to record).
    """

    state_name: str
    new_state: bool = False


class StateUpdateResult(BaseModel):
    """Updates to apply to a state given a new summary.

    ``description`` and/or ``event`` may be empty when no update is needed.
    For a newly created state both are expected (initial description + the
    first milestone event).
    """

    description: str
    event: str


