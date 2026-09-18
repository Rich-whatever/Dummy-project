"""
Expectation persistence — validate and store extracted behavioral expectations.

An *expectation* is a rule the user expects the agent to follow in future
conversations (e.g. "when spending, supervise spending"). It is distinct from
a preference note or a state: it captures expected agent behavior, not a fact
about the user.

Extraction itself lives in the merged pipeline
(``note_injection.run_preference_extraction``); this module owns the
deterministic guard and the writes:

    persist_expectation_candidates(candidates, expectations_file)
        |-- empty content/trigger/action --> skipped
        |-- trigger == "Always"           --> append_behavior_guide(...)
        `-- otherwise                     --> append_expectation(...)
                                              |-- duplicate --> no-op
                                              `-- appended  --> stored
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import ExpectationExtractionResult

logger = logging.getLogger("preference_system.expectation_injection")

DEFAULT_EXPECTATIONS_FILE = "expectations.json"

# Expectation store capacity. When the total number of expectations in the file
# reaches this, the user is warned that they should manage the file directly.
EXPECTATION_LIMIT = 20

# General behavior guide (Always-trigger expectations only).
# Always-trigger expectations are NOT stored in expectations.json; instead only
# their expectation_content is appended to this separate guide file, which is
# later sent to the main agent for response.
DEFAULT_BEHAVIOR_GUIDE_FILE = "agent_general_behavior_guide.json"

# If the guide's cumulative content length exceeds this, warn the user that the
# guide is growing long/complex and they should review the file directly.
BEHAVIOR_GUIDE_LENGTH_LIMIT = 3000

# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------


def append_expectation(
    expectations_file: str | Path,
    expectation_content: str,
    trigger: str,
    action: str,
) -> bool:
    """
    Append a validated expectation to the expectations JSON file.

    The file is re-read fresh on every call to avoid clobbering user edits.
    Exact duplicates (by ``expectation_content``) are treated as a no-op.

    Returns ``True`` if the expectation is present in the file afterwards
    (whether newly appended or already existing), ``False`` on write failure.
    """
    path = Path(expectations_file)

    # Preserve or default to the wrapper structure. A corrupt file is
    # quarantined (not silently discarded) before we start fresh.
    from json_io import load_or_quarantine, write_json_atomic

    data, backup = load_or_quarantine(
        path, default={"expectations": []}, validate=lambda d: isinstance(d, dict)
    )
    if backup:
        logger.error(
            "Expectations file %s was corrupt — quarantined to %s", path, backup
        )
    if not isinstance(data, dict):
        data = {"expectations": []}

    expectations = data.get("expectations")
    if not isinstance(expectations, list):
        expectations = []
    expectations.append(
        {
            "trigger": trigger,
            "action": action,
            "expectation_content": expectation_content,
        }
    )
    data["expectations"] = expectations

    try:
        write_json_atomic(path, data)
    except Exception:
        logger.exception("Failed to write expectation to %s", path)
        return False

    count = len(expectations)
    if count >= EXPECTATION_LIMIT:
        logger.warning(
            "Expectation store reached %d entries (limit %d) - please manage "
            "%s directly to stay under the limit.",
            count,
            EXPECTATION_LIMIT,
            path,
        )

    logger.info(
        "Appended expectation to %s: content=%r (trigger=%r, action=%r)",
        path,
        expectation_content,
        trigger,
        action,
    )
    return True


def append_behavior_guide(
    guide_content: str,
    guide_file: str | Path = DEFAULT_BEHAVIOR_GUIDE_FILE,
) -> bool:
    """
    Append an Always-trigger expectation's content to the general behavior guide.

    Unlike ``append_expectation``, this stores ONLY the ``guide_content`` string
    (no trigger/action). The file uses the structure::

        {"length": <int>, "guides": [<str>, ...]}

    where ``length`` is the running total character count of all guide entries,
    updated by ``+ len(guide_content)`` after each append.

    After writing, if ``length`` exceeds ``BEHAVIOR_GUIDE_LENGTH_LIMIT`` a
    warning is printed telling the user the guide is growing long/complex and
    should be reviewed directly.

    Returns ``True`` on success, ``False`` on write failure.
    """
    path = Path(guide_file)

    # Preserve or default to the wrapper structure. A corrupt file is
    # quarantined (not silently discarded) before we start fresh.
    from json_io import load_or_quarantine, write_json_atomic

    data, backup = load_or_quarantine(
        path, default={"length": 0, "guides": []}, validate=lambda d: isinstance(d, dict)
    )
    if backup:
        logger.error(
            "Behavior guide %s was corrupt — quarantined to %s", path, backup
        )
    if not isinstance(data, dict):
        data = {"length": 0, "guides": []}

    guides = data.get("guides")
    if not isinstance(guides, list):
        guides = []

    guides.append(guide_content)
    data["guides"] = guides
    data["length"] = int(data.get("length", 0) or 0) + len(guide_content)

    try:
        write_json_atomic(path, data)
    except Exception:
        logger.exception("Failed to write behavior guide to %s", path)
        return False

    logger.info(
        "Appended Always expectation to behavior guide %s (length=%d): %r",
        path,
        data["length"],
        guide_content,
    )

    if data["length"] > BEHAVIOR_GUIDE_LENGTH_LIMIT:
        print(
            f"  [WARN] agent_general_behavior_guide is growing long and complex "
            f"(length={data['length']} > {BEHAVIOR_GUIDE_LENGTH_LIMIT}). "
            f"Please review/trim {path} directly."
        )
        logger.warning(
            "Behavior guide length %d exceeds limit %d - user should review %s",
            data["length"],
            BEHAVIOR_GUIDE_LENGTH_LIMIT,
            path,
        )

    return True


# ---------------------------------------------------------------------
# Candidate persistence
# ---------------------------------------------------------------------


def persist_expectation_candidates(
    candidates: list[ExpectationExtractionResult],
    expectations_file: str | Path = DEFAULT_EXPECTATIONS_FILE,
) -> list[str]:
    """
    Validate and persist ALREADY-EXTRACTED expectation candidates.

    Applies the deterministic guard (non-empty content / trigger / action) and
    the ``Always`` routing, then writes via :func:`append_behavior_guide`
    (Always) or :func:`append_expectation` (everything else). Exact duplicates
    are a no-op in ``append_expectation`` and count as persisted.

    Returns the list of ``expectation_content`` strings that were persisted.
    """
    stored: list[str] = []
    for candidate in candidates or []:
        content = str(getattr(candidate, "expectation_content", "") or "").strip()
        trigger = str(getattr(candidate, "trigger", "") or "").strip()
        action = str(getattr(candidate, "action", "") or "").strip()

        if not content:
            logger.info("[OK] Empty expectation candidate skipped")
            continue
        if not trigger or not action:
            logger.info(
                "Rejecting extracted expectation with empty trigger/action - "
                "not recorded"
            )
            continue

        if trigger.lower() == "always":
            ok = append_behavior_guide(content)
        else:
            ok = append_expectation(expectations_file, content, trigger, action)

        if not ok:
            logger.error("Failed to persist expectation - not recorded")
            continue
        stored.append(content)
    return stored
