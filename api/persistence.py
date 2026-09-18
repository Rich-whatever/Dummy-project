"""File-backed read/write helpers for the settings API.

These operate on the SAME JSON files the agent pipeline reads and writes, so
edits made in the web UI are honoured by the running agent, and notes /
expectations the agent injects automatically appear back in the UI on reload.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import PROJECT_DIR
from json_io import write_json_atomic

PREFERENCES_FILE = PROJECT_DIR / "preferences.json"
EXPECTATIONS_FILE = PROJECT_DIR / "expectations.json"
BEHAVIOR_GUIDE_FILE = PROJECT_DIR / "agent_general_behavior_guide.json"
CHAT_HISTORY_FILE = PROJECT_DIR / "chat_history.json"

# ─────────────────────────────────────────────────────────────────────
# Chat history (web UI conversation panel) — 30-round scrolling window
# ─────────────────────────────────────────────────────────────────────

MAX_CHAT_HISTORY_ROUNDS = 30


def get_chat_history() -> list[dict]:
    """Return the persisted chat history rounds (oldest → newest).

    Empty list when the file is missing or corrupt."""
    data = _read_json(CHAT_HISTORY_FILE, {})
    rounds = data.get("rounds", [])
    return [r for r in rounds if isinstance(r, dict)] if isinstance(rounds, list) else []


def append_chat_history(round_record: dict) -> list[dict]:
    """Append one round to the chat history and trim to the latest
    ``MAX_CHAT_HISTORY_ROUNDS`` rounds (drop oldest first).

    Returns the full trimmed history (oldest → newest)."""
    rounds = get_chat_history()
    rounds.append(round_record)
    rounds = rounds[-MAX_CHAT_HISTORY_ROUNDS:]
    _write_json(CHAT_HISTORY_FILE, {"rounds": rounds})
    return rounds


def _read_json(path: Path, default: dict) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else default
    except Exception:
        return dict(default)


def _write_json(path: Path, data: dict) -> None:
    """Atomic JSON write (temp file + os.replace)."""
    write_json_atomic(path, data)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_preferences() -> dict:
    """Return a list-shaped preferences document for the UI."""
    data = _read_json(PREFERENCES_FILE, {})
    states = data.get("states", {}) if isinstance(data.get("states"), dict) else {}
    categories = (
        data.get("note_categories", {})
        if isinstance(data.get("note_categories"), dict)
        else {}
    )

    state_list = []
    for name, s in states.items():
        if not isinstance(s, dict):
            continue
        state_list.append(
            {
                "state_name": str(s.get("state_name", name)),
                "description": str(s.get("description", "")),
                "event_list": [e for e in s.get("event_list", []) if isinstance(e, str)],
                "active_time": str(s.get("active_time", "")),
            }
        )

    category_list = []
    for name, c in categories.items():
        if not isinstance(c, dict):
            continue
        notes = []
        for n in c.get("notes", []):
            if isinstance(n, str):
                notes.append({"content": n, "id": "", "saved_time": ""})
            elif isinstance(n, dict):
                notes.append(
                    {
                        "content": str(n.get("content", "")),
                        "id": str(n.get("id", "")),
                        "saved_time": str(n.get("saved_time", "")),
                    }
                )
        category_list.append(
            {
                "category_name": str(c.get("category_name", name)),
                "category_description": str(c.get("category_description", "")),
                "notes": notes,
            }
        )

    return {
        "states": state_list,
        "categories": category_list,
        "expectations_file": str(data.get("expectations_file", "expectations.json")),
    }


def put_preferences(payload: dict) -> dict:
    """Persist a list-shaped preferences document back to preferences.json."""
    states = {}
    for s in payload.get("states", []) or []:
        if not isinstance(s, dict) or not s.get("state_name"):
            continue
        name = str(s["state_name"])
        states[name] = {
            "state_name": name,
            "description": str(s.get("description", "")),
            "event_list": [e for e in s.get("event_list", []) if isinstance(e, str)],
            "active_time": str(s.get("active_time", "")) or _now_iso(),
        }

    categories = {}
    for c in payload.get("categories", []) or []:
        if not isinstance(c, dict) or not c.get("category_name"):
            continue
        name = str(c["category_name"])
        notes = []
        for n in c.get("notes", []) or []:
            if isinstance(n, str):
                notes.append({"content": n, "id": "", "saved_time": ""})
            elif isinstance(n, dict) and n.get("content") is not None:
                notes.append(
                    {
                        "content": str(n.get("content", "")),
                        "id": str(n.get("id", "")) or uuid4().hex,
                        "saved_time": str(n.get("saved_time", "")) or _now_iso(),
                    }
                )
        categories[name] = {
            "category_name": name,
            "category_description": str(c.get("category_description", "")),
            "notes": notes,
        }

    data = _read_json(PREFERENCES_FILE, {})
    data["states"] = states
    data["note_categories"] = categories
    data["expectations_file"] = str(
        payload.get("expectations_file")
        or data.get("expectations_file", "expectations.json")
    )
    _write_json(PREFERENCES_FILE, data)
    return get_preferences()


# ─────────────────────────────────────────────────────────────────────
# Expectations + general behavior guide
# ─────────────────────────────────────────────────────────────────────


def get_expectations() -> dict:
    """Return guides + expectations (backend field names)."""
    guide = _read_json(BEHAVIOR_GUIDE_FILE, {"length": 0, "guides": []})
    exp = _read_json(EXPECTATIONS_FILE, {"expectations": []})

    guides = [g for g in guide.get("guides", []) if isinstance(g, str)]
    expectations = []
    for e in exp.get("expectations", []) or []:
        if isinstance(e, dict) and e.get("expectation_content") is not None:
            expectations.append(
                {
                    "trigger": str(e.get("trigger", "")),
                    "action": str(e.get("action", "")),
                    "expectation_content": str(e.get("expectation_content", "")),
                }
            )

    return {"guides": guides, "expectations": expectations}


def put_expectations(payload: dict) -> dict:
    """Persist guides + expectations, recomputing the guide ``length``."""
    dedup_guides = []
    seen: set[str] = set()
    for g in payload.get("guides", []) or []:
        gs = str(g).strip()
        if gs and gs not in seen:
            seen.add(gs)
            dedup_guides.append(gs)
    _write_json(
        BEHAVIOR_GUIDE_FILE,
        {"length": sum(len(g) for g in dedup_guides), "guides": dedup_guides},
    )

    expectations = []
    for e in payload.get("expectations", []) or []:
        if not isinstance(e, dict):
            continue
        content = str(e.get("expectation_content", "")).strip()
        if not content:
            continue
        expectations.append(
            {
                "trigger": str(e.get("trigger", "")).strip(),
                "action": str(e.get("action", "")).strip(),
                "expectation_content": content,
            }
        )
    _write_json(EXPECTATIONS_FILE, {"expectations": expectations})

    return get_expectations()


# ─────────────────────────────────────────────────────────────────────
# Runtime config (General + Model overrides)
# ─────────────────────────────────────────────────────────────────────

RUNTIME_CONFIG_FILE = PROJECT_DIR / "runtime_config.json"


def _read_runtime() -> dict:
    data = _read_json(RUNTIME_CONFIG_FILE, {})
    if not isinstance(data.get("general"), dict):
        data["general"] = {}
    if not isinstance(data.get("model"), dict):
        data["model"] = {}
    return data


def get_runtime() -> dict:
    """Return the raw overrides document plus effective resolved values."""
    import config as config_mod

    raw = _read_runtime()
    return {
        "general": config_mod.effective_general_config(),
        "general_overrides": raw.get("general", {}),
        "model": config_mod.effective_model_settings(),
        "model_overrides": raw.get("model", {}),
    }


def put_runtime_general(overrides: dict) -> dict:
    """Merge general overrides into runtime_config.json and apply them."""
    import config as config_mod

    raw = _read_runtime()
    clean = {}
    for key in config_mod.GENERAL_CONFIG_KEYS:
        if key in overrides and overrides[key] is not None:
            clean[key] = overrides[key]
    raw["general"] = clean
    _write_json(RUNTIME_CONFIG_FILE, raw)
    config_mod.reload_runtime_config()
    return get_runtime()


def put_runtime_model(settings: dict) -> dict:
    """Merge model overrides into runtime_config.json (read lazily by get_llm)."""

    raw = _read_runtime()
    model = raw.get("model", {})
    if "model_name" in settings and settings["model_name"]:
        model["model_name"] = str(settings["model_name"]).strip()
    if "temperature" in settings and settings["temperature"] is not None:
        model["temperature"] = float(settings["temperature"])
    if "context_window" in settings and settings["context_window"]:
        model["context_window"] = str(settings["context_window"]).strip()
    if (
        "reasoning_effort" in settings
        and settings["reasoning_effort"] in ("low", "medium", "high")
    ):
        model["reasoning_effort"] = str(settings["reasoning_effort"])
    raw["model"] = model
    _write_json(RUNTIME_CONFIG_FILE, raw)
    return get_runtime()


