"""
Global configuration constants for the memory system.
Centralized here so all modules reference the same values.

This module is imported early by every entry point, so it is also where the
optional ``.env`` file is loaded.
"""

from pathlib import Path

# ── .env loading ───────────────────────────────────────────────────
# Load variables from a local ``.env`` if one exists (see ``.env.example``).
# The OS environment always wins: load_dotenv() does NOT override variables
# that are already set, so `setx`/shell exports keep working unchanged.
# The import is guarded so a missing optional dependency cannot take down
# every entry point.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:  # pragma: no cover - python-dotenv is optional
    pass

# ── Project Paths ──────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DB_DIR = str(PROJECT_DIR / "chroma_db")          # ChromaDB persistent directory
CHROMA_COLLECTION_NAME = "ltm_summaries"
THREAD_STATE_FILE = str(PROJECT_DIR / "thread_state.json")  # persisted ThreadBar
# Root of the agent's filesystem sandbox. The filesystem MCP server and the
# shell tool are both locked to this directory (created on first use).
SANDBOX_DIR = str(PROJECT_DIR / "Dummy_Folder")

# ── Embedding Model ────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Preference Vector Store ────────────────────────────────────────
PREFERENCE_DB_DIR = str(PROJECT_DIR / "preference_chroma_db")
PREFERENCE_COLLECTION_NAME = "preference_notes"
PREFERENCE_NOTE_SIMILARITY_THRESHOLD = 0.3   # cosine distance; lower = more similar
PREFERENCE_NOTE_QUERY_K = 3                 # max similar notes returned by dedupe search

# ── LLM Model ──────────────────────────────────────────────────────
LLM_MODEL_NAME = "deepseek-v4-flash-vision-exp"
#zai-org/GLM-5.3-Flash
#deepseek:deepseek-v4-flash

# Substrings that identify a vision-capable main model. Used to reject
# image-bearing messages when the active model cannot see images.
VISION_MODEL_MARKERS = ("vision", "-vl", "vl-", "multimodal", "omni")

# ── Small/Cheap LLM (preference system classification) ─────────────
# Qwen 3.5 9B served via SiliconFlow's OpenAI-compatible endpoint.
# NOTE: the .com (international) and .cn platforms issue different keys —
# SMALL_LLM_BASE_URL must match whichever platform your key came from.
SMALL_LLM_MODEL_NAME = "Qwen/Qwen3.5-9B"
SMALL_LLM_BASE_URL = "https://api.siliconflow.com/v1"
SMALL_LLM_API_KEY_ENV = "SILICONFLOW_API_KEY"


# ── Small LLM: request timeout + retry (hosted-SiliconFlow variance guard) ──
# If a response takes longer than SMALL_LLM_TIMEOUT_SECONDS, cut the connection
# and resend (OpenAI SDK retries up to SMALL_LLM_MAX_RETRIES times, exponential
# backoff). These are mitigation for the HOSTED SiliconFlow service's latency
# spikes. If this model is later replaced by a self-hosted deployment, this
# timeout/retry section can be removed (the model block above stays).
SMALL_LLM_TIMEOUT_SECONDS = 3
SMALL_LLM_MAX_RETRIES = 2


# ── Thread Limits ──────────────────────────────────────────────────
MAX_MESSAGES_PER_THREAD = 8
MAX_ACTIVE_THREADS = 5
BRIDGE_SUMMARY_WORD_LIMIT = 150     # ~150 words per summary block

# ── LTM Retrieval ──────────────────────────────────────────────────
LTM_RETRIEVE_K = 3                  # max results from vector search
LTM_SCORE_THRESHOLD = 0.7           # discard results with score >= 1.0
LTM_LATEST_K = 2                    # keep the 2 lowest-score objects as latest_ltm

# ── Summary Counter ────────────────────────────────────────────────
SUMMARY_COUNTER_INITIAL = 0        # first summary block starts at 0

# ── Worker Round Limit ───────────────────────────────────────────────
WORKER_ROUND_LIMIT = 8             # max tool invocations per worker before forced failure

# ── Main-Agent Task Cap ─────────────────────────────────────────────
MAX_ASSIGN_TASKS = 8             # max assign_task calls the main agent may make per round

# ── Main-Agent Memory Search Cap ────────────────────────────────────
MAX_MEMORY_SEARCH = 5             # max memory_search calls the main agent may make per round

# ── Tool Message Size Limit ──────────────────────────────────────────
TOOLMESSAGE_LIMIT = 40000          # max characters per tool response message before truncation

# ═══════════════════════════════════════════════════════════════════
# Runtime overrides (edited live from the web settings UI)
#
# Static defaults live above. The web UI persists changes to
# ``runtime_config.json`` (next to this module). ``config.py`` applies any
# overrides present in that file at import time, and ``reload_runtime_config()``
# re-applies them if the file changes while the process is running.
#
#   {"general": {MAX_MESSAGES_PER_THREAD: 10, ...},
#    "model":   {model_name, temperature, context_window, reasoning_effort}}
#
# Consumers that do ``from config import X`` bind X at import; consumers that
# do ``import config`` and read ``config.X`` pick up reloaded values live.
# ═══════════════════════════════════════════════════════════════════
RUNTIME_CONFIG_FILE = str(PROJECT_DIR / "runtime_config.json")

# General keys that the web "General" settings tab can override.
GENERAL_CONFIG_KEYS = (
    "MAX_MESSAGES_PER_THREAD",
    "MAX_ACTIVE_THREADS",
    "BRIDGE_SUMMARY_WORD_LIMIT",
    "LTM_RETRIEVE_K",
    "LTM_SCORE_THRESHOLD",
    "LTM_LATEST_K",
    "SUMMARY_COUNTER_INITIAL",
    "WORKER_ROUND_LIMIT",
    "MAX_ASSIGN_TASKS",
    "MAX_MEMORY_SEARCH",
    "TOOLMESSAGE_LIMIT",
)

# Defaults for the main-agent model (get_llm). These are the values the web
# "Model" settings tab edits; they only ever affect get_llm(), never the
# small/GLM/no-thinking singletons.
MAIN_MODEL_NAME = LLM_MODEL_NAME
MAIN_MODEL_TEMPERATURE = 0.0
MAIN_MODEL_CONTEXT_WINDOW = "500k"
MAIN_MODEL_REASONING_EFFORT = "low"   # "low" | "medium" | "high"

# Snapshot of the pristine general defaults (before any runtime overrides are
# applied). reload_runtime_config() resets to these, then reapplies the file.
DEFAULT_GENERAL: dict = {key: globals()[key] for key in GENERAL_CONFIG_KEYS}


def load_runtime_overrides() -> dict:
    """Read ``runtime_config.json`` into a dict. Never raises."""
    try:
        with open(RUNTIME_CONFIG_FILE, "r", encoding="utf-8") as f:
            import json

            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _set_general(mapping: dict) -> None:
    """Overwrite matching module-level general constants from ``mapping``."""
    for key in GENERAL_CONFIG_KEYS:
        if key in mapping and mapping[key] is not None:
            globals()[key] = mapping[key]


def _apply_general_overrides(overrides: dict) -> None:
    """Reset to defaults, then apply any persisted overrides on top."""
    merged = dict(DEFAULT_GENERAL)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    _set_general(merged)


# Apply persisted overrides at import time (so startup already honours them).
_apply_general_overrides(load_runtime_overrides().get("general", {}))


def reload_runtime_config() -> None:
    """Re-read ``runtime_config.json`` and re-apply general overrides.

    Resets mutated globals back to their defaults first, so clearing an
    override truly restores the original value. Call this at the start of a
    round to pick up General-tab changes without a restart. Model settings are
    read by ``get_llm()`` lazily, so they do not need this call.
    """
    _apply_general_overrides(load_runtime_overrides().get("general", {}))


def effective_model_settings() -> dict:
    """Return the effective main-agent model settings (overrides merged over
    defaults). The web Model tab and ``get_llm()`` both read from this."""
    model = load_runtime_overrides().get("model", {}) or {}
    return {
        "model_name": model.get("model_name", MAIN_MODEL_NAME),
        "temperature": float(model.get("temperature", MAIN_MODEL_TEMPERATURE)),
        "context_window": str(model.get("context_window", MAIN_MODEL_CONTEXT_WINDOW)),
        "reasoning_effort": str(
            model.get("reasoning_effort", MAIN_MODEL_REASONING_EFFORT)
        ),
    }


def effective_general_config() -> dict:
    """Return all general keys with any runtime overrides applied."""
    overrides = load_runtime_overrides().get("general", {}) or {}
    out: dict = {}
    for key in GENERAL_CONFIG_KEYS:
        out[key] = overrides[key] if key in overrides else globals().get(key)
    return out
