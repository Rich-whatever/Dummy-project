"""
LLM call abstraction for the memory system.

* ``get_llm()``       — primary model (DeepSeek v4 Flash via ``config.LLM_MODEL_NAME``)
* ``get_small_llm()`` — cheap classification model (Qwen 3.5 9B via SiliconFlow's
                        OpenAI-compatible endpoint, thinking mode disabled)

Both are singletons so the model is only initialised once.
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from config import (
    LLM_MODEL_NAME,
    SMALL_LLM_API_KEY_ENV,
    SMALL_LLM_BASE_URL,
    SMALL_LLM_MAX_RETRIES,
    SMALL_LLM_MODEL_NAME,
    SMALL_LLM_TIMEOUT_SECONDS,
    effective_model_settings,
)

# Module-level cache so we only initialise the models once.
_llm: BaseChatModel | None = None
_llm_key: tuple | None = None
_llm_no_thinking: BaseChatModel | None = None
_small_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """
    Return a DeepSeek chat model for the main agent.

    This is a caching factory keyed on the effective settings
    (``model_name`` / ``temperature`` / ``reasoning_effort``). Settings are read
    live from ``runtime_config.json`` (via ``config.effective_model_settings()``)
    on every call, so a Model-tab change from the web UI is picked up on the next
    call with a cheap re-initialisation — no process restart needed.

    Only this function is affected by the Model settings tab; the no-thinking,
    small, and GLM singletons are untouched.
    """
    global _llm, _llm_key
    settings = effective_model_settings()
    key = (
        settings["model_name"],
        settings["temperature"],
        settings["reasoning_effort"],
    )
    if _llm is None or _llm_key != key:
        kwargs = dict(
            model=settings["model_name"],
            reasoning_effort=settings["reasoning_effort"],
            extra_body={"thinking": {"type": "enabled"}},
        )
        # Temperature is threaded only when the provider supports it cleanly.
        # NOTE: context_window is recorded in the effective settings and shown in
        # the UI, but init_chat_model has no portable knob for it yet, so it is
        # intentionally not passed to avoid provider errors.
        if settings["temperature"] is not None:
            kwargs["temperature"] = settings["temperature"]
        _llm = init_chat_model(**kwargs)
        _llm_key = key
    return _llm

def get_glm_llm() -> BaseChatModel:
        api_key = os.getenv(SMALL_LLM_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {SMALL_LLM_API_KEY_ENV!r} is not set. "
                "Set your SiliconFlow API key (e.g. `setx SILICONFLOW_API_KEY ...` "
                "then restart the terminal) before calling get_small_llm()."
            )
        glm_llm = init_chat_model(
            "zai-org/GLM-5.3-Flash",
            model_provider="openai",
            base_url=SMALL_LLM_BASE_URL,
            api_key=api_key,
        )
        return glm_llm


def get_llm_no_thinking() -> BaseChatModel:
    """
    Return a singleton DeepSeek chat model with thinking DISABLED.

    Used for mechanical structured-output tasks (worker, router, preference
    injection/extraction, preference CLI).  Thinking must be off here because
    the DeepSeek endpoint rejects tool/function calls while thinking is enabled.
    """
    global _llm_no_thinking
    if _llm_no_thinking is None:
        _llm_no_thinking = init_chat_model(
            model=LLM_MODEL_NAME,
            reasoning_effort="low",
            extra_body={"thinking": {"type": "disabled"}},
        )
    return _llm_no_thinking


def get_small_llm() -> BaseChatModel:
    """
    Return a singleton Qwen 3.5 9B chat model instance served through
    SiliconFlow's OpenAI-compatible endpoint (``https://api.siliconflow.com/v1``).

    Thinking mode is disabled via ``extra_body={"enable_thinking": False}``
    so the model answers directly without emitting reasoning output —
    which keeps JSON-mode structured output clean.

    The API key is read from the ``SILICONFLOW_API_KEY`` environment
    variable (see ``config.SMALL_LLM_API_KEY_ENV``).
    """
    global _small_llm
    if _small_llm is None:
        api_key = os.getenv(SMALL_LLM_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {SMALL_LLM_API_KEY_ENV!r} is not set. "
                "Set your SiliconFlow API key (e.g. `setx SILICONFLOW_API_KEY ...` "
                "then restart the terminal) before calling get_small_llm()."
            )
        _small_llm = init_chat_model(
            SMALL_LLM_MODEL_NAME,
            model_provider="openai",
            base_url=SMALL_LLM_BASE_URL,
            api_key=api_key,
            extra_body={"enable_thinking": False},
            # ── Hosted-SiliconFlow latency guard (see config) ──────────
            # Cut the connection + resend if a call exceeds the timeout.
            # This section can be removed if the model is self-hosted later.
            request_timeout=SMALL_LLM_TIMEOUT_SECONDS,
            max_retries=SMALL_LLM_MAX_RETRIES,
        )
    return _small_llm


def is_vision_capable(model_name: str | None = None) -> bool:
    """Whether the (effective) main model can accept image content blocks.

    Checks the model name against ``config.VISION_MODEL_MARKERS``. Pass an
    explicit ``model_name`` to override; otherwise the effective main-model
    setting (runtime override merged over default) is used.
    """
    from config import VISION_MODEL_MARKERS

    name = (model_name or effective_model_settings()["model_name"] or "").lower()
    return any(marker in name for marker in VISION_MODEL_MARKERS)
