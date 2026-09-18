"""Web-facing agent session: run a full pipeline round for a chat message.

Mirrors what the CLI REPL (``main.py``) does per message, but hosted in the
API server so the web UI can:
  - POST a user message (returns a ``run_id`` immediately),
  - poll the run's status/result,
  - cancel a running round cooperatively (at the next agent/worker turn
    boundary).

One agent session is shared across requests (thread_bar / vector store /
toolkit / preference store), so the conversation persists across messages just
like the CLI.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from config import THREAD_STATE_FILE
from memory_backend.models import load_thread_bar, save_thread_bar
from workflow.run_control import (
    CancelledRun,
    begin_run,
    clear_run,
    request_cancel,
)

logger = logging.getLogger("api.agent")

_active_run_id: str | None = None
_runs: dict[str, dict] = {}

# Lazily-initialised session components (heavy).
_session = None


class _Session:
    """Holds the long-lived components used by every pipeline round."""

    def __init__(self) -> None:
        self.ready = False
        self.thread_bar = None
        self.vector_store = None
        self.toolkit = None
        self.preference_store = None
        self.preference_vector_store = None
        self._client_list = []

    async def ensure_ready(self) -> None:
        if self.ready:
            return
        from preference_system.models import PreferenceStore
        from preference_system.vector_store import PreferenceVectorStore
        from tool.toolkit_registry import build_toolkit
        from vector_store.client import MemoryVectorStore
        from workflow.llm import get_llm

        logger.info("Agent session: initialising components...")
        self.vector_store = MemoryVectorStore()
        self.thread_bar = load_thread_bar(THREAD_STATE_FILE)
        if self.thread_bar.count:
            logger.info(
                "Agent session loaded %d persisted thread(s)", self.thread_bar.count
            )
        self.preference_store = PreferenceStore.load("preferences.json")
        self.preference_vector_store = PreferenceVectorStore()
        logger.info("Agent session: building toolkits (MCP clients)...")
        # Build toolkits (MCP clients) and keep them alive for the process.
        self._client_list, self.toolkit = await build_toolkit()
        get_llm()  # warm the model singleton
        self.ready = True
        logger.info("Agent session: ready.")

    async def run_message(
        self, user_message: str, human_message_content: list[dict] | None = None
    ) -> dict:
        """Run one full pipeline round. Returns the response payload."""
        await self.ensure_ready()

        # Re-read persisted state at the start of EVERY round so external
        # edits (settings UI, hand edits to JSON, CLI deletions) take effect
        # without a restart. This is safe because every completed round
        # re-persists these files, so disk mirrors memory between rounds.
        from config import reload_runtime_config
        from preference_system.models import PreferenceStore

        reload_runtime_config()
        self.thread_bar = load_thread_bar(THREAD_STATE_FILE)
        self.preference_store = PreferenceStore.load("preferences.json")
        logger.info(
            "Agent session reloaded state: %d thread(s), %d note categories",
            self.thread_bar.count,
            len(self.preference_store.note_categories),
        )

        from preference_system.graph_nodes import wait_for_background_injections
        from workflow.graph import run_pipeline

        begin_run()
        t0 = time.perf_counter()
        logger.info("CHAT → starting run_pipeline (len=%d)", len(user_message))
        try:
            result = await run_pipeline(
                user_message=user_message,
                thread_bar=self.thread_bar,
                vector_store=self.vector_store,
                categorized_toolkit=self.toolkit,
                llm=None,  # run_pipeline uses get_llm() internally when None
                preference_store=self.preference_store,
                preference_vector_store=self.preference_vector_store,
                human_message_content=human_message_content,
            )
            # Let fire-and-forget preference injection finish before persisting.
            try:
                await wait_for_background_injections()
            except Exception:
                logger.warning("wait_for_background_injections raised", exc_info=True)

            # Persist the thread bar (append/summarise/evict happened inside).
            try:
                save_thread_bar(self.thread_bar, THREAD_STATE_FILE)
            except Exception:
                logger.warning("Failed to persist thread bar", exc_info=True)

            payload = {
                "user_message": result.get("user_message", ""),
                "agent_response": result.get("agent_response", ""),
                "underlying_content": result.get("underlying_content"),
                "router_result": result.get("router_result", ""),
                "thread_was_evicted": result.get("thread_was_evicted", ""),
                "response_time_s": round(time.perf_counter() - t0, 2),
            }
            logger.info(
                "CHAT → run finished in %.2fs (router=%s)",
                payload["response_time_s"],
                payload["router_result"],
            )
            return payload
        finally:
            clear_run()


async def _ensure_session() -> "_Session":
    global _session
    if _session is None:
        _session = _Session()
    await _session.ensure_ready()
    return _session


async def start_run(
    user_message: str,
    upload_ids: list[str] | None = None,
    human_message_content: list[dict] | None = None,
    reference_labels: list[dict] | None = None,
) -> str:
    """Start a chat round in the background. Raises if one is already running.

    ``user_message`` is the ABBREVIATED input (typed text + file manifest) used
    by every non-main-agent subsystem and for memory. ``human_message_content``
    is the full content-block list (file content + user text) injected as the
    HumanMessage for the main agent. Uploaded files are cleaned up once the run
    finishes.
    """
    global _active_run_id
    if _active_run_id is not None:
        raise RuntimeError("A message is already being processed. Please wait.")

    run_id = uuid.uuid4().hex
    _active_run_id = run_id
    _runs[run_id] = {
        "status": "running",
        "result": None,
        "error": "",
        "upload_ids": list(upload_ids or []),
        "reference_labels": list(reference_labels or []),
    }

    async def _runner() -> None:
        rec = _runs[run_id]
        try:
            session = await _ensure_session()
            result = await session.run_message(user_message, human_message_content)
            rec["status"] = "done"
            rec["result"] = result
        except CancelledRun:
            logger.info("CHAT → run %s cancelled", run_id)
            rec["status"] = "cancelled"
        except Exception as exc:  # noqa: BLE001 - surface to the client
            logger.exception("Chat round %s failed", run_id)
            rec["status"] = "failed"
            rec["error"] = str(exc)
        else:
            # Persist the completed round to the web-UI chat history
            # (30-round scrolling window). Failures never break the run.
            try:
                from datetime import datetime, timezone

                from api.persistence import append_chat_history

                r = rec["result"] or {}
                append_chat_history(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "user_message": r.get("user_message", ""),
                        "agent_response": r.get("agent_response", ""),
                        "underlying_content": r.get("underlying_content"),
                        "response_time_s": r.get("response_time_s", 0),
                        "thread_was_evicted": r.get("thread_was_evicted", ""),
                        "references": rec.get("reference_labels", []),
                    }
                )
            except Exception:
                logger.warning(
                    "Failed to append chat history for run %s", run_id, exc_info=True
                )
        finally:
            # Free the uploaded files once the round is over, so a discarded
            # set of uploads never leaks on disk / in the store.
            try:
                if rec.get("upload_ids"):
                    from api import files as api_files

                    api_files.discard(rec["upload_ids"])
            except Exception:
                logger.warning(
                    "Failed to clean up uploads for run %s", run_id, exc_info=True
                )

            global _active_run_id
            if _active_run_id == run_id:
                _active_run_id = None

    asyncio.create_task(_runner())
    return run_id


def get_run(run_id: str) -> dict | None:
    rec = _runs.get(run_id)
    if rec is None:
        return None
    return {
        "run_id": run_id,
        "status": rec["status"],
        "result": rec["result"],
        "error": rec["error"],
    }


def cancel_run(run_id: str) -> bool:
    rec = _runs.get(run_id)
    if rec is None or rec["status"] != "running":
        return False
    request_cancel()
    return True
