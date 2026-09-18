"""FastAPI server exposing settings + agent persistence to the web UI.

Run with::

    python -m api.server          # serves http://127.0.0.1:8000

CORS is enabled for the Vite dev server origin so the React app can call it.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api import persistence


def _configure_logging() -> None:
    """Mirror ``main.setup_logging`` for the API process so the agent pipeline
    logs (INFO) print to this process's stderr — otherwise they are suppressed
    because Python's root logger defaults to WARNING."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Quiet uvicorn's own HTTP access/error chatter (each request prints a line
    # otherwise); our pipeline logs still flow through the root INFO handler.
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    for name in (
        "preference_system.note_injection",
        "preference_system.expectation_injection",
        "preference_system.retrieve_branch",
        "preference_system.state_injection",
        "preference_system.vector_store",
        "preference_system.models",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

app = FastAPI(title="Dummy Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PreferencesBody(BaseModel):
    states: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    expectations_file: str | None = None


class ExpectationsBody(BaseModel):
    guides: list[str] = []
    expectations: list[dict[str, Any]] = []


class GeneralBody(BaseModel):
    general: dict[str, Any] = {}


class ModelBody(BaseModel):
    model: dict[str, Any] = {}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# ── Preferences ──────────────────────────────────────────────────────


@app.get("/api/preferences")
def read_preferences() -> dict:
    return persistence.get_preferences()


@app.put("/api/preferences")
def write_preferences(body: PreferencesBody) -> dict:
    return persistence.put_preferences(body.dict())


# ── Expectations + behavior guide ────────────────────────────────────


@app.get("/api/expectations")
def read_expectations() -> dict:
    return persistence.get_expectations()


@app.put("/api/expectations")
def write_expectations(body: ExpectationsBody) -> dict:
    return persistence.put_expectations(body.dict())


# ── Runtime config (General + Model) ────────────────────────────────


@app.get("/api/config")
def read_runtime() -> dict:
    return persistence.get_runtime()


@app.get("/api/chat/history")
def read_chat_history() -> dict:
    return {"rounds": persistence.get_chat_history()}


@app.put("/api/config/general")
def write_general(body: GeneralBody) -> dict:
    return persistence.put_runtime_general(body.general)


@app.put("/api/config/model")
def write_model(body: ModelBody) -> dict:
    return persistence.put_runtime_model(body.model)


# ── File uploads (web UI attachments) ───────────────────────────────


@app.post("/api/upload")
async def upload_file(file: UploadFile) -> dict:
    from api import files as api_files

    return await api_files.save_upload(file)


@app.get("/api/upload/status")
async def upload_status(ids: str) -> dict:
    """Per-upload parse status. ``ids`` is a comma-separated list of upload_ids."""
    from api import files as api_files

    upload_ids = [i for i in ids.split(",") if i]
    if not upload_ids:
        return {"uploads": [], "ready": True}
    return api_files.status_of(upload_ids)


@app.delete("/api/upload/{upload_id}")
async def upload_delete(upload_id: str) -> dict:
    from api import files as api_files

    api_files.discard([upload_id])
    return {"discarded": True}


# ── Chat (agent round) ───────────────────────────────────────────────


class ChatMessageBody(BaseModel):
    user_message: str = ""
    upload_ids: list[str] = []
    references: list[dict] = []


# Referenced-round composition (Option A): the referenced CONTENT goes only into
# the main agent's human message; the subsystems + memory see a metadata marker.
MAX_REFERENCES = 3
_MAX_REF_FIELD_CHARS = 4000
_REFERENCE_HEADER = "=== Referenced Earlier Conversation ==="


def _reference_rounds(references: list[dict]):
    """Parse client references into ConversationRound objects (capped/truncated)."""
    from memory_backend.models import ConversationRound

    rounds = []
    for r in references[:MAX_REFERENCES]:
        if not isinstance(r, dict):
            continue
        rounds.append(
            ConversationRound(
                user_message=str(r.get("user_message", ""))[:_MAX_REF_FIELD_CHARS],
                agent_response=str(r.get("agent_message", ""))[:_MAX_REF_FIELD_CHARS],
                underlying_content=None,
                timestamp=str(r.get("timestamp", "") or ""),
            )
        )
    return rounds


def _reference_content_blocks(rounds) -> list[dict]:
    """Human-message block carrying the full referenced round(s)."""
    if not rounds:
        return []
    from memory_backend.models import format_rounds

    return [{"type": "text",
             "text": f"{_REFERENCE_HEADER}\n\n{format_rounds(rounds)}"}]


def _reference_marker(rounds) -> str:
    """Metadata-only marker for the abbreviated user_message (subsystems/memory)."""
    if not rounds:
        return ""
    from datetime import datetime, timezone

    from memory_backend.models import format_round_time

    now = datetime.now(timezone.utc)
    return "\n".join(
        f"[referenced earlier round @ {format_round_time(r.timestamp, now)}]"
        for r in rounds
    )


@app.post("/api/chat")
async def chat_send(body: ChatMessageBody) -> dict:
    from api import agent
    from api import files as api_files
    from file_input.manifest import abbreviate

    user_text = body.user_message.strip()
    upload_ids = list(body.upload_ids or [])
    ref_rounds = _reference_rounds([r for r in (body.references or []) if isinstance(r, dict)])

    if not user_text and not upload_ids and not ref_rounds:
        raise HTTPException(status_code=422, detail="Empty user_message")

    # File content blocks (the current user message is appended separately, last,
    # so the order is: files → referenced rounds → current user message).
    file_blocks: list[dict] = []
    manifest = ""
    if upload_ids:
        from file_input.config import IMAGE_SEND_METHOD
        from file_input.pipeline import files_to_llm_input

        result = files_to_llm_input(
            api_files.get_store(),
            upload_ids,
            user_message=None,
            image_method=IMAGE_SEND_METHOD,
        )
        if not result.ok:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot send — {result.notice}",
            )
        file_blocks = result.content_blocks or []
        manifest = result.manifest

    human_content = (
        file_blocks
        + _reference_content_blocks(ref_rounds)
        + ([{"type": "text", "text": f"\nUser Message\n{user_text}"}] if user_text else [])
    )

    # Abbreviated user_message seen by router/preferences/LTM/memory:
    # typed text + file manifest + reference marker (never the referenced body).
    abbreviated = abbreviate(user_text, manifest)
    marker = _reference_marker(ref_rounds)
    if marker:
        abbreviated = f"{abbreviated}\n\n{marker}" if abbreviated else marker

    # Image-bearing messages require a vision-capable model; otherwise error out
    # with a clear note instead of sending images the model cannot read.
    if any(b.get("type") != "text" for b in human_content):
        from workflow.llm import is_vision_capable

        if not is_vision_capable():
            raise HTTPException(
                status_code=422,
                detail=(
                    "Images cannot be processed: the active model is not "
                    "vision-capable. Switch the Model setting to a vision model."
                ),
            )

    reference_labels = [
        {
            "timestamp": r.timestamp,
            "user_message": r.user_message,
            "agent_message": r.agent_response,
        }
        for r in ref_rounds
    ]

    try:
        run_id = await agent.start_run(
            abbreviated,
            upload_ids=upload_ids,
            human_message_content=human_content,
            reference_labels=reference_labels,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run_id}


@app.get("/api/chat/runs/{run_id}")
async def chat_status(run_id: str) -> dict:
    from api import agent

    rec = agent.get_run(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return rec


@app.post("/api/chat/runs/{run_id}/cancel")
async def chat_cancel(run_id: str) -> dict:
    from api import agent

    if not agent.cancel_run(run_id):
        raise HTTPException(status_code=404, detail="Run not running or unknown")
    return {"cancelled": True}


def main() -> None:
    import uvicorn

    _configure_logging()
    uvicorn.run(
        "api.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        access_log=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
