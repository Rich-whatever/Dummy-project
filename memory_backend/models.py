"""
Core data structures for the memory system.

Defines:
    ConversationRound — one user input + one agent response in a thread.
    Thread            — a conversation thread with up to 8 rounds, bridge summary, LTM refs.
    ThreadBar         — a container managing up to 4 active threads.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import (
    MAX_ACTIVE_THREADS,
    MAX_MESSAGES_PER_THREAD,
    SUMMARY_COUNTER_INITIAL,
)

# ── helpers ────────────────────────────────────────────────────────

def _now_utc() -> str:
    """Return a sortable ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    """Return a fresh UUID4 string."""
    return str(uuid.uuid4())


# ── Round formatting (single source for every LLM-facing extraction) ──
#
# Every place that renders past conversation rounds for an LLM (STM context,
# summarisation, eviction, cross-thread context, pre-router, preference
# retrieve) uses ``format_rounds`` so the labels, structure and time info are
# identical everywhere. Callers only choose WHICH rounds and whether to include
# the underlying execution content.
#
# Rendered per round:
#
#     ===== Conversation round | 2026-09-10 18:03 Thursday (2 hours ago) | +15 min after previous round =====
#     [USER]
#     <user_message>
#
#     [ASSISTANT]
#     <agent_response>
#
# The time is shown in LOCAL time and with a relative age so the model can
# judge recency/gaps without doing arithmetic. The stored ``timestamp`` stays
# ISO-8601 UTC.

_ROUND_HEADER = "===== Conversation round"


def _parse_ts(ts: str) -> datetime | None:
    """Parse an ISO-8601 timestamp; assume UTC when it has no tz. None on failure."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" + ("" if n == 1 else "s")


def humanize_age(ts: str, now: datetime) -> str:
    """Relative age of a timestamp, e.g. "just now", "3 hours ago", "2 days ago"."""
    then = _parse_ts(ts)
    if then is None:
        return "time unknown"
    seconds = max(0.0, (now - then).total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        return _plural(int(minutes), "minute") + " ago"
    hours = minutes / 60
    if hours < 24:
        return _plural(int(hours), "hour") + " ago"
    days = hours / 24
    if days < 7:
        return _plural(int(days), "day") + " ago"
    weeks = days / 7
    if weeks < 5:
        return _plural(int(weeks), "week") + " ago"
    months = days / 30
    if months < 12:
        return _plural(int(months), "month") + " ago"
    return _plural(int(days / 365), "year") + " ago"


def format_round_time(ts: str, now: datetime) -> str:
    """Local absolute time + relative age, e.g. "2026-09-10 18:03 Thursday (2 hours ago)"."""
    then = _parse_ts(ts)
    if then is None:
        return "(time unknown)"
    absolute = then.astimezone().strftime("%Y-%m-%d %H:%M %A")
    return f"{absolute} ({humanize_age(ts, now)})"


def _humanize_delta(prev_ts: str, ts: str) -> str:
    """Compact elapsed time between two timestamps ("" if either is unparseable)."""
    a, b = _parse_ts(prev_ts), _parse_ts(ts)
    if a is None or b is None:
        return ""
    secs = max(0.0, (b - a).total_seconds())
    if secs < 60:
        return f"{int(secs)} sec"
    if secs < 3600:
        return f"{int(secs // 60)} min"
    if secs < 86400:
        return f"{int(secs // 3600)} h"
    return f"{int(secs // 86400)} d"


def format_rounds(
    rounds: Sequence[ConversationRound],
    *,
    include_underlying: bool = False,
    show_gap: bool = True,
    now: datetime | None = None,
) -> str:
    """Render conversation rounds in the shared, LLM-friendly format.

    Parameters
    ----------
    rounds : Sequence[ConversationRound]
        The rounds to render (callers choose the slice). Any object exposing
        ``user_message`` / ``agent_response`` (and optionally
        ``underlying_content`` / ``timestamp``) works.
    include_underlying : bool
        Include each round's ``underlying_content`` (execution log) when present.
    show_gap : bool
        Annotate each round after the first with the elapsed time since the
        previous round (e.g. "+15 min after previous round").
    now : datetime | None
        Reference "now" for relative ages; captured once so all rounds agree.

    Returns the formatted block, or "" when ``rounds`` is empty.
    """
    rounds = list(rounds)
    if not rounds:
        return ""
    if now is None:
        now = datetime.now(timezone.utc)

    blocks: list[str] = []
    prev_ts = ""
    for r in rounds:
        ts = getattr(r, "timestamp", "") or ""
        header = f"{_ROUND_HEADER} | {format_round_time(ts, now)}"
        if show_gap and prev_ts:
            delta = _humanize_delta(prev_ts, ts)
            if delta:
                header += f" | +{delta} after previous round"
        header += " ====="

        lines = [
            header,
            "[USER]",
            str(getattr(r, "user_message", "")),
            "",
            "[ASSISTANT]",
            str(getattr(r, "agent_response", "")),
        ]
        if include_underlying:
            underlying = getattr(r, "underlying_content", None)
            if underlying:
                lines += ["", "[UNDERLYING EXECUTION CONTENT]", str(underlying)]
        blocks.append("\n".join(lines))
        prev_ts = ts

    return "\n\n".join(blocks)


# ── ConversationRound ──────────────────────────────────────────────

@dataclass
class ConversationRound:
    """
    A single round of conversation: one user input + one agent response.

    Attributes:
        user_message   : the user's input text
        agent_response : the assistant's reply text
        timestamp      : ISO-8601 UTC string (auto-set on creation)
    """
    user_message: str
    agent_response: str
    underlying_content: str | None = None
    timestamp: str = field(default_factory=_now_utc)


# ── Thread ─────────────────────────────────────────────────────────

@dataclass
class Thread:
    """
    A single conversation thread.

    Lifecycle:
        1. Created by the Intelligent Router (Track 1).
        2. Conversation rounds appended up to MAX_MESSAGES_PER_THREAD (8).
        3. When full, the earliest 6 rounds are summarised into a
           150-word summary block, which is:
             a) upserted into the LTM vector store,
             b) saved as this thread's new ``bridge_summary``.
           The latest 2 rounds remain, and ``summary_counter`` increments.
        4. On eviction, the thread may be summarised and pushed to LTM
           or deleted entirely.

    Attributes:
        thread_id         : UUID string, unique identifier.
        summary_counter   : integer distinguishing summary blocks with the same
                            thread_id (1, 2, 3, …).
        topic_description : LLM-generated dynamic string describing the focus.
        conversation_rounds : list of ConversationRound objects (max 8).
        bridge_summary    : the most recent ~150-word summary of earlier rounds
                            in this thread.  Pre-pended to LTM context when the
                            thread is continued.
        last_active       : ISO-8601 UTC timestamp, updated **only** when a new
                            round is appended (not on reads or touch).
        latest_ltm        : the two lowest-score LTM objects retrieved for this
                            thread — list of dicts:
                                [{"thread_id": str, "block#": int, "score": float}, …]
    """
    thread_id: str = field(default_factory=_new_uuid)
    summary_counter: int = SUMMARY_COUNTER_INITIAL
    topic_description: str = ""
    conversation_rounds: list[ConversationRound] = field(default_factory=list)
    bridge_summary: str = ""
    last_active: str = field(default_factory=_now_utc)
    latest_ltm: list[dict] = field(default_factory=list)

    # ── convenience ────────────────────────────────────────────────

    @property
    def round_count(self) -> int:
        return len(self.conversation_rounds)

    @property
    def is_full(self) -> bool:
        """True when this thread has reached the round limit."""
        return self.round_count >= MAX_MESSAGES_PER_THREAD

    @property
    def age_seconds(self) -> float:
        """Seconds since ``last_active`` (useful for LRU comparisons)."""
        then = datetime.fromisoformat(self.last_active)
        now = datetime.now(timezone.utc)
        return (now - then).total_seconds()

    def append_round(self, user_message: str, agent_response: str, underlying_content: str | None = None) -> None:
        """
        Add a full conversation round (user input + agent response) to this
        thread and update ``last_active``.

        Does NOT check the capacity — that is handled by the summarisation
        policy before appending.

        Parameters
        ----------
        user_message : str
            The user's input text.
        agent_response : str
            The assistant's reply text.
        underlying_content : str | None, optional
            Detailed plan/execution data (for execute-path responses).
        """
        self.conversation_rounds.append(
            ConversationRound(
                user_message=user_message,
                agent_response=agent_response,
                underlying_content=underlying_content,
            )
        )
        self.last_active = _now_utc()

    def to_summary_input(self) -> str:
        """
        Format the thread's last 5 rounds for LLM consumption (STM context /
        rolling summarisation). Delegates to the shared ``format_rounds`` so the
        labels, structure and time info match every other extraction site.

        Includes each round's ``underlying_content`` (execution log) when present.
        """
        return format_rounds(self.conversation_rounds[-5:], include_underlying=True)

    def get_last_n_rounds(self, n: int) -> list[ConversationRound]:
        """Return the last *n* conversation rounds (or all, if fewer exist)."""
        return self.conversation_rounds[-n:]

    def __repr__(self) -> str:
        return (
            f"Thread(id={self.thread_id[:8]}…, "
            f"rounds={self.round_count}, "
            f"sum#={self.summary_counter}, "
            f"active={self.last_active})"
        )


# ── ThreadBar ──────────────────────────────────────────────────────

@dataclass
class ThreadBar:
    """
    A container managing up to MAX_ACTIVE_THREADS (4) active Thread objects.

    Provides:
        - add / remove / get by thread_id
        - LRU eviction candidate selection
        - occupancy check
    """
    threads: list[Thread] = field(default_factory=list)

    # ── queries ────────────────────────────────────────────────────

    @property
    def is_full(self) -> bool:
        return len(self.threads) >= MAX_ACTIVE_THREADS

    @property
    def count(self) -> int:
        return len(self.threads)

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Return thread by ID, or ``None``."""
        for t in self.threads:
            if t.thread_id == thread_id:
                return t
        return None

    def has_thread(self, thread_id: str) -> bool:
        return self.get_thread(thread_id) is not None

    # ── mutations ──────────────────────────────────────────────────

    def add_thread(self, thread: Thread) -> bool:
        """
        Add a thread to the bar.

        Returns ``True`` on success, ``False`` if the bar is full.
        (The caller should handle eviction before calling this.)
        """
        if self.is_full:
            return False
        self.threads.append(thread)
        return True

    def remove_thread(self, thread_id: str) -> bool:
        """
        Remove a thread by ID.

        Returns ``True`` if removed, ``False`` if not found.
        """
        for i, t in enumerate(self.threads):
            if t.thread_id == thread_id:
                self.threads.pop(i)
                return True
        return False

    # ── LRU ────────────────────────────────────────────────────────

    def get_lru_thread(self) -> Optional[Thread]:
        """
        Return the thread with the oldest ``last_active`` timestamp
        (i.e. the best eviction candidate).

        Returns ``None`` if the bar is empty.
        """
        if not self.threads:
            return None
        return min(self.threads, key=lambda t: t.last_active)

    def get_lru_thread_id(self) -> Optional[str]:
        """Return the ``thread_id`` of the LRU thread."""
        lru = self.get_lru_thread()
        return lru.thread_id if lru else None

    def get_most_recent_thread(self) -> Optional[Thread]:
        """Return the most recently active thread (max ``last_active``)."""
        if not self.threads:
            return None
        return max(self.threads, key=lambda t: t.last_active)

    def get_most_recent_thread_id(self) -> Optional[str]:
        """Return the ``thread_id`` of the most recently active thread."""
        recent = self.get_most_recent_thread()
        return recent.thread_id if recent else None

    # ── iteration ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.threads)

    def __getitem__(self, index: int) -> Thread:
        return self.threads[index]

    def __iter__(self):
        return iter(self.threads)

    def __repr__(self) -> str:
        return f"ThreadBar({self.count}/{MAX_ACTIVE_THREADS} threads)"


# ── Persistence (ThreadBar <-> JSON) ────────────────────────────────


def _round_to_dict(r: ConversationRound) -> dict:
    return {
        "user_message": r.user_message,
        "agent_response": r.agent_response,
        "underlying_content": r.underlying_content,
        "timestamp": r.timestamp,
    }


def _round_from_dict(d: dict) -> ConversationRound:
    return ConversationRound(
        user_message=d.get("user_message", ""),
        agent_response=d.get("agent_response", ""),
        underlying_content=d.get("underlying_content"),
        # "" (not now) for legacy rounds missing a timestamp, so the formatter
        # renders "(time unknown)" instead of a misleading "just now".
        timestamp=d.get("timestamp", ""),
    )


def _thread_to_dict(t: Thread) -> dict:
    return {
        "thread_id": t.thread_id,
        "summary_counter": t.summary_counter,
        "topic_description": t.topic_description,
        "bridge_summary": t.bridge_summary,
        "last_active": t.last_active,
        "latest_ltm": t.latest_ltm,
        "conversation_rounds": [_round_to_dict(r) for r in t.conversation_rounds],
    }


def _thread_from_dict(d: dict) -> Thread:
    return Thread(
        thread_id=d.get("thread_id", _new_uuid()),
        summary_counter=d.get("summary_counter", SUMMARY_COUNTER_INITIAL),
        topic_description=d.get("topic_description", ""),
        conversation_rounds=[_round_from_dict(r) for r in d.get("conversation_rounds", [])],
        bridge_summary=d.get("bridge_summary", ""),
        last_active=d.get("last_active", _now_utc()),
        latest_ltm=list(d.get("latest_ltm", [])),
    )


def thread_bar_to_dict(bar: ThreadBar) -> dict:
    """Serialize a ``ThreadBar`` (and its threads/rounds) to JSON-safe dicts."""
    return {"threads": [_thread_to_dict(t) for t in bar.threads]}


def thread_bar_from_dict(data: dict) -> ThreadBar:
    """Rebuild a ``ThreadBar`` from the dict produced by :func:`thread_bar_to_dict`."""
    bar = ThreadBar()
    bar.threads = [_thread_from_dict(t) for t in data.get("threads", [])]
    return bar


def save_thread_bar(bar: ThreadBar, path) -> None:
    """Persist ``bar`` to ``path`` as JSON (atomic write)."""
    from json_io import write_json_atomic

    write_json_atomic(path, thread_bar_to_dict(bar))


def load_thread_bar(path):
    """Load a ``ThreadBar`` from ``path``.

    Returns a ``ThreadBar`` with the persisted threads. If the file is corrupt
    or the wrong shape it is **quarantined** (moved to ``<path>.corrupt-<ts>``)
    so the data is preserved, a loud error is printed, and an empty
    ``ThreadBar`` is returned.
    """
    from pathlib import Path

    from json_io import load_or_quarantine

    path = Path(path)
    if not path.exists():
        return ThreadBar()

    data, backup = load_or_quarantine(
        path, default={}, validate=lambda d: isinstance(d, dict)
    )
    if backup:
        print(
            f"  [ERROR] thread state file {path} was corrupt — quarantined to "
            f"{backup}; starting with an empty thread bar."
        )
        return ThreadBar()
    if not isinstance(data, dict):
        return ThreadBar()
    return thread_bar_from_dict(data)
