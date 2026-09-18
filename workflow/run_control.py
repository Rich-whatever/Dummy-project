"""Cooperative run-cancellation for the agent pipeline.

The web UI can stop a running agent round via a Cancel button. Aborting the
main-agent/worker turns must happen cooperatively — a single in-flight LLM or
tool call cannot be preempted, so the round is stopped at the next turn
boundary (the next main-agent iteration, before dispatching another worker, or
at the top of a worker's tool call).

These helpers use a single module-level ``asyncio.Event`` (the system runs one
round at a time), set up per run and cleared when the run finishes.
"""

from __future__ import annotations

import asyncio


class CancelledRun(Exception):
    """Raised inside the pipeline when a cancel was requested."""


_cancel_event: asyncio.Event | None = None


def begin_run() -> None:
    """Reset the cancellation token for a new run (no cancel requested)."""
    global _cancel_event
    _cancel_event = asyncio.Event()


def request_cancel() -> None:
    """Signal the in-progress run to stop at the next turn boundary."""
    global _cancel_event
    if _cancel_event is None:
        _cancel_event = asyncio.Event()
    _cancel_event.set()


def clear_run() -> None:
    """Drop the cancellation token once a run has finished/cancelled."""
    global _cancel_event
    _cancel_event = None


def cancel_requested() -> bool:
    """Return True if a cancel has been requested for the current run."""
    return _cancel_event is not None and _cancel_event.is_set()


def check_cancel() -> None:
    """Raise ``CancelledRun`` if a cancel has been requested."""
    if cancel_requested():
        raise CancelledRun()
