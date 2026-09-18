"""
LLM latency tracking — simple function wrappers (no proxy classes).

Every structured/plain LLM ``invoke``/``ainvoke`` call site is wrapped with
``timed_invoke`` / ``timed_ainvoke`` so the elapsed wall-clock time of each LLM
response is recorded into a registry (used by the end-of-round latency table in
``main.py``).  Per-node durations are surfaced on each node's own log line
rather than a separate ``latency`` logger.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable

# (label, seconds) entries for the current conversation round.
_timings: list[tuple[str, float]] = []


def reset_timings() -> None:
    """Clear the timing registry (call once at the start of a round)."""
    _timings.clear()


def record_timing(label: str, seconds: float) -> None:
    """Append a single LLM call duration to the registry."""
    _timings.append((label, seconds))


def timed_invoke(label: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Wrap a synchronous LLM invoke: time it, record it, return the result.

    Parameters
    ----------
    label : str
        Human-readable name for this LLM call (usually the schema class name).
    func : Callable
        The bound ``.invoke`` method (or any callable to time).
    """
    t0 = time.perf_counter()
    try:
        return func(*args, **kwargs)
    finally:
        record_timing(label, time.perf_counter() - t0)


async def timed_ainvoke(label: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Wrap an async LLM invoke (``.ainvoke``): time it, record it."""
    t0 = time.perf_counter()
    try:
        return await func(*args, **kwargs)
    finally:
        record_timing(label, time.perf_counter() - t0)


def print_latency_table(total_seconds: float | None = None) -> None:
    """
    Print the aggregated latency table for the completed conversation round.

    One row per LLM call label (call count, total, average), followed by the
    combined LLM time and — when ``total_seconds`` is given — the end-to-end
    time from user input to the final agent response.
    """
    if not _timings:
        print("\n  ── LLM Latency (this round) ──")
        print("  (no LLM calls recorded)")
        return

    per_label: dict[str, list[float]] = defaultdict(list)
    for label, seconds in _timings:
        per_label[label].append(seconds)

    # Build rows: (label, calls, total, avg)
    rows: list[tuple[str, int | None, float | None, float | None]] = []
    for label in sorted(per_label):
        durations = per_label[label]
        rows.append(
            (label, len(durations), sum(durations), sum(durations) / len(durations))
        )

    llm_call_count = sum(len(v) for v in per_label.values())
    llm_total = sum(sum(v) for v in per_label.values())
    rows.append(("LLM total", llm_call_count, llm_total, None))
    if total_seconds is not None:
        rows.append(("Total (input → response)", None, total_seconds, None))

    w_label = max(len("LLM call"), *(len(r[0]) for r in rows))
    w_calls = max(len("calls"), *(len(str(r[1])) for r in rows if r[1] is not None))
    w_total = max(len("total (s)"), *(len(f"{r[2]:.2f}") for r in rows if r[2] is not None))
    w_avg = len("avg (s)")
    widths = (w_label, w_calls, w_total, w_avg)
    aligns = ("<", ">", ">", ">")

    def sep() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(cells: tuple[str, str, str, str]) -> str:
        parts = [
            f"{cell:{align}{width}}"
            for cell, width, align in zip(cells, widths, aligns)
        ]
        return "| " + " | ".join(parts) + " |"

    print("\n  ── LLM Latency (this round) ──")
    print(sep())
    print(fmt_row(("LLM call", "calls", "total (s)", "avg (s)")))
    print(sep())
    for label, n, total, avg in rows:
        n_s = "—" if n is None else str(n)
        avg_s = "—" if avg is None else f"{avg:.2f}"
        print(fmt_row((label, n_s, f"{total:.2f}", avg_s)))
    print(sep())
