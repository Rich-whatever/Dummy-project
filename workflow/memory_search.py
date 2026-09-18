"""
Agent-initiated memory search.

Backs the main agent's ``memory_search`` tool. Given one or more questions the
agent may search the long-term-memory (LTM) summary vector store and/or the
preference-note vector store. Every question is searched against every
requested source concurrently, results are de-duplicated, and everything is
packaged into a single text block that is fed back to the agent as a
``ToolMessage`` — the same report-back channel used by ``assign_task``.

Only vector search is used here (no LLM):

- LTM reuses :func:`workflow.ltm_retriever.retrieve_ltm` (score-gated).
- Preference notes reuse :meth:`PreferenceVectorStore.query_similar_note`.

The blocking Chroma calls are dispatched via ``asyncio.to_thread`` so all
queries across both sources run in parallel.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from langchain_core.documents import Document

from vector_store.client import MemoryVectorStore
from workflow.ltm_retriever import retrieve_ltm

logger = logging.getLogger("memory_search")


_NO_QUESTIONS = (
    "memory_search error: 'questions' must contain at least one non-empty "
    "question. Provide one or more short, targeted questions."
)
_NO_SOURCE = (
    "memory_search error: enable at least one of search_user_preference or "
    "search_long_term_memory."
)


# ── Formatting helpers ──────────────────────────────────────────────


def _format_ltm_hits(query: str, hits: list[tuple[Document, float]]) -> str:
    """Render the LTM hits for a single query as a text block."""
    lines = [f'[Query] "{query}"']
    if not hits:
        lines.append("  (no match)")
    for doc, score in hits:
        thread_id = doc.metadata.get("thread_id", "?")
        block = doc.metadata.get("block#", "?")
        lines.append(
            f"  - [thread={thread_id}, block#{block}, score={score:.3f}] "
            f"{doc.page_content}"
        )
    return "\n".join(lines)


def _format_pref_hits(query: str, hits: list[tuple[Document, float]]) -> str:
    """Render the preference-note hits for a single query as a text block."""
    lines = [f'[Query] "{query}"']
    if not hits:
        lines.append("  (no match)")
    for doc, score in hits:
        category = doc.metadata.get("category_name", "?")
        content = doc.metadata.get("note_content") or doc.page_content
        lines.append(f"  - [category={category}, score={score:.3f}] {content}")
    return "\n".join(lines)


def _dedupe_across(
    per_query: list[list[tuple[Document, float]]],
    key_fn: Callable[[Document], Any],
) -> list[list[tuple[Document, float]]]:
    """
    Drop duplicate documents across all queries, keeping the first occurrence.

    A document surfaced by an earlier query is not repeated under a later one,
    which keeps the report compact when questions overlap.
    """
    seen: set = set()
    out: list[list[tuple[Document, float]]] = []
    for hits in per_query:
        unique: list[tuple[Document, float]] = []
        for doc, score in hits:
            key = key_fn(doc)
            if key in seen:
                continue
            seen.add(key)
            unique.append((doc, score))
        out.append(unique)
    return out


def _ltm_key(doc: Document) -> Any:
    return (doc.metadata.get("thread_id"), doc.metadata.get("block#"))


def _pref_key(doc: Document) -> Any:
    return doc.metadata.get("note_id") or doc.page_content


# ── Per-source async search ─────────────────────────────────────────


async def _search_source(
    source_name: str,
    questions: list[str],
    search_fn: Callable[[str], list[tuple[Document, float]]],
) -> list[list[tuple[Document, float]]]:
    """
    Run ``search_fn`` for every question concurrently.

    Returns one hit-list per question, in question order. A failing query
    yields an empty list for that question rather than aborting the others, so
    a single source failure never loses the rest of the report.
    """
    gathered = await asyncio.gather(
        *[asyncio.to_thread(search_fn, q) for q in questions],
        return_exceptions=True,
    )
    per_query: list[list[tuple[Document, float]]] = []
    for query, result in zip(questions, gathered):
        if isinstance(result, BaseException):
            logger.error(
                "memory_search: %s query failed for %r: %s",
                source_name,
                query,
                result,
            )
            per_query.append([])
        else:
            per_query.append(result)
    return per_query


# ── Public entry point ──────────────────────────────────────────────


async def run_memory_search(
    questions: list[str],
    *,
    search_user_preference: bool,
    search_long_term_memory: bool,
    vector_store: MemoryVectorStore | None,
    preference_vector_store: Any | None,
) -> str:
    """
    Execute the main agent's ``memory_search`` request and return a text report.

    Parameters
    ----------
    questions : list[str]
        The questions to search for. Each is embedded and searched separately.
    search_user_preference : bool
        Search the preference-note vector store.
    search_long_term_memory : bool
        Search the LTM summary vector store.
    vector_store : MemoryVectorStore | None
        The LTM vector store. May be ``None`` (reported as unavailable).
    preference_vector_store : object | None
        The preference vector store (must expose ``query_similar_note``). May be
        ``None`` (reported as unavailable).

    Returns
    -------
    str
        A packaged report of all hits, or a short guidance/error string when
        the request is invalid or a requested store is unavailable.
    """
    cleaned = [
        q.strip()
        for q in (questions or [])
        if isinstance(q, str) and q.strip()
    ]
    if not cleaned:
        return _NO_QUESTIONS
    if not (search_user_preference or search_long_term_memory):
        return _NO_SOURCE

    # ── Dispatch both sources concurrently ──────────────────────────
    ltm_task = None
    pref_task = None
    if search_long_term_memory and vector_store is not None:
        ltm_task = _search_source(
            "ltm",
            cleaned,
            lambda q: retrieve_ltm(q, vector_store),
        )
    if search_user_preference and preference_vector_store is not None:
        pref_task = _search_source(
            "preference",
            cleaned,
            lambda q: preference_vector_store.query_similar_note(q),
        )

    results = await asyncio.gather(
        *[t for t in (ltm_task, pref_task) if t is not None],
        return_exceptions=True,
    )
    ltm_results: list[list[tuple[Document, float]]] | None = None
    pref_results: list[list[tuple[Document, float]]] | None = None

    idx = 0
    if ltm_task is not None:
        res = results[idx]
        idx += 1
        if isinstance(res, BaseException):
            logger.error("memory_search: LTM search failed: %s", res)
            res = [[] for _ in cleaned]
        ltm_results = res
    if pref_task is not None:
        res = results[idx]
        if isinstance(res, BaseException):
            logger.error("memory_search: preference search failed: %s", res)
            res = [[] for _ in cleaned]
        pref_results = res

    return _package_report(
        cleaned,
        pref_results=pref_results,
        ltm_results=ltm_results,
        search_user_preference=search_user_preference,
        search_long_term_memory=search_long_term_memory,
        preference_available=preference_vector_store is not None,
        ltm_available=vector_store is not None,
    )


def _package_report(
    questions: list[str],
    *,
    pref_results: list[list[tuple[Document, float]]] | None,
    ltm_results: list[list[tuple[Document, float]]] | None,
    search_user_preference: bool,
    search_long_term_memory: bool,
    preference_available: bool,
    ltm_available: bool,
) -> str:
    """Render the collected hits into the final agent-facing report string."""
    lines = ["=== Memory Search Results ==="]
    lines.append("Queries: " + " | ".join(f'"{q}"' for q in questions))
    lines.append(
        "Sources: "
        f"user_preference={'yes' if search_user_preference else 'no'}, "
        f"long_term_memory={'yes' if search_long_term_memory else 'no'}"
    )
    lines.append("")

    if search_user_preference:
        lines.append("--- User Preference (vector matches) ---")
        if not preference_available:
            lines.append("(preference vector store unavailable)")
        else:
            deduped = _dedupe_across(pref_results or [], _pref_key)
            if not any(deduped):
                lines.append("(no relevant user preference found)")
            else:
                for query, hits in zip(questions, deduped):
                    lines.append(_format_pref_hits(query, hits))
        lines.append("")

    if search_long_term_memory:
        lines.append("--- Long-Term Memory (summary blocks) ---")
        if not ltm_available:
            lines.append("(long-term memory store unavailable)")
        else:
            deduped = _dedupe_across(ltm_results or [], _ltm_key)
            if not any(deduped):
                lines.append("(no relevant long-term memory found)")
            else:
                for query, hits in zip(questions, deduped):
                    lines.append(_format_ltm_hits(query, hits))
        lines.append("")

    logger.info(
        "memory_search → queries=%d | preference=%s(%d) | ltm=%s(%d)",
        len(questions),
        "yes" if search_user_preference else "no",
        sum(len(h) for h in (pref_results or [])),
        "yes" if search_long_term_memory else "no",
        sum(len(h) for h in (ltm_results or [])),
    )
    return "\n".join(lines).rstrip() + "\n"

