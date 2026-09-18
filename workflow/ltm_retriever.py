"""
Track 2 — LTM Retriever.

Executes **in parallel** with the Intelligent Router (Track 1).

Process:
    1. Query the ChromaDB vector store with the user message.
    2. Apply the score gate: keep only results with score < 1.0.
    3. Return at most 3 ``(Document, score)`` pairs.

The ID filter (which removes the current thread's own latest summary)
is handled in ``context_assembly.py`` after both tracks finish, because
it needs the router's output (``thread_id``) which isn't available
until Track 1 completes.
"""

from __future__ import annotations

from langchain_core.documents import Document

from config import LTM_RETRIEVE_K
from vector_store.client import MemoryVectorStore
from vector_store.operations import query_with_score


def retrieve_ltm(
    user_message: str,
    vector_store: MemoryVectorStore,
    k: int = LTM_RETRIEVE_K,
) -> list[tuple[Document, float]]:
    """
    Retrieve up to *k* summary blocks from LTM that are relevant to the
    user message, filtering by the score gate (score < 1.0).

    Returns
    -------
    list[tuple[Document, float]]
        ``(Document, score)`` pairs, sorted by score ascending (most
        relevant first).  May be empty if nothing passes the gate.
    """
    return query_with_score(
        store=vector_store,
        query=user_message,
        k=k,
        exclude_filter=None,  # ID filter is applied later in context_assembly
    )
