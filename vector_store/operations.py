"""
CRUD operations for the LTM (long-term memory) vector store.

All functions accept a MemoryVectorStore instance as their first argument,
keeping the client initialisation separate from business logic.
"""

from typing import Optional

from langchain_core.documents import Document

from config import LTM_RETRIEVE_K, LTM_SCORE_THRESHOLD

from .client import MemoryVectorStore


def upsert_summary(
    store: MemoryVectorStore,
    thread_id: str,
    block_number: int,
    summary_text: str,
    timestamp: str,
) -> str:
    """
    Upsert a summary block into the LTM vector store.

    Metadata stored:
        - thread_id   : links the summary to its origin thread
        - block#      : distinguishes multiple summaries with the same thread_id
        - timestamp   : when the summary was created
        - type        : always "summary_block"

    Returns the generated ID of the stored document.
    """
    doc = Document(
        page_content=summary_text,
        metadata={
            "thread_id": thread_id,
            "block#": block_number,
            "timestamp": timestamp,
            "type": "summary_block",
        },
    )
    ids = store.vectorstore.add_documents([doc])
    return ids[0] if ids else ""


def query_with_score(
    store: MemoryVectorStore,
    query: str,
    k: int = LTM_RETRIEVE_K,
    exclude_filter: Optional[dict] = None,
) -> list[tuple[Document, float]]:
    """
    Retrieve the top-k summary blocks matching *query*, filtered by score.

    Score gate:  only results with score < LTM_SCORE_THRESHOLD (1.0) are returned.

    ID filter (optional):
        Pass ``exclude_filter`` to skip results whose metadata matches certain
        key/value pairs.  This is used to exclude the current thread's own
        latest summary block so it doesn't provide redundant context.

        Example:  exclude_filter = {"thread_id": "abc", "block#": 3}

    Returns a list of (Document, score) tuples, sorted by score (ascending).
    The score is a cosine distance — *lower* means *more similar*.
    """
    # Step 1: raw similarity search
    raw_results = store.vectorstore.similarity_search_with_score(query, k=k)

    # Step 2: apply score gate
    gated = [(doc, score) for doc, score in raw_results if score < LTM_SCORE_THRESHOLD]

    # Step 3: apply ID filter (exclude specific metadata combinations)
    if exclude_filter is not None:
        filtered = []
        for doc, score in gated:
            match = all(doc.metadata.get(key) == value for key, value in exclude_filter.items())
            if not match:
                filtered.append((doc, score))
        gated = filtered

    return gated


def get_summary_by_id(
    store: MemoryVectorStore,
    thread_id: str,
    block_number: int,
) -> Optional[Document]:
    """
    Retrieve a specific summary block by its ``thread_id`` + ``block#``.

    ChromaDB requires ``$and`` syntax for compound metadata filters.
    Returns the Document if found, otherwise ``None``.
    """
    results = store.vectorstore.similarity_search(
        query="",            # dummy query — we filter by metadata only
        k=1,
        filter={
            "$and": [
                {"thread_id": thread_id},
                {"block#": block_number},
            ]
        },
    )
    return results[0] if results else None


def delete_summaries_by_thread(
    store: MemoryVectorStore,
    thread_id: str,
) -> int:
    """
    Delete ALL summary blocks belonging to a given ``thread_id``.

    Returns the number of deleted documents.
    """
    collection = store.collection
    existing = collection.get(where={"thread_id": thread_id})
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)
