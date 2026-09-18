"""
PreferenceVectorStore — ChromaDB wrapper for note-level preference vectors.

Mirrors ``MemoryVectorStore`` (``vector_store/client.py``) but stores individual
preference notes instead of LTM thread summaries.

Usage::

    store = PreferenceVectorStore()
    note_id = store.upsert_note("User prefers concise answers", "communication_style")
    results = store.query_similar_note("concise answers please")
"""

from __future__ import annotations

import logging
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL_NAME,
    PREFERENCE_COLLECTION_NAME,
    PREFERENCE_DB_DIR,
    PREFERENCE_NOTE_QUERY_K,
)

logger = logging.getLogger("preference_system.vector_store")


class PreferenceVectorStore:
    """
    Wraps ChromaDB holding preference notes as vectors.

    Notes are stored as single documents with metadata:
        - ``note_content``: the plain-text note string
        - ``category_name``: the category this note belongs to
        - ``note_id``: the unique id returned by :meth:`upsert_note`
        - ``type``: always ``"preference_note"``

    Cosine distance is used, matching the memory vector store, so lower scores
    mean higher similarity.
    """

    def __init__(
        self,
        persist_directory: str = PREFERENCE_DB_DIR,
        collection_name: str = PREFERENCE_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        client: object | None = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        kwargs: dict = {
            "collection_name": collection_name,
            "embedding_function": self.embeddings,
            "collection_metadata": {"hnsw:space": "cosine"},
        }
        if client is not None:
            kwargs["client"] = client
        else:
            kwargs["persist_directory"] = persist_directory

        self.vectorstore = Chroma(**kwargs)
        logger.debug(
            "Initialised PreferenceVectorStore(collection='%s', model='%s')",
            collection_name,
            embedding_model,
        )

    @property
    def collection(self):
        """Expose the underlying Chroma collection for direct metadata queries."""
        return self.vectorstore._collection

    def upsert_note(
        self,
        note_content: str,
        category_name: str,
        note_id: str | None = None,
    ) -> str:
        """Embed and store a single preference note. Returns the note_id.

        If ``note_id`` is omitted, a new UUID is generated. Pass an explicit
        ``note_id`` when the caller keeps the same id in another store so the
        two stay in sync.
        """
        if note_id is None:
            note_id = uuid.uuid4().hex
        self.vectorstore.add_texts(
            texts=[note_content],
            metadatas=[
                {
                    "note_content": note_content,
                    "category_name": category_name,
                    "note_id": note_id,
                    "type": "preference_note",
                }
            ],
            ids=[note_id],
        )
        logger.debug("Upserted preference note %s into '%s'", note_id, category_name)
        return note_id

    def query_similar_note(
        self,
        note_content: str,
        k: int = PREFERENCE_NOTE_QUERY_K,
    ) -> list[tuple[Document, float]]:
        """
        Return up to ``k`` stored notes most similar to ``note_content``.

        Each result is a ``(Document, score)`` pair where ``score`` is a cosine
        distance (lower = more similar). No score gate is applied here — the
        caller (dedupe logic) decides what threshold is acceptable.
        """
        return self.vectorstore.similarity_search_with_score(note_content, k=k)

    def clear(self) -> int:
        """
        Delete ALL documents from the preference-note collection.

        Returns the number of documents removed (0 if the collection was
        already empty or the delete failed).
        """
        try:
            ids = self.collection.get()["ids"]
        except Exception:
            logger.exception(
                "Failed to read ids from preference collection '%s' during clear",
                self.collection_name,
            )
            return 0
        if ids:
            try:
                self.vectorstore.delete(ids=ids)
            except Exception:
                logger.exception(
                    "Failed to clear preference collection '%s' (%d docs)",
                    self.collection_name,
                    len(ids),
                )
                return 0
            logger.info(
                "Cleared %d note(s) from preference vector store '%s'",
                len(ids),
                self.collection_name,
            )
        else:
            logger.info("Preference vector store '%s' already empty", self.collection_name)
        return len(ids)

    def __repr__(self) -> str:
        return (
            f"PreferenceVectorStore(collection='{self.collection_name}', "
            f"persist='{self.persist_directory}', "
            f"model='{self.embedding_model_name}')"
        )
