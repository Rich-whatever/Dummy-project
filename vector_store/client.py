"""
MemoryVectorStore — the single ChromaDB client wrapper.

Initialises a persistent Chroma vector store using HuggingFace embeddings.
All vector operations in the memory system go through this class.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_COLLECTION_NAME, DB_DIR, EMBEDDING_MODEL_NAME


class MemoryVectorStore:
    """
    Wraps ChromaDB with a persistent directory and HuggingFace embeddings.

    Usage:
        store = MemoryVectorStore()
        store.upsert_summary(...)
        results = store.query_with_score(...)
    """

    def __init__(
        self,
        persist_directory: str = DB_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL_NAME,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Initialise embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Initialise Chroma vector store (creates persist_directory if needed).
        # We use cosine distance so scores < 1.0 indicate semantic relevance,
        # matching the spec's score gate requirement.
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        """Expose the underlying Chroma collection for direct metadata queries."""
        return self.vectorstore._collection

    def __repr__(self) -> str:
        return (
            f"MemoryVectorStore(collection='{self.collection_name}', "
            f"persist='{self.persist_directory}', "
            f"model='{self.embedding_model_name}')"
        )
