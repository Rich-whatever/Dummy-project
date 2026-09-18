"""
Pipeline state definition for the memory pipeline graph.

``PipelineState`` is extracted into its own module to avoid circular
imports when ``graph.py`` imports node implementations from
``memory_backend.memory_node``, which in turn needs to reference the
state type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from memory_backend.models import ThreadBar
from vector_store.client import MemoryVectorStore


@dataclass
class PipelineState:
    """
    Mutable state passed between graph nodes.

    Attributes
    ----------
    user_message : str
        The original user input that triggered the pipeline.
    thread_bar : ThreadBar
        The active threads container (mutated in-place by various nodes).
    vector_store : MemoryVectorStore
        ChromaDB-backed vector store for LTM retrieval.
    llm : BaseChatModel | None
        Optional LLM instance (defaults to module singleton).
    router_result : str
        Output of the router node — either a ``thread_id`` or ``NEW_THREAD``.
    ltm_results : list[tuple[Document, float]]
        Output of the LTM retriever node.
    messages : list[BaseMessage]
        The accumulated LangChain message list (system + user + assistant).
        Managed via LangGraph's ``add_messages`` reducer.
    underlying_content : str | None
        The worker activity log (assigned tasks + reports) produced by the
        main agent during the round; stored for cross-thread/STM context.
    thread_was_evicted : str
        Set by update_memory_node if a thread was evicted (for logging).
    execution_path : str
        (reserved) Router execution-path field; currently unused.
    thread_for_context : list[str]
        Additional thread IDs whose context should be fetched (cross-thread
        reference).  Populated by the router when the LLM outputs the
        ``| context:`` suffix.
    agent_response : str
        The main agent's final response to the user.
    """

    user_message: str = ""
    # Prebuilt human-message content blocks (file content + user text), built by
    # the caller. Appended as a HumanMessage after the system message by
    # ``human_message_node``. None → fall back to a plain-text HumanMessage.
    human_message_content: list[dict] | None = None
    messages: Annotated[list[BaseMessage], add_messages] = field(
        default_factory=list
    )
    thread_bar: ThreadBar = field(default_factory=ThreadBar)
    vector_store: MemoryVectorStore = field(default_factory=lambda: None)  # type: ignore[arg-type]
    llm: BaseChatModel | None = None
    preference_store: object | None = None
    preference_vector_store: object | None = None

    # Track outputs (set by nodes)
    router_result: str = ""
    execution_path: str = ""
    thread_for_context: list[str] = field(default_factory=list)
    ltm_results: list[tuple[Document, float]] = field(default_factory=list)
    # Whether the preference-retrieve should include the last round of the
    # target thread. Set by the async router node (len heuristic + small-LLM
    # probe racing the router; default True when the probe is not ready).
    retrieve_with_last_round: bool = True
    # Retrieved preference context block (set by preference_retrieve_node).
    retrieved_preference: str = ""
    # General behavior expectations (Always-trigger rules from the behavior
    # guide file), assembled for the agent prompt.
    general_behavior_expectation: str = ""

    # Generation output
    agent_response: str = ""
    underlying_content: str | None = None

    # Memory management logging
    thread_was_evicted: str = ""
    categorized_toolkit: dict[str, list] = field(default_factory=dict)
