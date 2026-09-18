"""
Workflow — LangGraph agentic pipeline for the memory system.

Sub-modules:
    llm              — LLM call abstraction (DeepSeek)
    router           — Track 1: Intelligent Router
    ltm_retriever    — Track 2: LTM vector store retrieval (score-gated)
    context_assembly — Post-processing: ID filter + prompt builder
    graph            — LangGraph state definition + execution graph

The ``graph`` sub-module is intentionally NOT imported here: it pulls in the
whole execution/preference/memory-backend subtree, which would create a
circular import when any of those modules import ``workflow.timing`` first.
Import it explicitly via ``from workflow.graph import ...``.
"""

from .context_assembly import assemble_context
from .llm import get_llm
from .ltm_retriever import retrieve_ltm
from .router import route_message

__all__ = [
    "get_llm",
    "route_message",
    "retrieve_ltm",
    "assemble_context",
]
