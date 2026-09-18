"""
High-level tools exposed to the main agent for its tool-calling ReAct loop.

The main agent no longer uses structured JSON output. Instead it is given three
tools and may call them freely:

- ``response(report)``      — completion signal: deliver the final reply.
- ``assign_task(context_explain, instruction, assigned_tool_category)``
                            — delegate a task to the worker execution loop.
- ``memory_search(questions, search_user_preference, search_long_term_memory)``
                            — on-demand vector search over the user-preference
                              notes and/or the long-term-memory summaries.

These tools are SCHEMA-ONLY: the function bodies are never actually executed.
``agent_node`` in ``workflow/graph.py`` intercepts the LLM's tool calls and
implements the real logic (running the worker loop for ``assign_task``, the
vector searches for ``memory_search``, capturing the reply for ``response``).
We define them as tools purely so the model sees the correct callable
signatures/descriptions via ``bind_tools``.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def response(report: str) -> str:
    """Use this tool to deliver your final reply to the user and complete this conversation round.

    Call this ONLY when you have finished what you want to do and want response to
    user directly. The ``report`` argument is the exact text of your reply.
    """
    # Schema-only stub — handled by agent_node.
    return report


@tool
def assign_task(
    context_explain: str,
    instruction: str,
    assigned_tool_category: str,
) -> str:
    """
    This tool allows you to delegate task to sub-worker agents.
    You don't directly access tools in tool system, but you can
    assign workers corresponding tool categories on tool map.
    Then the worker can use the tools in the assigned category,
    pursue the task and return its findings/progress to you so you can decide what to do next.
    This is how you're able to use tools or make execution as agent.

    Args:
        context_explain: Background context and the ultimate goal you are trying
            to accomplish. Helps the worker understand its scope and how much
            effort to invest.
        instruction: The concrete task description — what you want it to do, how to do it,
            what counts as complete, and what information should be return, etc (any instruction
            or guide should be in this field).
        assigned_tool_category: the tool category the worker will likely use.
            Must be one of the existing categories on tool map.
    """
    # Schema-only stub — the actual worker loop is run by agent_node.
    return ""


@tool
def memory_search(
    questions: list[str],
    search_user_preference: bool,
    search_long_term_memory: bool,
) -> str:
    """
    This tool allows you to query information about user and your long-term
    conversation history. Use it when you need to recall more information about user
    or relevant past conversation memory. If the search didn't return helpful info,
    it is more likely that there's no relevant info recorded, so another rephrased search is unliekly to help

    Args:
        questions: One or more short, targeted questions to search for. Keep
            this small and specific — each question will be searched independently
        search_user_preference: Set True to search recorded facts, tastes, and personal information about the user
        search_long_term_memory: Set True to search the long-term memory conversation summaries

    At least one of ``search_user_preference`` / ``search_long_term_memory``
    must be True. Prefer the fewest questions that fully cover what you need.
    """
    # Schema-only stub — the actual vector searches are run by agent_node.
    return ""
