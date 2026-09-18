"""
LangGraph State + Execution Graph for the memory pipeline.

Builds a state graph with these nodes:

    **router_node**            — Track 1: calls ``route_message()`` (async) to
                                 classify the user message into an existing
                                 thread or ``NEW_THREAD``.
    **ltm_node**               — Track 2: calls ``retrieve_ltm()`` to fetch
                                 relevant LTM summary blocks from ChromaDB.
    **preference_retrieve_node**  — retrieves relevant user preferences.
    **preference_injection_node** — fire-and-forget note/expectation injection.
    **assembly_node**          — post-processing: ID filter, fallback, builds
                                 the assembled context string.
    **agent_node**             — main agent: an async tool-calling ReAct loop
                                 with ``response``, ``assign_task`` and
                                 ``memory_search`` tools. ``assign_task``
                                 spawns a worker loop; ``memory_search`` runs
                                 on-demand vector searches over preference
                                 notes and/or long-term memory.
    **update_memory_node**     — persists the conversation round to the thread.

The router and ltm tracks run in parallel via LangGraph fan-out. After
assembly, the main agent (``agent_node``) runs a tool-calling loop; it calls
``assign_task`` to delegate work to a worker loop (see ``execution/worker.py``)
and ``response`` (or free text) to finish the round.

Node implementations live in:
    - ``memory_backend/memory_node.py``   — router_node, ltm_node, update_memory_node
    - ``preference_system/graph_nodes.py`` — preference_retrieve/injection nodes
    - ``workflow/context_assembly.py``     — assemble_context (imported here)
    - ``execution/worker.py``              — the worker loop (invoked by agent_node)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from memory_backend.memory_node import arouter_node, ltm_node, update_memory_node
from memory_backend.models import ThreadBar
from preference_system.graph_nodes import (
    preference_injection_node,
    preference_retrieve_node,
)
from text_utils import clip
from vector_store.client import MemoryVectorStore

from .context_assembly import SYSTEM_INSTRUCTIONS, assemble_context
from .router import NEW_THREAD
from .state import PipelineState
from .timing import timed_ainvoke

# ── Logger ───────────────────────────────────────────────────────────
logger = logging.getLogger("memory_pipeline")


# ── Nodes (defined locally: assembly + agent) ─


def assembly_node(state: PipelineState) -> dict:
    """
    LangGraph node for Context Assembly.

    Applies the ID filter, manages fallback, updates ``latest_ltm``,
    and builds the final ``(system_prompt, human_message)`` tuple.
    """
    system_prompt = assemble_context(
        thread_bar=state.thread_bar,
        vector_store=state.vector_store,
        router_result=state.router_result,
        ltm_results=state.ltm_results,
        thread_for_context=state.thread_for_context,
    )

    # ── Append retrieved preference + general behavior expectations ──
    # These are reference-only context; the agent decides whether to use them.
    if state.retrieved_preference:
        system_prompt += (
            "\n\n=== Retrieved User Preferences (reference only — may be "
            "irrelevant; rely on your own judgement) ===\n"
            f"{state.retrieved_preference}"
        )
    if state.general_behavior_expectation:
        system_prompt += (
            "\n\n=== General Behavior Expectations (guidelines to follow) ===\n"
            f"{state.general_behavior_expectation}"
        )

    # ── logging ─────────────────────────────────────────────────────
    tid = state.router_result[:8] if state.router_result != NEW_THREAD else "NEW"
    thread = (
        state.thread_bar.get_thread(state.router_result)
        if state.router_result != NEW_THREAD
        else None
    )
    has_bridge = "yes" if thread and thread.bridge_summary else "no"

    # Count LTM blocks in system prompt by inspecting header
    ltm_count = 0
    if "=== Long-Term Memory Context ===" in system_prompt:
        for line in system_prompt.split("\n"):
            if line.strip().startswith("  - [") and "score=" in line:
                ltm_count += 1

    stm_count = system_prompt.count("[USER]")
    logger.info(
        "ASSEMBLY → thread=%s… | LTM_blocks=%d | bridge=%s | STM_rounds=%d",
        tid,
        ltm_count,
        has_bridge,
        stm_count,
    )

    '''system_prompt +=(
        "\n\n=== Agent execution history ===\
        \nThe following records describe previous actions taken by you and the results returned by external systems.")'''
    #print(f"Assembled_Content\n{system_prompt}")

    # ── Current time / date / day / timezone at the very top of the message ──
    now = datetime.now().astimezone()
    tz = now.strftime("%z")  # e.g. "+0800"
    tz_label = f"UTC{tz[:3]}:{tz[3:]}" if tz else "UTC"
    time_line = (
        f"Current time: {now.strftime('%H:%M:%S')} | "
        f"Date: {now.strftime('%Y-%m-%d')} | "
        f"Day: {now.strftime('%A')} | "
        f"Timezone: {tz_label} |"
        "Use this for current time reference. This is the time you got invoked!"
    )

    system_msg = SystemMessage(
        content=f"{time_line}\n{SYSTEM_INSTRUCTIONS}\n\n{system_prompt}"
    )

    return {"messages": system_msg}


def human_message_node(state: PipelineState) -> dict:
    """
    LangGraph node: append the caller-built human message.

    Runs AFTER ``assembly_node`` so the agent receives the conventional order
    ``[SystemMessage, HumanMessage, ...]``. The content blocks (file content +
    user text) are prebuilt by the caller and carried in
    ``state.human_message_content``; when absent, falls back to a plain-text
    HumanMessage built from ``state.user_message``.
    """
    content = state.human_message_content or [
        {"type": "text", "text": state.user_message}
    ]
    return {"messages": HumanMessage(content=content)}


async def agent_node(state: PipelineState) -> dict:
    """
    Main-agent ReAct tool loop.

    The agent no longer uses structured JSON output. It is given three tools and
    outputs freely:

    - ``response(report)`` — completion signal; the report is the final reply.
    - ``assign_task(context_explain, instruction, assigned_tool_category)``
      — delegates a task to the worker loop, which runs and returns a report
        (or failure summary) that is fed back as a ``ToolMessage``.
    - ``memory_search(questions, search_user_preference, search_long_term_memory)``
      — runs on-demand vector searches over the preference notes and/or the
        long-term-memory summaries and feeds the packaged hits back as a
        ``ToolMessage``.

    The agent may call ``assign_task`` up to ``MAX_ASSIGN_TASKS`` times and
    ``memory_search`` up to ``MAX_MEMORY_SEARCH`` times, seeing each result
    before deciding its next action. It finishes by calling ``response`` or by
    emitting free text with no tool call.

    Returns ``decision="response"`` with ``agent_response``, the accumulated
    ``messages`` (incl. worker ``ToolMessage`` activity log), and
    ``underlying_content`` (the formatted activity log).
    """
    if state.llm is None:
        from .llm import get_llm

        llm = get_llm()
    else:
        llm = state.llm
    from config import MAX_ASSIGN_TASKS, MAX_MEMORY_SEARCH
    from execution.worker import run_worker_task
    from tool.agent_tools import assign_task, memory_search, response
    from tool.toolkit_registry import TOOL_CATEGORY_NAMES
    from workflow.memory_search import run_memory_search

    agent_start = time.perf_counter()
    messages = list(state.messages)
    agent_llm = llm.bind_tools([response, assign_task, memory_search])

    assign_count = 0
    memory_search_count = 0
    activity_parts: list[str] = []
    over_limit_told = False

    def _finish(agent_response: str) -> dict:
        return {
            "agent_response": agent_response,
            "messages": messages,
            "underlying_content": (
                "\n\n".join(activity_parts) if activity_parts else None
            ),
        }

    while True:
        from workflow.run_control import check_cancel

        check_cancel()  # Stop here if the user hit Cancel (turn boundary)
        ai = await timed_ainvoke("AgentToolLoop", agent_llm.ainvoke, messages)
        messages = messages + [ai]

        tool_calls = getattr(ai, "tool_calls", [])

        # ── Log this agent turn: its notes + the tool commands it issued ──
        ai_text = str(getattr(ai, "content", "") or "").strip()
        if ai_text:
            logger.info("AGENT → note: %s", clip(ai_text))
        if tool_calls:
            if ai_text:
                activity_parts.append(f"[Agent Note] {ai_text}")
            for _tc in tool_calls:
                _name = _tc.get("name")
                _a = _tc.get("args") or {}
                if _name == "assign_task":
                    instruction = clip(_a.get("instruction", ""), 800)
                    context_explain = clip(_a.get("context_explain", ""), 500)
                    logger.info(
                        "AGENT → tool assign_task: category=%s | instruction: %s | context: %s",
                        _a.get("assigned_tool_category"),
                        instruction,
                        context_explain,
                    )
                    append = f"[Agent Assign task] | assigned tool category: {_a.get("assigned_tool_category")}\
                                instruction: {instruction}"
                    activity_parts.append(append)
                elif _name == "memory_search":
                    logger.info(
                        "AGENT → tool memory_search: preference=%s | ltm=%s | questions: %s",
                        _a.get("search_user_preference"),
                        _a.get("search_long_term_memory"),
                        clip(_a.get("questions", ""), 500),
                    )
                    append = (
                        "[Agent Memory Search] | preference: "
                        f"{_a.get('search_user_preference')} | ltm: "
                        f"{_a.get('search_long_term_memory')} | questions: "
                        f"{clip(_a.get('questions', ''), 300)}"
                    )
                    activity_parts.append(append)
                else:
                    logger.info(
                        "AGENT → tool %s: %s",
                        _name,
                        clip(_a),
                    )

        # Free text with NO tool call → the response.
        if not tool_calls:
            agent_response = str(getattr(ai, "content", "") or "").strip()
            logger.info(
                "AGENT → response (free text) | took %.2fs | %s",
                time.perf_counter() - agent_start,
                clip(agent_response),
            )
            return _finish(agent_response)

        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}
            tc_id = tc.get("id") or "unknown"

            if name == "response":
                agent_response = str(args.get("report", "")).strip()
                logger.info(
                    "AGENT → response (tool) | took %.2fs | %s",
                    time.perf_counter() - agent_start,
                    clip(agent_response),
                )
                return _finish(agent_response)

            if name == "assign_task":
                if assign_count >= MAX_ASSIGN_TASKS:
                    if not over_limit_told:
                        over_limit_told = True
                        messages = messages + [ToolMessage(
                            content=(
                                f"You have hit the assign task limit ({MAX_ASSIGN_TASKS} attempts) for this round of conversation. "
                                "You're highly recommended to response to user and explain progress, failure, what you been doing." \
                                "Basically response, they have being waited long enough."
                            ),
                            tool_call_id=tc_id,
                            name="assign_task",
                        )]
                category = str(args.get("assigned_tool_category", "")).strip()
                if category and category not in TOOL_CATEGORY_NAMES:
                    messages = messages + [ToolMessage(
                        content=(
                            f"Invalid assigned_tool_category '{category}'. "
                            f"Valid categories: {TOOL_CATEGORY_NAMES}. "
                            "Choose a valid one."
                        ),
                        tool_call_id=tc_id,
                        name="assign_task",
                    )]
                    continue

                assign_count += 1
                from workflow.run_control import check_cancel

                check_cancel()  # Don't start a new worker if the user cancelled
                status, report = await run_worker_task(
                    instruction=str(args.get("instruction", "")),
                    context_explain=str(args.get("context_explain", "")),
                    assigned_tool_category=category,
                    categorized_toolkit=state.categorized_toolkit,
                )
                activity_parts.append(report)
                messages = messages + [ToolMessage(
                    content=report,
                    tool_call_id=tc_id,
                    name="assign_task",
                )]
                logger.info(
                    "AGENT → assign_task #%d (%s) → %s | took %.2fs",
                    assign_count,
                    category,
                    report,
                    time.perf_counter() - agent_start,
                )

            if name == "memory_search":
                if memory_search_count >= MAX_MEMORY_SEARCH:
                    messages = messages + [ToolMessage(
                        content=(
                            f"You have hit the memory search limit "
                            f"({MAX_MEMORY_SEARCH} searches) for this round of "
                            "conversation. Work with what you already have and "
                            "respond to the user."
                        ),
                        tool_call_id=tc_id,
                        name="memory_search",
                    )]
                    continue

                raw_questions = args.get("questions") or []
                if isinstance(raw_questions, str):
                    raw_questions = [raw_questions]
                search_pref = bool(args.get("search_user_preference", False))
                search_ltm = bool(args.get("search_long_term_memory", False))

                memory_search_count += 1
                from workflow.run_control import check_cancel

                check_cancel()  # Don't start a search if the user cancelled
                report = await run_memory_search(
                    questions=raw_questions,
                    search_user_preference=search_pref,
                    search_long_term_memory=search_ltm,
                    vector_store=state.vector_store,
                    preference_vector_store=state.preference_vector_store,
                )
                messages = messages + [ToolMessage(
                    content=report,
                    tool_call_id=tc_id,
                    name="memory_search",
                )]
                logger.info(
                    "AGENT → memory_search #%d (pref=%s, ltm=%s, %d question(s)) | took %.2fs",
                    memory_search_count,
                    search_pref,
                    search_ltm,
                    len(raw_questions),
                    time.perf_counter() - agent_start,
                )

        # Absolute safety guard against an infinite tool loop.
        ai_count = sum(
            1 for m in messages if getattr(m, "type", "") == "ai"
        )
        if ai_count > MAX_ASSIGN_TASKS + MAX_MEMORY_SEARCH + 3:
            logger.warning("AGENT → loop guard hit; forcing response")
            return _finish(
                "I've hit my task limit for this round. "
                "Here is what I have so far."
            )


def build_pipeline_graph() -> StateGraph:
    """
    Build and return the LangGraph ``StateGraph`` for the memory
    pipeline.

    Graph structure::

        [START]
           |
           ├── router_node ── preference_retrieve_node ──┐
           │                └─ preference_injection_node → END
           ├── ltm_node ─────────────────────────────────┤
           └─────────────────────────────────────────────┘
                                   |
                              assembly_node
                                   |
                              agent_node
                                   |
                              update_memory_node
                                   |
                                 [END]

    ``router_node`` and ``ltm_node`` run in parallel via LangGraph fan-out.
    After assembly, ``agent_node`` runs a tool-calling ReAct loop — it may
    delegate tasks to the worker loop (via ``assign_task``) and always ends by
    producing a response. ``update_memory_node`` then persists the round.
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("router_node", arouter_node)
    graph.add_node("ltm_node", ltm_node)
    graph.add_node("preference_retrieve_node", preference_retrieve_node)
    graph.add_node("preference_injection_node", preference_injection_node)
    graph.add_node("assembly_node", assembly_node)
    graph.add_node("human_message_node", human_message_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("update_memory_node", update_memory_node)

    # Edges: start → router + ltm (parallel fan-out)
    graph.add_edge(START, "router_node")
    graph.add_edge("router_node", "ltm_node")

    # Router → preference retrieve + injection (parallel)
    graph.add_edge("router_node", "preference_retrieve_node")
    graph.add_edge("router_node", "preference_injection_node")

    # Retrieve → assembly; injection is fire-and-forget → END
    graph.add_edge("preference_retrieve_node", "assembly_node")
    graph.add_edge("preference_injection_node", END)

    # Both tracks → assembly
    graph.add_edge("ltm_node", "assembly_node")
    # assembly (system message) → human message → agent, so the agent sees
    # [SystemMessage, HumanMessage, ...] in the conventional order.
    graph.add_edge("assembly_node", "human_message_node")
    graph.add_edge("human_message_node", "agent_node")
    # agent_node always resolves to a response → update_memory_node
    graph.add_edge("agent_node", "update_memory_node")
    graph.add_edge("update_memory_node", END)

    return graph


# ── Convenience Runner ─────────────────────────────────────────────


async def run_pipeline(
    user_message: str,
    thread_bar: ThreadBar,
    vector_store: MemoryVectorStore,
    categorized_toolkit: dict,
    llm: BaseChatModel | None = None,
    preference_store: object | None = None,
    preference_vector_store: object | None = None,
    human_message_content: list[dict] | None = None,
) -> dict:
    """
    One-shot convenience function: build the graph, run it, return
    the full results dict.

    This is the **main entry point** for the full Phase 3 pipeline.

    Parameters
    ----------
    user_message : str
        The user's input.
    thread_bar : ThreadBar
        Active threads container.
    vector_store : MemoryVectorStore
        Long-term memory vector store.
    categorized_toolkit : dict
        Mapping of tool category name → list of tool instances, made
        available to the main agent (for ``assign_task``) and workers.
    llm : BaseChatModel, optional
        Override the default LLM (used by the main agent and, via
        ``assign_task``, the worker loops).
    preference_store : object, optional
        The preference system's ``PreferenceStore`` (states + categories).
        When provided, the preference retrieve/injection nodes run.
    preference_vector_store : object, optional
        The preference system's ``PreferenceVectorStore`` (Chroma-backed
        notes), used by the preference injection node.
    human_message_content : list[dict], optional
        Prebuilt human-message content blocks (file content + user text),
        appended as the HumanMessage after the system message.

    Returns
    -------
    dict
        A dictionary with keys:
            - "user_message": str
            - "agent_response": str
            - "underlying_content": str | None
            - "router_result": str
            - "thread_was_evicted": str
            - "retrieved_preference": str
            - "general_behavior_expectation": str
    """
    graph = build_pipeline_graph()
    compiled = graph.compile()

    initial_state = PipelineState(
        user_message=user_message,
        human_message_content=human_message_content,
        thread_bar=thread_bar,
        vector_store=vector_store,
        llm=llm,
        categorized_toolkit = categorized_toolkit,
        preference_store=preference_store,
        preference_vector_store=preference_vector_store,
    )

    final_state = await compiled.ainvoke(initial_state)
    return {
        "user_message": final_state["user_message"],
        "agent_response": final_state["agent_response"],
        "underlying_content": final_state.get("underlying_content"),
        "router_result": final_state["router_result"],
        "thread_was_evicted": final_state["thread_was_evicted"],
        "retrieved_preference": final_state.get("retrieved_preference", ""),
        "general_behavior_expectation": final_state.get(
            "general_behavior_expectation", ""
        ),
    }
