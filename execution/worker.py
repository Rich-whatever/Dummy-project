"""
Worker Agent — a LangGraph-based tool-calling ReAct loop that executes a task.

The worker runs an observe → think → act cycle using a category's tools, plus
the always-available ``report`` tool. The loop terminates in one of two ways:

1. **Report** — the worker calls ``report(...)`` to hand a report back to
   Main_agent: either a finished deliverable, or an explanation of why it
   could not finish (blocked, impossible, or confusing). Free text with no
   tool call is treated as a report too.

2. **Failure** — the number of tool invocations reaches ``round_limit``.
   ``failure_summary_node`` records the failure (status → ``"failed"``).

The graph structure::

    [START] → llm_node
                  │
                  ▼  (conditional)
             ┌────┴──────┐
        tool │          report
             ▼             ▼
        tool_node   report_node
             │           │
             ▼           ▼
        llm_node ←┘    [END]
             │
      failure │
             ▼
    failure_summary_node
             │
             ▼
           [END]
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Annotated, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import TOOLMESSAGE_LIMIT, WORKER_ROUND_LIMIT
from text_utils import clip
from workflow.timing import timed_invoke

# ── Logger ───────────────────────────────────────────────────────────
logger = logging.getLogger("worker")


def _log_tool_result(category: str | None, tool_round: int, tool_name: str, content: str) -> None:
    """Log a worker tool result as a clean, clipped block.

    ``SafeShellTool``-style content contains ``[STDOUT]`` / ``[STDERR]`` /
    ``[EXIT CODE]`` markers; when present those are surfaced separately so the
    worker's actual side output is legible.
    """
    content = content or ""
    header = f"Worker[{category}] | round:{tool_round} | tool '{tool_name}' result:"

    if "[STDOUT]" in content or "[STDERR]" in content or "[EXIT CODE]" in content:
        # Shell-style output — show stdout, stderr, and exit code separately.
        def _section(marker: str) -> str | None:
            start = content.find(marker)
            if start == -1:
                return None
            start += len(marker)
            end_markers = ["[STDOUT]", "[STDERR]", "[EXIT CODE]"]
            end = len(content)
            for m in end_markers:
                p = content.find(m, start)
                if p != -1 and p < end:
                    end = p
            return content[start:end].strip()

        parts = [header]
        for marker in ("[STDOUT]", "[STDERR]"):
            sec = _section(marker)
            if sec:
                parts.append(f"  {marker.strip('[]')}:")
                parts.append(f"    {clip(sec)}")
        ex = _section("[EXIT CODE]")
        if ex is not None:
            parts.append(f"  exit code: {ex.splitlines()[0] if ex else ''}")
        logger.info("\n".join(parts))
        return

    clipped = clip(content)
    if clipped:
        logger.info("%s\n%s", header, clipped)
    else:
        logger.info("%s (no output)", header)


# ── Report Tool (always available, schema-only) ──────────────────────
# The worker hands its result back to Main_agent by calling ``report`` rather
# than returning an empty response. ``report_node`` intercepts the call and
# captures the report — the body here is never executed.


@tool
def report(report: str) -> str:
    """Hand a report back to Main_agent and end this task.

    Call this when you have a result to deliver — whether you finished the
    task, hit an impossible/blocked situation, or the task is confusing and you
    need to return control. ``report`` should concisely say what you did and
    the outcome (or explain why you could not complete it).
    """
    # Schema-only stub — handled by report_node.
    return report


# ── Tool Category Resolver ────────────────────────────────────────────


def get_tools_for_category(
    categorized_toolkit: dict[str, list],
    category: str | None,
) -> list:
    """
    Resolve a tool category name to its list of tool objects.

    Parameters
    ----------
    categorized_toolkit : dict[str, list]
        Mapping of category name → list of tool instances
        (e.g. ``{"github": [...], "filesystem": [...]}``).
    category : str | None
        The predicted tool category for this step.

    Returns
    -------
    list
        The tool list for the given category, or ``[]`` if
        the category is ``None`` or not found.
    """
    if category is None:
        return []
    tools = categorized_toolkit.get(category, [])
    logger.debug("TOOLS  → get_tools_for_category(%s)  → %d tools", category, len(tools))
    return tools


# ── State Definition ─────────────────────────────────────────────────


@dataclass
class WorkerState:
    """
    Mutable state for a single Worker Agent ReAct loop.

    The first nine fields mirror ``PlanStep`` fields and are treated as
    immutable during execution.  The remaining fields track the ReAct loop
    progress and final output.
    """

    # ── From PlanStep (immutable during execution) ──────────────────
    task: str = ""
    predicted_tool_category: Optional[str] = None
    """Summaries from predecessor steps this worker depends on."""
    instruction: Optional[str] = None


    # ── ReAct loop state ──────────────────────────────────────────
    round: int = 0
    """Number of LLM invocations so far.  Incremented by ``llm_node``."""

    round_limit: int = field(default_factory=lambda: WORKER_ROUND_LIMIT)
    """Maximum tool invocations before the worker is forced to fail."""

    status: str = "running"
    """Current status: ``"running"`` → ``"success"`` | ``"failed"``."""

    messages: Annotated[list[BaseMessage], add_messages] = field(
        default_factory=list
    )
    """
    Conversation history for the ReAct loop.
    LangGraph's ``add_messages`` reducer merges/updates messages by ID.
    """

    llm: Optional[BaseChatModel] = None
    """LLM instance for this worker.  Falls back to the module singleton."""

    categorized_toolkit: dict[str, list] = field(default_factory=dict)
    """Mapping of category name → list of tool instances for this step."""

    tools: list | None = None
    """Tools bound for this worker round (category tools + ``report``).
    Set by ``llm_node`` and consumed by ``tool_node``."""

    # ── Outputs (set on termination) ─────────────────────────
    report: str = ""     # set by report_node or failure_summary_node



# ── Prompt Builder ───────────────────────────────────────────────────


# Single, readable system-prompt template for the worker. The static rules
# live here verbatim; the per-task "context" and "instruction" are injected
# at the placeholders when a worker initializes. This keeps the exact prompt
# easy to visualize and inspect in one place.
#
# Placeholders are substituted with str.replace() (not str.format()) so that
# task text containing literal braces can never break formatting.
WORKER_SYSTEM_PROMPT_TEMPLATE = """\
You are a worker helping Main_agent accomplish a task. Use the tools available to you to complete the assigned instruction.

You are given:
context: explains why your work matters
Instruction: explain what to do and some guidance

For every round, you must output beside calling tool to record your progress, thoughts, and reflect.

In first round of tool calling, you must define a short task completion criteria and a general outline / execution plan. output them along with your tool call

After receiving ToolMessage/results, briefly analyze progress or information, reflect on whether task completion criteria is met, and decide whether you are ready to hand a report back to Main_agent. Once you have a deliverable — or once you conclude the task cannot be completed (blocked, impossible, or confusing) — call the `report` tool with your findings.
and what you are doing next and why. Keep these notes short; record decisions, findings, or next steps, not a reasoning transcript.

If a tool keeps failing (return nothing or error), try different approaches/tools. Do not repeatedly retry the same broken tool, and remember to mention the tool error in your report.

Do not confirm, expand, or explore beyond what is required by the instruction. Stay focused on your assigned task and use the minimum work necessary. The overall context is not an instruction to take over the larger task. When you cannot complete the task, stop and report the blocker instead of guessing or doing unrequested work.

When reporting, include only information relevant to the instruction. Try to keep the report short and straightforward. It can be long if there's lots of important content.

=== Context ===
{context_explain}

=== Instruction ===
{instruction}
"""


def initialize_worker(state: WorkerState) -> dict:
    """
    Called once when the worker starts. Builds the initial message list.
    llm_node takes over from here using the message history directly.
    """
    context_explain = state.instruction or ""
    instruction = state.task or ""

    prompt = (
        WORKER_SYSTEM_PROMPT_TEMPLATE
        .replace("{context_explain}", context_explain)
        .replace("{instruction}", instruction)
    )

    return {"messages": SystemMessage(content=prompt)}


def llm_node(state: WorkerState) -> dict:
    """
    LangGraph node: invoke the LLM using existing message history.
    Does not rebuild prompts — initialization already handled that.
    """

    from workflow.run_control import check_cancel

    check_cancel()

    category_tools = get_tools_for_category(
        state.categorized_toolkit, state.predicted_tool_category
    )
    # The worker ALWAYS has `report` available regardless of category.
    tools = list(category_tools) + [report]
    llm_with_tools = state.llm.bind_tools(tools)

    response = timed_invoke(
        "WorkerToolLoop",
        llm_with_tools.invoke,
        state.messages,
    )

    usage = getattr(response, "usage_metadata", {}) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    token_str = f" | in:{input_tokens/1000:.1f}k out:{output_tokens/1000:.1f}k"

    tool_calls = getattr(response, "tool_calls", [])

    # Log the worker's side output (narrative text written alongside tool calls).
    side_output = str(getattr(response, "content", "") or "").strip()
    if side_output:
        logger.info(
            "Worker[%s] | round:%d  | note: %s",
            state.predicted_tool_category,
            state.round + 1,
            clip(side_output),
        )

    if tool_calls:
        simplified = []
        for tc in tool_calls:
            args = tc.get("args", {})
            if isinstance(args, dict):
                clipped_args = {k: clip(v) for k, v in args.items()}
            else:
                clipped_args = clip(args)
            simplified.append({tc.get("name"): clipped_args})
        logger.info(
            "Worker[%s] | round:%d  | tool_calls: %s%s",
            state.predicted_tool_category,
            state.round+1,
            simplified,
            token_str,
        )

    return {
        "messages": [response],
        "tools": tools,
    }




async def tool_node(state: WorkerState) -> dict:
    """Execute tool calls from the last message."""
    from workflow.run_control import check_cancel

    check_cancel()

    if not state.tools or not state.messages:
        return {"messages": []}

    last_message = state.messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    try:
        node = ToolNode(state.tools)
        result = await node.ainvoke({"messages": [last_message]})
    except Exception as e:
        # MCP tool errors (e.g. access denied, path not allowed) are caught here
        # and returned as ToolMessage error content so the LLM can retry or adapt.
        error_content = (
            f"❌ Tool execution error: {e}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        # Build a synthetic ToolMessage matching each tool_call so LangGraph
        # can route it back into the message history properly.
        synthetic_messages = []
        for tc in tool_calls:
            synthetic_messages.append(
                ToolMessage(
                    content=error_content,
                    tool_call_id=tc.get("id", "unknown"),
                    name=tc.get("name", "unknown"),
                )
            )
        return {"messages": synthetic_messages, "round": state.round + 1}

    # Truncate oversized tool response messages to prevent token bloat
    truncated_messages = []
    for msg in result.get("messages", []):
        content = str(getattr(msg, "content", ""))
        if len(content) > TOOLMESSAGE_LIMIT:
            truncated_content = (
                f"[TRUNCATED] Tool response too large: length={len(content)} "
                f"exceeded limit={TOOLMESSAGE_LIMIT}. "
                "Try a different approach."
            )
            msg.content = truncated_content
        truncated_messages.append(msg)

    # Log results after execution (clean, clipped, stdout/stderr separated).
    '''for msg in truncated_messages:
        _log_tool_result(
            state.predicted_tool_category,
            state.round + 1,
            getattr(msg, "name", "unknown"),
            str(getattr(msg, "content", "")),
        )'''

    return {**result, "messages": truncated_messages, "round": state.round + 1}


def report_node(state: WorkerState) -> dict:
    """
    LangGraph node: handle a ``report`` tool call (or free text) and mark the
    worker as done — WITHOUT calling a success-summarizer LLM.

    The report is taken from the ``report(report=...)`` args. If the last
    message is free text with no tool call, that text is used as the report
    (safety fallback). A worker reports whether it finished OR hit a blocker /
    could not complete the task.
    """
    report_text = ""
    if state.messages:
        last_msg = state.messages[-1]
        for tc in getattr(last_msg, "tool_calls", []):
            if tc.get("name") == "report":
                report_text = str((tc.get("args") or {}).get("report", "")).strip()
                break
        if not report_text:
            report_text = str(getattr(last_msg, "content", "") or "").strip()

    logger.info(
        "WORKER  → REPORT  category=%s  rounds_used=%d/%d",
        state.predicted_tool_category,
        state.round,
        state.round_limit,
    )
    return {
        "status": "success",
        "report": f"['{state.predicted_tool_category}' Worker Report]\n {report_text}",
    }


# ── Failure Summarizer Prompt (no state_records) ─────────────────────

FAILURE_SUMMARIZER_SYSTEM_PROMPT = """\
Your task is to read the worker agent activity provided above and provide an report to main agent in worker's tone perspective
Worker has hit the round limit for tool_calling.
Now you MUST report to Main agent regarding your current progress, failures you encountered,
, any content/info demanded by main_agent (see instruction provided earlier),
and why you cannot complete task under expected round limit.
Do not output anything else, do not try to call any tool.
"""

def failure_summary_node(state: WorkerState) -> dict:
    """
    LangGraph node: record a failure due to round limit.
    """
    llm = state.llm
    summary = timed_invoke(
        "WorkerSummary(failure)",
        llm.invoke,
        [
            *state.messages[1:],  # pass conversation history directly
            SystemMessage(content=FAILURE_SUMMARIZER_SYSTEM_PROMPT)
        ],
    )
    report_text = f"['{state.predicted_tool_category}' Worker Report]\n {summary.content}"
    logger.info(
        "WORKER  → FAILURE  category=%s  rounds_used=%d/%d",
        state.predicted_tool_category,
        state.round,
        state.round_limit,
    )

    return {
        "status": "failed",
        "report": report_text,
    }


# ── Conditional Router ──────────────────────────────────────────────


def success_or_continue(state: WorkerState) -> str:
    """
    Conditional edge router for report judgement.

    Decision logic (in priority order):

    1. If the last message calls ``report`` → **report**.
    2. Else if it has other ``tool_calls`` → **tool**.
    3. Otherwise (free text, no tool call) → **report** (treated as a
       free-text report via ``report_node``).
    """

    # Priority 1: report tool call → hand report back
    if state.messages:
        last_msg = state.messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", [])
        if tool_calls:
            names = [tc.get("name") for tc in tool_calls]
            if "report" in names:
                logger.debug(
                    "WORKER  → route  category=%s  report  → report",
                    state.predicted_tool_category,
                )
                return "report"
            logger.debug(
                "WORKER  → route  category=%s  tool_calls=%d  → tool",
                state.predicted_tool_category,
                len(tool_calls),
            )
            return "tool"

    # Priority 2: no tool call → treat as free-text report
    logger.debug(
        "WORKER  → route  category=%s  → report (free text)",
        state.predicted_tool_category,
    )
    return "report"

def failure_or_continue(state: WorkerState) -> str:
    """
    Conditional edge router for failure judgement

    Decision logic:

    If ``round >= round_limit`` → **failure_summary**.
    else -> "llm_node" to start a new round
    """

    # Priority 1: round limit reached
    if state.round >= state.round_limit and state.messages:
        logger.debug(
            "WORKER  → route  category=%s  round=%d >= limit=%d  → failure_summary",
            state.predicted_tool_category,
            state.round,
            state.round_limit,
        )
        return "failure_summary"
    else:
        return "llm"

# ── Graph Builder ────────────────────────────────────────────────────


def build_worker_graph() -> StateGraph:
    """
    Build and return the LangGraph ``StateGraph`` for a single Worker
    Agent ReAct loop.

    Graph structure::

[START] → initialize_worker -> llm_node
                                  │
                                  ▼  (route_worker)
                             ┌────┴────────┐
                        tool │         report
                             ▼            ▼
                        tool_node   report_node
                             │            │
                             ▼            ▼
                        llm_node ←┘     [END]
                             │
                      failure│
                             ▼
                    failure_summary_node
                             │
                             ▼
                           [END]
    """
    graph = StateGraph(WorkerState)

    # ── Add nodes ─────────────────────────────────────────────
    graph.add_node("initialize_worker_node", initialize_worker)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("report_node", report_node)
    graph.add_node("failure_summary_node", failure_summary_node)

    # ── Edges ─────────────────────────────────────────────────
    graph.add_edge(START, "initialize_worker_node")
    graph.add_edge("initialize_worker_node", "llm_node")

    graph.add_conditional_edges(
        "llm_node",
        success_or_continue,
        {
            "tool": "tool_node",
            "report": "report_node",
        }
    )

    # After tool execution, check failure or loop back to LLM for next ReAct cycle
    graph.add_conditional_edges(
        "tool_node",
        failure_or_continue,
        {
            "failure_summary": "failure_summary_node",
            "llm": "llm_node",
        }

    )

    # Terminal nodes → END
    graph.add_edge("report_node", END)
    graph.add_edge("failure_summary_node", END)

    return graph


# ── Convenience Runner ──────────────────────────────────────────────


async def run_worker(state: WorkerState) -> WorkerState:
    """
    Convenience function: build the worker graph, compile it, and run.

    Parameters
    ----------
    state : WorkerState
        Pre-configured worker state (must include at minimum ``task`` and
        ``messages`` with a system/human prompt to start).

    Returns
    -------
    WorkerState
        Final state after the ReAct loop terminates.
    """
    graph = build_worker_graph()
    compiled = graph.compile()
    return await compiled.ainvoke(state)


async def run_worker_task(
    instruction: str,
    context_explain: str,
    assigned_tool_category: str,
    categorized_toolkit: dict[str, list],
) -> tuple[str, str]:
    """
    Run a single worker loop for a task delegated by the main agent.

    Directly connects the main agent to the worker loop (no planner/checker).

    Parameters
    ----------
    instruction : str
        The concrete task description (what to do, what counts as complete).
    context_explain : str
        Background context and the ultimate goal the worker should aim at.
    assigned_tool_category : str
        The tool category the worker should use.
    categorized_toolkit : dict[str, list]
        Mapping of category name → list of tool instances.

    Returns
    -------
    tuple[str, str]
        ``(status, report)`` where ``status`` is ``"success"`` (the worker
        reported back) or ``"failed"`` (round limit hit), and ``report`` is the
        single worker report in either case.
    """
    from workflow.llm import get_llm

    worker_state = WorkerState(
        task=instruction,
        instruction=context_explain,
        predicted_tool_category=assigned_tool_category,
        categorized_toolkit=categorized_toolkit,
        llm=get_llm(),
        round_limit=WORKER_ROUND_LIMIT,
    )

    final = await run_worker(worker_state)
    status = final.get("status", "unknown")
    if status == "success":
        return "success", final.get("report", "")
    return "failed", final.get("report", "")
