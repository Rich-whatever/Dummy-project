"""
Execution package — the Worker Agent (tool-calling ReAct loop).

The worker is invoked directly by the main agent (via ``assign_task``) to
accomplish a task using a category's tools. It hands a single report back to
the main agent with the ``report`` tool, or is marked failed after the round
limit. The worker report is stored in the single ``WorkerState.report`` field.
"""

from .worker import (
    WorkerState,
    build_worker_graph,
    failure_or_continue,
    failure_summary_node,
    llm_node,
    report,
    report_node,
    run_worker,
    run_worker_task,
    success_or_continue,
    tool_node,
)

__all__ = [
    "WorkerState",
    "build_worker_graph",
    "run_worker",
    "run_worker_task",
    "llm_node",
    "tool_node",
    "report_node",
    "report",
    "failure_summary_node",
    "success_or_continue",
    "failure_or_continue",
]
