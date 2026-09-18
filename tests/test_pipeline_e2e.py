"""
End-to-end pipeline tests — the refactoring safety net.

Runs the REAL compiled LangGraph through ``workflow.graph.run_pipeline`` with a
scripted mock LLM and fake tool boundaries, so the entire node wiring is
exercised without any network, LLM, or MCP dependency.

Purpose: **characterisation / regression safety.** These tests pin down what
the pipeline *does* (what it returns, what it writes into the ThreadBar) so a
refactor that changes wiring — a moved function, a renamed import, a split
module — fails loudly instead of silently.

Mocked boundaries (everything else is real):
  * the LLM(s)                     — scripted, no network
  * LTM vector retrieval           — returns nothing
  * the worker task                — canned report
  * ``memory_search``              — canned report
  * the preference retrieve branch — empty context block
  * the preference store           — MagicMock (never touches disk)

Determinism notes:
  * An EMPTY ThreadBar makes ``route_message`` and ``is_continuation_of_recent``
    return without calling the LLM at all, so the first-round tests are fully
    deterministic.
  * Messages longer than 40 chars skip the small-LLM retrieve probe.
"""

from __future__ import annotations

import asyncio
import contextlib
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

import execution.worker as worker_mod
import memory_backend.memory_node as memory_node_mod
import preference_system.graph_nodes as pref_nodes_mod
import workflow.llm as llm_mod
import workflow.memory_search as memory_search_mod
from memory_backend.memory_node import RetrieveNeedsLastRound
from memory_backend.models import Thread, ThreadBar
from tool.toolkit_registry import TOOL_CATEGORY_NAMES
from workflow.graph import run_pipeline
from workflow.router import NEW_THREAD, ContinuationCheck

CATEGORY = TOOL_CATEGORY_NAMES[0]


# ─────────────────────────────────────────────────────────────────────
# Mock builders
# ─────────────────────────────────────────────────────────────────────


def _agent_llm(responses: list[AIMessage]) -> MagicMock:
    """LLM whose ``bind_tools(...).ainvoke`` replays ``responses`` in order."""
    llm = MagicMock(name="scripted_agent_llm")
    bound = MagicMock(name="bound_llm")
    bound.ainvoke = AsyncMock(side_effect=list(responses))
    llm.bind_tools.return_value = bound
    return llm


def _no_think_llm(continues_recent: bool = False) -> MagicMock:
    """Pre-router / router LLM: structured continuation answer + text router."""
    llm = MagicMock(name="no_think_llm")
    structured = MagicMock(name="continuation_structured")
    structured.invoke.return_value = ContinuationCheck(
        continues_recent_conversation=continues_recent
    )
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value = MagicMock(content=NEW_THREAD)
    return llm


def _small_llm() -> MagicMock:
    """Small-LLM probe used when the user message is <= 40 characters."""
    llm = MagicMock(name="small_llm")
    structured = MagicMock(name="probe_structured")
    structured.ainvoke = AsyncMock(
        return_value=RetrieveNeedsLastRound(needs_last_round=False)
    )
    llm.with_structured_output.return_value = structured
    return llm


def _branch_result() -> MagicMock:
    """Stand-in for ``RetrieveBranchResult`` from the preference branch."""
    result = MagicMock(name="branch_result")
    result.to_context_block.return_value = ""
    return result


def _response_tool(report: str, call_id: str = "call_r") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "response", "args": {"report": report}, "id": call_id}
        ],
    )


def _assign_tool(
    category: str = CATEGORY, call_id: str = "call_a"
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "assign_task",
                "args": {
                    "context_explain": "ctx",
                    "instruction": "do the thing",
                    "assigned_tool_category": category,
                },
                "id": call_id,
            }
        ],
    )


def _search_tool(call_id: str = "call_s") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "memory_search",
                "args": {
                    "questions": ["what does the user like?"],
                    "search_user_preference": True,
                    "search_long_term_memory": False,
                },
                "id": call_id,
            }
        ],
    )



# ─────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _pipeline_env(
    agent_responses: list[AIMessage],
    *,
    continues_recent: bool = False,
    worker_result: tuple[str, str] = ("success", "WORKER REPORT"),
    search_report: str = "SEARCH REPORT",
):
    """Patch every external boundary the pipeline touches."""
    agent = _agent_llm(agent_responses)
    worker_mock = AsyncMock(return_value=worker_result)
    search_mock = AsyncMock(return_value=search_report)

    patches = [
        patch.object(
            llm_mod, "get_llm_no_thinking",
            return_value=_no_think_llm(continues_recent),
        ),
        patch.object(llm_mod, "get_small_llm", return_value=_small_llm()),
        patch.object(memory_node_mod, "retrieve_ltm", return_value=[]),
        patch.object(memory_node_mod, "generate_topic", return_value="Test topic"),
        patch.object(memory_node_mod, "rolling_summarize", return_value="SUMMARY"),
        patch.object(memory_node_mod, "evict_lru_thread", return_value=None),
        patch.object(worker_mod, "run_worker_task", new=worker_mock),
        patch.object(memory_search_mod, "run_memory_search", new=search_mock),
        patch.object(
            pref_nodes_mod, "arun_retrieve_branch",
            new=AsyncMock(return_value=_branch_result()),
        ),
        patch.object(
            pref_nodes_mod, "load_general_behavior_expectations", return_value=""
        ),
        patch.object(pref_nodes_mod, "run_preference_injection", return_value=None),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield agent, worker_mock, search_mock


def _run(
    agent_responses: list[AIMessage],
    user_message: str = "hello there, this is a test message",
    thread_bar: ThreadBar | None = None,
    **kwargs,
):
    """Run one full pipeline round against a mocked environment."""
    bar = thread_bar if thread_bar is not None else ThreadBar()
    with _pipeline_env(agent_responses, **kwargs) as mocks:
        agent, worker_mock, search_mock = mocks
        result = asyncio.run(
            run_pipeline(
                user_message=user_message,
                thread_bar=bar,
                vector_store=MagicMock(name="vector_store"),
                categorized_toolkit={},
                llm=agent,
                preference_store=None,
                preference_vector_store=None,
            )
        )
    return result, bar, worker_mock, search_mock


# ─────────────────────────────────────────────────────────────────────
# Round 1 — routing + persistence
# ─────────────────────────────────────────────────────────────────────


class TestFirstRound(unittest.TestCase):
    def test_free_text_becomes_agent_response(self):
        result, _, _, _ = _run([AIMessage(content="Hello there!")])
        self.assertEqual(result["agent_response"], "Hello there!")

    def test_response_tool_becomes_agent_response(self):
        result, _, _, _ = _run([_response_tool("Tool answer.")])
        self.assertEqual(result["agent_response"], "Tool answer.")

    def test_empty_bar_routes_to_new_thread(self):
        result, _, _, _ = _run([AIMessage(content="hi")])
        self.assertEqual(result["router_result"], NEW_THREAD)

    def test_round_is_persisted_to_a_new_thread(self):
        result, bar, _, _ = _run(
            [AIMessage(content="reply text")], user_message="question text"
        )
        self.assertEqual(bar.count, 1)
        thread = bar.get_most_recent_thread()
        self.assertIsNotNone(thread)
        self.assertEqual(thread.round_count, 1)
        rnd = thread.conversation_rounds[0]
        self.assertEqual(rnd.user_message, "question text")
        self.assertEqual(rnd.agent_response, "reply text")

    def test_no_tool_use_leaves_underlying_content_empty(self):
        result, _, _, _ = _run([AIMessage(content="plain")])
        self.assertIsNone(result["underlying_content"])

    def test_result_exposes_the_documented_keys(self):
        result, _, _, _ = _run([AIMessage(content="x")])
        for key in (
            "user_message",
            "agent_response",
            "underlying_content",
            "router_result",
            "thread_was_evicted",
            "retrieved_preference",
            "general_behavior_expectation",
        ):
            self.assertIn(key, result)


# ─────────────────────────────────────────────────────────────────────
# Tool loop — assign_task / memory_search
# ─────────────────────────────────────────────────────────────────────


class TestToolLoop(unittest.TestCase):
    def test_assign_task_report_is_fed_back_then_responded(self):
        result, _, worker_mock, _ = _run(
            [_assign_tool(), _response_tool("Done after worker.")]
        )
        self.assertEqual(result["agent_response"], "Done after worker.")
        self.assertEqual(worker_mock.await_count, 1)
        self.assertIn("WORKER REPORT", result["underlying_content"] or "")

    def test_assign_task_passes_instruction_and_category(self):
        _, _, worker_mock, _ = _run([_assign_tool(), _response_tool("ok")])
        kwargs = worker_mock.await_args.kwargs
        self.assertEqual(kwargs["assigned_tool_category"], CATEGORY)
        self.assertEqual(kwargs["instruction"], "do the thing")
        self.assertEqual(kwargs["context_explain"], "ctx")

    def test_memory_search_report_is_fed_back_then_responded(self):
        result, _, _, search_mock = _run(
            [_search_tool(), _response_tool("After search.")]
        )
        self.assertEqual(result["agent_response"], "After search.")
        self.assertEqual(search_mock.await_count, 1)
        kwargs = search_mock.await_args.kwargs
        self.assertEqual(kwargs["questions"], ["what does the user like?"])
        self.assertTrue(kwargs["search_user_preference"])
        self.assertFalse(kwargs["search_long_term_memory"])

    def test_invalid_category_does_not_run_a_worker(self):
        result, _, worker_mock, _ = _run(
            [_assign_tool(category="not_a_category"), _response_tool("recovered")]
        )
        self.assertEqual(worker_mock.await_count, 0)
        self.assertEqual(result["agent_response"], "recovered")

    def test_free_text_without_tool_call_ends_the_loop(self):
        result, _, worker_mock, _ = _run(
            [_assign_tool(), AIMessage(content="I will just answer.")]
        )
        self.assertEqual(result["agent_response"], "I will just answer.")
        self.assertEqual(worker_mock.await_count, 1)



# ─────────────────────────────────────────────────────────────────────
# Round 2+ — routing into existing threads
# ─────────────────────────────────────────────────────────────────────


def _bar_with_one_thread(user_message: str = "first question", reply: str = "first reply"):
    bar = ThreadBar()
    thread = Thread()
    thread.append_round(user_message, reply)
    bar.add_thread(thread)
    return bar, thread


class TestRoutingIntoExistingThreads(unittest.TestCase):
    def test_continuation_appends_to_the_most_recent_thread(self):
        bar, thread = _bar_with_one_thread()
        result, bar, _, _ = _run(
            [_response_tool("second reply")],
            user_message="and another follow up question here",
            thread_bar=bar,
            continues_recent=True,
        )
        self.assertEqual(bar.count, 1, "should NOT have created a second thread")
        self.assertEqual(result["router_result"], thread.thread_id)
        self.assertEqual(thread.round_count, 2)
        self.assertEqual(
            thread.conversation_rounds[1].agent_response, "second reply"
        )

    def test_non_continuation_can_start_a_new_thread(self):
        bar, thread = _bar_with_one_thread()
        result, bar, _, _ = _run(
            [_response_tool("brand new answer")],
            user_message="completely unrelated brand new topic here",
            thread_bar=bar,
            continues_recent=False,
        )
        self.assertEqual(bar.count, 2, "expected a brand new thread")
        self.assertEqual(result["router_result"], NEW_THREAD)
        self.assertEqual(thread.round_count, 1, "old thread must be untouched")

    def test_existing_threads_keep_their_rounds_intact(self):
        bar, thread = _bar_with_one_thread(user_message="q1", reply="a1")
        _run(
            [_response_tool("a2")],
            user_message="another follow up question goes here",
            thread_bar=bar,
            continues_recent=True,
        )
        self.assertEqual(thread.conversation_rounds[0].user_message, "q1")
        self.assertEqual(thread.conversation_rounds[0].agent_response, "a1")


if __name__ == "__main__":
    unittest.main()

