"""
Tests for the main agent's ``memory_search`` tool and its orchestration module
``workflow.memory_search``.

All stores are mocked (``MagicMock``) — no ChromaDB / embeddings are touched.
The async entry point is driven with ``asyncio.run`` so no pytest-asyncio
configuration is required.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from workflow import memory_search as ms
from workflow.memory_search import (
    _NO_QUESTIONS,
    _NO_SOURCE,
    run_memory_search,
)


def _run(coro):
    """Drive the async entry point from a synchronous test."""
    return asyncio.run(coro)


def _ltm_doc(thread_id: str, block: int, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"thread_id": thread_id, "block#": block},
    )


def _pref_doc(note_id: str, category: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "note_id": note_id,
            "category_name": category,
            "note_content": text,
        },
    )


class TestValidation(unittest.TestCase):
    def test_no_questions_returns_guidance(self):
        result = _run(
            run_memory_search(
                [],
                search_user_preference=True,
                search_long_term_memory=True,
                vector_store=MagicMock(),
                preference_vector_store=MagicMock(),
            )
        )
        self.assertEqual(result, _NO_QUESTIONS)

    def test_blank_questions_returns_guidance(self):
        result = _run(
            run_memory_search(
                ["   ", ""],
                search_user_preference=True,
                search_long_term_memory=True,
                vector_store=MagicMock(),
                preference_vector_store=MagicMock(),
            )
        )
        self.assertEqual(result, _NO_QUESTIONS)

    def test_no_source_returns_guidance(self):
        result = _run(
            run_memory_search(
                ["who is the user?"],
                search_user_preference=False,
                search_long_term_memory=False,
                vector_store=MagicMock(),
                preference_vector_store=MagicMock(),
            )
        )
        self.assertEqual(result, _NO_SOURCE)


class TestLtmOnly(unittest.TestCase):
    def test_searches_each_question_separately(self):
        def fake_retrieve(query, store):
            block = {"alpha": 0, "beta": 1}[query]
            return [(_ltm_doc("t1", block, f"answer for {query}"), 0.42)]

        with patch.object(ms, "retrieve_ltm", side_effect=fake_retrieve) as spy:
            result = _run(
                run_memory_search(
                    ["alpha", "beta"],
                    search_user_preference=False,
                    search_long_term_memory=True,
                    vector_store=MagicMock(),
                    preference_vector_store=None,
                )
            )

        self.assertEqual(spy.call_count, 2)
        self.assertIn("answer for alpha", result)
        self.assertIn("answer for beta", result)
        self.assertIn("--- Long-Term Memory", result)
        self.assertNotIn("User Preference", result)

    def test_empty_results_message(self):
        with patch.object(ms, "retrieve_ltm", return_value=[]):
            result = _run(
                run_memory_search(
                    ["nothing here"],
                    search_user_preference=False,
                    search_long_term_memory=True,
                    vector_store=MagicMock(),
                    preference_vector_store=None,
                )
            )
        self.assertIn("(no relevant long-term memory found)", result)

    def test_unavailable_store_message(self):
        result = _run(
            run_memory_search(
                ["anything"],
                search_user_preference=False,
                search_long_term_memory=True,
                vector_store=None,
                preference_vector_store=None,
            )
        )
        self.assertIn("(long-term memory store unavailable)", result)


class TestPreferenceOnly(unittest.TestCase):
    def test_queries_preference_store_per_question(self):
        pref = MagicMock()
        pref.query_similar_note.side_effect = lambda q: [
            (_pref_doc(f"n-{q}", "Hobbies", f"likes {q}"), 0.15)
        ]

        result = _run(
            run_memory_search(
                ["chess", "coffee"],
                search_user_preference=True,
                search_long_term_memory=False,
                vector_store=None,
                preference_vector_store=pref,
            )
        )

        self.assertEqual(pref.query_similar_note.call_count, 2)
        self.assertIn("likes chess", result)
        self.assertIn("likes coffee", result)
        self.assertIn("category=Hobbies", result)
        self.assertIn("--- User Preference", result)
        self.assertNotIn("Long-Term Memory", result)

    def test_empty_results_message(self):
        pref = MagicMock()
        pref.query_similar_note.return_value = []
        result = _run(
            run_memory_search(
                ["nothing"],
                search_user_preference=True,
                search_long_term_memory=False,
                vector_store=None,
                preference_vector_store=pref,
            )
        )
        self.assertIn("(no relevant user preference found)", result)

    def test_unavailable_store_message(self):
        result = _run(
            run_memory_search(
                ["anything"],
                search_user_preference=True,
                search_long_term_memory=False,
                vector_store=None,
                preference_vector_store=None,
            )
        )
        self.assertIn("(preference vector store unavailable)", result)


class TestBothSources(unittest.TestCase):
    def test_runs_both_and_packages_results(self):
        def fake_retrieve(query, store):
            return [(_ltm_doc("t1", 1, f"ltm for {query}"), 0.5)]

        pref = MagicMock()
        pref.query_similar_note.side_effect = lambda q: [
            (_pref_doc(f"n-{q}", "Preferences", f"pref for {q}"), 0.2)
        ]

        with patch.object(ms, "retrieve_ltm", side_effect=fake_retrieve):
            result = _run(
                run_memory_search(
                    ["query one"],
                    search_user_preference=True,
                    search_long_term_memory=True,
                    vector_store=MagicMock(),
                    preference_vector_store=pref,
                )
            )

        self.assertIn("Sources: user_preference=yes, long_term_memory=yes", result)
        self.assertIn("ltm for query one", result)
        self.assertIn("pref for query one", result)


class TestDedupe(unittest.TestCase):
    def test_same_ltm_doc_across_queries_appears_once(self):
        shared = _ltm_doc("t1", 3, "shared summary")
        with patch.object(ms, "retrieve_ltm", return_value=[(shared, 0.4)]):
            result = _run(
                run_memory_search(
                    ["q1", "q2"],
                    search_user_preference=False,
                    search_long_term_memory=True,
                    vector_store=MagicMock(),
                    preference_vector_store=None,
                )
            )
        self.assertEqual(result.count("shared summary"), 1)

    def test_same_preference_doc_across_queries_appears_once(self):
        shared = _pref_doc("n1", "Cat", "shared note")
        pref = MagicMock()
        pref.query_similar_note.return_value = [(shared, 0.1)]
        result = _run(
            run_memory_search(
                ["q1", "q2"],
                search_user_preference=True,
                search_long_term_memory=False,
                vector_store=None,
                preference_vector_store=pref,
            )
        )
        self.assertEqual(result.count("shared note"), 1)


class TestFailureIsolation(unittest.TestCase):
    def test_preference_failure_keeps_ltm_results(self):
        pref = MagicMock()
        pref.query_similar_note.side_effect = RuntimeError("pref store down")

        with patch.object(
            ms, "retrieve_ltm", return_value=[(_ltm_doc("t1", 0, "ltm survives"), 0.3)]
        ):
            result = _run(
                run_memory_search(
                    ["q"],
                    search_user_preference=True,
                    search_long_term_memory=True,
                    vector_store=MagicMock(),
                    preference_vector_store=pref,
                )
            )

        self.assertIn("ltm survives", result)
        self.assertIn("(no relevant user preference found)", result)

    def test_ltm_query_failure_does_not_abort_other_queries(self):
        calls = {"n": 0}

        def flaky(query, store):
            calls["n"] += 1
            if query == "boom":
                raise RuntimeError("ltm error")
            return [(_ltm_doc("t1", 0, f"ok {query}"), 0.2)]

        with patch.object(ms, "retrieve_ltm", side_effect=flaky):
            result = _run(
                run_memory_search(
                    ["boom", "fine"],
                    search_user_preference=False,
                    search_long_term_memory=True,
                    vector_store=MagicMock(),
                    preference_vector_store=None,
                )
            )

        self.assertEqual(calls["n"], 2)
        self.assertIn("ok fine", result)
        self.assertIn("(no match)", result)


class TestToolSchema(unittest.TestCase):
    def test_memory_search_tool_exposes_expected_args(self):
        from tool.agent_tools import memory_search

        self.assertEqual(
            set(memory_search.args.keys()),
            {"questions", "search_user_preference", "search_long_term_memory"},
        )
        self.assertEqual(memory_search.name, "memory_search")


if __name__ == "__main__":
    unittest.main()

