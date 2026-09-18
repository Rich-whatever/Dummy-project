"""
Tests for the v2 preference-injection pipeline:
merged extraction → vector prefilter → admission gate → persistence.

All LLMs and the vector store are mocked; the PreferenceStore is real so the
category/note bookkeeping is exercised. `run_preference_injection` is driven
synchronously (no asyncio needed).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from preference_system import expectation_injection as ei
from preference_system import note_injection as ni
from preference_system import preference_injection as pj
from preference_system.models import (
    AdmissionResult,
    ExpectationExtractionResult,
    MergedExtractionResult,
    PreferenceStore,
)


def _expectation(content="Be concise", trigger="When replying", action="short"):
    return ExpectationExtractionResult(
        expectation_content=content, trigger=trigger, action=action
    )


def _admission(
    notes=None,
    accepted_expectation_indices=None,
    created_category=None,
) -> AdmissionResult:
    return AdmissionResult(
        accepted_expectation_indices=accepted_expectation_indices or [],
        notes=notes or {},
        created_category=created_category or {},
    )


def _mock_store():
    store = MagicMock()
    store.query_similar_note.return_value = []
    return store


class TestOrchestration(unittest.TestCase):
    def _run(self, extraction, admission, store=None, preference_store=None,
             expectations_file=None):
        store = store or _mock_store()
        preference_store = preference_store or PreferenceStore()
        with patch.object(pj, "run_preference_extraction", return_value=extraction), \
             patch.object(pj, "run_admission_gate", return_value=admission):
            result = pj.run_preference_injection(
                MagicMock(),
                "some user message",
                store,
                preference_store,
                expectations_file or Path("expectations.json"),
            )
        return result, store, preference_store

    def test_extraction_failure_returns_none(self):
        result, store, _ = self._run(None, None)
        self.assertIsNone(result)
        store.upsert_note.assert_not_called()

    def test_empty_extraction_returns_zero_counts(self):
        extraction = MergedExtractionResult(notes=[], expectations=[])
        result, _, _ = self._run(extraction, None)
        self.assertIsNotNone(result)
        self.assertEqual(result.notes_extracted, 0)
        self.assertEqual(result.notes_stored, 0)

    def test_gate_failure_persists_nothing(self):
        extraction = MergedExtractionResult(notes=["User likes tea"], expectations=[])
        result, store, pref = self._run(extraction, None)
        self.assertIsNone(result)
        store.upsert_note.assert_not_called()
        self.assertEqual(pref.note_categories, {})

    def test_accepted_note_persisted_to_both_stores(self):
        extraction = MergedExtractionResult(notes=["User likes tea"], expectations=[])
        admission = _admission(
            notes={0: "Food Preference"},
            created_category={
                "Food Preference": "What the user likes to eat and drink."
            },
        )
        result, store, pref = self._run(extraction, admission)

        self.assertEqual(result.notes_stored, 1)
        self.assertIn("Food Preference", pref.note_categories)
        note = pref.note_categories["Food Preference"].notes[0]
        self.assertEqual(note.content, "User likes tea")
        store.upsert_note.assert_called_once()
        _, kwargs = store.upsert_note.call_args
        self.assertEqual(kwargs.get("note_id"), note.id)

    def test_rejected_note_not_persisted(self):
        extraction = MergedExtractionResult(
            notes=["User is tired today"], expectations=[]
        )
        admission = _admission()
        result, store, pref = self._run(extraction, admission)

        self.assertEqual(result.notes_rejected, 1)
        self.assertEqual(result.notes_stored, 0)
        store.upsert_note.assert_not_called()
        self.assertEqual(pref.note_categories, {})

    def test_existing_category_reused_case_insensitively(self):
        pref = PreferenceStore()
        pref.add_category("food preference", "Food stuff")
        extraction = MergedExtractionResult(notes=["User likes tea"], expectations=[])
        admission = _admission(notes={0: "Food Preference"})
        _, _, pref = self._run(extraction, admission, preference_store=pref)

        self.assertEqual(list(pref.note_categories.keys()), ["food preference"])
        self.assertEqual(len(pref.note_categories["food preference"].notes), 1)

    def test_duplicate_note_prefiltered_before_gate(self):
        pref = PreferenceStore()
        pref.add_category("Hobbies")
        pref.note_categories["Hobbies"].add_note("User likes chess", "n1")

        store = _mock_store()
        store.query_similar_note.return_value = [
            (
                Document(
                    page_content="User likes chess",
                    metadata={"note_id": "n1", "category_name": "Hobbies"},
                ),
                0.1,
            )
        ]
        extraction = MergedExtractionResult(notes=["User likes chess"], expectations=[])

        with patch.object(pj, "run_preference_extraction", return_value=extraction), \
             patch.object(pj, "run_admission_gate") as gate:
            result = pj.run_preference_injection(
                MagicMock(), "msg", store, pref, Path("expectations.json")
            )

        gate.assert_not_called()
        self.assertEqual(result.notes_duplicate, 1)
        self.assertEqual(result.notes_stored, 0)

    def test_accepted_expectation_persisted_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_file = Path(tmp) / "expectations.json"
            extraction = MergedExtractionResult(
                notes=[], expectations=[_expectation()]
            )
            admission = _admission(accepted_expectation_indices=[0])
            result, _, _ = self._run(
                extraction, admission, expectations_file=exp_file
            )

            self.assertEqual(result.expectations_stored, 1)
            data = json.loads(exp_file.read_text(encoding="utf-8"))
            self.assertEqual(
                data["expectations"][0]["expectation_content"], "Be concise"
            )

    def test_rejected_expectation_not_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_file = Path(tmp) / "expectations.json"
            extraction = MergedExtractionResult(
                notes=[], expectations=[_expectation()]
            )
            admission = _admission()
            result, _, _ = self._run(
                extraction, admission, expectations_file=exp_file
            )

            self.assertEqual(result.expectations_rejected, 1)
            self.assertEqual(result.expectations_stored, 0)
            self.assertFalse(exp_file.exists())


def _structured_llm(results: dict):
    """Mock LLM whose .with_structured_output(schema) returns a schema-keyed result.

    The per-schema structured mocks are collected on ``llm.structured_mocks``
    (in call order) so tests can inspect the prompts they received.
    """
    llm = MagicMock()
    created: list = []

    def _factory(schema, method=None):
        m = MagicMock()
        m.invoke.return_value = results[schema]
        created.append(m)
        return m

    llm.with_structured_output.side_effect = _factory
    llm.structured_mocks = created
    return llm


class TestExtractionAndGate(unittest.TestCase):
    def test_run_preference_extraction_uses_merged_schema(self):
        expected = MergedExtractionResult(notes=["n"], expectations=[])
        llm = _structured_llm({MergedExtractionResult: expected})

        out = ni.run_preference_extraction(llm, "msg")

        self.assertIs(out, expected)
        self.assertIs(
            llm.with_structured_output.call_args.args[0], MergedExtractionResult
        )

    def test_run_admission_gate_numbers_candidates_in_prompt(self):
        admission = _admission(accepted_expectation_indices=[0])
        llm = _structured_llm({AdmissionResult: admission})

        out = ni.run_admission_gate(
            llm, ["note a"], [_expectation()], "=== Note Category Names ===", "ctx"
        )

        self.assertIs(out, admission)
        messages = llm.structured_mocks[0].invoke.call_args.args[0]
        human = messages[1].content
        self.assertIn("0. note a", human)
        self.assertIn("Candidate Expectations", human)

    def test_run_admission_gate_failure_returns_none(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("x")
        self.assertIsNone(
            ni.run_admission_gate(llm, ["note"], [], "(none)")
        )

    def test_run_preference_extraction_failure_returns_none(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("x")
        self.assertIsNone(ni.run_preference_extraction(llm, "msg"))


class TestPrefilter(unittest.TestCase):
    def test_keeps_candidates_and_builds_neighbor_context(self):
        kept, ctx = ni.prefilter_note_candidates(
            ["a", "b"], _mock_store(), PreferenceStore()
        )
        self.assertEqual(kept, [0, 1])
        self.assertIn('Note 1: "a"', ctx)
        self.assertIn('Note 2: "b"', ctx)

    def test_drops_within_batch_duplicates(self):
        kept, _ = ni.prefilter_note_candidates(
            ["a", "a"], _mock_store(), PreferenceStore()
        )
        self.assertEqual(kept, [0])

    def test_drops_duplicate_present_in_preference_store(self):
        pref = PreferenceStore()
        pref.add_category("Hobbies")
        pref.note_categories["Hobbies"].add_note("likes chess", "n1")
        store = _mock_store()
        store.query_similar_note.return_value = [
            (
                Document(
                    page_content="likes chess",
                    metadata={"note_id": "n1", "category_name": "Hobbies"},
                ),
                0.05,
            )
        ]
        kept, _ = ni.prefilter_note_candidates(["likes chess"], store, pref)
        self.assertEqual(kept, [])


class TestPersistenceHelpers(unittest.TestCase):
    def test_persist_categorized_notes_appends_to_existing_category(self):
        pref = PreferenceStore()
        pref.add_category("Hobbies")
        stored = ni.persist_categorized_notes(
            MagicMock(), [("likes chess", "Hobbies", "")], _mock_store(), pref
        )
        self.assertEqual(len(stored), 1)
        self.assertEqual(len(pref.note_categories["Hobbies"].notes), 1)

    def test_persist_categorized_notes_creates_new_category_with_description(self):
        pref = PreferenceStore()
        stored = ni.persist_categorized_notes(
            MagicMock(),
            [("likes chess", "Hobbies", "Games and pastimes.")],
            _mock_store(),
            pref,
        )
        self.assertEqual(len(stored), 1)
        self.assertEqual(
            pref.note_categories["Hobbies"].category_description,
            "Games and pastimes.",
        )

    def test_persist_expectation_candidates_rejects_empty_trigger(self):
        bad = _expectation(content="whatever", trigger="", action="")
        self.assertEqual(ei.persist_expectation_candidates([bad], Path("nope.json")), [])

    def test_persist_expectation_candidates_writes_valid_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_file = Path(tmp) / "expectations.json"
            stored = ei.persist_expectation_candidates(
                [_expectation()], exp_file
            )
            self.assertEqual(stored, ["Be concise"])
            self.assertTrue(exp_file.exists())


if __name__ == "__main__":
    unittest.main()

