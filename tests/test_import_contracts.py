"""
Contract tests — structural guards against accidental API drift.

These lock down things that are easy to break silently while reorganising code
and that a functional test would not always notice:

1. **Every application module still imports.** Catches a broken import left
   behind by a moved/renamed/deleted module.
2. **The pipeline graph still has all of its nodes and edges.** Catches a node
   that was accidentally dropped while splitting a module.
3. **The documented public API still resolves at its documented path.** Catches
   a symbol that moved without a backward-compatible re-export.
4. **Key entry-point signatures are unchanged.** Catches accidental parameter
   renames while moving functions between modules.

These tests intentionally do NOT mock anything — they are about structure.
"""

from __future__ import annotations

import importlib
import inspect
import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Packages whose every module must import cleanly.
APP_PACKAGES = (
    "api",
    "execution",
    "file_input",
    "memory_backend",
    "preference_system",
    "tool",
    "vector_store",
    "workflow",
)

# Root-level modules that are safe to import (no module-level side effects).
# NOTE: test_worker_perf.py is deliberately excluded — it calls setup_logging()
# at import time, and it lives at the repo root rather than in tests/.
ROOT_MODULES = ("config", "json_io")

_SKIP_DIR_NAMES = {"__pycache__"}

# Top-level names that belong to THIS project. A missing module from this set is
# always a real failure; a missing third-party module is an uninstalled optional
# dependency and is skipped (reported, not failed).
LOCAL_TOP_LEVEL = set(APP_PACKAGES) | {"config", "json_io", "main", "dev"}


def _is_missing_optional_dep(exc: ModuleNotFoundError) -> bool:
    """True when the missing module is third-party rather than ours."""
    missing = (exc.name or "").split(".")[0]
    return missing not in LOCAL_TOP_LEVEL


def _discover_modules() -> list[str]:
    """Every importable app module, discovered from disk.

    Discovery (rather than a hardcoded list) means a newly added module is
    covered automatically.
    """
    found: list[str] = []
    for pkg in APP_PACKAGES:
        pkg_dir = REPO_ROOT / pkg
        if not pkg_dir.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(pkg_dir):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                full = Path(dirpath) / filename
                rel = full.relative_to(REPO_ROOT).with_suffix("")
                parts = list(rel.parts)
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                if not parts:
                    continue
                found.append(".".join(parts))
    found.extend(ROOT_MODULES)
    return sorted(set(found))


class TestModulesImport(unittest.TestCase):
    def test_every_app_module_imports(self):
        """A broken import anywhere in the app fails this test.

        A missing THIRD-PARTY module (an uninstalled optional dependency) is
        reported as a skip rather than a failure; a missing module from this
        project is always a failure.
        """
        failures: list[str] = []
        skipped: list[str] = []
        for module_name in _discover_modules():
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                if _is_missing_optional_dep(exc):
                    skipped.append(f"{module_name} <- missing '{exc.name}'")
                else:
                    failures.append(f"{module_name}: {exc}")
            except Exception as exc:  # noqa: BLE001 - report every failure
                failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
        if skipped:
            print(
                "\n[info] modules skipped because an optional dependency is "
                "not installed in this environment:\n  " + "\n  ".join(skipped)
            )
        self.assertEqual(
            failures, [], "modules failed to import:\n" + "\n".join(failures)
        )

    def test_discovery_actually_found_modules(self):
        """Guards against the discovery logic silently returning nothing."""
        modules = _discover_modules()
        self.assertGreater(len(modules), 40)
        self.assertIn("workflow.graph", modules)
        self.assertIn("preference_system.note_injection", modules)


class TestPipelineGraphShape(unittest.TestCase):
    """The graph's node set is part of the system's contract."""

    EXPECTED_NODES = {
        "router_node",
        "ltm_node",
        "preference_retrieve_node",
        "preference_injection_node",
        "assembly_node",
        "human_message_node",
        "agent_node",
        "update_memory_node",
    }

    def test_all_expected_nodes_are_present(self):
        from workflow.graph import build_pipeline_graph

        graph = build_pipeline_graph()
        missing = self.EXPECTED_NODES - set(graph.nodes)
        self.assertEqual(missing, set(), f"graph is missing nodes: {missing}")

    def test_graph_compiles(self):
        from workflow.graph import build_pipeline_graph

        build_pipeline_graph().compile()


class TestPublicApiResolves(unittest.TestCase):
    """Documented public symbols must keep resolving at their documented path.

    This is the guard for moving code between modules: if a function is moved,
    either re-export it from its old location or update this list deliberately.
    """

    # package/symbol-name pairs, as documented by preference_system/__init__.py
    PREFERENCE_SYSTEM_EXPORTS = (
        "PreferenceState",
        "Note",
        "NoteCategory",
        "PreferenceStore",
        "StateMapDecision",
        "NoteCategoryMapDecision",
        "ExpectationMapDecision",
        "StateRetrievalResult",
        "NoteRetrievalResult",
        "ExpectationExtractionResult",
        "run_retrieve_branch",
        "append_expectation",
        "persist_expectation_candidates",
        "run_preference_extraction",
        "run_admission_gate",
        "prefilter_note_candidates",
        "persist_categorized_notes",
        "run_preference_injection",
        "PreferenceInjectionResult",
    )

    # module path -> symbol names that must exist there
    SYMBOL_LOCATIONS = {
        "preference_system.note_injection": (
            "run_preference_extraction",
            "run_admission_gate",
            "prefilter_note_candidates",
            "persist_categorized_notes",
            "run_note_maintenance",
            "backfill_category_descriptions",
        ),
        "preference_system.preference_injection": (
            "run_preference_injection",
            "PreferenceInjectionResult",
        ),
        "preference_system.expectation_injection": (
            "append_expectation",
            "persist_expectation_candidates",
        ),
        "preference_system.state_injection": (
            "run_state_injection",
            "run_state_classifier",
            "run_state_update",
        ),
        "preference_system.retrieve_branch": (
            "run_retrieve_branch",
            "arun_retrieve_branch",
        ),
        "preference_system.models": (
            "PreferenceStore",
            "PreferenceState",
            "Note",
            "NoteCategory",
            "MergedExtractionResult",
            "AdmissionResult",
            "StateClassificationResult",
            "StateUpdateResult",
        ),
        "file_input": ("process_file", "process_files", "FileUploadStore"),
        "file_input.processing": ("process_file", "process_files"),
        "workflow.graph": ("run_pipeline", "build_pipeline_graph"),
        "workflow.state": ("PipelineState",),
        "workflow.router": (
            "route_message",
            "is_continuation_of_recent",
            "NEW_THREAD",
        ),
        "execution.worker": ("run_worker_task", "run_worker", "WorkerState"),
        "memory_backend.models": (
            "Thread",
            "ThreadBar",
            "ConversationRound",
            "load_thread_bar",
            "save_thread_bar",
        ),
        "memory_backend.memory_node": (
            "arouter_node",
            "ltm_node",
            "update_memory_node",
        ),
        "tool.agent_tools": ("response", "assign_task", "memory_search"),
        "workflow.memory_search": ("run_memory_search",),
    }

    def test_preference_system_package_exports(self):
        pkg = importlib.import_module("preference_system")
        missing = [n for n in self.PREFERENCE_SYSTEM_EXPORTS if not hasattr(pkg, n)]
        self.assertEqual(missing, [], f"preference_system no longer exports: {missing}")

    def test_symbols_resolve_at_their_documented_module(self):
        missing: list[str] = []
        for module_path, symbols in self.SYMBOL_LOCATIONS.items():
            module = importlib.import_module(module_path)
            for symbol in symbols:
                if not hasattr(module, symbol):
                    missing.append(f"{module_path}.{symbol}")
        self.assertEqual(missing, [], "symbols missing:\n" + "\n".join(missing))



class TestEntryPointSignatures(unittest.TestCase):
    """Key entry points must keep accepting the parameters callers rely on.

    This catches an accidental parameter rename while moving a function between
    modules. Extra/new parameters are allowed — only removal or renaming fails.
    """

    EXPECTED_PARAMS = {
        "workflow.graph.run_pipeline": (
            "user_message",
            "thread_bar",
            "vector_store",
            "categorized_toolkit",
            "llm",
            "preference_store",
            "preference_vector_store",
            "human_message_content",
        ),
        "execution.worker.run_worker_task": (
            "instruction",
            "context_explain",
            "assigned_tool_category",
            "categorized_toolkit",
        ),
        # NOTE: run_state_injection fetches its own no-thinking LLM internally
        # (``get_llm_no_thinking()``) and therefore takes NO ``llm`` parameter.
        "preference_system.state_injection.run_state_injection": (
            "summary_text",
            "preference_store",
        ),
        "preference_system.note_injection.prefilter_note_candidates": (
            "notes",
            "store",
            "preference_store",
        ),
        "workflow.memory_search.run_memory_search": (
            "questions",
            "search_user_preference",
            "search_long_term_memory",
            "vector_store",
            "preference_vector_store",
        ),
    }

    def test_entry_point_parameters_are_stable(self):
        problems: list[str] = []
        for qualified, expected in self.EXPECTED_PARAMS.items():
            module_path, _, attr = qualified.rpartition(".")
            func = getattr(importlib.import_module(module_path), attr)
            actual = set(inspect.signature(func).parameters)
            missing = [p for p in expected if p not in actual]
            if missing:
                problems.append(f"{qualified} is missing parameters: {missing}")
        self.assertEqual(problems, [], "\n".join(problems))


if __name__ == "__main__":
    unittest.main()

