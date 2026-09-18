"""
Standalone CLI for testing the preference-system pipelines.

Four pipelines can be exercised directly:

    /retrieve <message>            note_retrieve — retrieve notes + states + expectations
                                   (run_retrieve_branch)
    /inject   <message>            preference injection — merged extract → gate → persist
                                   (run_preference_injection)
    /state    <summary text>       state injection — classify + create/update a state
                                   (run_state_injection; expects ~150-word summary text)

The PreferenceStore is persisted to JSON (default: preferences.json, override with
PREFERENCE_FILE or --pref-file) and auto-saved after every mutating pipeline and on
exit, so data survives restarts.

Usage::

    python -m preference_system.cli
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from workflow.llm import get_llm_no_thinking, get_small_llm

from .models import PreferenceStore, load_expectations
from .preference_injection import run_preference_injection
from .retrieve_branch import run_retrieve_branch, run_retrieve_branch_detailed
from .state_injection import run_state_injection
from .vector_store import PreferenceVectorStore

logger = logging.getLogger("preference_system.cli")


# ─────────────────────────────────────────────────────────────────────
# Printer helpers
# ─────────────────────────────────────────────────────────────────────

def print_store(store: PreferenceStore) -> None:
    """Pretty-print the entire store."""
    print("\n╔════════════════════════════════════════════╗")
    print("║          Preference Store                  ║")
    print("╚════════════════════════════════════════════╝")

    print(f"\n── States ({len(store.states)}) ──")
    if not store.states:
        print("  (none)")
    for name, state in store.states.items():
        print(f"  [{name}]")
        print(f"    Description: {state.description}")
        print(f"    Active:      {state.active_time}")
        print(f"    Events ({len(state.event_list)}):")
        for i, evt in enumerate(state.event_list, 1):
            print(f"      {i}. {evt}")

    print(f"\n── Note Categories ({len(store.note_categories)}) ──")
    if not store.note_categories:
        print("  (none)")
    for name, cat in store.note_categories.items():
        print(f"  [{name}] ({len(cat.notes)} notes)")
        for i, note in enumerate(cat.notes, 1):
            print(f"      {i}. [{note.saved_time}] {note.content}")

    print(f"\n── Expectations (file: {store.expectations_file}) ──")
    print(store.get_expectations())


# ─────────────────────────────────────────────────────────────────────
# REPL
# ─────────────────────────────────────────────────────────────────────

def print_help() -> None:
    print("""
Pipeline test commands:
  /help                          Show this help
  /print                         Print the full store (states, categories, notes, expectations)
  /map                           Show the preference map (names only)
  /reset                         Wipe EVERYTHING (states, categories, vector notes, expectations) after confirmation

  /retrieve <message>            Test note_retrieve — retrieve notes + states + expectations
  /retrieve_detailed <message>   Same as /retrieve, plus a per-LLM-call timing trace
  /inject <message>              Test preference injection — merged extract → gate → persist
  /state <summary text>          Test state injection — classify + create/update a state

  /quit  /exit                   Save and exit
""")


def main() -> None:
    # Preference sub-modules log per-note/per-inject detail at INFO — keep it
    # visible in the standalone CLI (the main pipeline silences these loggers).
    for name in (
        "preference_system.note_injection",
        "preference_system.expectation_injection",
        "preference_system.retrieve_branch",
        "preference_system.state_injection",
        "preference_system.vector_store",
        "preference_system.models",
    ):
        logging.getLogger(name).setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Preference System Pipeline Test CLI")
    parser.add_argument("--pref-file", default=None, help="Path to preference JSON file")
    args = parser.parse_args()

    # Determine persistence path
    pref_path = args.pref_file or os.environ.get("PREFERENCE_FILE", "preferences.json")
    pref_path = Path(pref_path)

    # ── Load or create store ────────────────────────────────────────
    store = PreferenceStore.load(pref_path)
    print(f"Loaded store from {pref_path} ({len(store.states)} states, {len(store.note_categories)} categories)")
    print("Type /help for commands.")

    llm = None  # Lazy init (normal model — note/expectation/state injection)
    small_llm = None  # Lazy init (small model — retrieve branch)
    vector_store = None  # Lazy init

    def get_llm_lazy():
        nonlocal llm
        if llm is None:
            print("  Initialising LLM...")
            llm = get_llm_no_thinking()
        return llm

    def get_small_llm_lazy():
        nonlocal small_llm
        if small_llm is None:
            print("  Initialising small LLM...")
            small_llm = get_small_llm()
        return small_llm

    def get_vector_store():
        nonlocal vector_store
        if vector_store is None:
            print("  Initialising vector store...")
            vector_store = PreferenceVectorStore()
        return vector_store

    def save_store(msg: str = "") -> None:
        """Persist the PreferenceStore to JSON."""
        try:
            store.save(pref_path)
            if msg:
                print(msg)
            else:
                print(f"  ✓ Saved to {pref_path}")
        except Exception:
            logger.exception("Failed to save store to %s", pref_path)
            print("  ✗ Save failed — see logs")

    # ── REPL ────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            save_store("Saved on exit.")
            print("\nBye!")
            break

        if not user_input:
            continue

        # ──── /help ─────────────────────────────────────────────────
        if user_input == "/help":
            print_help()

        # ──── /quit /exit ───────────────────────────────────────────
        elif user_input in ("/quit", "/exit"):
            save_store("Saved before exit.")
            print("Bye!")
            break

        # ──── /print ────────────────────────────────────────────────
        elif user_input == "/print":
            print_store(store)

        # ──── /map ──────────────────────────────────────────────────
        elif user_input == "/map":
            print(store.get_preference_map())

        # ──── /reset ───────────────────────────────────────────────
        elif user_input == "/reset":
            n_states = len(store.states)
            n_categories = len(store.note_categories)
            n_expectations = len(load_expectations(store.expectations_file))
            print(
                f"  This will wipe: {n_states} states, {n_categories} note "
                f"categories, ALL vector-store notes, and {n_expectations} "
                f"expectations (file: {store.expectations_file})."
            )
            confirm = input("  Are you sure? (y/N) ").strip().lower()
            if confirm not in ("y", "yes"):
                print("  Reset cancelled.")
                continue

            # 1) Clear the in-memory PreferenceStore
            store.states.clear()
            store.note_categories.clear()

            # 2) Clear the vector store (persisted Chroma notes)
            vs = get_vector_store()
            n_vector = vs.clear()

            # 3) Reset the expectations JSON file
            try:
                from json_io import write_json_atomic

                exp_path = Path(store.expectations_file)
                write_json_atomic(exp_path, {"expectations": []})
            except Exception:
                logger.exception(
                    "Failed to reset expectations file %s", store.expectations_file
                )
                n_expectations = -1  # flag error in summary

            # 4) Persist the emptied store
            save_store()
            cleared_exp = (
                f"{n_expectations} expectations"
                if n_expectations >= 0
                else "expectations (FILE RESET FAILED — see logs)"
            )
            print(
                f"  ✓ Reset complete — removed {n_states} states, "
                f"{n_categories} categories, {n_vector} vector notes, "
                f"{cleared_exp}."
            )


        # ──── /retrieve ─────────────────────────────────────────────
        elif user_input.startswith("/retrieve "):
            message = user_input[len("/retrieve "):].strip()
            if not message:
                print("Usage: /retrieve <message>")
                continue
            model = get_small_llm_lazy()
            print(f"  Running retrieve branch for: {message[:60]}...")
            result = run_retrieve_branch(model, message, store)
            context_block = result.to_context_block()
            if context_block:
                print("\n  ── Retrieved Context ──")
                for line in context_block.split("\n"):
                    print(f"  {line}")
            else:
                print("  (nothing retrieved)")
            print(f"\n  Summary: {result}")

        # ──── /retrieve_detailed ──────────────────────────────────────
        elif user_input.startswith("/retrieve_detailed "):
            message = user_input[len("/retrieve_detailed "):].strip()
            if not message:
                print("Usage: /retrieve_detailed <message>")
                continue
            model = get_small_llm_lazy()
            print(f"  Running detailed retrieve branch for: {message[:60]}...")
            result, timing = run_retrieve_branch_detailed(model, message, store)
            context_block = result.to_context_block()
            if context_block:
                print("\n  ── Retrieved Context ──")
                for line in context_block.split("\n"):
                    print(f"  {line}")
            else:
                print("  (nothing retrieved)")

            print("\n  ── Retrieve Timeline ──")
            if not timing:
                print("  (no timing recorded)")
            for entry in timing:
                print(
                    f"  {entry.name:<28} "
                    f"{entry.started_at[11:23]} → {entry.completed_at[11:23]}  "
                    f"{entry.duration_ms:8.1f} ms  {entry.output}"
                )

        # ──── /inject ───────────────────────────────────────────────
        elif user_input.startswith("/inject "):
            message = user_input[len("/inject "):].strip()
            if not message:
                print("Usage: /inject <message>")
                continue
            model = get_llm_lazy()
            vs = get_vector_store()
            print(f"  Running preference injection for: {message[:60]}...")
            result = run_preference_injection(
                llm=model,
                user_message=message,
                store=vs,
                preference_store=store,
                expectations_file=Path(store.expectations_file),
            )
            if result is None:
                print(
                    "  ✗ Injection FAILED "
                    "(extraction or admission-gate LLM call failed)"
                )
            else:
                print(f"  ✓ {result.summary()}")
                save_store()  # Persist category/note bookkeeping

        # ──── /state ────────────────────────────────────────────────
        elif user_input.startswith("/state "):
            summary = user_input[len("/state "):].strip()
            if not summary:
                print("Usage: /state <summary text>")
                continue
            print(f"  Running state injection for {len(summary)}-char summary...")
            result = run_state_injection(summary, store)
            if result is None:
                print("  (not state-worthy or nothing changed)")
            else:
                action = result.get("action", "?")
                state_name = result.get("state_name", "?")

                print(f"  ✓ State injection: {action} — '{state_name}'")
                if action == "created":
                    print(f"      description: {result.get('description', '')!r}")
                    if result.get("event"):
                        print(f"      initial event: {result.get('event')!r}")
                    if result.get("evicted_state"):
                        print(
                            f"      (evicted oldest state: {result.get('evicted_state')!r})"
                        )
                elif action == "updated":
                    if result.get("description_changed"):
                        print(
                            f"      description CHANGED → {result.get('new_description', '')!r}"
                        )
                    else:
                        print("      description: unchanged")
                    if result.get("new_event"):
                        print(f"      appended event: {result.get('new_event')!r}")
                    if result.get("evicted_state"):
                        print(
                            f"      (evicted oldest state: {result.get('evicted_state')!r})"
                        )
                elif action == "unchanged":
                    print("      description: unchanged, no new events")
                else:
                    print(f"      {result}")
                save_store()  # Persist states

        # ──── Unknown ───────────────────────────────────────────────
        elif user_input.startswith("/"):
            print(f"Unknown command: {user_input}")
            print("Type /help for commands.")

        # ──── Plain text → treat as retrieve ────────────────────────
        else:
            model = get_small_llm_lazy()
            print(f"  Running retrieve branch for: {user_input[:60]}...")
            result = run_retrieve_branch(model, user_input, store)
            context_block = result.to_context_block()
            if context_block:
                print("\n  ── Retrieved Context ──")
                for line in context_block.split("\n"):
                    print(f"  {line}")
            else:
                print("  (nothing retrieved)")
            print(f"\n  Summary: {result}")


if __name__ == "__main__":
    main()
