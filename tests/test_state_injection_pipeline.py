"""
Runnable MANUAL test for the state-injection pipeline.

This is a script, not a pytest case: it has no ``test_*`` functions, so
``pytest tests/`` imports it harmlessly without running the pipeline. Run it
directly::

    python tests/test_state_injection_pipeline.py
    python tests/test_state_injection_pipeline.py --pref-file other.json
    python tests/test_state_injection_pipeline.py --summary "<~150-word summary>"

It calls ``run_state_injection`` (which uses the real no-thinking LLM
internally) and prints:

  * the raw classifier/updater debug lines (state name, new_state, description
    length, event),
  * the returned action dict (created / updated / unchanged / None),
  * the resulting state store, so you can see the pipeline take action.

With ``--summary`` it runs exactly ONE test case and exits; otherwise it loops
on your input until ``exit`` / ``quit`` / Ctrl+C.

WARNING: it writes to the REAL ``preferences.json`` (or the ``--pref-file`` you
pass), so point it at a throwaway file if you do not want to touch your store.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the repo root importable when run as `python tests/<this file>.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Imports must follow the sys.path bootstrap above, hence noqa: E402.
from preference_system.models import PreferenceStore  # noqa: E402
from preference_system.state_injection import run_state_injection  # noqa: E402


def setup_logging() -> None:
    """Show the state-injection debug lines; silence everything else noisy."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    # The pipeline emits its classifier/updater decisions on this logger.
    logging.getLogger("preference_system.state_injection").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def print_states(store: PreferenceStore) -> None:
    """Print every state currently held in the store."""
    print(f"\n  States in store ({len(store.states)}):")
    if not store.states:
        print("    (none)")
        return
    for name, state in store.states.items():
        print(f"    [{name}]")
        print(f"      description : {state.description or '(none)'}")
        print(f"      active_time : {state.active_time or '(none)'}")
        print(f"      events ({len(state.event_list)}):")
        for i, event in enumerate(state.event_list, 1):
            print(f"        {i}. {event}")


def describe_result(result: dict[str, object] | None) -> None:
    """Print the meaning of what ``run_state_injection`` returned."""
    if result is None:
        print(
            "\n  RESULT: None — summary was not state-worthy, "
            "or a classifier/updater LLM call failed."
        )
        return

    action = result.get("action")
    print(f"\n  RESULT: action={action!r}")
    if action == "created":
        print(f"    new state      : {result.get('state_name')!r}")
        print(f"    description    : {result.get('description')!r}")
        print(f"    initial event  : {result.get('event')!r}")
        print(f"    evicted state  : {result.get('evicted_state')!r}")
    elif action == "updated":
        print(f"    state          : {result.get('state_name')!r}")
        print(f"    desc changed   : {result.get('description_changed')}")
        print(f"    new description: {result.get('new_description')!r}")
        print(f"    new event      : {result.get('new_event')!r}")
    elif action == "unchanged":
        print(f"    state          : {result.get('state_name')!r} — no changes needed")


def inject_one(store: PreferenceStore, summary: str) -> dict[str, object] | None:
    """Run a single state-injection round and print everything it did."""
    print(f"\n  Summary ({len(summary.split())} words):\n    {summary}")
    print("  ── injecting ──")
    result = run_state_injection(
        summary_text=summary,
        preference_store=store,
    )
    describe_result(result)
    print_states(store)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually run the state-injection pipeline (real LLM)."
    )
    parser.add_argument(
        "--pref-file",
        default="preferences.json",
        help="PreferenceStore JSON path (default: preferences.json)",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Run exactly this one summary as a test case, then exit.",
    )
    args = parser.parse_args()

    setup_logging()

    pref_path = Path(args.pref_file)
    preference_store = PreferenceStore.load(pref_path)

    print(f"Writing to REAL store: {pref_path.resolve()}")
    print_states(preference_store)

    # ── One-shot test case ──────────────────────────────────────────
    if args.summary.strip():
        try:
            inject_one(preference_store, args.summary.strip())
        except (KeyboardInterrupt, EOFError):
            print("\nInterrupted — exiting.")
        finally:
            preference_store.save(pref_path)
            print(f"\nSaved → {pref_path.resolve()}")
        return

    # ── Interactive loop ────────────────────────────────────────────
    print("\nType a ~150-word summary to inject; 'exit' (or Ctrl+C) to quit.")
    try:
        while True:
            summary = input("\nSummary to inject:\n> ").strip()
            if not summary:
                continue
            if summary.lower() in ("exit", "quit"):
                break

            try:
                inject_one(preference_store, summary)
            except Exception:
                logging.getLogger("preference_system.state_injection").exception(
                    "State injection raised unexpectedly"
                )

            preference_store.save(pref_path)
            print(f"  Saved → {pref_path.resolve()}")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted — exiting.")
    finally:
        preference_store.save(pref_path)


if __name__ == "__main__":
    main()

