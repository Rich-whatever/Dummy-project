"""
Runnable MANUAL test for the v2 preference-injection pipeline.

This is a script, not a pytest case: it has no ``test_*`` functions, so
``pytest tests/`` imports it harmlessly without running the pipeline. Run it
directly::

    python tests/test_preference_injection_pipeline.py
    python tests/test_preference_injection_pipeline.py --pref-file other.json

It calls ``run_preference_injection`` on your input in a loop (store + LLM are
loaded once), printing the raw LLM output of each call (extraction, then
admission gate) plus a result summary. Type ``exit`` or press Ctrl+C to quit.

WARNING: it writes to the REAL ``preferences.json`` and ``preference_chroma_db``.
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
from preference_system.preference_injection import run_preference_injection  # noqa: E402
from preference_system.vector_store import PreferenceVectorStore  # noqa: E402
from workflow.llm import get_llm_no_thinking  # noqa: E402


def setup_logging() -> None:
    """Print ONLY the raw LLM outputs — silence every other pipeline log."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    # The pipeline emits each raw LLM output on this dedicated logger.
    logging.getLogger("preference_system.llm_output").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually run the v2 preference-injection pipeline."
    )
    parser.add_argument(
        "--pref-file",
        default="preferences.json",
        help="PreferenceStore JSON path (default: preferences.json)",
    )
    args = parser.parse_args()

    setup_logging()

    pref_path = Path(args.pref_file)
    preference_store = PreferenceStore.load(pref_path)
    vector_store = PreferenceVectorStore()
    llm = get_llm_no_thinking()
    expectations_file = Path(preference_store.expectations_file)

    print(f"Writing to REAL store: {pref_path.resolve()}")
    print(
        f"Vector store: {vector_store.persist_directory} "
        f"(collection '{vector_store.collection_name}')"
    )
    print("Type a message to inject; 'exit' (or Ctrl+C) to quit.")

    try:
        while True:
            user_message = input("\nUser message to inject:\n> ").strip()
            if not user_message:
                continue
            if user_message.lower() in ("exit", "quit"):
                break

            result = run_preference_injection(
                llm=llm,
                user_message=user_message,
                store=vector_store,
                preference_store=preference_store,
                expectations_file=expectations_file,
            )

            if result is None:
                print(
                    "\nInjection FAILED "
                    "(extraction or admission-gate LLM call failed)."
                )
            else:
                print(f"\nResult: {result.summary()}")
                for category, note in result.stored_notes:
                    print(f"  + note [{category}]: {note}")
                for expectation in result.stored_expectations:
                    print(f"  + expectation: {expectation}")

            preference_store.save(pref_path)
            print(f"Saved → {pref_path.resolve()}")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted — exiting.")
    finally:
        preference_store.save(pref_path)


if __name__ == "__main__":
    main()

