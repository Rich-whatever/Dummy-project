#!/usr/bin/env python3
"""
Memory Pipeline — CLI REPL.

A simple interactive shell for the memory system pipeline.

Usage:
    python main.py              # Normal mode (no pipeline logging)
    python main.py -v           # Verbose mode (INFO logging for memory_pipeline)
    python main.py -vv          # Very verbose (DEBUG for everything)

Commands (inside the REPL):
    /help           Show available commands
    /threads        List all active threads
    /thread <id>    Show detailed info for a specific thread
    /ltm_lookup <id> [block#]   Query LTM vector store for summaries from a thread
    /state          Dump full pipeline state (thread_bar, etc.)
    /quit           Exit the REPL
    /exit           Alias for /quit
"""
import argparse
import asyncio
import logging
import sys
import time

from config import MAX_ACTIVE_THREADS, THREAD_STATE_FILE
from memory_backend.models import ThreadBar, load_thread_bar, save_thread_bar
from preference_system.graph_nodes import wait_for_background_injections
from preference_system.models import PreferenceStore
from preference_system.vector_store import PreferenceVectorStore
from tool.toolkit_registry import build_toolkit
from vector_store.client import MemoryVectorStore
from vector_store.operations import query_with_score
from workflow.graph import run_pipeline
from workflow.llm import get_llm
from workflow.timing import print_latency_table, reset_timings

logger = logging.getLogger("main")


# ── Logging Setup ──────────────────────────────────────────────────

def setup_logging(verbose_level: int = 0) -> None:
    """
    Configure logging based on verbosity.

    * 0: Only warnings+ (no pipeline logging)
    * 1 (-v): INFO for all modules
    * 2 (-vv): DEBUG for everything
    """
    root = logging.getLogger()

    # Remove any pre-existing handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if verbose_level >= 2:
        root.setLevel(logging.DEBUG)
    elif verbose_level >= 1:
        root.setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
    else:
        root.setLevel(logging.WARNING)

    # Preference-system sub-modules log per-note/per-inject detail at INFO —
    # in the main pipeline that detail is noise since each graph node emits one
    # consolidated line.  The standalone preference CLI re-enables it.
    if verbose_level < 2:
        for name in (
            "preference_system.note_injection",
            "preference_system.expectation_injection",
            "preference_system.retrieve_branch",
            "preference_system.state_injection",
            "preference_system.vector_store",
            "preference_system.models",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)


# ── REPL ───────────────────────────────────────────────────────────

def print_help() -> None:
    """Print available commands."""
    help_text = """
╔═══════════════════════════════════════════════════════╗
║          Memory Pipeline — REPL                       ║
╠═══════════════════════════════════════════════════════╣
║ Commands:                                             ║
║  /help           Show this help                       ║
║  /threads        List active threads                  ║
║  /thread <id>    Show thread details                  ║
║  /delete_thread <id>   Delete a thread from the bar    ║
║  /ltm_lookup <id> [block#]    Query LTM for thread    ║
║  /check_all_thread          Dump pipeline state       ║
║  /quit  /exit    Exit the REPL                        ║
║                                                       ║
║ Anything else is treated as a message                 ║
║ sent through the memory pipeline.                     ║
╚═══════════════════════════════════════════════════════╝
"""
    print(help_text)


def cmd_threads(thread_bar: ThreadBar) -> None:
    """List all active threads with summary info."""
    if thread_bar.count == 0:
        print("  (no active threads)")
        return

    print(f"  Active threads ({thread_bar.count}/{MAX_ACTIVE_THREADS}):")
    for i, thread in enumerate(thread_bar):
        tid = thread.thread_id
        topic = thread.topic_description or "(no topic)"
        rounds = thread.round_count
        last_active = (thread.last_active[11:19] if len(thread.last_active) >= 19 else thread.last_active) if thread.last_active else "?"
        bridge = "yes" if thread.bridge_summary else "no"
        ltm_count = len(thread.latest_ltm)
        print(
            f"  [{i+1}] {tid} | rounds={rounds} | "
            f"bridge={bridge} | ltm_cache={ltm_count} | "
            f"active={last_active}"
        )
        print(f"       topic: {topic[:70]}")


def cmd_thread_detail(thread_bar: ThreadBar, search_id: str) -> None:
    """Show detailed info for a specific thread by ID prefix or index."""
    # Try index first (1-based)
    if search_id.isdigit():
        idx = int(search_id) - 1
        if 0 <= idx < thread_bar.count:
            thread = thread_bar[idx]
            _print_thread_detail(thread, idx + 1)
            return
        print(f"  No thread at index {search_id}")
        return

    # Try full thread_id match
    for i, thread in enumerate(thread_bar):
        if thread.thread_id == search_id:
            _print_thread_detail(thread, i + 1)
            return

    # Try prefix match
    for i, thread in enumerate(thread_bar):
        if thread.thread_id.startswith(search_id):
            _print_thread_detail(thread, i + 1)
            return

    print(f"  No thread found matching '{search_id}'")


def _print_thread_detail(thread, index: int) -> None:
    """Print full detail for a single thread."""
    print(f"\n  Thread #{index}")
    print("  ─────────────────────────────")
    print(f"  thread_id:       {thread.thread_id}")
    print(f"  topic:           {thread.topic_description or '(no topic)'}")
    print(f"  rounds:          {thread.round_count}")
    print(f"  is_full:         {thread.is_full}")
    print(f"  summary_counter: {thread.summary_counter}")
    print(f"  last_active:     {thread.last_active}")
    print(f"  latest_ltm:      {len(thread.latest_ltm)} entries")
    if thread.latest_ltm:
        for i, entry in enumerate(thread.latest_ltm):
            print(
                f"     [{i+1}] thread={entry.get('thread_id','?')[:12]}… "
                f"sum#{entry.get('block#','?')} "
                f"score={entry.get('score','?'):.3f}"
            )
    print(f"\n  bridge_summary ({len(thread.bridge_summary)} chars):")
    if thread.bridge_summary:
        for line in thread.bridge_summary.split("\n"):
            print(f"    | {line}")
    else:
        print("    (empty)")

    print(f"\n  Conversation Rounds ({thread.round_count}):")
    for i, round_obj in enumerate(thread.conversation_rounds):
        user_msg = round_obj.user_message.replace("\n", " ")
        agent_msg = round_obj.agent_response.replace("\n", " ")
        print(f"    [{i+1}] [USER] {user_msg}")
        if round_obj.underlying_content:
            print(f"         [Exec detail] {round_obj.underlying_content.replace("##", "\n")}")
        print(f"         [ASST] {agent_msg}")
        print(f"(length: user={len(round_obj.user_message)} asst={len(round_obj.agent_response)})")


def _resolve_thread_id(thread_bar: ThreadBar, search_id: str) -> str | None:
    """Resolve a user-provided id/index/prefix to a full thread_id."""
    if search_id.isdigit():
        idx = int(search_id) - 1
        if 0 <= idx < thread_bar.count:
            return thread_bar[idx].thread_id
        return None
    for thread in thread_bar:
        if thread.thread_id == search_id or thread.thread_id.startswith(search_id):
            return thread.thread_id
    return None


def cmd_delete_thread(thread_bar: ThreadBar, search_id: str) -> None:
    """
    Delete a thread from the active ThreadBar by index / prefix / full id.

    Note: this only removes the thread from the thread bar. Its LTM summaries
    (a separate store) are intentionally left intact.
    """
    full_id = _resolve_thread_id(thread_bar, search_id)
    if full_id is None:
        print(f"  No thread found matching '{search_id}'")
        return
    removed = thread_bar.remove_thread(full_id)
    if removed:
        save_thread_bar(thread_bar, THREAD_STATE_FILE)
        print(f"  ✓ Deleted thread {full_id}…  ({thread_bar.count}/{MAX_ACTIVE_THREADS} threads remain)")
    else:
        print(f"  No thread found matching '{search_id}'")


def cmd_ltm_lookup(vector_store: MemoryVectorStore, thread_id: str, block_number: str | None = None) -> None:
    """Query LTM vector store for summaries related to a thread_id.

    If ``block_number`` is provided, only the summary block with that
    block# is shown.  Otherwise all summary blocks for the thread are
    returned.
    """
    try:
        filter_dict = {"thread_id": thread_id}
        if block_number is not None:
            filter_dict["block#"] = block_number
        results = vector_store.vectorstore.similarity_search_with_score(
            query="",
            k=20,
            filter=filter_dict,
        )
    except Exception:
        results = query_with_score(
            vector_store,
            query="",
            k=20,
            exclude_filter=None,
        )
        results = [(d, s) for d, s in results if d.metadata.get("thread_id") == thread_id]
        if block_number is not None:
            results = [(d, s) for d, s in results if d.metadata.get("block#") == block_number]

    if not results:
        label = f"'{thread_id[:12]}…'" if block_number is None else f"'{thread_id[:12]}…' block#{block_number}"
        print(f"  No LTM summaries found for {label}")
        return

    label = f"'{thread_id[:12]}…'" if block_number is None else f"'{thread_id[:12]}…' block#{block_number}"
    print(f"  LTM summaries for {label} ({len(results)} found):")
    for doc, score in results:
        block_num = doc.metadata.get("block#", "?")
        ts = doc.metadata.get("timestamp", "?")
        content_preview = doc.page_content[:100].replace("\n", " ")
        print(f"\n  ── block#={block_num} | score={score:.3f} | timestamp={ts}")
        print(f"     content: {content_preview}…")


def cmd_state(thread_bar: ThreadBar) -> None:
    """Dump full pipeline state information."""
    print("\n  ── Pipeline State ──")
    print(f"  ThreadBar: {thread_bar.count}/{MAX_ACTIVE_THREADS} threads active")

    for i, thread in enumerate(thread_bar):
        tid = thread.thread_id
        topic = thread.topic_description or "(none)"
        rounds = thread.round_count
        summary_ctr = thread.summary_counter
        latest_ltm = len(thread.latest_ltm)
        last_active = thread.last_active if thread.last_active else "N/A"
        bridge_len = len(thread.bridge_summary) if thread.bridge_summary else 0

        print(f"\n  ── Thread #{i+1}: {tid}")
        print(f"     topic:          {topic[:80]}")
        print(f"     rounds:         {rounds}")
        print(f"     summary_ctr:    {summary_ctr}")
        print(f"     bridge_len:     {bridge_len}")
        print(f"     latest_ltm:     {latest_ltm}")
        print(f"     last_active:    {last_active}")


# ── Main ───────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory Pipeline — REPL",
    )
    parser.add_argument(
        "-v",
        action="count",
        default=0,
        dest="verbose",
        help="Verbosity: -v (pipeline logging), -vv (debug all)",
    )
    args = parser.parse_args()

    setup_logging(verbose_level=args.verbose)

    # ── Initialise core components ──────────────────────────────────
    print("Initialising Memory Pipeline...", file=sys.stderr)
    try:
        vector_store = MemoryVectorStore()
    except Exception as e:
        print(f"FATAL: Failed to initialise vector store: {e}", file=sys.stderr)
        sys.exit(1)
    thread_bar = load_thread_bar(THREAD_STATE_FILE)
    if thread_bar.count:
        print(
            f"Loaded {thread_bar.count} persisted thread(s) from {THREAD_STATE_FILE}.",
            file=sys.stderr,
        )
    print("Initialising preference system...", file=sys.stderr)
    try:
        preference_store = PreferenceStore.load("preferences.json")

        # One-shot backfill: give every note category that is missing a
        # description an LLM-generated one (<15 words), then persist.
        try:
            from preference_system.note_injection import (
                backfill_category_descriptions,
            )

            if backfill_category_descriptions(None, preference_store) > 0:
                preference_store.save("preferences.json")
        except Exception:
            print(
                "WARN: Category-description backfill failed (non-fatal).",
                file=sys.stderr,
            )

        preference_vector_store = PreferenceVectorStore()
    except Exception as e:
        print(f"WARN: Failed to initialise preference system: {e}", file=sys.stderr)
        preference_store = None
        preference_vector_store = None
    print("Initialising agent toolkits....")
    try:
        client_list, categorized_toolkit = await build_toolkit()
    except Exception as e:
        print(f"FATAL: Failed to initialise agent toolkits: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("Ready. Type /help for commands, or just type a message.", file=sys.stderr)
    print(file=sys.stderr)

    # ── REPL Loop ───────────────────────────────────────────────────
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            save_thread_bar(thread_bar, THREAD_STATE_FILE)
            break

        if not user_input:
            continue

        # Pick up General-tab changes made via the settings API without a
        # restart. (Model settings are already read lazily by get_llm().)
        # NOTE: fully live General knobs require consumers to read config as
        # `import config` attributes; `from config import X`-style modules pick
        # changes up only after a restart.
        try:
            from config import reload_runtime_config
            reload_runtime_config()
        except Exception:
            pass

        # ── Commands ────────────────────────────────────────────────
        if user_input == "/help":
            print_help()

        elif user_input in ("/quit", "/exit"):
            print("Bye!")
            save_thread_bar(thread_bar, THREAD_STATE_FILE)
            break

        elif user_input == "/threads":
            cmd_threads(thread_bar)

        elif user_input.startswith("/thread "):
            arg = user_input[len("/thread "):].strip()
            if not arg:
                print("  Usage: /thread <id or index>")
            else:
                cmd_thread_detail(thread_bar, arg)

        elif user_input.startswith("/ltm_lookup "):
            args = user_input[len("/ltm_lookup "):].strip().split(None, 1)
            thread_id = args[0] if args else ""
            block_number = args[1] if len(args) > 1 else None
            if not thread_id:
                print("  Usage: /ltm_lookup <thread_id> [block#]")
            else:
                cmd_ltm_lookup(vector_store, thread_id, block_number)

        elif user_input == "/check_all_thread":
            cmd_state(thread_bar)

        elif user_input.startswith("/delete_thread "):
            arg = user_input[len("/delete_thread "):].strip()
            if not arg:
                print("  Usage: /delete_thread <id or index>")
            else:
                cmd_delete_thread(thread_bar, arg)

        elif user_input.startswith("/"):
            print(f"  Unknown command: {user_input}")
            print("  Type /help for available commands.")

        # ── Normal message through the pipeline ─────────────────────
        else:
            print("  ⏳ Processing message...")
            reset_timings()
            _round_start = time.perf_counter()
            try:
                result = await run_pipeline(
                    user_message=user_input,
                    thread_bar=thread_bar,
                    vector_store=vector_store,
                    llm = get_llm(),
                    categorized_toolkit = categorized_toolkit,
                    preference_store=preference_store,
                    preference_vector_store=preference_vector_store,
                )
                agent_response = result["agent_response"]
                router_result = result["router_result"]
                evicted = result.get("thread_was_evicted", "")

                # Print the agent's response
                print(f"\n  🤖 {agent_response}")


                # Brief status line (always shown, non-verbose)
                if router_result == "NEW_THREAD":
                    tid = thread_bar.threads[-1].thread_id[:12] if thread_bar.count > 0 else "?"
                    status = f"  [new thread {tid}…]"
                else:
                    status = f"  [continued thread {router_result[:12]}…]"
                if evicted:
                    status += f" (evicted: {evicted})"

                # Count total threads
                status += f" ({thread_bar.count}/{MAX_ACTIVE_THREADS} threads)"
                print(status)

                # Wait for fire-and-forget preference injection to finish so the
                # latency table + next input are not shown before the background
                # note/expectation writes (and preferences.json save) complete.
                await wait_for_background_injections()

                # Persist the thread bar (append/summarise/evict happened during
                # run_pipeline) so threads survive a crash/restart.
                try:
                    save_thread_bar(thread_bar, THREAD_STATE_FILE)
                except Exception as _e:
                    logger.error("Failed to persist thread state: %s", _e)

            except Exception as e:
                print(f"  ❌ Error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
            finally:
                print_latency_table(total_seconds=time.perf_counter() - _round_start)


if __name__ == "__main__":
    asyncio.run(main())
