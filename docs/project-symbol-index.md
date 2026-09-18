# Project Symbol Index

> Auto-generated index of all public/private classes, functions, and constants.
> **Update this file whenever you create or modify code.**

---

## config.py
| Symbol | Line | Description |
|--------|------|-------------|
| `PROJECT_DIR` / `DB_DIR` / `THREAD_STATE_FILE` | — | Repo root, ChromaDB dir, persisted `ThreadBar` |
| `SANDBOX_DIR` | — | Agent filesystem sandbox root (`<repo>/Dummy_Folder`) shared by the filesystem MCP server and `SafeShellTool` |
| `CHROMA_COLLECTION_NAME` | — | ChromaDB LTM collection (`ltm_summaries`) |
| `PREFERENCE_DB_DIR` / `PREFERENCE_COLLECTION_NAME` | — | Preference vector store dir + collection |
| `MAX_MESSAGES_PER_THREAD` | — | Thread round limit (8) |
| `MAX_ACTIVE_THREADS` | — | Thread bar capacity (5) |
| `BRIDGE_SUMMARY_WORD_LIMIT` | — | Summary word limit (150) |
| `LTM_RETRIEVE_K` | — | Max vector search results (3) |
| `LTM_SCORE_THRESHOLD` | — | Discard results scoring at or above this (0.7) |
| `LTM_LATEST_K` | — | Latest LTM cache size (2) |
| `SUMMARY_COUNTER_INITIAL` | — | Summary counter start (0) |
| `WORKER_ROUND_LIMIT` | — | Max tool invocations per worker task (8) |
| `MAX_ASSIGN_TASKS` / `MAX_MEMORY_SEARCH` | — | Per-round main-agent tool caps (8 / 5) |
| `TOOLMESSAGE_LIMIT` | — | Max chars per tool message before truncation (40000) |
| `EMBEDDING_MODEL_NAME` | — | Sentence transformer model (`all-MiniLM-L6-v2`) |
| `LLM_MODEL_NAME` | — | Default main model (OpenAI-compatible endpoint, not Ollama) |
| `SMALL_LLM_*` | — | Cheap classification model: name / base URL / API-key env var / timeout / retries |
| `RUNTIME_CONFIG_FILE` / `GENERAL_CONFIG_KEYS` | — | Web-editable overrides persisted to `runtime_config.json` |
| `load_runtime_overrides()` | — | Read `runtime_config.json` into a dict; never raises |
| `reload_runtime_config()` | — | Reset to defaults then re-apply the file's general overrides (no restart needed) |

---

## dev.py (one-command dev launcher — NOT part of the app runtime)
Runs the backend + frontend together in one terminal and auto-restarts ONLY the
backend when a watched `.py` changes (frontend/launcher stay up). Run: `python dev.py`
(flags: `--no-reload`, `--backend-only`, `--frontend-only`, `--open`, `--frontend CMD`).

| Symbol | Line | Description |
|--------|------|-------------|
| `PROJECT_ROOT` / `FRONTEND_DIR` | 40 | Repo root + `frontend/` paths |
| `BACKEND_HOST` / `BACKEND_PORT` / `BACKEND_URL` | 43 | `127.0.0.1:8000` + base URL |
| `FRONTEND_URL` / `HEALTH_URL` | 46 | `http://localhost:5173` + `/api/health` |
| `WATCH_POLL_SECONDS` / `WATCH_DEBOUNCE_SECONDS` / `HEALTH_TIMEOUT_SECONDS` | 49 | 1.0 / 0.6 / 180.0 |
| `SKIP_DIR_NAMES` | 54 | Dirs the `.py` watcher never descends into (keeps the mtime walk fast) |
| `_log(tag, message)` | 71 | Prefixed, colorized console line (`[backend]`/`[frontend]`/`[dev]`/`[warn]`) |
| `_raise_keyboard_interrupt(signum, frame)` | 81 | Routes SIGBREAK (Ctrl+Break) into the Ctrl+C cleanup path |
| `_popen(cmd, cwd, env)` | 91 | Spawn a child with piped stdout + its own process group (Ctrl+C stays with the launcher) |
| `_stream(proc, tag)` | 111 | Reader thread forwarding child output with a tag prefix |
| `_kill_tree(proc)` | 122 | Kill child + descendants (`taskkill /F /T` on Windows, `killpg` on POSIX) |
| `_start_backend()` | 150 | Launch `python -u -m api.server` from the repo root |
| `_frontend_shell_command(override)` | 157 | Build the shell-wrapped frontend command (`cmd /c npm run dev`) |
| `_start_frontend(override)` | 165 | Launch the Vite dev server in `frontend/` |
| `_wait_for_health(proc)` | 183 | Poll `/api/health` until ready or the process dies |
| `_announce_ready(proc, show_frontend)` | 199 | Background thread printing the ready banner |
| `_watch_snapshot()` | 209 | Map every backend `.py` (pruned walk, excludes `dev.py`) to its mtime |
| `_changed_files(before, after)` | 225 | mtime diff incl. added/removed files |
| `_parse_args()` | 237 | CLI flags |
| `main()` | 255 | Start both → watch backend → restart backend on change → clean shutdown (finally + atexit) |

---

## main.py (CLI/REPL)
| Symbol | Line | Description |
|--------|------|-------------|
| `setup_logging(verbose_level)` | 35 | Configure logging level |
| `print_help()` | 69 | Print help text for REPL commands |
| `cmd_threads(thread_bar)` | 90 | List all threads in bar |
| `cmd_thread_detail(thread_bar, search_id)` | 112 | Display detail for a specific thread |
| `_print_thread_detail(thread, index)` | 139 | Format and print a single thread |
| `cmd_ltm_lookup(vector_store, thread_id, block_number)` | 174 | Retrieve LTM summary by thread/block |
| `cmd_state(thread_bar)` | 216 | Show current thread bar state |
| `main()` | 241 | CLI entry point (REPL loop) |

---

## text_utils.py (Shared text helpers)
| Symbol | Line | Description |
|--------|------|-------------|
| `DEFAULT_LOG_LIMIT` | 12 | Default truncation cap (2000) for log lines carrying long content |
| `clip(text, limit)` | 15 | Shorten text for log lines, appending `… [truncated: <n> chars, showing <limit>]`; `None` → `""`. Shared by `workflow/graph.py` and `execution/worker.py` (was duplicated verbatim in both). Tested by `tests/test_text_utils.py` |

---

## json_io.py (Atomic + corruption-tolerant JSON helpers)
| Symbol | Line | Description |
|--------|------|-------------|
| `write_json_atomic(path, data, *, indent)` | 33 | Write JSON via `<path>.tmp` + `os.replace` (atomic; a crash cannot leave a half-written file) |
| `read_json(path, default)` | 45 | Tolerant reader (returns `default` when missing) |
| `quarantine(path)` | 52 | Move a corrupt file aside to `<path>.corrupt-<utc-ts>`; returns backup path |
| `load_or_quarantine(path, default, *, validate)` | 66 | Returns `(data, backup\|None)`; unparseable/invalid-shape file is quarantined (data never silently lost) |

---

## memory_backend/models.py (Core Data Structures)
| Symbol | Line | Description |
|--------|------|-------------|
| `_now_utc()` | 26 | Get current ISO-8601 UTC timestamp |
| `_new_uuid()` | 31 | Generate UUID4 string |
| `ConversationRound` | 39 | Dataclass: user_message, agent_response, timestamp |
| `Thread` | 56 | Dataclass: thread_id, summary_counter, topic_description, conversation_rounds, bridge_summary, last_active, latest_ltm |
| `Thread.round_count` | 97 | Property: len(conversation_rounds) |
| `Thread.is_full` | 101 | Property: round_count >= 8 |
| `Thread.age_seconds` | 106 | Property: seconds since last_active |
| `Thread.append_round(user_message, agent_response)` | 112 | Add a conversation round |
| `Thread.to_summary_input()` | 292 | Format last 5 rounds (delegates to `format_rounds`, includes underlying content) |
| `format_rounds(rounds, *, include_underlying, show_gap, now)` | 129 | Shared LLM-facing round formatter: `===== Conversation round \| <local time> (<age>) [\| +gap] =====` + `[USER]`/`[ASSISTANT]`/`[UNDERLYING EXECUTION CONTENT]` blocks |
| `humanize_age(ts, now)` | 79 | Relative age: "just now" / "3 hours ago" / "2 days ago" / "(time unknown)" |
| `format_round_time(ts, now)` | 105 | Local absolute time + relative age, e.g. `2026-09-10 18:03 Thursday (2 hours ago)` |
| `Thread.get_last_n_rounds(n)` | 140 | Get last n ConversationRound objects |
| `ThreadBar` | 156 | Dataclass: container for up to 4 threads |
| `ThreadBar.is_full` | 170 | Property: len(threads) >= 4 |
| `ThreadBar.count` | 174 | Property: len(threads) |
| `ThreadBar.get_thread(thread_id)` | 177 | Get thread by ID |
| `ThreadBar.has_thread(thread_id)` | 184 | Check if thread exists |
| `ThreadBar.add_thread(thread)` | 189 | Add thread (returns False if full) |
| `ThreadBar.remove_thread(thread_id)` | 202 | Remove thread by ID |
| `ThreadBar.get_lru_thread()` | 216 | Get thread with oldest last_active |
| `ThreadBar.get_lru_thread_id()` | 227 | Get oldest thread ID (or None) |
| `ThreadBar.__len__` | 234 | Support len() |
| `ThreadBar.__getitem__` | 237 | Support indexing |
| `ThreadBar.__iter__` | 240 | Support iteration |
| `ThreadBar.__repr__` | 243 | String representation |

---

## memory_backend/compression.py (Rolling Summarization)
| Symbol | Line | Description |
|--------|------|-------------|
| `_call_summarizer(conversation_text, llm)` | 55 | Call LLM to produce summary |
| `rolling_summarize(thread, vector_store, llm)` | 76 | Compress 6 oldest → summary, upsert to LTM, update bridge_summary, increment counter_

---

## memory_backend/eviction.py (LRU Eviction)
| Symbol | Line | Description |
|--------|------|-------------|
| `_is_valuable(thread, llm)` | 83 | Ask LLM if thread is valuable or noise |
| `EvictionResult` | — | Class: evicted_thread_id, was_summarized, summary_text |
| `_summarize_and_store(thread, vector_store, llm)` | 96 | Summarize thread and upsert to LTM |
| `evict_lru_thread(thread_bar, vector_store, llm)` | 131 | Evict LRU thread (dump ≤2 rounds, valuate ≥3) |

---

## memory_backend/topic.py (Topic Generation)
| Symbol | Line | Description |
|--------|------|-------------|
| `generate_topic(conversation_text, llm)` | 32 | LLM-based topic description update |

---

## memory_backend/memory_node.py (LangGraph Nodes for Memory Pipeline)
| Symbol | Line | Description |
|--------|------|-------------|
| `router_node(state)` | 32 | Track 1: LLM-based thread/execution path classification (falls back to `get_llm()` when `state.llm` is None) |
| `ltm_node(state)` | 72 | Track 2: Score-gated ChromaDB query |
| `update_memory_node(state)` | 325 | Append round, rolling summarization, LRU eviction. Resolves `llm = state.llm or get_llm()`; logs an error if the new thread could not be added |

---

## vector_store/client.py (ChromaDB Wrapper)
| Symbol | Line | Description |
|--------|------|-------------|
| `MemoryVectorStore` | 15 | ChromaDB wrapper class |
| `__init__(persist_directory, collection_name)` | 25 | Initialize client + collection |
| `collection` | 49 | Property: get or create collection |
| `__repr__` | 53 | String representation |

---

## vector_store/operations.py (Vector Store Operations)
| Symbol | Line | Description |
|--------|------|-------------|
| `upsert_summary(store, thread_id, summary_text, block_number, timestamp)` | 17 | Upsert a summary block to LTM |
| `query_with_score(store, query_text, n_results, exclude_filter)` | 48 | Query with score and optional exclude |
| `get_summary_by_id(store, thread_id, block_number)` | 87 | Retrieve summary by thread_id + block# |
| `delete_summaries_by_thread(store, thread_id)` | 111 | Delete all summaries for a thread |

---

## workflow/state.py (LangGraph State)
| Symbol | Line | Description |
|--------|------|-------------|
| `PipelineState` | 23 | Main pipeline state: user_message (ABBREVIATED: typed text + file manifest), human_message_content (content blocks: file content + user text), messages, thread_bar, vector_store, llm, preference_store, preference_vector_store, router_result, execution_path, thread_for_context, ltm_results, retrieve_with_last_round, retrieved_preference, general_behavior_expectation, agent_response, underlying_content, thread_was_evicted, categorized_toolkit |
| `PlannerState` | 92 | Planner sub-graph state: user_message, system_prompt, tool_map, plan, plan_attempts, checker_warning, plan_approved, max_attempts, llm, goal, agent_response, thread_bar, vector_store |

---

## workflow/router.py (Track 1 — Intelligent Router)
| Symbol | Line | Description |
|--------|------|-------------|
| `_build_router_input_message(thread_bar, user_message)` | 92 | Build LLM input: topic + 2 latest per thread |
| `route_message(thread_bar, user_message, llm)` | 117 | Router: classify thread + execution path |

---

## workflow/ltm_retriever.py (Track 2 — LTM Retriever)
| Symbol | Line | Description |
|--------|------|-------------|
| `retrieve_ltm(vector_store, user_message, thread, llm)` | 28 | Score-gated ChromaDB query with ID filter + fallback |

---

## workflow/context_assembly.py (Prompt Building)
| Symbol | Line | Description |
|--------|------|-------------|
| `_apply_exclude_filter(results, thread_id, last_block)` | 50 | Exclude results matching thread_id + last block# |
| `_update_latest_ltm(thread, results, k)` | 77 | Cache top k best-scoring results |
| `_get_ltm_fallback(thread)` | 99 | Return latest_ltm cache as formatted string |
| `_build_final_prompt(ltm_text, cross_thread_text, bridge_summary, stm_text)` | 228 | Assemble final system prompt from all sections |
| `assemble_context(thread_bar, vector_store, router_result, ltm_results, thread_for_context)` | 272 | Build SYSTEM prompt only (returns str); current user message is delivered separately as the HumanMessage |

---

## workflow/graph.py (Main LangGraph Pipeline)
| Symbol | Line | Description |
|--------|------|-------------|
| `assembly_node(state)` | 90 | Graph node: wraps assemble_context → emits the SystemMessage |
| `human_message_node(state)` | 168 | Graph node: appends the caller-built HumanMessage (content blocks) so the agent sees [System, Human, …] |
| `agent_node(state)` | 184 | Graph node: main-agent ReAct tool loop (response / assign_task / memory_search) |
| `build_pipeline_graph()` | 424 | Build full StateGraph: parallel router/ltm → assembly → human_message → agent → update_memory |
| `run_pipeline(user_message, …, human_message_content=None)` | 491 | Entry point: compile graph + invoke; seeds abbreviated user_message + human content blocks |

---

## workflow/memory_search.py (Agent-initiated vector memory search)
| Symbol | Line | Description |
|--------|------|-------------|
| `_NO_QUESTIONS` | 34 | Guidance string returned when no non-empty question is supplied |
| `_NO_SOURCE` | 38 | Guidance string returned when neither source flag is True |
| `_format_ltm_hits(query, hits)` | 47 | Render LTM `(Document, score)` hits for one query |
| `_format_pref_hits(query, hits)` | 62 | Render preference-note hits (category + score) for one query |
| `_dedupe_across(per_query, key_fn)` | 74 | Drop documents already surfaced by an earlier query |
| `_ltm_key(doc)` / `_pref_key(doc)` | 98 / 102 | Dedupe keys: `(thread_id, block#)` for LTM; `note_id` for preference notes |
| `_search_source(source_name, questions, search_fn)` | 109 | Runs `search_fn` per question via `asyncio.to_thread`; per-query failure → empty list |
| `run_memory_search(questions, *, search_user_preference, search_long_term_memory, vector_store, preference_vector_store)` | 143 | Entry point: validate → concurrent per-question search of both stores → packaged report string |
| `_package_report(...)` | 233 | Render collected hits into the final agent-facing report (headers, per-query hits, empty/unavailable notes) |

---

## tool/agent_tools.py (Main-Agent Tool Schemas)
| Symbol | Line | Description |
|--------|------|-------------|
| `response(report)` | 28 | Schema-only: completion signal carrying the final reply |
| `assign_task(context_explain, instruction, assigned_tool_category)` | 39 | Schema-only: delegate a task to the worker loop |
| `memory_search(questions, search_user_preference, search_long_term_memory)` | 67 | Schema-only: on-demand vector search over preference notes and/or LTM (run by `agent_node` via `run_memory_search`) |

---

## file_input/ (Standalone file-ingestion subsystem + web-upload bridge)
| Symbol | Line | Description |
|--------|------|-------------|
| `builder.build_content_blocks(processed, *, image_method, user_message)` | 100 | Build LLM content blocks (text/table/image) + labelled User Message; always a `list[dict]` |
| `manifest.render_manifest(processed)` | 40 | Metadata-only manifest string (`[attached file: name (kind)]`); no file content |
| `manifest.manifest_line(pf)` | 33 | One manifest line for a processed file |
| `manifest.abbreviate(user_message, manifest)` | 47 | Combine typed text + manifest → the abbreviated `user_message` |
| `pipeline.PipelineResult` | 22 | ok, content_blocks (list[dict]), manifest (str), notice |
| `pipeline.files_to_llm_input(store, upload_ids, *, user_message, image_method)` | 30 | Gated: ready → content blocks + manifest (graph wraps blocks into the HumanMessage) |
| `builder` image cap | 55 | Only the first `MAX_IMAGES_PER_MESSAGE` (5) images are sent inline; the rest become name-only `[OMITTED]` annotations + a notice asking the user to re-upload if needed |
| `readers/image_reader` size guard | 45 | Raw image > `MAX_IMAGE_BYTES` (32 MB) → `FilePartError` → upload shows ✗ and send is blocked |
| `readers/docling_reader` picture guard | 69 | docling figure PNG > 32 MB → inline text annotation (does not fail the whole document) |
| `upload_store.FileUploadStore` | — | Background parse store (add_file/get_status/get_processed/remove) |
| `processing.process_file(path)` | 20 | Route one file to the right reader (text / image / docling) → `ProcessedFile` |
| `processing.process_files(paths)` | 31 | Batch variant; errors become `FilePartError` entries, never raise |
| `__init__.process_file(s)` | — | Re-exported from `file_input.processing` (public API; lives in its own module to avoid a cycle with `upload_store`) |


---

## workflow/llm.py (LLM Singleton)
| Symbol | Line | Description |
|--------|------|-------------|
| `get_llm()` | 33 | Factory: returns the main DeepSeek chat model (name from `effective_model_settings()`, currently a vision model) |
| `get_llm_no_thinking()` | 86 | Singleton main model with thinking disabled |
| `get_small_llm()` | 104 | Singleton Qwen 3.5 9B via SiliconFlow (thinking disabled) |
| `is_vision_capable(model_name=None)` | 140 | Whether the (effective) main model can accept image blocks, via `config.VISION_MODEL_MARKERS` |

---

## workflow/timing.py (LLM Latency Tracking)
| Symbol | Line | Description |
|--------|------|-------------|
| `reset_timings()` | 27 | Clear the per-round timing registry |
| `record_timing(label, seconds)` | 32 | Append one LLM call duration to the registry |
| `timed_invoke(label, func, *args, **kwargs)` | 40 | Wrap a sync LLM invoke: time + log + record, return result |
| `timed_ainvoke(label, func, *args, **kwargs)` | 58 | Wrap an async LLM invoke (`.ainvoke`): time + log + record |
| `print_latency_table(total_seconds)` | 76 | Print aggregated latency table (per-label count/total/avg + LLM total + end-to-end) |

Wraps every production LLM call site (~26): router, agent, planner init/revision, checker,
worker tool loop + summarizers, compression/eviction/topic, note/expectation/state injection,
retrieve-branch retrievers. The main REPL loop prints the table after each conversation round.

---

## execution/worker.py (Worker Agent)
| Symbol | Line | Description |
|--------|------|-------------|
| `get_tools_for_category(category)` | 56 | Get tools for a predicted category |
| `WorkerState` | 147 | Dataclass: task, predicted_tool_category, instruction, round/round_limit, status, messages, llm, categorized_toolkit, tools, report (single output) |
| `_build_worker_system_prompt(state)` | 164 | Build system prompt for Worker |
| `_build_worker_initial_human_prompt(state)` | 189 | Build initial human prompt for Worker |
| `initialize_worker(state)` | 206 | Initialize worker state from PlanStep |
| `llm_node(state)` | 216 | Invoke LLM with current messages |
| `tool_node(state)` | 325 | Execute tool calls from the last message via `ToolNode`; MCP errors are caught into synthetic `ToolMessage`s |
| `report(report: str)` tool | ~121 | Always-available schema-only tool: worker hands its single report to Main_agent (finished OR blocked/could-not-complete) |
| `report_node(state)` | ~389 | Handle `report` tool call / free text → status "success", sets `WorkerState.report` |
| `failure_summary_node(state)` | ~432 | Budget-limit failure path → status "failed", writes the same `WorkerState.report` |
| `route_worker(state)` | 453 | Route: "tool", "success", or "failure" |
| `build_worker_graph()` | 496 | Build Worker LangGraph |
| `run_worker(state)` | 557 | Compile and run worker graph |

---

## preference_system/models.py (Data Models)
| Symbol | Line | Description |
|--------|------|-------------|
| `PreferenceState` | 30 | Dataclass: state_name, description, event_list, active_time; add_event() |
| `Note` | 46 | Dataclass: content, id, saved_time |
| `NoteCategory` | 56 | Dataclass: category_name, notes, category_description (short <15 words); add_note(content, id), remove_note(index) |
| `load_expectations(path)` | 72 | Read `{expectations: [{trigger, action, expectation_content}]}` JSON → list[dict]; never raises |
| `PreferenceStore` | 91 | Dataclass: states, note_categories, expectations_file |
| `PreferenceStore.add_state/update_state/remove_state/add_event` | — | State operations |
| `PreferenceStore.add_category/remove_category/add_note/remove_note` | — | Note-category ops; add_category(name, description=""); add_note(category, content, id="") auto-generates uuid id |
| `PreferenceStore.get_preference_map()` | — | State + categories + expectations map string for LLM |
| `PreferenceStore.get_state()` | — | `=== State Names ===` section (`name: description`) |
| `PreferenceStore.get_categories()` | — | `=== Note Category Names ===` section (`name: description`, `(no description)` fallback) |
| `PreferenceStore.get_expectations()` | — | `=== Expectations ===` indexed `trigger | action` lines (fresh from JSON file) |
| `PreferenceStore.save/load` | — | Persist/restore full store (incl. expectations_file); load handles Note dicts + legacy raw-string entries |
| `StateMapDecision(BaseModel)` | 299 | Map-retriever decision: state_names (list[str]) |
| `NoteCategoryMapDecision(BaseModel)` | 306 | Map-retriever decision: category_names (list[str]) — exact names only |
| `ExpectationMapDecision(BaseModel)` | 313 | Map-retriever decision: expectation_indices (list[int]) |
| `StateRetrievalResult(BaseModel)` | — | Structured output: include_state, include_history |
| `NoteRetrievalResult(BaseModel)` | — | Structured output: selected_indices (list[int]) |
| `MergedExtractionResult(BaseModel)` | 396 | v2: notes (flat list[str], UNCATEGORIZED) + expectations (list[ExpectationExtractionResult]) from ONE call |
| `AdmissionResult(BaseModel)` | 408 | v2 gate output (3 fields): accepted_expectation_indices (0-based), notes (dict[note_text → category_name]), created_category (dict[new category → <15-word description]); rejected candidates are omitted entirely |

---

## preference_system/retrieve_branch.py (LLM Retrievers)
| Symbol | Line | Description |
|--------|------|-------------|
| `PREFERENCE_MAP_SYSTEM_PROMPT` | — | Map-retriever prompt: state/note/expectation selection independent |
| `STATE_RETRIEVER_SYSTEM_PROMPT` | — | State relevance prompt |
| `NOTE_RETRIEVER_SYSTEM_PROMPT` | — | Note relevance prompt |
| `RETRIEVE_BRANCH_FINAL_PROMPT` | — | Prefix for the retrieved-context block |
| `_run_sync(coro)` | 47 | Run an async coroutine from a sync context (raises if called inside a running loop) |
| `_note_content(note)` | 64 | Extract display text from a Note dataclass or dict |
| `arun_preference_map_retriever(llm, user_message, preference_map_str)` | 105 | Async: LLM selects relevant state/category names + expectation indices |
| `arun_state_retriever(llm, user_message, state_name, state_description, state_event_list)` | 169 | Async: decide include_state/include_history |
| `arun_note_retriever(llm, category_name, user_message, notes)` | 245 | Async: select relevant note indices |
| `RetrievedStateContext` | — | Wrapper: state_name, description, include_state, include_history, event_list; to_context_str() |
| `RetrieveBranchResult` | — | Holds selected_states, selected_notes, selected_expectation (list[str]); to_context_block(), __bool__, __repr__ |
| `arun_retrieve_branch(llm, user_message, store)` | 379 | Async orchestration: map → expectations resolve → concurrent state/note tier-2 retrieval via asyncio.gather |
| `run_retrieve_branch(llm, user_message, store)` | 511 | Sync wrapper around `arun_retrieve_branch` (used by CLI) |

---

## preference_system/note_injection.py (Note Injection)
| Symbol | Line | Description |
|--------|------|-------------|
| `NOTE_MAINTENANCE_THRESHOLD` | — | Maintenance threshold (20 notes per category) |
| `CategoryDescriptionResult(BaseModel)` | — | Structured output: descriptions (dict category name → <15-word description) |
| `CATEGORY_DESCRIPTION_SYSTEM_PROMPT` | — | Backfill prompt: one <15-word description per listed category |
| `backfill_category_descriptions(llm, preference_store)` | — | One structured LLM call fills category_description for every category missing one (llm=None → get_llm_no_thinking); returns count filled; never raises |
| `NoteMaintenanceResult(BaseModel)` | — | Structured output: deleted_notes / merged_notes / merged_content |
| `run_preference_extraction(llm, user_message)` | 438 | v2: ONE structured LLM call → MergedExtractionResult (flat note candidates + expectation candidates, mutual-exclusion routing); None on failure |
| `run_admission_gate(llm, candidate_notes, candidate_expectations, categories_text, neighbor_context="")` | 477 | v2: ONE structured LLM call → AdmissionResult (3 fields: accepted_expectation_indices + notes{note→category} + created_category{name→desc}); rejected candidates omitted; None on failure |
| `_resolve_category_name(preference_store, raw_name)` | 645 | Match a gate-supplied category name case/whitespace-insensitively to an existing category; else treat as new |
| `_format_neighbor_context(position, note, results)` | 665 | Render nearest existing notes for one candidate (gate context only) |
| `prefilter_note_candidates(notes, store, preference_store)` | 683 | v2: hard-drop duplicate/orphan notes against the vector store BEFORE the gate; returns (kept_indices, neighbor_context) |
| `run_note_maintenance(llm, category, store)` | 760 | Review/prune/merge a category's notes when ≥ threshold; None when skipped or LLM fails |
| `persist_categorized_notes(llm, categorized_notes, store, preference_store)` | 891 | Shared writer: (note, category, description) → create/append category, PreferenceStore-first, vector store SAME note_id, rollback, maintenance |

---

## preference_system/preference_injection.py (v2 merged pipeline)
| Symbol | Line | Description |
|--------|------|-------------|
| `PreferenceInjectionResult` | 49 | Dataclass telemetry: notes extracted/duplicate/stored/rejected, expectations extracted/stored/rejected; `summary()` |
| `_clean_notes(raw_notes)` | 71 | Drop empty/whitespace candidates, preserving order |
| `run_preference_injection(llm, user_message, store, preference_store, expectations_file=...)` | 81 | Orchestrator: merged extraction → note prefilter → admission gate → persist notes + expectations; rejected candidates dropped silently (counts derived); None on extraction/gate failure |

---

## expectations.json (Expectations store)
| Symbol | Line | Description |
|--------|------|-------------|
| `expectations` | 2 | User-editable list of `{trigger, action, expectation_content}` behavior rules; indexed 0-based, read fresh each round via `PreferenceStore.expectations_file` |

---

## preference_system/expectation_injection.py (Expectation Injection)
| Symbol | Line | Description |
|--------|------|-------------|
| `DEFAULT_EXPECTATIONS_FILE` | — | Default expectations JSON path (`expectations.json`) |
| `EXPECTATION_LIMIT` | — | Capacity warning threshold (20); warn-only, no auto-eviction |
| `EXPECTATION_EXTRACTION_SYSTEM_PROMPT` | — | Extraction prompt: stable/recurring behavioral rules only; Always trigger; content-first; trigger+action < 15 words |
| `run_expectation_extraction(llm, user_message)` | 98 | Structured LLM call → ExpectationExtractionResult (empty fields = none) or None on failure |
| `append_expectation(expectations_file, expectation_content, trigger, action)` | 142 | Re-read + append to JSON (duplicate by content = no-op); warns when count ≥ EXPECTATION_LIMIT; False on write failure |
| `append_behavior_guide(guide_content, guide_file=...)` | 211 | Append an Always-trigger expectation's content to the behavior guide; warns over length limit |
| `persist_expectation_candidates(candidates, expectations_file=...)` | 291 | Shared writer: deterministic guard (non-empty content/trigger/action) + Always→guide routing + append; returns persisted contents |
| `run_expectation_injection(llm, user_message, expectations_file)` | 338 | LEGACY orchestration: extract → `persist_expectation_candidates`; None (failure) / (False, None) none / (False, reason) invalid / (True, content) recorded |

---

## preference_system/state_injection.py (State Injection — memory pipeline bridge)
| Symbol | Line | Description |
|--------|------|-------------|
| `MAX_STATES` | — | State cap (10); after a new-state create, the earliest-active_time state is evicted when over |
| `STATE_CLASSIFIER_SYSTEM_PROMPT` | — | Classifier prompt: existing state / new state / not state-worthy |
| `STATE_UPDATE_SYSTEM_PROMPT` | — | Updater prompt: revise description + append event (new vs existing state) |
| `run_state_classifier(llm, summary_text, preference_store)` | — | Structured LLM → StateClassificationResult (state_name, new_state) or None on failure |
| `run_state_update(llm, summary_text, state_name, is_new_state, description, event_list)` | — | Structured LLM → StateUpdateResult (description, event) or None on failure |
| `run_state_injection(llm, summary_text, preference_store)` | — | Orchestration: create/update/unchanged/skip; refreshes active_time on change; evicts oldest when over MAX_STATES |

---

## preference_system/cli.py (Standalone Pipeline Test CLI)
| Symbol | Line | Description |
|--------|------|-------------|
| `print_store(store)` | — | Pretty-print states, categories, notes, and expectations |
| `print_help()` | — | Print help text for pipeline test commands |
| `main()` | — | CLI entry point: pipeline test REPL with auto-save persistence |

### CLI pipeline test commands
| Command | Pipeline | Description |
|---------|----------|-------------|
| `/retrieve <message>` | note_retrieve | `run_retrieve_branch` — notes + states + expectations |
| `/note <message>` | note injection | `run_note_injection` — ONE call extract+categorize (dict category→notes), dedupe, create-or-append categories, PreferenceStore-first, vector store SAME note_id, returns list of (note_id, note_content, category_name) |
| `/expectation <message>` | expectation injection | `run_expectation_injection` — extract + persist to expectations.json |
| `/state <summary>` | state injection | `run_state_injection` — classify/create/update state, auto-save |
| `/print`, `/map`, `/help`, `/quit` | — | Store print, preference map, help, save-and-exit |

---

## Test Files

### tests/test_text_utils.py (Shared `clip()` log-truncation helper)
8 tests: `None` → `""`, empty/at-limit text unchanged, over-limit truncation marker, original length reported, non-string coercion, and the `DEFAULT_LOG_LIMIT` default. Exists because the end-to-end net mocks `run_worker_task` and therefore never executes the worker's call sites.

### tests/test_memory_search.py (Agent `memory_search` tool)
15 tests: validation, per-question LTM/preference search, both sources, cross-query dedupe, failure isolation, and tool schema.

### tests/test_preference_injection.py (Preference Injection v2)
20 tests: orchestration (gate failure, accepted/rejected notes, category reuse, prefilter-before-gate, expectation persistence), extraction/gate helpers, prefilter, shared writers, and legacy-path contracts.

### tests/test_preference_injection_pipeline.py (MANUAL runner — not a pytest case)
Runnable script (no `test_*` functions, so pytest only imports it): loads the REAL `PreferenceStore` + `PreferenceVectorStore`, prints a pre-run warning + BEFORE snapshot, prompts for a user message, calls `run_preference_injection(...)`, prints the result telemetry + AFTER snapshot, then saves the store. Run with `python tests/test_preference_injection_pipeline.py [--debug] [--pref-file <path>]`.

### tests/test_state_injection_pipeline.py (MANUAL runner — not a pytest case)
Runnable script (no `test_*` functions, so pytest only imports it): loads the REAL `PreferenceStore`, calls `run_state_injection(summary, store)` (the function fetches its own no-thinking LLM internally), and prints the classifier/updater debug lines, the returned action dict (created/updated/unchanged/None), and the resulting state store before/after saving. Helpers: `setup_logging()`, `print_states(store)`, `describe_result(result)`, `inject_one(store, summary)`, `main()`. Run with `python tests/test_state_injection_pipeline.py [--pref-file <path>] [--summary "<~150-word summary>"]`; without `--summary` it loops interactively.

### tests/test_pipeline_e2e.py (END-TO-END safety net — 14 tests)
Runs the REAL compiled LangGraph through `workflow.graph.run_pipeline` with a scripted mock LLM and fake tool boundaries, so the whole node wiring is exercised with no network/LLM/MCP. Mocked boundaries: agent LLM (`_agent_llm` replays `AIMessage`s), pre-router/router LLM (`_no_think_llm`), small-LLM probe (`_small_llm`), LTM retrieval, `run_worker_task`, `run_memory_search`, the preference retrieve branch, and the preference store. Harness: `_pipeline_env(...)` / `_run(...)`. Classes: `TestFirstRound` (routing + round persistence), `TestToolLoop` (`assign_task` / `memory_search` / invalid category / limits), `TestRoutingIntoExistingThreads` (continuation vs new thread). Validated by mutation testing — disabling `update_memory_node` or replacing `agent_node` makes it fail.

### tests/test_import_contracts.py (STRUCTURAL safety net — 7 tests)
Turns the refactor hazards into failing tests. `TestModulesImport` discovers every app module from disk (`_discover_modules`) and imports it — a missing local module fails, a missing third-party optional dep is reported as an info skip (`_is_missing_optional_dep`, `LOCAL_TOP_LEVEL`). `TestPipelineGraphShape` asserts the graph still has all 8 expected nodes and compiles. `TestPublicApiResolves` asserts the `preference_system` package exports and that symbols still resolve at their documented module (`SYMBOL_LOCATIONS`, `PREFERENCE_SYSTEM_EXPORTS`). `TestEntryPointSignatures` asserts key entry points still accept their documented parameters (`EXPECTED_PARAMS`) — this is what caught the broken `run_state_injection` callers.



## api/ (Web API server) + agent run service
| Symbol | Line | Description |
|--------|------|-------------|
| `api/server.py` | — | FastAPI app (CORS 5173); settings + config + expectation/preference endpoints; **chat**: `POST /api/chat`, `GET /api/chat/runs/{run_id}`, `POST /api/chat/runs/{run_id}/cancel`, `GET /api/chat/history` |
| `api/persistence.py` `get_chat_history/append_chat_history` | 29/38 | Web-UI chat history in `chat_history.json`; 30-round scrolling window (`MAX_CHAT_HISTORY_ROUNDS`), appended server-side when a run finishes |
| `api/server.py` upload/chat endpoints | — | `POST /api/upload`, `GET /api/upload/status?ids=`, `DELETE /api/upload/{id}`; `POST /api/chat` accepts `upload_ids` + `references` → gates readiness, composes human content in order files → referenced rounds → user message, and (if images) requires a vision-capable model else 422 |
| `api/server.py` reference helpers | — | `_reference_rounds` (cap `MAX_REFERENCES=3`, truncate), `_reference_content_blocks` (full referenced rounds → human block), `_reference_marker` (metadata-only `[referenced earlier round @ …]` for subsystems/memory) |
| `api/files.py` `get_store/save_upload/original_name/status_of/discard` | — | File-upload bridge: writes each upload to `uploads/<token>/<original-name>` (keeps the real filename), feeds the process-wide `FileUploadStore`, maps upload_id→filename/path, status snapshot, discard |
| `api/agent.py` `AgentSession` | — | Lazily built long-lived session (thread_bar/vector_store/toolkit/preference store) mirroring `main.py` init; `run_message()` **reloads thread_bar + PreferenceStore + runtime config from disk each round** (honors external edits), runs `run_pipeline`, waits for background injections, persists thread bar; on run success appends the round to chat history (non-fatal on failure) |
| `api/agent.py` `start_run(user_message, upload_ids, human_message_content, reference_labels)` | — | run-id registry + single-active-run guard; `user_message` is the ABBREVIATED input, `human_message_content` the full content blocks, `reference_labels` the referenced-round labels for history display; frees uploads (store+disk) in `finally` |
| `file_input/builder.py` composition | — | Files block FIRST (`FILES_UPLOADED_HEADER`) with text/image/table sections, then labelled `User Message` appended after; `build_content_blocks(processed, user_message=...)` returns the content-block list (the graph's `human_message_node` wraps it into the HumanMessage) |
| `api/agent.py` `start_run/get_run/cancel_run` | — | run-id registry + single-active-run guard (409 when busy); result payload: agent_response, underlying_content, router_result, thread_was_evicted, response_time_s |
| `workflow/run_control.py` | — | Cooperative cancellation: `CancelledRun`, `begin_run/request_cancel/clear_run/cancel_requested/check_cancel` |

63 tests covering all PreferenceState, Note, NoteCategory, PreferenceStore operations, all retrieval models, run_preference_map_retriever, run_state_retriever, run_note_retriever, run_retrieve_branch, and edge cases.