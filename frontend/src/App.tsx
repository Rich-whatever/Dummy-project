import { useCallback, useEffect, useRef, useState } from "react";
import { Sidebar, LocalClock } from "./components/Sidebar";
import { NeuralOrb } from "./components/NeuralOrb";
import { HistoryWindow } from "./components/HistoryWindow";
import { ChatComposer } from "./components/ChatComposer";
import { AgentStatusBar } from "./components/AgentStatusBar";
import { ResponseCard } from "./components/ResponseCard";
import { SettingsPage } from "./components/settings/SettingsPage";
import type { Attachment, ConversationRound, OrbState, ReferencedRound, View } from "./types";
import { sendChat, getChatStatus, cancelChat, fetchChatHistory, uploadFile, fetchUploadStatus, removeUpload } from "./api";

function DarkModeToggle({ isDark, onToggle }: { isDark: boolean; onToggle: () => void }) {
  return (
    <button onClick={onToggle} className="flex items-center gap-2 px-3 py-1.5 rounded-full transition-all"
      style={{ background: "var(--muted)", color: "var(--muted-foreground)", border: "1px solid var(--border)", fontSize: 11 }} aria-label="Toggle dark mode">
      <span style={{ fontSize: 13 }}>{isDark ? "◑" : "○"}</span>
      <span className="font-mono tracking-widest uppercase" style={{ fontSize: 9 }}>{isDark ? "Dark" : "Light"}</span>
    </button>
  );
}


// ─── Root ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [isDark, setIsDark] = useState(true);
  const [view, setView] = useState<View>("chat");
  const [orbState, setOrbState] = useState<OrbState>("idle");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [rounds, setRounds] = useState<ConversationRound[]>([]);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [references, setReferences] = useState<ReferencedRound[]>([]);
  const mainRef = useRef<HTMLDivElement>(null);
  const [activeResponseId, setActiveResponseId] = useState<string | null>(null);
  const activeRequest = useRef<{ roundId: string; runId: string; poll: ReturnType<typeof setInterval> | null } | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
  }, [isDark]);

  // Hydrate the persisted conversation history (30-round scrolling window,
  // stored backend-side in chat_history.json) once on mount.
  useEffect(() => {
    let cancelled = false;
    fetchChatHistory()
      .then(({ rounds: history }) => {
        if (cancelled) return;
        setRounds(
          history.map((r, i) => ({
            id: `hist-${i}-${r.timestamp}`,
            user_message: r.user_message,
            agent_message: r.agent_response,
            attachments: [],
            references: (r.references ?? []).map((x, j) => ({
              id: `histref-${i}-${j}`,
              timestamp: x.timestamp,
              user_message: x.user_message,
              agent_message: x.agent_message,
            })),
            timestamp: new Date(r.timestamp),
            meta: {
              response_time_s: r.response_time_s,
              tokens_in: 0,
              tokens_out: 0,
              underlying_content: r.underlying_content ?? "",
              worker_report: "",
            },
          })),
        );
      })
      .catch((err) => console.error("Failed to load chat history:", err));
    return () => {
      cancelled = true;
    };
  }, []);

  const patchAttachment = useCallback((id: string, patch: Partial<Attachment>) => {
    setAttachments((prev) => prev.map((a) => (a.id === id ? { ...a, ...patch } : a)));
  }, []);

  // Upload each dropped/picked file to the backend and start tracking its
  // parse status. Uploads happen one-per-file; the polling effect below picks
  // them up as soon as they carry an uploadId.
  const addFiles = useCallback((files: FileList | File[]) => {
    const list = Array.from(files);
    list.forEach((f) => {
      const localId = `${Date.now()}-${f.name}-${Math.random().toString(36).slice(2, 7)}`;
      setAttachments((prev) => [
        ...prev,
        { id: localId, name: f.name, size: f.size, status: "processing" },
      ]);
      uploadFile(f)
        .then(({ upload_id }) => patchAttachment(localId, { uploadId: upload_id }))
        .catch(() =>
          patchAttachment(localId, {
            status: "error",
            errorReason: "Upload to the agent backend failed. Is it running?",
          }),
        );
    });
  }, [patchAttachment]);

  // Poll server-side parse status for every in-flight (processing) upload that
  // already has an upload_id. Stops automatically once none remain.
  const processingIds = attachments
    .filter((a) => a.uploadId && a.status === "processing")
    .map((a) => a.uploadId as string);
  const processingKey = processingIds.join(",");

  useEffect(() => {
    if (!processingIds.length) return;
    const poll = setInterval(async () => {
      try {
        const { uploads } = await fetchUploadStatus(processingIds);
        setAttachments((prev) =>
          prev.map((a) => {
            if (!a.uploadId || a.status !== "processing") return a;
            const rec = uploads.find((u) => u.upload_id === a.uploadId);
            if (!rec) return a;
            if (rec.status === "done") return { ...a, status: "ready", errorReason: undefined };
            if (rec.status === "error")
              return { ...a, status: "error", errorReason: rec.reason || "File could not be processed." };
            return a; // still pending
          }),
        );
      } catch {
        /* transient network error — retry next tick */
      }
    }, 800);
    return () => clearInterval(poll);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [processingKey]);

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => {
      const target = prev.find((a) => a.id === id);
      if (target?.uploadId) removeUpload(target.uploadId).catch(() => {});
      return prev.filter((a) => a.id !== id);
    });
  }, []);

  // Referenced conversation rounds (Option A: content goes to the agent only;
  // the backend adds a metadata marker to the stored/classified user_message).
  const addReference = useCallback((ref: ReferencedRound) => {
    setReferences((prev) =>
      prev.some((r) => r.id === ref.id) ? prev : [...prev, ref].slice(0, 3),
    );
  }, []);

  const removeReference = useCallback((id: string) => {
    setReferences((prev) => prev.filter((r) => r.id !== id));
  }, []);

  const handleDropRound = useCallback(
    (r: { timestamp: string; user_message: string; agent_message: string }) => {
      addReference({
        id: `drag-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        timestamp: String(r.timestamp ?? ""),
        user_message: String(r.user_message ?? ""),
        agent_message: String(r.agent_message ?? ""),
      });
    },
    [addReference],
  );

  const finalizeRound = (roundId: string, update: Partial<ConversationRound>) => {
    setRounds((prev) => prev.map((r) => (r.id === roundId ? { ...r, ...update } : r)));
  };

  const stopPolling = () => {
    const a = activeRequest.current;
    if (a && a.poll) { clearInterval(a.poll); a.poll = null; }
    activeRequest.current = null;
  };

  const discardPending = (roundId: string) => {
    setRounds((prev) => prev.filter((r) => r.id !== roundId));
  };

  const failRound = (roundId: string, message: string) => {
    setOrbState("idle");
    finalizeRound(roundId, {
      agent_message: message,
      meta: { response_time_s: 0, tokens_in: 0, tokens_out: 0, underlying_content: "", worker_report: "" },
    });
    setActiveResponseId(roundId);
  };

  const handleSend = (text: string) => {
    const roundId = "r" + Date.now();
    const roundAttachments = attachments;
    const roundReferences = references;
    setRounds((prev) => [
      ...prev,
      {
        id: roundId,
        user_message: text,
        agent_message: "",
        attachments: roundAttachments,
        references: roundReferences,
        timestamp: new Date(),
        meta: { response_time_s: 0, tokens_in: 0, tokens_out: 0, underlying_content: "", worker_report: "" },
      },
    ]);
    setAttachments([]);
    setReferences([]);
    setOrbState("thinking");
    setActiveResponseId(null);

    // Only successfully-parsed files are sent; blocked/errored ones never make
    // it here (the composer disables send while any attachment isn't ready).
    const uploadIds = roundAttachments
      .filter((a) => a.status === "ready" && a.uploadId)
      .map((a) => a.uploadId as string);

    const refPayload = roundReferences.map((r) => ({
      timestamp: r.timestamp,
      user_message: r.user_message,
      agent_message: r.agent_message,
    }));

    sendChat(text, uploadIds, refPayload)
      .then(({ run_id }) => {
        const started = Date.now();
        activeRequest.current = { roundId, runId: run_id, poll: null };
        const poll = setInterval(async () => {
          try {
            const st = await getChatStatus(run_id);
            if (st.status === "done" && st.result) {
              stopPolling();
              const elapsed = st.result.response_time_s || (Date.now() - started) / 1000;
              finalizeRound(roundId, {
                agent_message: st.result.agent_response,
                meta: {
                  response_time_s: elapsed,
                  tokens_in: 0,
                  tokens_out: 0,
                  underlying_content: st.result.underlying_content || "",
                  worker_report: "",
                },
              });
              setOrbState("idle");
              setActiveResponseId(roundId);
            } else if (st.status === "cancelled") {
              stopPolling();
              setOrbState("idle");
              discardPending(roundId);
            } else if (st.status === "failed") {
              stopPolling();
              failRound(roundId, "(agent error) " + (st.error || "run failed"));
            }
          } catch {
            stopPolling();
            failRound(roundId, "(connection error talking to the agent backend)");
          }
        }, 800);
        if (activeRequest.current) activeRequest.current.poll = poll;
      })
      .catch(() => {
        setOrbState("idle");
        discardPending(roundId);
        finalizeRound(roundId, {
          agent_message: "(could not start the agent run - is the backend running?)",
          meta: { response_time_s: 0, tokens_in: 0, tokens_out: 0, underlying_content: "", worker_report: "" },
        });
        setActiveResponseId(roundId);
      });
  };

  const handleCancel = () => {
    const a = activeRequest.current;
    if (a) { cancelChat(a.runId).catch(() => {}); }
    stopPolling();
    setRounds((prev) => prev.filter((r) => r.agent_message !== ""));
    setOrbState("idle");
    setActiveResponseId(null);
  };

  const activeRound = activeResponseId ? rounds.find((r) => r.id === activeResponseId) ?? null : null;

  if (view === "settings") {
    return (
      <div className="flex h-full overflow-hidden" style={{ background: "var(--background)" }}>
        <Sidebar onOpenSettings={() => setView("settings")} />
        <div className="flex-1 min-w-0 overflow-hidden relative">
          <div className="absolute top-4 right-6 z-20"><DarkModeToggle isDark={isDark} onToggle={() => setIsDark((d) => !d)} /></div>
          <SettingsPage onBack={() => setView("chat")} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full overflow-hidden" style={{ background: "var(--background)" }}>
      <Sidebar onOpenSettings={() => setView("settings")} />

      {/* Main orb area (floating windows are clamped inside this container) */}
      <div
        ref={mainRef}
        className="flex-1 min-w-0 overflow-hidden relative"
        style={{ background: "var(--background)" }}
      >

        {/* Full-area neural orb canvas */}
        <NeuralOrb orbState={orbState} isDark={isDark} />

        {/* Vignette edges */}
        <div className="absolute inset-0 pointer-events-none" style={{
          background: isDark
            ? "radial-gradient(ellipse at (50%) 50%, transparent 35%, rgba(6,10,18,0.75) 100%)"
            : "radial-gradient(ellipse at (50%) 50%, transparent 52%, rgba(255,255,255,0.30) 100%)",
          zIndex: 1,
        }} />

        {/* Top bar */}
        <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-6 py-4" style={{ zIndex: 10 }}>
          {/* History toggle */}
          <button
            onClick={() => setHistoryOpen((o) => !o)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full transition-all"
            style={{
              background: historyOpen ? "var(--accent)" : "var(--muted)",
              color: historyOpen ? "var(--accent-foreground)" : "var(--muted-foreground)",
              border: "1px solid var(--border)",
              fontSize: 11,
            }}
          >
            <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.3">
              <circle cx="5.5" cy="4.5" r="3.5" />
              <path d="M5.5 3V5l1.2 1.2" strokeLinecap="round" />
              <path d="M1 9.5c1-1.5 2.5-2 4.5-2s3.5.5 4.5 2" strokeLinecap="round" opacity="0.5" />
            </svg>
            <span className="font-mono tracking-widest uppercase" style={{ fontSize: 9 }}>Conversation History</span>
          </button>

          {/* Local time / date above the orb (absolutely centered on the orb) */}
          <div className="absolute left-1/2 -translate-x-1/2" style={{ top: 10, zIndex: 10 }}>
            <LocalClock />
          </div>

          <DarkModeToggle isDark={isDark} onToggle={() => setIsDark((d) => !d)} />
        </div>

        {/* Active status under the orb */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ zIndex: 2 }}>
          <div className="flex items-center gap-2.5" style={{ transform: `translateY(${220}px)` }}>
            <div className="relative w-3 h-3 flex items-center justify-center">
              <div
                className="absolute w-3 h-3 rounded-full border"
                style={{ borderColor: "var(--accent)", animation: "pulse-ring 2s ease-out infinite" }}
              />
              <div
                className="w-1.5 h-1.5 rounded-full"
                style={{
                  background: "var(--accent)",
                  animation: "breathe 3s ease-in-out infinite",
                  boxShadow: orbState === "thinking" ? "0 0 6px var(--accent)" : "none",
                }}
              />
            </div>
            <p className="font-mono tracking-widest uppercase" style={{ fontSize: 16, color: "var(--muted-foreground)" }}>
              {orbState === "thinking" ? "Thinking" : "Active"}
            </p>
          </div>
        </div>

        {/* Draggable history window */}
        {historyOpen && (
          <HistoryWindow
            rounds={rounds}
            onClose={() => setHistoryOpen(false)}
            containerRef={mainRef}
          />
        )}

        {/* Chat composer / execution status bar */}
        {orbState === "thinking" ? (
          <AgentStatusBar onCancel={handleCancel} />
        ) : (
          <ChatComposer
            onSend={handleSend}
            isThinking={false}
            isDark={isDark}
            attachments={attachments}
            onAddFiles={addFiles}
            onRemoveAttachment={removeAttachment}
            references={references}
            onRemoveReference={removeReference}
            onDropRound={handleDropRound}
          />
        )}

        {/* Fresh response overlay */}
        {activeRound && activeRound.agent_message && (
          <ResponseCard
            round={activeRound}
            onMinimize={() => setActiveResponseId(null)}
            containerRef={mainRef}
          />
        )}

        {/* Detail page (underlying content / worker report) */}
      </div>
    </div>
  );
}
