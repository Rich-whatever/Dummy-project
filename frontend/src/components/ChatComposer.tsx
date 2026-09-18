import React, { useCallback, useRef, useState } from "react";
import type { Attachment, ReferencedRound } from "../types";

// ── Drop-zone sizing ────────────────────────────────────────────────
// The drop/hit area is bigger than the visible composer so an intentional
// drag is easy, but it stays bounded to the panel width (never full-screen).
// Tweak these two numbers to resize the zone.
const DROP_W = "min(1152px, 100%)"; // ≈2× the 576px composer, capped to the panel
const DROP_PAD_Y = 70;              // extra hit area above the composer (px)

export function ChatComposer({
  onSend,
  isThinking,
  isDark,
  attachments,
  onAddFiles,
  onRemoveAttachment,
  references,
  onRemoveReference,
  onDropRound,
}: {
  onSend: (text: string) => void;
  isThinking: boolean;
  isDark: boolean;
  attachments: Attachment[];
  onAddFiles: (files: FileList) => void;
  onRemoveAttachment: (id: string) => void;
  references: ReferencedRound[];
  onRemoveReference: (id: string) => void;
  onDropRound: (r: { timestamp: string; user_message: string; agent_message: string }) => void;
}) {
  const [draft, setDraft] = useState("");
  const [dropActive, setDropActive] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Send gating: disabled while the agent is thinking, while any attachment is
  // still processing (parse not finished), or while any attachment errored
  // (user must remove it). Allowed when there's text OR a ready file.
  const hasText = draft.trim().length > 0;
  const hasReadyFile = attachments.some((a) => a.status === "ready");
  const processingFile = attachments.some((a) => a.status === "processing");
  const erroredFile = attachments.some((a) => a.status === "error");
  const canSend =
    !isThinking && !processingFile && !erroredFile &&
    (hasText || hasReadyFile || references.length > 0);

  const handleSend = useCallback(() => {
    if (!canSend) return;
    const text = draft.trim();
    setDraft("");
    onSend(text);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }, [draft, canSend, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setDraft(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
  };

  return (
    <div className="absolute bottom-0 left-0 right-0 px-8 py-10" style={{ zIndex: 10 }}>
      {/* Drop zone: bigger than the composer, bounded to the panel (not full screen) */}
      <div
        className="relative mx-auto rounded-2xl transition-all"
        onDragOver={(e) => {
          e.preventDefault();
          setDropActive(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          if (e.currentTarget.contains(e.relatedTarget as Node)) return;
          setDropActive(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDropActive(false);
          const dt = e.dataTransfer;
          if (dt.files && dt.files.length) {
            onAddFiles(dt.files);
            return;
          }
          const raw = dt.getData("application/x-dummy-round");
          if (raw) {
            try {
              onDropRound(JSON.parse(raw));
            } catch {
              /* ignore malformed payload */
            }
          }
        }}
        style={{
          width: DROP_W,
          paddingTop: DROP_PAD_Y,
          ...(dropActive ? { boxShadow: "0 0 0 2px var(--accent)", borderRadius: 20 } : {}),
        }}
      >
        <div className="max-w-xl mx-auto">
        {/* Referenced-round chips */}
        {references.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-2 px-1">
            {references.map((r) => {
              const d = new Date(r.timestamp);
              const label = isNaN(d.getTime())
                ? r.timestamp
                : d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
              return (
                <div
                  key={r.id}
                  title={r.user_message.slice(0, 200)}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg"
                  style={{ background: "var(--muted)", border: "1px solid var(--border)" }}
                >
                  <span className="font-mono" style={{ fontSize: 14, color: "var(--accent)" }}>@</span>
                  <span className="text-sm" style={{ color: "var(--foreground)" }}>
                    conversation at {label}
                  </span>
                  <button
                    onClick={() => onRemoveReference(r.id)}
                    className="flex items-center justify-center w-4 h-4 rounded hover:opacity-100"
                    style={{ color: "var(--muted-foreground)", opacity: 0.7 }}
                    aria-label="Remove reference"
                  >
                    <svg width="9" height="9" viewBox="0 0 8 8" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                      <line x1="1" y1="1" x2="7" y2="7" />
                      <line x1="7" y1="1" x2="1" y2="7" />
                    </svg>
                  </button>
                </div>
              );
            })}
          </div>
        )}

        {/* Attachment chips */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-2 px-1">
            {attachments.map((a) => {
              const st = a.status;
              const statusColor =
                st === "error" ? "#D9574E" : st === "ready" ? "#4CB874" : "var(--muted-foreground)";
              const borderColor =
                st === "error" ? "rgba(217,87,78,0.55)" : st === "ready" ? "rgba(76,184,116,0.5)" : "var(--border)";
              return (
                <div
                  key={a.id}
                  title={st === "error" ? a.errorReason : undefined}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg"
                  style={{ background: "var(--muted)", border: `1px solid ${borderColor}` }}
                >
                  <svg width="12" height="12" viewBox="0 0 10 12" fill="none" stroke="currentColor" strokeWidth="1.2" style={{ color: "var(--accent)" }}>
                    <path d="M6 1H2a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4L6 1z" />
                    <path d="M6 1v3h3" />
                  </svg>
                  <span className="text-sm" style={{ color: "var(--foreground)", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{a.name}</span>
                  <span className="font-mono" style={{ fontSize: 11, color: "var(--muted-foreground)" }}>{(a.size / 1024).toFixed(0)} KB</span>
                  {/* status indicator: pulsing dot = processing, ✓ = ready, ✗ = error */}
                  {st === "processing" ? (
                    <span className="w-2 h-2 rounded-full inline-block" style={{ background: statusColor, animation: "breathe 1.2s ease-in-out infinite" }} />
                  ) : st === "ready" ? (
                    <svg width="13" height="13" viewBox="0 0 12 12" fill="none" stroke={statusColor} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="2,6 4.8,8.8 10,3" />
                    </svg>
                  ) : st === "error" ? (
                    <svg width="13" height="13" viewBox="0 0 12 12" fill="none" stroke={statusColor} strokeWidth="1.8" strokeLinecap="round">
                      <line x1="3" y1="3" x2="9" y2="9" />
                      <line x1="9" y1="3" x2="3" y2="9" />
                    </svg>
                  ) : null}
                  <button
                    onClick={() => onRemoveAttachment(a.id)}
                    className="flex items-center justify-center w-4 h-4 rounded hover:opacity-100"
                    style={{ color: "var(--muted-foreground)", opacity: 0.7 }}
                    aria-label={`Remove ${a.name}`}
                  >
                    <svg width="9" height="9" viewBox="0 0 8 8" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                      <line x1="1" y1="1" x2="7" y2="7" />
                      <line x1="7" y1="1" x2="1" y2="7" />
                    </svg>
                  </button>
                </div>
              );
            })}
          </div>
        )}

        <div
          className="flex items-end gap-3 rounded-2xl px-4 py-5"
          style={{
            background: isDark ? "rgba(10,16,32,0.88)" : "rgba(250,246,234,0.92)",
            border: "1px solid var(--border)",
            backdropFilter: "blur(16px)",
            boxShadow: isDark
              ? "0 0 0 1px rgba(79,144,232,0.08), 0 8px 32px rgba(0,0,0,0.5)"
              : "0 0 0 1px rgba(217,164,6,0.12), 0 8px 24px rgba(0,0,0,0.1)",
          }}
        >
          {/* Attach (+) button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isThinking}
            className="shrink-0 w-8 h-8 flex items-center justify-center rounded-lg transition-all mb-0.5 hover:opacity-100"
            style={{
              background: "var(--muted)",
              color: "var(--muted-foreground)",
              opacity: 0.85,
            }}
            aria-label="Attach files"
            title="Attach files"
          >
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
              <line x1="6.5" y1="2" x2="6.5" y2="11" />
              <line x1="2" y1="6.5" x2="11" y2="6.5" />
            </svg>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) onAddFiles(e.target.files);
              e.target.value = ""; // allow re-picking the same file
            }}
          />
          {isThinking && (
            <div className="flex items-center gap-1.5 mr-1 mb-1.5">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className={`typing-dot w-1.5 h-1.5 rounded-full`}
                  style={{ background: "var(--accent)" }}
                />
              ))}
            </div>
          )}
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={isThinking ? "Dummy is thinking…" : "Message Dummy…"}
            rows={1}
            disabled={isThinking}
            className="flex-1 text-sm leading-relaxed overflow-hidden"
            style={{ minHeight: 24, maxHeight: 140 }}
          />
          <button
            onClick={handleSend}
            disabled={!canSend}
            className="shrink-0 w-8 h-8 flex items-center justify-center rounded-lg transition-all"
            style={{
              background: canSend ? "var(--accent)" : "var(--muted)",
              color: canSend ? "var(--accent-foreground)" : "var(--muted-foreground)",
              boxShadow: canSend ? "0 0 14px var(--accent-glow)" : "none",
            }}
            aria-label="Send"
            title={!canSend && processingFile ? "Waiting for file to finish processing…"
              : !canSend && erroredFile ? "Remove the failed file before sending"
              : "Send"}
          >
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <line x1="6.5" y1="11" x2="6.5" y2="2" />
              <polyline points="2.5,6 6.5,2 10.5,6" />
            </svg>
          </button>
        </div>
        <p className="text-center font-mono mt-2" style={{ fontSize:13, color: "var(--muted-foreground)", opacity: 0.4 }}>
          ↵ send · ⇧↵ newline
        </p>
        </div>
      </div>
    </div>
  );
}
