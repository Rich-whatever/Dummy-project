import React, { useEffect, useRef, useState } from "react";
import type { ConversationRound } from "../types";
import { useFloatingWindow } from "../hooks/useFloatingWindow";

export function HistoryWindow({
  rounds,
  onClose,
  containerRef,
}: {
  rounds: ConversationRound[];
  onClose: () => void;
  containerRef: React.RefObject<HTMLElement | null>;
}) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { rect, onTitleMouseDown, onResizeMouseDown } = useFloatingWindow(
    { x: 80, y: 100, w: 820, h: 580 },
    containerRef,
    { minW: 420, minH: 260 },
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [rounds]);

  const fmtTok = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

  return (
    <div
      className="absolute rounded-xl overflow-hidden flex flex-col"
      style={{
        left: rect.x,
        top: rect.y,
        width: rect.w,
        height: rect.h,
        zIndex: 50,
        background: "var(--card)",
        border: "1px solid var(--border)",
        boxShadow: "0 24px 64px rgba(0,0,0,0.35), 0 0 0 1px rgba(79,144,232,0.06)",
        backdropFilter: "blur(20px)",
      }}
    >
      {/* Title bar */}
      <div
        onMouseDown={onTitleMouseDown}
        className="flex items-center justify-between px-4 py-3 shrink-0 select-none"
        style={{
          borderBottom: "1px solid var(--border)",
          cursor: "grab",
          background: "var(--secondary)",
        }}
      >
        <div className="flex items-center gap-2">
          <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.3" style={{ color: "var(--muted-foreground)" }}>
            <circle cx="5.5" cy="4" r="2.5" />
            <path d="M1 10c0-2.5 2-4 4.5-4s4.5 1.5 4.5 4" strokeLinecap="round" />
          </svg>
          <span className="font-mono tracking-widest uppercase" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>
            Conversation history
          </span>
        </div>
        <button
          onClick={onClose}
          className="w-5 h-5 flex items-center justify-center rounded transition-opacity hover:opacity-100"
          style={{ color: "var(--muted-foreground)", opacity: 0.6 }}
          aria-label="Close"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <line x1="1" y1="1" x2="9" y2="9" />
            <line x1="9" y1="1" x2="1" y2="9" />
          </svg>
        </button>
      </div>

      {/* Conversation rounds */}
      <div className="flex-1 overflow-y-auto py-2">
        {rounds.map((round) => {
          const expanded = expandedId === round.id;
          const hovered = hoveredId === round.id;
          const time = round.timestamp.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
          return (
            <div
              key={round.id}
              onMouseEnter={() => setHoveredId(round.id)}
              onMouseLeave={() => setHoveredId((h) => (h === round.id ? null : h))}
              className="relative px-4 pt-3 pb-1"
            >
              {/* Small drag grip — the ONLY drag source for referencing a round */}
              <span
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData(
                    "application/x-dummy-round",
                    JSON.stringify({
                      timestamp: round.timestamp.toISOString(),
                      user_message: round.user_message,
                      agent_message: round.agent_message,
                    }),
                  );
                  e.dataTransfer.effectAllowed = "copy";
                }}
                onMouseDown={(e) => e.stopPropagation()}
                title="Drag into the chat box to reference this round"
                aria-label="Drag to reference"
                className="absolute left-0.5 top-3.5 select-none"
                style={{
                  cursor: "grab",
                  color: "var(--muted-foreground)",
                  opacity: hovered ? 0.7 : 0,
                  transition: "opacity 120ms",
                  fontSize: 27,
                  lineHeight: 1,
                }}
              >
                ⠿
              </span>
              {expanded ? (
                /* ── Expanded: replace round content with the execution detail ── */
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono tracking-widest uppercase" style={{ fontSize: 12, color: "var(--accent)" }}>
                      Execution detail
                    </span>
                    <button
                      onClick={() => setExpandedId(null)}
                      className="font-mono px-2 py-0.5 rounded transition-colors hover:bg-[color:var(--secondary)]"
                      style={{ fontSize: 12, color: "var(--muted-foreground)", border: "1px solid var(--border)" }}
                    >
                      Fold
                    </button>
                  </div>
                  <pre
                    className="text-xs leading-relaxed whitespace-pre-wrap font-mono"
                    style={{
                      color: "var(--foreground)",
                      background: "var(--secondary)",
                      borderRadius: 8,
                      padding: "10px 12px",
                      maxHeight: 280,
                      overflowY: "auto",
                    }}
                  >
                    {round.meta.underlying_content || "(no execution detail recorded)"}
                  </pre>
                </div>
              ) : (
                /* ── Normal round: user bubble + agent message ── */
                <>
                  <div className="flex justify-end">
                    <div style={{ maxWidth: 400 }}>
                      {(round.references?.length ?? 0) > 0 && (
                        <div className="flex flex-wrap gap-1 justify-end mb-1">
                          {round.references!.map((r) => {
                            const d = new Date(r.timestamp);
                            const label = isNaN(d.getTime())
                              ? r.timestamp
                              : d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
                            return (
                              <span key={r.id} className="font-mono px-2 py-0.5 rounded" style={{ fontSize: 9, background: "var(--muted)", color: "var(--accent)", border: "1px solid var(--border)" }}>
                                @ {label}
                              </span>
                            );
                          })}
                        </div>
                      )}
                      {round.attachments.length > 0 && (
                        <div className="flex flex-wrap gap-1 justify-end mb-1">
                          {round.attachments.map((a) => (
                            <span key={a.id} className="font-mono px-2 py-0.5 rounded" style={{ fontSize: 9, background: "var(--muted)", color: "var(--muted-foreground)", border: "1px solid var(--border)" }}>
                              {a.name}
                            </span>
                          ))}
                        </div>
                      )}
                      <div className="px-3 py-2 rounded-xl rounded-tr-sm inline-block" style={{ background: "var(--secondary)" }}>
                        <p className="text-xs leading-relaxed whitespace-pre-wrap" style={{ color: "var(--foreground)" }}>{round.user_message}</p>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 mt-2">
                    <div className="w-px self-stretch mt-1 shrink-0" style={{ background: "var(--accent)", minWidth: 1 }} />
                    <div className="flex-1 min-w-0">
                      {round.agent_message ? (
                        <p className="text-xs leading-relaxed whitespace-pre-wrap" style={{ color: "var(--foreground)" }}>{round.agent_message}</p>
                      ) : (
                        <p className="text-xs font-mono" style={{ color: "var(--muted-foreground)" }}>…</p>
                      )}
                    </div>
                  </div>
                </>
              )}

              {/* Hover / meta bar (hides on mouse-leave unless expanded) */}
              {(hovered || expanded) && (
                <div className="flex items-center gap-3 mt-1.5 ml-1 px-3 py-1.5 rounded-lg" style={{ background: "var(--muted)" }}>
                  <span className="font-mono" style={{ fontSize: 12, color: "var(--muted-foreground)" }}>{time}</span>
                  <span className="font-mono" style={{ fontSize: 12, color: "var(--muted-foreground)" }}>{round.meta.response_time_s.toFixed(1)}s</span>
                  <span className="font-mono" style={{ fontSize: 12, color: "var(--muted-foreground)" }}>↑ {fmtTok(round.meta.tokens_in)} in</span>
                  <span className="font-mono" style={{ fontSize: 12, color: "var(--muted-foreground)" }}>↓ {fmtTok(round.meta.tokens_out)} out</span>
                  <div className="flex-1" />
                  <button
                    onClick={() => setExpandedId(expanded ? null : round.id)}
                    className="font-mono px-2 py-0.5 rounded transition-colors hover:bg-[color:var(--secondary)]"
                    style={{ fontSize: 12, color: "var(--accent)", border: "1px solid var(--border)" }}
                  >
                    {expanded ? "Fold" : "View detail"}
                  </button>
                </div>
              )}
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Resize grip (bottom-right corner) */}
      <span
        onMouseDown={onResizeMouseDown}
        title="Resize"
        className="absolute bottom-0 right-0 w-4 h-4 z-[60]"
        style={{ cursor: "nwse-resize" }}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" style={{ display: "block", color: "var(--muted-foreground)", opacity: 0.5 }}>
          <path d="M13 5 L5 13 M13 9 L9 13" stroke="currentColor" strokeWidth="1.2" fill="none" strokeLinecap="round" />
        </svg>
      </span>
    </div>
  );
}
