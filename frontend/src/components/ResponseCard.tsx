import React, { useEffect, useRef, useState } from "react";
import type { ConversationRound } from "../types";
import { useFloatingWindow } from "../hooks/useFloatingWindow";

export function ResponseCard({
  round,
  onMinimize,
  containerRef,
}: {
  round: ConversationRound;
  onMinimize: () => void;
  containerRef: React.RefObject<HTMLElement | null>;
}) {
  const [open, setOpen] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);

  const { rect, onTitleMouseDown, onResizeMouseDown } = useFloatingWindow(
    { x: 0, y: 0, w: 640, h: 460 },
    containerRef,
    { minW: 360, minH: 200, center: true },
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onMinimize();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onMinimize]);

  // When detail opens, grow downward by scrolling the body to the bottom.
  useEffect(() => {
    if (open && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [open]);

  const hasTokens = round.meta.tokens_in > 0 || round.meta.tokens_out > 0;
  const detail = round.meta.underlying_content || round.meta.worker_report || "(no details recorded)";

  return (
    <div className="absolute inset-0" style={{ zIndex: 25, pointerEvents: "none" }}>
      <div
        className="flex flex-col rounded-2xl overflow-hidden"
        style={{
          position: "absolute",
          left: rect.x,
          top: rect.y,
          width: rect.w,
          height: rect.h,
          pointerEvents: "auto",
          background: "color-mix(in srgb, var(--card) 20%, transparent)",
          backdropFilter: "blur(6px)",
          WebkitBackdropFilter: "blur(6px)",
          border: "1px solid var(--border)",
          boxShadow: "0 18px 60px rgba(0,0,0,0.25)",
        }}
      >
        {/* Header: label + close (cross) */}
        <div
          onMouseDown={onTitleMouseDown}
          className="flex items-center justify-between px-4 py-2.5 shrink-0 select-none"
          style={{ borderBottom: "1px solid var(--border)", cursor: "grab" }}
        >
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--accent)" }} />
            <span className="font-mono tracking-widest uppercase" style={{ fontSize: 15, color: "var(--muted-foreground)" }}>Agent response</span>
          </div>
          <button
            onClick={onMinimize}
            className="w-6 h-6 flex items-center justify-center rounded transition-colors hover:bg-[color:var(--secondary)]"
            style={{ color: "var(--muted-foreground)" }}
            aria-label="Dismiss response"
          >
            <svg width="11" height="11" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
              <line x1="1" y1="1" x2="9" y2="9" />
              <line x1="9" y1="1" x2="1" y2="9" />
            </svg>
          </button>
        </div>

        {/* Scrollable body: message, then appended detail when open */}
        <div ref={bodyRef} className="flex-1 overflow-y-auto px-4 py-3.5">
          <p className="text-xs leading-relaxed whitespace-pre-wrap" style={{ color: "var(--foreground)" }}>
            {round.agent_message}
          </p>

          {open && (
            <div className="mt-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-mono tracking-widest uppercase" style={{ fontSize: 12, color: "var(--accent)" }}>
                  Execution detail
                </span>
              </div>
              <pre
                className="text-xs leading-relaxed whitespace-pre-wrap font-mono"
                style={{ color: "var(--foreground)", background: "color-mix(in srgb, var(--secondary) 55%, transparent)", borderRadius: 8, padding: "10px 12px" }}
              >
                {detail}
              </pre>
            </div>
          )}
        </div>

        {/* Footer: bigger metadata + detail toggle */}
        <div className="flex items-center justify-between gap-2 px-4 py-2.5 shrink-0" style={{ borderTop: "1px solid var(--border)" }}>
          <p className="font-mono truncate" style={{ fontSize: 12, color: "var(--muted-foreground)" }}>
            {round.meta.response_time_s.toFixed(1)}s{hasTokens ? " | " + round.meta.tokens_in + " in / " + round.meta.tokens_out + " out" : ""}
          </p>
          <button
            onClick={() => setOpen((o) => !o)}
            className="font-mono px-2.5 py-1 rounded transition-colors hover:bg-[color:var(--secondary)] shrink-0"
            style={{ fontSize: 12, color: "var(--accent)", border: "1px solid var(--border)" }}
          >
            {open ? "Fold" : "View details"}
          </button>
        </div>

        {/* Resize grip (bottom-right corner) */}
        <span
          onMouseDown={onResizeMouseDown}
          title="Resize"
          className="absolute bottom-0 right-0 w-4 h-4"
          style={{ cursor: "nwse-resize" }}
        >
          <svg width="14" height="14" viewBox="0 0 14 14" style={{ display: "block", color: "var(--muted-foreground)", opacity: 0.5 }}>
            <path d="M13 5 L5 13 M13 9 L9 13" stroke="currentColor" strokeWidth="1.2" fill="none" strokeLinecap="round" />
          </svg>
        </span>
      </div>
    </div>
  );
}
