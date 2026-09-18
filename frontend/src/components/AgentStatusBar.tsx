import { useEffect, useState } from "react";

const STATUS_PHRASES = [
  "Parsing request",
  "Routing through memory",
  "Retrieving long-term context",
  "Planning execution",
  "Worker executing tool",
  "Cross-checking results",
  "Assembling response",
];

export function AgentStatusBar({ onCancel }: { onCancel: () => void }) {
  const [idx, setIdx] = useState(() => Math.floor(Math.random() * STATUS_PHRASES.length));

  useEffect(() => {
    const t = setInterval(() => {
      setIdx((i) => {
        let n = Math.floor(Math.random() * STATUS_PHRASES.length);
        if (n === i) n = (n + 1) % STATUS_PHRASES.length;
        return n;
      });
    }, 1600);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex items-center gap-3 px-5 py-3.5 rounded-full"
      style={{
        width: "min(680px, 92%)",
        background: "var(--card)",
        border: "1px solid var(--border)",
        boxShadow: "0 8px 32px rgba(0,0,0,0.18)",
        zIndex: 20,
      }}
    >
      {/* pulsing indicator */}
      <div className="relative w-3 h-3 shrink-0 flex items-center justify-center">
        <div className="absolute w-3 h-3 rounded-full border" style={{ borderColor: "var(--accent)", animation: "pulse-ring 2s ease-out infinite" }} />
        <div className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--accent)", animation: "breathe 3s ease-in-out infinite", boxShadow: "0 0 6px var(--accent)" }} />
      </div>

      <p className="font-mono tracking-widest uppercase flex-1 truncate" style={{ fontSize: 11, color: "var(--muted-foreground)" }}>
        {STATUS_PHRASES[idx]}
      </p>

      <button
        onClick={onCancel}
        className="text-xs px-3 py-1.5 rounded-full transition-colors hover:bg-[color:var(--secondary)] shrink-0"
        style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)", fontSize: 11 }}
        aria-label="Cancel agent execution"
      >
        Cancel
      </button>
    </div>
  );
}
