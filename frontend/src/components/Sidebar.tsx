import { useEffect, useRef, useState } from "react";

export function DummyMark({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 22 22" fill="none">
      <circle cx="11" cy="11" r="10" stroke="var(--accent)" strokeWidth="1" opacity="0.3" />
      <circle cx="11" cy="11" r="6" stroke="var(--accent)" strokeWidth="1" opacity="0.6" />
      <circle cx="11" cy="11" r="2.5" fill="var(--accent)" />
      <line x1="11" y1="1" x2="11" y2="5" stroke="var(--accent)" strokeWidth="1" opacity="0.5" />
      <line x1="11" y1="17" x2="11" y2="21" stroke="var(--accent)" strokeWidth="1" opacity="0.5" />
      <line x1="1" y1="11" x2="5" y2="11" stroke="var(--accent)" strokeWidth="1" opacity="0.5" />
      <line x1="17" y1="11" x2="21" y2="11" stroke="var(--accent)" strokeWidth="1" opacity="0.5" />
    </svg>
  );
}

export function Sidebar({ onOpenSettings }: { onOpenSettings: () => void }) {
  return (
    <aside
      className="flex flex-col items-center h-full shrink-0 py-5"
      style={{ width: 76, borderRight: "1px solid var(--border)", background: "var(--card)" }}
    >
      {/* Logo */}
      <div className="flex flex-col items-center gap-2">
        <DummyMark size={26} />
        <span style={{ fontFamily: "Instrument Serif, serif", fontSize: 15, color: "var(--foreground)" }}>Dummy</span>
      </div>

      <div className="flex-1" />

      {/* Settings */}
      <button
        onClick={onOpenSettings}
        className="flex items-center justify-center w-9 h-9 rounded transition-all group hover:bg-[color:var(--muted)]"
        style={{ color: "var(--muted-foreground)" }}
        aria-label="Settings"
      >
        <svg width="13" height="13" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.3">
          <circle cx="5.5" cy="5.5" r="2" />
          <path d="M5.5 1v1.5M5.5 8.5V10M1 5.5h1.5M8.5 5.5H10M2.3 2.3l1 1M7.7 7.7l1 1M8.7 2.3l-1 1M3.3 7.7l-1 1" strokeLinecap="round" />
        </svg>
      </button>
    </aside>
  );
}

export function LocalClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  const timeStr = now.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
  const dateStr = now.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
  return (
    <div className="flex flex-col items-center gap-1 pointer-events-none">
      <p className="font-mono font-semibold" style={{ fontSize: 45, fontWeight: 600, color: "var(--foreground)", letterSpacing: "-0.02em" }}>{timeStr}</p>
      <p className="font-mono" style={{ fontSize: 16, color: "var(--muted-foreground)" }}>{dateStr}</p>
    </div>
  );
}
