export function ResponseDetailPage({
  title, content, onBack,
}: {
  title: string;
  content: string;
  onBack: () => void;
}) {
  return (
    <div className="absolute inset-0 flex flex-col" style={{ background: "var(--background)", zIndex: 40 }}>
      {/* header */}
      <div className="flex items-center justify-between px-6 py-4 shrink-0" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="flex items-center gap-2.5">
          <button onClick={onBack} className="flex items-center justify-center w-5 h-5 rounded" style={{ color: "var(--muted-foreground)" }} aria-label="Back">
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M6.5 1 2.5 5l4 4" /></svg>
          </button>
          <span className="font-mono tracking-widest uppercase" style={{ fontSize: 10, color: "var(--foreground)" }}>{title}</span>
        </div>
        <span className="font-mono" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>{content.split("\n").length} lines</span>
      </div>

      {/* scrollable content */}
      <div className="flex-1 overflow-y-auto px-6 py-5">
        <pre className="font-mono text-xs leading-relaxed whitespace-pre-wrap" style={{ color: "var(--foreground)", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>
          {content || "(empty)"}
        </pre>
      </div>
    </div>
  );
}
