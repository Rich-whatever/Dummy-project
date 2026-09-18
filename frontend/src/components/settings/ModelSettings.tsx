import { useEffect, useState } from "react";
import { fetchConfig, saveConfigModel, type ModelConfig } from "../../api";

const CONTEXT_OPTIONS = ["32k", "100k", "200k", "500k", "1M"];

export function ModelSettings() {
  const [model, setModel] = useState<ModelConfig | null>(null);
  const [draft, setDraft] = useState<ModelConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchConfig()
      .then((c) => { setModel(c.model); setDraft(c.model); })
      .catch(() => setError("Could not load backend config — is the API running?"));
  }, []);

  if (!model || !draft) {
    return (
      <p className="text-xs" style={{ color: "var(--muted-foreground)" }}>
        {error || "Loading model settings…"}
      </p>
    );
  }

  const dirty =
    draft.temperature !== model.temperature ||
    draft.context_window !== model.context_window ||
    draft.reasoning_effort !== model.reasoning_effort ||
    draft.model_name !== model.model_name;

  const apply = async () => {
    setSaving(true);
    try {
      const next = await saveConfigModel(draft);
      setModel(next.model);
    } catch {
      setError("Failed to save — is the API running?");
    } finally {
      setSaving(false);
    }
  };

  const numStyle: React.CSSProperties = {
    background: "var(--secondary)", border: "1px solid var(--border)",
    borderRadius: 4, padding: "4px 8px", fontFamily: "JetBrains Mono, monospace",
    fontSize: 11, color: "var(--foreground)",
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-y-auto pr-2">
        <div className="mb-6">
          <p className="font-mono tracking-widest uppercase mb-2" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Model name</p>
          <input
            value={draft.model_name}
            onChange={(e) => setDraft((d) => (d ? { ...d, model_name: e.target.value } : d))}
            style={{ ...numStyle, width: "100%" }}
          />
        </div>

        <div className="mb-6">
          <p className="font-mono tracking-widest uppercase mb-2" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Temperature</p>
          <div className="flex items-center gap-3">
            <input type="range" min={0} max={1} step={0.05} value={draft.temperature}
              onChange={(e) => setDraft((d) => (d ? { ...d, temperature: parseFloat(e.target.value) } : d))}
              className="w-48" style={{ accentColor: "var(--accent)" }} />
            <span className="font-mono text-xs" style={{ color: "var(--foreground)" }}>{draft.temperature.toFixed(2)}</span>
          </div>
        </div>

        <div className="mb-6">
          <p className="font-mono tracking-widest uppercase mb-2" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Context window</p>
          <select
            value={draft.context_window}
            onChange={(e) => setDraft((d) => (d ? { ...d, context_window: e.target.value } : d))}
            style={numStyle}
          >
            {CONTEXT_OPTIONS.map((o) => <option key={o}>{o}</option>)}
          </select>
        </div>

        <div className="mb-6">
          <p className="font-mono tracking-widest uppercase mb-2" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Thinking effort</p>
          <div className="flex gap-1.5">
            {(["low", "medium", "high"] as const).map((level) => (
              <button
                key={level}
                onClick={() => setDraft((d) => (d ? { ...d, reasoning_effort: level } : d))}
                className="text-xs px-3 py-1 rounded transition-all capitalize"
                style={{
                  background: draft.reasoning_effort === level ? "var(--accent)" : "var(--muted)",
                  color: draft.reasoning_effort === level ? "var(--accent-foreground)" : "var(--muted-foreground)",
                  border: "1px solid var(--border)",
                }}
              >
                {level}
              </button>
            ))}
          </div>
        </div>

        <p className="text-[10px] leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
          These settings apply to the main agent model (<code>get_llm</code>) and are picked
          up on the next message — no restart needed. Other models (no-thinking, small, GLM) are unaffected.
        </p>
      </div>

      <div className="flex items-center justify-end gap-2 pt-3 mt-2 shrink-0" style={{ borderTop: "1px solid var(--border)" }}>
        {error && <span className="text-[10px]" style={{ color: "var(--accent)" }}>{error}</span>}
        {dirty && (
          <button
            onClick={() => setDraft(model)}
            className="text-xs px-3 py-1.5 rounded transition-colors hover:bg-[color:var(--secondary)]"
            style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }}
          >Reset</button>
        )}
        <button
          onClick={apply}
          disabled={!dirty || saving}
          className="text-xs px-4 py-1.5 rounded transition-all"
          style={{
            background: dirty && !saving ? "var(--accent)" : "var(--muted)",
            color: dirty && !saving ? "var(--accent-foreground)" : "var(--muted-foreground)",
            opacity: dirty && !saving ? 1 : 0.5,
            cursor: dirty && !saving ? "pointer" : "default",
          }}
        >
          {saving ? "Saving…" : dirty ? "Apply changes" : "Applied ✓"}
        </button>
      </div>
    </div>
  );
}

