import React, { useEffect, useState } from "react";
import { fetchConfig, saveConfigGeneral } from "../../api";

type GeneralConfig = {
  MAX_MESSAGES_PER_THREAD: number;
  MAX_ACTIVE_THREADS: number;
  BRIDGE_SUMMARY_WORD_LIMIT: number;
  LTM_RETRIEVE_K: number;
  LTM_SCORE_THRESHOLD: number;
  LTM_LATEST_K: number;
  SUMMARY_COUNTER_INITIAL: number;
  WORKER_ROUND_LIMIT: number;
  MAX_ASSIGN_TASKS: number;
  TOOLMESSAGE_LIMIT: number;
};

const GENERAL_GROUPS: { title: string; fields: { key: keyof GeneralConfig; label: string; step: number }[] }[] = [
  { title: "Thread Limits", fields: [
    { key: "MAX_MESSAGES_PER_THREAD", label: "Max messages per thread", step: 1 },
    { key: "MAX_ACTIVE_THREADS", label: "Max active threads", step: 1 },
    { key: "BRIDGE_SUMMARY_WORD_LIMIT", label: "Bridge summary word limit", step: 10 },
  ]},
  { title: "LTM Retrieval", fields: [
    { key: "LTM_RETRIEVE_K", label: "Retrieve K", step: 1 },
    { key: "LTM_SCORE_THRESHOLD", label: "Score threshold", step: 0.1 },
    { key: "LTM_LATEST_K", label: "Latest K", step: 1 },
  ]},
  { title: "Summary Counter", fields: [
    { key: "SUMMARY_COUNTER_INITIAL", label: "Initial summary counter", step: 1 },
  ]},
  { title: "Worker Round Limit", fields: [
    { key: "WORKER_ROUND_LIMIT", label: "Worker round limit", step: 1 },
  ]},
  { title: "Main-Agent Task Cap", fields: [
    { key: "MAX_ASSIGN_TASKS", label: "Max assign_task calls per round", step: 1 },
  ]},
  { title: "Tool Message Size Limit", fields: [
    { key: "TOOLMESSAGE_LIMIT", label: "Tool message char limit", step: 1000 },
  ]},
];

export function GeneralSettings() {
  const [saved, setSaved] = useState<GeneralConfig | null>(null);
  const [draft, setDraft] = useState<GeneralConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    fetchConfig()
      .then((c) => {
        const g = c.general as unknown as GeneralConfig;
        setSaved(g);
        setDraft(g);
      })
      .catch(() => setLoadError("Could not load backend config — is the API running?"));
  }, []);

  if (!saved || !draft) {
    return (
      <p className="text-xs" style={{ color: "var(--muted-foreground)" }}>
        {loadError || "Loading config…"}
      </p>
    );
  }

  const dirty = JSON.stringify(saved) !== JSON.stringify(draft);

  const numStyle: React.CSSProperties = {
    width: 110, background: "var(--secondary)", border: "1px solid var(--border)",
    borderRadius: 4, padding: "4px 8px", fontFamily: "JetBrains Mono, monospace",
    fontSize: 11, color: "var(--foreground)",
  };

  const apply = async () => {
    setSaving(true);
    try {
      const next = await saveConfigGeneral(draft as unknown as Record<string, number>);
      setSaved(next.general as unknown as GeneralConfig);
    } catch {
      setLoadError("Failed to save config — is the API running?");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-y-auto pr-2">
        {GENERAL_GROUPS.map((g) => (
          <div key={g.title} className="mb-6">
            <p className="font-mono tracking-widest uppercase mb-2.5" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>{g.title}</p>
            <div className="flex flex-col gap-2">
              {g.fields.map((f) => (
                <div key={f.key} className="flex items-center justify-between">
                  <span className="text-xs" style={{ color: "var(--foreground)" }}>{f.label}</span>
                  <input
                    type="number"
                    step={f.step}
                    value={draft[f.key]}
                    onChange={(e) => setDraft((d) => (d ? { ...d, [f.key]: parseFloat(e.target.value) || 0 } : d))}
                    style={numStyle}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-end gap-2 pt-3 mt-2 shrink-0" style={{ borderTop: "1px solid var(--border)" }}>
        {loadError && <span className="text-[10px]" style={{ color: "var(--accent)" }}>{loadError}</span>}
        {dirty && (
          <button
            onClick={() => setDraft(saved)}
            className="text-xs px-3 py-1.5 rounded transition-colors hover:bg-[color:var(--secondary)]"
            style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }}
          >
            Reset
          </button>
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

