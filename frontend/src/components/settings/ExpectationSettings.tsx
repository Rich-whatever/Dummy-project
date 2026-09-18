import React, { useEffect, useRef, useState } from "react";
import { fetchExpectations, saveExpectations, type ApiExpectations } from "../../api";

function toFrontend(api: ApiExpectations): ExpectationData {
  return {
    guides: api.guides.map((g, i) => ({ id: `g${i}`, content: g })),
    expectations: api.expectations.map((e, i) => ({
      id: `e${i}`,
      trigger: e.trigger,
      action: e.action,
      description: e.expectation_content,
    })),
  };
}

function toBackend(d: ExpectationData): ApiExpectations {
  return {
    guides: d.guides.map((g) => g.content),
    expectations: d.expectations.map((e) => ({
      trigger: e.trigger,
      action: e.action,
      expectation_content: e.description,
    })),
  };
}

type GuideItem = { id: string; content: string };
type ExpectationItem = { id: string; trigger: string; action: string; description: string };
type ExpectationData = { guides: GuideItem[]; expectations: ExpectationItem[] };

const EXPECTATION_SEED: ExpectationData = {
  guides: [
    { id: "g1", content: "Think independently and critically, don't mindlessly agree with user. Keep response simple and direct as the situation allows" },
    { id: "g2", content: "You may ask any questions if you're curious. However, never end your response with a question merely to keep the conversation going." },
    { id: "g3", content: "Your output shouldn't be too long, at most 500 words, unless for exception cases when you need longer response to generate helpful response, or conversation topic is sophisticated" },
    { id: "g4", content: "Do not end your response with a question or 'want me to do sth' just to keep conversation going." },
    { id: "g5", content: "Stay humorous and human-like don't be robotic, don't speak like general assistant ai" },
  ],
  expectations: [
    {
      id: "e1",
      trigger: "User says let's do u ask i answer",
      action: "Ask user questions",
      description: "When the user initiates a Q&A where the agent asks and the user answers, the agent should ask questions about the user.",
    },
  ],
};

const expInputStyle: React.CSSProperties = {
  width: "100%", background: "var(--secondary)", border: "1px solid var(--border)",
  borderRadius: 4, padding: "6px 8px", fontSize: 11, color: "var(--foreground)",
  fontFamily: "inherit", resize: "vertical",
};

const expBtnStyle = { background: "var(--accent)", color: "var(--accent-foreground)" };
const expCancelBtnStyle = { color: "var(--muted-foreground)", border: "1px solid var(--border)" };

function GuideEditor({ initial, onSave, onCancel, saveLabel }: { initial: string; onSave: (v: string) => void; onCancel: () => void; saveLabel: string }) {
  const [draft, setDraft] = useState(initial);
  return (
    <div className="p-2.5 rounded-lg" style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}>
      <textarea
        autoFocus
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && draft.trim()) { onSave(draft.trim()); }
        }}
        rows={3}
        placeholder="Write the guidance…"
        style={expInputStyle}
      />
      <div className="flex gap-2 justify-end mt-2">
        <button onClick={onCancel} className="text-xs px-2 py-1 rounded" style={expCancelBtnStyle}>Cancel</button>
        <button onClick={() => { if (draft.trim()) onSave(draft.trim()); }} className="text-xs px-2 py-1 rounded" style={expBtnStyle}>{saveLabel}</button>
      </div>
    </div>
  );
}

function ExpectationForm({
  initial, onSave, onCancel, saveLabel, onDelete,
}: {
  initial: ExpectationItem;
  onSave: (v: { trigger: string; action: string; description: string }) => void;
  onCancel: () => void;
  saveLabel: string;
  onDelete?: () => void;
}) {
  const [trigger, setTrigger] = useState(initial.trigger);
  const [action, setAction] = useState(initial.action);
  const [description, setDescription] = useState(initial.description);
  const valid = trigger.trim() && action.trim() && description.trim();
  return (
    <div className="p-2.5 rounded-lg flex flex-col gap-2" style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}>
      <div>
        <p className="font-mono tracking-widest uppercase mb-1" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Trigger</p>
        <input autoFocus value={trigger} onChange={(e) => setTrigger(e.target.value)} placeholder="When does this apply…" style={expInputStyle} />
      </div>
      <div>
        <p className="font-mono tracking-widest uppercase mb-1" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Action</p>
        <input value={action} onChange={(e) => setAction(e.target.value)} placeholder="What the agent should do…" style={expInputStyle} />
      </div>
      <div>
        <p className="font-mono tracking-widest uppercase mb-1" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Description</p>
        <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} placeholder="Full expectation content…" style={expInputStyle} />
      </div>
      <div className="flex gap-2 justify-end">
        {onDelete && (
          <button onClick={onDelete} className="text-xs px-2 py-1 rounded mr-auto" style={expCancelBtnStyle}>Delete</button>
        )}
        <button onClick={onCancel} className="text-xs px-2 py-1 rounded" style={expCancelBtnStyle}>Cancel</button>
        <button
          onClick={() => { if (valid) onSave({ trigger: trigger.trim(), action: action.trim(), description: description.trim() }); }}
          className="text-xs px-2 py-1 rounded"
          style={{ ...expBtnStyle, opacity: valid ? 1 : 0.5 }}
        >{saveLabel}</button>
      </div>
    </div>
  );
}

export function ExpectationSettings() {
  const [data, setData] = useState<ExpectationData>({ guides: [], expectations: [] });
  const [editingGuide, setEditingGuide] = useState<number | "new" | null>(null);
  const [editingExp, setEditingExp] = useState<number | "new" | null>(null);

  const loadedRef = useRef(false);
  useEffect(() => {
    fetchExpectations()
      .then((api) => { loadedRef.current = true; setData(toFrontend(api)); })
      .catch(() => { loadedRef.current = true; });
  }, []);

  const savingRef = useRef(false);
  useEffect(() => {
    if (!loadedRef.current || savingRef.current) return;
    savingRef.current = true;
    saveExpectations(toBackend(data)).catch(() => {
      /* keep local state; retries on next change */
    }).finally(() => { savingRef.current = false; });
  }, [data]);

  return (
    <div className="flex flex-col gap-6">
      {/* General Guidance */}
      <div className="p-4 rounded-xl" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
        <p className="font-mono tracking-widest uppercase mb-1" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>General Guidance</p>
        <p className="text-xs mb-3" style={{ color: "var(--muted-foreground)" }}>Always included in the agent's system prompt.</p>
        <div className="flex flex-col gap-1.5">
          {data.guides.map((g, i) =>
            editingGuide === i ? (
              <GuideEditor
                key={g.id}
                initial={g.content}
                onCancel={() => setEditingGuide(null)}
                onSave={(v) => {
                  setData((d) => ({ ...d, guides: d.guides.map((x, j) => (j === i ? { ...x, content: v } : x)) }));
                  setEditingGuide(null);
                }}
                saveLabel="Save"
              />
            ) : (
              <div key={g.id} className="group flex items-start gap-2 px-3 py-2.5 rounded-lg" style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}>
                <button onClick={() => setEditingGuide(i)} className="flex-1 text-left text-xs leading-relaxed" style={{ color: "var(--foreground)" }}>
                  {g.content}
                </button>
                <button
                  onClick={() => setData((d) => ({ ...d, guides: d.guides.filter((_, j) => j !== i) }))}
                  className="opacity-0 group-hover:opacity-70 transition-opacity shrink-0"
                  style={{ color: "var(--muted-foreground)" }}
                  aria-label="Delete guidance"
                >
                  <svg width="7" height="7" viewBox="0 0 8 8" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                    <line x1="1" y1="1" x2="7" y2="7" /><line x1="7" y1="1" x2="1" y2="7" />
                  </svg>
                </button>
              </div>
            )
          )}

          {editingGuide === "new" ? (
            <GuideEditor
              initial=""
              onCancel={() => setEditingGuide(null)}
              onSave={(v) => {
                setData((d) => ({ ...d, guides: [...d.guides, { id: `g${Date.now()}`, content: v }] }));
                setEditingGuide(null);
              }}
              saveLabel="Add guidance"
            />
          ) : (
            <button
              onClick={() => setEditingGuide("new")}
              className="flex items-center justify-center gap-2 py-3 rounded-lg transition-colors hover:bg-[color:var(--secondary)]"
              style={{ border: "1px dashed var(--border)", color: "var(--muted-foreground)" }}
            >
              <span style={{ fontSize: 14 }}>+</span>
              <span className="text-xs">Click to add new guidance</span>
            </button>
          )}
        </div>
      </div>

      {/* Expectations */}
      <div className="p-4 rounded-xl" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
        <p className="font-mono tracking-widest uppercase mb-1" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Expectations</p>
        <p className="text-xs mb-3" style={{ color: "var(--muted-foreground)" }}>Trigger-conditioned behaviors — the agent follows these when the trigger matches.</p>
        <div className="flex flex-col gap-1.5">
          {data.expectations.map((e, i) =>
            editingExp === i ? (
              <ExpectationForm
                key={e.id}
                initial={e}
                onCancel={() => setEditingExp(null)}
                onDelete={() => {
                  setData((d) => ({ ...d, expectations: d.expectations.filter((_, j) => j !== i) }));
                  setEditingExp(null);
                }}
                onSave={(v) => {
                  setData((d) => ({ ...d, expectations: d.expectations.map((x, j) => (j === i ? { ...x, ...v } : x)) }));
                  setEditingExp(null);
                }}
                saveLabel="Save"
              />
            ) : (
              <div key={e.id} className="group px-3 py-2.5 rounded-lg" style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}>
                <button onClick={() => setEditingExp(i)} className="w-full text-left">
                  <p className="text-xs font-medium" style={{ color: "var(--foreground)" }}>{e.trigger}</p>
                  <p className="text-xs mt-0.5" style={{ color: "var(--accent)" }}>{e.action}</p>
                  <p className="text-xs mt-0.5 leading-relaxed" style={{ color: "var(--muted-foreground)" }}>{e.description}</p>
                </button>
                <button
                  onClick={() => setData((d) => ({ ...d, expectations: d.expectations.filter((_, j) => j !== i) }))}
                  className="opacity-0 group-hover:opacity-70 transition-opacity mt-1"
                  style={{ color: "var(--muted-foreground)" }}
                  aria-label="Delete expectation"
                >
                  <svg width="7" height="7" viewBox="0 0 8 8" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                    <line x1="1" y1="1" x2="7" y2="7" /><line x1="7" y1="1" x2="1" y2="7" />
                  </svg>
                </button>
              </div>
            )
          )}

          {editingExp === "new" ? (
            <ExpectationForm
              initial={{ id: "", trigger: "", action: "", description: "" }}
              onCancel={() => setEditingExp(null)}
              onSave={(v) => {
                setData((d) => ({ ...d, expectations: [...d.expectations, { id: `e${Date.now()}`, ...v }] }));
                setEditingExp(null);
              }}
              saveLabel="Create expectation"
            />
          ) : (
            <button
              onClick={() => setEditingExp("new")}
              className="flex items-center justify-center gap-2 py-3 rounded-lg transition-colors hover:bg-[color:var(--secondary)]"
              style={{ border: "1px dashed var(--border)", color: "var(--muted-foreground)" }}
            >
              <span style={{ fontSize: 14 }}>+</span>
              <span className="text-xs">Click to add new expectation</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

