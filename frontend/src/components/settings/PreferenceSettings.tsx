import React, { useEffect, useRef, useState } from "react";
import { fetchPreferences, savePreferences } from "../../api";

type NoteItem = { content: string; id: string; saved_time: string };
type NoteCategoryData = { category_name: string; category_description: string; notes: NoteItem[] };
type StateItem = { state_name: string; description: string; event_list: string[]; active_time: string };
type PrefData = { categories: NoteCategoryData[]; states: StateItem[] };

const prefInputStyle: React.CSSProperties = {
  width: "100%", background: "var(--secondary)", border: "1px solid var(--border)",
  borderRadius: 4, padding: "6px 8px", fontSize: 11, color: "var(--foreground)",
  fontFamily: "inherit", resize: "vertical",
};


function PrefSubHeader({ title, onBack, onDelete }: { title: string; onBack: () => void; onDelete?: () => void }) {
  return (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <button onClick={onBack} className="w-5 h-5 flex items-center justify-center rounded" style={{ color: "var(--muted-foreground)" }} aria-label="Back">
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M6.5 1 2.5 5l4 4" /></svg>
        </button>
        <span className="font-mono tracking-widest uppercase" style={{ fontSize: 10, color: "var(--foreground)" }}>{title}</span>
      </div>
      {onDelete && (
        <button onClick={onDelete} className="flex items-center gap-1.5 text-xs px-2 py-1 rounded transition-colors hover:bg-[color:var(--secondary)]" style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }} aria-label="Delete">
          <svg width="9" height="10" viewBox="0 0 10 11" fill="none" stroke="currentColor" strokeWidth="1.2"><path d="M1 2.5h8M3.5 2.5V1h3v1.5M2 2.5l.5 7h5l.5-7" strokeLinecap="round" /></svg>
          Delete
        </button>
      )}
    </div>
  );
}


export function NameDescEditor({ value, onCommit, singleLine }: { value: string; onCommit: (v: string) => void; singleLine?: boolean }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);

  if (editing) {
    return (
      <div>
        {singleLine ? (
          <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} style={prefInputStyle} />
        ) : (
          <textarea autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} rows={2} style={prefInputStyle} />
        )}
        <div className="flex gap-2 justify-end mt-1.5">
          <button onClick={() => { setEditing(false); setDraft(value); }} className="text-xs px-2 py-1 rounded" style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }}>Cancel</button>
          <button
            onClick={() => { onCommit(draft.trim()); setEditing(false); }}
            className="text-xs px-2 py-1 rounded"
            style={{ background: "var(--accent)", color: "var(--accent-foreground)" }}
          >Save</button>
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={() => { setDraft(value); setEditing(true); }}
      className="text-left w-full px-2.5 py-2 rounded-lg transition-colors hover:bg-[color:var(--secondary)]"
      style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}
    >
      <p className={"text-xs leading-relaxed whitespace-pre-wrap " + (singleLine ? "font-medium" : "")} style={{ color: "var(--foreground)" }}>
        {value || "(empty - click to edit)"}
      </p>
    </button>
  );
}

function EventRow({ value, onCommit, onDelete }: { value: string; onCommit: (v: string) => void; onDelete: () => void }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);

  if (editing) {
    return (
      <div className="p-2 rounded-lg" style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}>
        <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} style={prefInputStyle} />
        <div className="flex gap-2 justify-end mt-1.5">
          <button onClick={() => { setEditing(false); setDraft(value); }} className="text-xs px-2 py-0.5 rounded" style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }}>Cancel</button>
          <button
            onClick={() => { onCommit(draft); setEditing(false); }}
            className="text-xs px-2 py-0.5 rounded"
            style={{ background: "var(--accent)", color: "var(--accent-foreground)" }}
          >Save</button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 px-2.5 py-2 rounded-lg group" style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}>
      <div className="w-1 h-1 rounded-full shrink-0" style={{ background: "var(--accent)" }} />
      <button onClick={() => { setDraft(value); setEditing(true); }} className="flex-1 text-left text-xs" style={{ color: "var(--foreground)" }}>
        {value}
      </button>
      <button onClick={onDelete} className="opacity-0 group-hover:opacity-70 transition-opacity" style={{ color: "var(--muted-foreground)" }} aria-label="Delete event">
        <svg width="7" height="7" viewBox="0 0 8 8" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
          <line x1="1" y1="1" x2="7" y2="7" /><line x1="7" y1="1" x2="1" y2="7" />
        </svg>
      </button>
    </div>
  );
}

function AddRowButton({ label, placeholder, onAdd }: { label: string; placeholder: string; onAdd: (v: string) => void }) {
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState("");

  if (adding) {
    return (
      <div className="p-2 rounded-lg" style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}>
        <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={placeholder} style={prefInputStyle} />
        <div className="flex gap-2 justify-end mt-1.5">
          <button onClick={() => { setAdding(false); setDraft(""); }} className="text-xs px-2 py-0.5 rounded" style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }}>Cancel</button>
          <button
            onClick={() => { if (draft.trim()) onAdd(draft.trim()); setAdding(false); setDraft(""); }}
            className="text-xs px-2 py-0.5 rounded"
            style={{ background: "var(--accent)", color: "var(--accent-foreground)" }}
          >Add</button>
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={() => { setAdding(true); setDraft(""); }}
      className="flex items-center justify-center gap-2 py-2.5 rounded-lg transition-colors hover:bg-[color:var(--secondary)]"
      style={{ border: "1px dashed var(--border)", color: "var(--muted-foreground)" }}
    >
      <span style={{ fontSize: 13 }}>+</span>
      <span className="text-xs">{label}</span>
    </button>
  );
}


export function PreferenceSettings() {
  const [data, setData] = useState<PrefData>({ categories: [], states: [] });
  const [openCat, setOpenCat] = useState<string | null>(null);
  const [openState, setOpenState] = useState<string | null>(null);
  const [addingCat, setAddingCat] = useState(false);
  const [newCatName, setNewCatName] = useState("");
  const [newCatDesc, setNewCatDesc] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [editingNote, setEditingNote] = useState<number | "new" | null>(null);
  const [noteDraft, setNoteDraft] = useState("");
  const [addingState, setAddingState] = useState(false);
  const [newStateName, setNewStateName] = useState("");
  const [newStateDesc, setNewStateDesc] = useState("");

  // Load persisted preferences from the backend on mount.
  const loadedRef = useRef(false);
  useEffect(() => {
    fetchPreferences()
      .then((p) => {
        loadedRef.current = true;
        setData(p);
      })
      .catch(() => loadedRef.current = true); // backend down → show empty lists
  }, []);

  // Persist every user edit back to the backend (agent-injected changes land in
  // the same JSON, so a reload re-syncs the UI).
  const savingRef = useRef(false);
  useEffect(() => {
    if (!loadedRef.current || savingRef.current) return;
    savingRef.current = true;
    savePreferences(data).catch(() => {
      /* keep local state; will retry on next change */
    }).finally(() => { savingRef.current = false; });
  }, [data]);

  const cat = data.categories.find((c) => c.category_name === openCat);
  const state = data.states.find((s) => s.state_name === openState);

  // ── Category drill-down ──
  if (cat) {
    return (
      <div>
        <PrefSubHeader
          title={cat.category_name}
          onBack={() => { setOpenCat(null); setEditingNote(null); }}
          onDelete={() => setConfirmDelete(true)}
        />

        {confirmDelete && (
          <div className="mb-3 p-3 rounded-lg" style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}>
            <p className="text-xs mb-2" style={{ color: "var(--foreground)" }}>
              Delete category “{cat.category_name}” and all {cat.notes.length} notes?
            </p>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setConfirmDelete(false)} className="text-xs px-3 py-1 rounded" style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }}>No</button>
              <button
                onClick={() => {
                  setData((d) => ({ ...d, categories: d.categories.filter((c) => c.category_name !== cat.category_name) }));
                  setConfirmDelete(false);
                  setOpenCat(null);
                }}
                className="text-xs px-3 py-1 rounded"
                style={{ background: "var(--accent)", color: "var(--accent-foreground)" }}
              >Yes</button>
            </div>
          </div>
        )}

        <div className="mb-3">
          <p className="font-mono tracking-widest uppercase mb-1.5" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Description</p>
          <NameDescEditor
            value={cat.category_description}
            onCommit={(v) =>
              setData((d) => ({
                ...d,
                categories: d.categories.map((c) =>
                  c.category_name === cat.category_name ? { ...c, category_description: v } : c
                ),
              }))
            }
          />
        </div>

        <div className="flex flex-col gap-2">
          {cat.notes.map((note, i) =>
            editingNote === i ? (
              <div key={note.id || i} className="p-2.5 rounded-lg" style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}>
                <textarea value={noteDraft} onChange={(e) => setNoteDraft(e.target.value)} rows={3} style={prefInputStyle} />
                <div className="flex gap-2 justify-end mt-2">
                  <button onClick={() => setEditingNote(null)} className="text-xs px-2 py-1 rounded" style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }}>Cancel</button>
                  <button
                    onClick={() => {
                      setData((d) => ({ ...d, categories: d.categories.map((c) => c.category_name !== cat.category_name ? c : { ...c, notes: c.notes.map((n, j) => j === i ? { ...n, content: noteDraft } : n) }) }));
                      setEditingNote(null);
                    }}
                    className="text-xs px-2 py-1 rounded"
                    style={{ background: "var(--accent)", color: "var(--accent-foreground)" }}
                  >Save</button>
                </div>
              </div>
            ) : (
              <button
                key={note.id || i}
                onClick={() => { setEditingNote(i); setNoteDraft(note.content); }}
                className="text-left p-2.5 rounded-lg transition-colors hover:bg-[color:var(--secondary)]"
                style={{ background: "var(--card)", border: "1px solid var(--border)" }}
              >
                <p className="text-xs leading-relaxed" style={{ color: "var(--foreground)" }}>{note.content}</p>
                {note.saved_time && (
                  <p className="font-mono mt-1" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>
                    {new Date(note.saved_time).toLocaleString()}
                  </p>
                )}
              </button>
            )
          )}

          {editingNote === "new" ? (
            <div className="p-2.5 rounded-lg" style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}>
              <textarea value={noteDraft} onChange={(e) => setNoteDraft(e.target.value)} rows={3} placeholder="Write the note…" style={prefInputStyle} />
              <div className="flex gap-2 justify-end mt-2">
                <button onClick={() => { setEditingNote(null); setNoteDraft(""); }} className="text-xs px-2 py-1 rounded" style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }}>Cancel</button>
                <button
                  onClick={() => {
                    if (!noteDraft.trim()) return;
                    setData((d) => ({ ...d, categories: d.categories.map((c) => c.category_name !== cat.category_name ? c : { ...c, notes: [...c.notes, { content: noteDraft.trim(), id: `n${Date.now()}`, saved_time: new Date().toISOString() }] }) }));
                    setEditingNote(null);
                    setNoteDraft("");
                  }}
                  className="text-xs px-2 py-1 rounded"
                  style={{ background: "var(--accent)", color: "var(--accent-foreground)" }}
                >Add note</button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => { setEditingNote("new"); setNoteDraft(""); }}
              className="flex items-center justify-center gap-2 py-3 rounded-lg transition-colors hover:bg-[color:var(--secondary)]"
              style={{ border: "1px dashed var(--border)", color: "var(--muted-foreground)" }}
            >
              <span style={{ fontSize: 14 }}>+</span>
              <span className="text-xs">Add note</span>
            </button>
          )}
        </div>
      </div>
    );
  }

  // ── State drill-down ──
  if (state) {
    const setState = (updater: (s: StateItem) => StateItem) =>
      setData((d) => ({ ...d, states: d.states.map((s) => (s.state_name === openState ? updater(s) : s)) }));

    return (
      <div>
        <PrefSubHeader title="State" onBack={() => setOpenState(null)} />

        <div className="mb-3">
          <p className="font-mono tracking-widest uppercase mb-1.5" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>State name</p>
          <NameDescEditor value={state.state_name} onCommit={(v) => setState((s) => ({ ...s, state_name: v }))} singleLine />
        </div>

        <div className="mb-3">
          <p className="font-mono tracking-widest uppercase mb-1.5" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Description</p>
          <NameDescEditor value={state.description} onCommit={(v) => setState((s) => ({ ...s, description: v }))} />
        </div>

        <div className="mb-4">
          <p className="font-mono tracking-widest uppercase mb-1.5" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Active time</p>
          <p className="font-mono text-xs" style={{ color: "var(--muted-foreground)" }}>
            {state.active_time ? new Date(state.active_time).toLocaleString() : "(unknown)"}
          </p>
        </div>

        <p className="font-mono tracking-widest uppercase mb-1.5" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Events</p>
        <div className="flex flex-col gap-1.5">
          {state.event_list.map((ev, i) => (
            <EventRow
              key={i}
              value={ev}
              onCommit={(v) => setState((s) => ({ ...s, event_list: s.event_list.map((e, j) => (j === i ? v : e)) }))}
              onDelete={() => setState((s) => ({ ...s, event_list: s.event_list.filter((_, j) => j !== i) }))}
            />
          ))}
          <AddRowButton label="Add event" placeholder="Event content…" onAdd={(v) => setState((s) => ({ ...s, event_list: [...s.event_list, v] }))} />
        </div>
      </div>
    );
  }


  // ── Main: Notes + State lists stacked ──
  return (
    <div className="flex flex-col gap-6">
      {/* Notes */}
      <div className="p-4 rounded-xl" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
        <p className="font-mono tracking-widest uppercase mb-3" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Notes</p>
        <div className="flex flex-col gap-1.5">
          {data.categories.map((c) => (
            <button
              key={c.category_name}
              onClick={() => { setOpenCat(c.category_name); setEditingNote(null); setConfirmDelete(false); }}
              className="flex items-center justify-between px-3 py-2.5 rounded-lg transition-colors hover:bg-[color:var(--secondary)]"
              style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}
            >
              <span className="text-xs" style={{ color: "var(--foreground)" }}>{c.category_name}</span>
              <span className="font-mono" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>({c.notes.length} notes)</span>
            </button>
          ))}

          {addingCat ? (
            <div className="flex flex-col gap-2 px-3 py-2">
              <input
                autoFocus
                value={newCatName}
                onChange={(e) => setNewCatName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && newCatName.trim()) {
                    setData((d) => ({ ...d, categories: [...d.categories, { category_name: newCatName.trim(), category_description: newCatDesc.trim(), notes: [] }] }));
                    setAddingCat(false);
                    setNewCatName("");
                    setNewCatDesc("");
                  }
                }}
                placeholder="Category name…"
                style={prefInputStyle}
              />
              <input
                value={newCatDesc}
                onChange={(e) => setNewCatDesc(e.target.value)}
                placeholder="Short description (what this category contains)…"
                style={prefInputStyle}
              />
              <div className="flex gap-2 justify-end">
                <button
                  onClick={() => {
                    if (newCatName.trim()) {
                      setData((d) => ({ ...d, categories: [...d.categories, { category_name: newCatName.trim(), category_description: newCatDesc.trim(), notes: [] }] }));
                    }
                    setAddingCat(false);
                    setNewCatName("");
                    setNewCatDesc("");
                  }}
                  className="text-xs px-2 py-1 rounded"
                  style={{ background: "var(--accent)", color: "var(--accent-foreground)" }}
                >Add</button>
                <button
                  onClick={() => { setAddingCat(false); setNewCatName(""); setNewCatDesc(""); }}
                  className="text-xs px-2 py-1 rounded"
                  style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }}
                >Cancel</button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => { setAddingCat(true); setNewCatName(""); }}
              className="flex items-center justify-center gap-2 py-3 rounded-lg transition-colors hover:bg-[color:var(--secondary)]"
              style={{ border: "1px dashed var(--border)", color: "var(--muted-foreground)" }}
            >
              <span style={{ fontSize: 14 }}>+</span>
              <span className="text-xs">Click to add new category</span>
            </button>
          )}
        </div>
      </div>

      {/* State */}
      <div className="p-4 rounded-xl" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
        <p className="font-mono tracking-widest uppercase mb-3" style={{ fontSize: 9, color: "var(--muted-foreground)" }}>State</p>
        <div className="flex flex-col gap-1.5">
          {data.states.map((s) => (
            <button
              key={s.state_name}
              onClick={() => setOpenState(s.state_name)}
              className="text-left px-3 py-2.5 rounded-lg transition-colors hover:bg-[color:var(--secondary)]"
              style={{ background: "var(--secondary)", border: "1px solid var(--border)" }}
            >
              <p className="text-xs font-medium" style={{ color: "var(--foreground)" }}>{s.state_name}</p>
              <p className="text-xs mt-0.5" style={{ color: "var(--muted-foreground)" }}>{s.description || "(no description)"}</p>
            </button>
          ))}

          {addingState ? (
            <div className="flex flex-col gap-2 px-3 py-2">
              <input
                autoFocus
                value={newStateName}
                onChange={(e) => setNewStateName(e.target.value)}
                placeholder="State name…"
                style={prefInputStyle}
              />
              <textarea
                value={newStateDesc}
                onChange={(e) => setNewStateDesc(e.target.value)}
                placeholder="Description…"
                rows={2}
                style={prefInputStyle}
              />
              <div className="flex gap-2 justify-end">
                <button
                  onClick={() => { setAddingState(false); setNewStateName(""); setNewStateDesc(""); }}
                  className="text-xs px-2 py-1 rounded"
                  style={{ color: "var(--muted-foreground)", border: "1px solid var(--border)" }}
                >Cancel</button>
                <button
                  onClick={() => {
                    if (!newStateName.trim()) return;
                    setData((d) => ({ ...d, states: [...d.states, { state_name: newStateName.trim(), description: newStateDesc.trim(), event_list: [], active_time: new Date().toISOString() }] }));
                    setAddingState(false);
                    setNewStateName("");
                    setNewStateDesc("");
                  }}
                  className="text-xs px-2 py-1 rounded"
                  style={{ background: "var(--accent)", color: "var(--accent-foreground)" }}
                >Create state</button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => { setAddingState(true); setNewStateName(""); setNewStateDesc(""); }}
              className="flex items-center justify-center gap-2 py-3 rounded-lg transition-colors hover:bg-[color:var(--secondary)]"
              style={{ border: "1px dashed var(--border)", color: "var(--muted-foreground)" }}
            >
              <span style={{ fontSize: 14 }}>+</span>
              <span className="text-xs">Click to add new state</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
