import { useState } from "react";
import { GeneralSettings } from "./GeneralSettings";
import { ModelSettings } from "./ModelSettings";
import { PreferenceSettings } from "./PreferenceSettings";
import { ExpectationSettings } from "./ExpectationSettings";

export function SettingsPage({ onBack }: { onBack: () => void }) {
  const [activeSection, setActiveSection] = useState("general");
  const sections = [
    { id: "general", label: "General" },
    { id: "model", label: "Model" },
    { id: "preference", label: "Preference" },
    { id: "expectation", label: "Expectation" },
    { id: "tool", label: "Tool" },
  ];

  return (
    <div className="flex h-full min-w-0">
      <div className="w-48 shrink-0 flex flex-col py-6 px-3" style={{ borderRight: "1px solid var(--border)" }}>
        <div className="flex items-center gap-2.5 px-3 mb-6">
          <button onClick={onBack} className="flex items-center justify-center w-5 h-5 rounded" style={{ color: "var(--muted-foreground)" }} aria-label="Back">
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M6.5 1 2.5 5l4 4" /></svg>
          </button>
          <span style={{ fontFamily: "Instrument Serif, serif", fontSize: 16, color: "var(--foreground)" }}>Settings</span>
        </div>
        <div className="flex flex-col gap-1">
          {sections.map((s) => (
            <button
              key={s.id}
              onClick={() => setActiveSection(s.id)}
              className="text-left text-xs px-3 py-1.5 rounded transition-colors"
              style={{
                background: activeSection === s.id ? "var(--secondary)" : "transparent",
                color: activeSection === s.id ? "var(--foreground)" : "var(--muted-foreground)",
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 min-w-0 overflow-y-auto px-6 py-6">
        {activeSection === "general" && <GeneralSettings />}
        {activeSection === "model" && <ModelSettings />}
        {activeSection === "preference" && <PreferenceSettings />}
        {activeSection === "expectation" && <ExpectationSettings />}
        {activeSection === "tool" && (
          <p className="text-xs" style={{ color: "var(--muted-foreground)" }}>Tool settings — coming soon.</p>
        )}
      </div>
    </div>
  );
}
