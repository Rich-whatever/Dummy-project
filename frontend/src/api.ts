// Thin client for the Dummy backend HTTP API (see api/ in the backend root).

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

// ── Preferences ──────────────────────────────────────────────────────

export interface ApiNote {
  content: string;
  id: string;
  saved_time: string;
}
export interface ApiCategory {
  category_name: string;
  category_description: string;
  notes: ApiNote[];
}
export interface ApiState {
  state_name: string;
  description: string;
  event_list: string[];
  active_time: string;
}
export interface ApiPreferences {
  states: ApiState[];
  categories: ApiCategory[];
}

export const fetchPreferences = () =>
  request<ApiPreferences>("/api/preferences");

export const savePreferences = (data: ApiPreferences) =>
  request<ApiPreferences>("/api/preferences", {
    method: "PUT",
    body: JSON.stringify(data),
  });

// ── Expectations + behavior guide ───────────────────────────────────

export interface ApiExpectation {
  trigger: string;
  action: string;
  expectation_content: string;
}
export interface ApiExpectations {
  guides: string[]; // plain strings in the behavior-guide file
  expectations: ApiExpectation[];
}

export const fetchExpectations = () =>
  request<ApiExpectations>("/api/expectations");

export const saveExpectations = (data: ApiExpectations) =>
  request<ApiExpectations>("/api/expectations", {
    method: "PUT",
    body: JSON.stringify(data),
  });

// ── Runtime config (General + Model) ────────────────────────────────

export type GeneralConfig = Record<string, number>;

export interface ModelConfig {
  model_name: string;
  temperature: number;
  context_window: string;
  reasoning_effort: "low" | "medium" | "high";
}
export interface RuntimeConfig {
  general: GeneralConfig;
  general_overrides: GeneralConfig;
  model: ModelConfig;
  model_overrides: Partial<ModelConfig>;
}

export const fetchConfig = () => request<RuntimeConfig>("/api/config");

export const saveConfigGeneral = (general: GeneralConfig) =>
  request<RuntimeConfig>("/api/config/general", {
    method: "PUT",
    body: JSON.stringify({ general }),
  });

export const saveConfigModel = (model: ModelConfig) =>
  request<RuntimeConfig>("/api/config/model", {
    method: "PUT",
    body: JSON.stringify({ model }),
  });

// ── Chat (agent round) ───────────────────────────────────────────────

export interface ChatResult {
  user_message: string;
  agent_response: string;
  underlying_content: string | null;
  router_result: string;
  thread_was_evicted: string;
  response_time_s: number;
}
export interface ChatRunStatus {
  run_id: string;
  status: "running" | "done" | "failed" | "cancelled";
  result?: ChatResult;
  error?: string;
}

export const sendChat = (
  user_message: string,
  upload_ids: string[] = [],
  references: { timestamp: string; user_message: string; agent_message: string }[] = [],
) =>
  request<{ run_id: string }>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ user_message, upload_ids, references }),
  });

// ── File uploads (web UI attachments) ───────────────────────────────

export interface UploadResult {
  upload_id: string;
  name: string;
  size: number;
}
export interface UploadStatusRecord {
  upload_id: string;
  name: string;
  status: "pending" | "done" | "error";
  reason: string;
}

// Upload a file (multipart). The JSON `request()` helper can't send files,
// so this uses its own fetch with a FormData body.
export const uploadFile = (file: File): Promise<UploadResult> =>
  fetch(`${BASE}/api/upload`, { method: "POST", body: formDataWith(file) }).then(
    (res) => {
      if (!res.ok) {
        throw new Error(`upload -> ${res.status} ${res.statusText}`);
      }
      return res.json() as Promise<UploadResult>;
    },
  );

function formDataWith(file: File): FormData {
  const fd = new FormData();
  fd.append("file", file, file.name);
  return fd;
}

export const fetchUploadStatus = (ids: string[]) =>
  request<{ uploads: UploadStatusRecord[]; ready: boolean }>(
    `/api/upload/status?ids=${encodeURIComponent(ids.join(","))}`,
  );

export const removeUpload = (id: string) =>
  request<{ discarded: boolean }>(`/api/upload/${id}`, { method: "DELETE" });

export const getChatStatus = (run_id: string) =>
  request<ChatRunStatus>(`/api/chat/runs/${run_id}`);

export const cancelChat = (run_id: string) =>
  request<{ cancelled: boolean }>(`/api/chat/runs/${run_id}/cancel`, {
    method: "POST",
  });

// ── Chat history (persisted 30-round scrolling window) ──────────────

export interface ChatHistoryRound {
  timestamp: string; // ISO-8601
  user_message: string;
  agent_response: string;
  underlying_content: string | null;
  response_time_s: number;
  thread_was_evicted: string;
  references?: { timestamp: string; user_message: string; agent_message: string }[];
}

export const fetchChatHistory = () =>
  request<{ rounds: ChatHistoryRound[] }>("/api/chat/history");

