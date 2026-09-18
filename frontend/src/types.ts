export type OrbState = "idle" | "thinking";
export type View = "chat" | "settings";

export interface Attachment {
  id: string;
  name: string;
  size: number; // bytes
  uploadId?: string;              // server upload_id once uploaded (Round 2)
  status?: "processing" | "ready" | "error"; // absent for historical rounds
  errorReason?: string;           // human-readable reason when status === "error"
}

export interface RoundMeta {
  response_time_s: number;
  tokens_in: number;
  tokens_out: number;
  underlying_content: string;
  worker_report: string;
}

export interface ReferencedRound {
  id: string;
  timestamp: string; // ISO-8601 of the referenced round
  user_message: string;
  agent_message: string;
}

export interface ConversationRound {
  id: string;
  user_message: string;
  agent_message: string; // "" while pending
  attachments: Attachment[];
  references?: ReferencedRound[]; // earlier rounds this message referenced
  timestamp: Date;
  meta: RoundMeta;
}

// ─── Sphere geometry (computed once) ─────────────────────────────────────────
