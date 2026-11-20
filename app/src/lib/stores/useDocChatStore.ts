import { create } from "zustand";
import type { Step } from "../artifacts/assistant.ts";

// The conversation beside a document: what the user asked, what Dara did, and Dara's thinking. It lives in memory for the app session
// (one list per document); the document itself is what is saved.

export interface UserMsg {
  id: string;
  role: "user";
  text: string;
}
export interface AssistantMsg {
  id: string;
  role: "assistant";
  text: string;
  steps: Step[];
  reasoning: string;
  reasoningStreaming: boolean;
  thinkingMs: number;
  status: "running" | "done" | "stopped" | "error";
  /** the document before this answer changed it, so one click puts it back; only the newest answer keeps it */
  undo: { markdown: string; title: string } | null;
}
export type DocChatMsg = UserMsg | AssistantMsg;

interface DocChatState {
  byDoc: Record<string, DocChatMsg[]>;
  push: (docId: string, msg: DocChatMsg) => void;
  patch: (docId: string, id: string, patch: Partial<AssistantMsg>) => void;
  clearUndo: (docId: string) => void;
}

const EMPTY: DocChatMsg[] = [];
export const selectMessages =
  (docId: string) =>
  (s: DocChatState): DocChatMsg[] =>
    s.byDoc[docId] ?? EMPTY;

export const useDocChatStore = create<DocChatState>((set) => ({
  byDoc: {},
  push: (docId, msg) =>
    set((s) => ({ byDoc: { ...s.byDoc, [docId]: [...(s.byDoc[docId] ?? []), msg] } })),
  patch: (docId, id, patch) =>
    set((s) => ({
      byDoc: {
        ...s.byDoc,
        [docId]: (s.byDoc[docId] ?? []).map((m) =>
          m.id === id && m.role === "assistant" ? { ...m, ...patch } : m
        )
      }
    })),
  clearUndo: (docId) =>
    set((s) => ({
      byDoc: {
        ...s.byDoc,
        [docId]: (s.byDoc[docId] ?? []).map((m) =>
          m.role === "assistant" && m.undo ? { ...m, undo: null } : m
        )
      }
    }))
}));
