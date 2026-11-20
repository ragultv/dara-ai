import { create } from "zustand";
import type { Step } from "../artifacts/assistant.ts";
import { storeApi } from "../api/store-api.ts";

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
  /** documents whose saved chat has been read from the server this session */
  loaded: Record<string, boolean>;
  /** Reads the saved chat of a document (once per session). Messages that were still running when the app closed are marked stopped. */
  load: (docId: string) => Promise<void>;
  /** Saves one message with the document. A failure is silent: the chat stays on screen and is saved with the next message. */
  persist: (docId: string, msg: DocChatMsg) => Promise<void>;
  /** Deletes the saved chat of a document and empties the panel. */
  clear: (docId: string) => Promise<void>;
  push: (docId: string, msg: DocChatMsg) => void;
  patch: (docId: string, id: string, patch: Partial<AssistantMsg>) => void;
  clearUndo: (docId: string) => void;
}

const EMPTY: DocChatMsg[] = [];
export const selectMessages =
  (docId: string) =>
  (s: DocChatState): DocChatMsg[] =>
    s.byDoc[docId] ?? EMPTY;

const MAX_SAVED_REASONING = 60_000;

/** What is saved of a message: everything except the undo snapshot (the document's version history covers going back). */
export function savedForm(msg: DocChatMsg): Record<string, unknown> {
  if (msg.role === "user") return { ...msg };
  const { undo: _undo, ...rest } = msg;
  return {
    ...rest,
    reasoning: rest.reasoning.slice(0, MAX_SAVED_REASONING),
    reasoningStreaming: false
  };
}

/** A saved row back into a message; anything unreadable is dropped. */
export function fromSaved(id: string, data: Record<string, unknown>): DocChatMsg | null {
  if (data["role"] === "user" && typeof data["text"] === "string")
    return { id, role: "user", text: data["text"] };
  if (data["role"] !== "assistant") return null;
  const status = data["status"];
  const interrupted = status === "running";
  return {
    id,
    role: "assistant",
    text: interrupted
      ? "Interrupted when the app was closed."
      : typeof data["text"] === "string"
        ? data["text"]
        : "",
    steps: Array.isArray(data["steps"]) ? (data["steps"] as Step[]) : [],
    reasoning: typeof data["reasoning"] === "string" ? data["reasoning"] : "",
    reasoningStreaming: false,
    thinkingMs: typeof data["thinkingMs"] === "number" ? data["thinkingMs"] : 0,
    status: interrupted || status === "stopped" ? "stopped" : status === "error" ? "error" : "done",
    undo: null
  };
}

export const useDocChatStore = create<DocChatState>((set, get) => ({
  byDoc: {},
  loaded: {},
  load: async (docId) => {
    if (get().loaded[docId]) return;
    set((s) => ({ loaded: { ...s.loaded, [docId]: true } }));
    try {
      const res = await storeApi.listArtifactChat(docId);
      const saved = res.data
        .map((r) => fromSaved(r.id, r.data))
        .filter((m): m is DocChatMsg => m !== null);
      // anything said while this was loading stays after the saved messages
      set((s) => ({ byDoc: { ...s.byDoc, [docId]: [...saved, ...(s.byDoc[docId] ?? [])] } }));
    } catch {
      set((s) => ({ loaded: { ...s.loaded, [docId]: false } })); // try again next time the panel opens
    }
  },
  persist: async (docId, msg) => {
    try {
      await storeApi.putArtifactChatMessage(docId, msg.id, savedForm(msg));
    } catch {
      // not saved this time
    }
  },
  clear: async (docId) => {
    try {
      await storeApi.clearArtifactChat(docId);
    } catch {
      // the panel is still emptied
    }
    set((s) => ({ byDoc: { ...s.byDoc, [docId]: [] } }));
  },
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
