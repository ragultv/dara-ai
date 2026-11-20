import { create } from "zustand";
import { newId, storeApi } from "../api/store-api.ts";
import type { DbArtifactComment } from "../api/types.ts";

// The comments of the open document (docs/ui/10-artifact-collab.md). The server is the source of truth; the list here follows it.

interface CommentsState {
  docId: string | null;
  items: DbArtifactComment[];
  error: string | null;
  /** the thread shown in the panel and highlighted in the page */
  activeId: string | null;
  load: (docId: string) => Promise<void>;
  reset: () => void;
  setActive: (id: string | null) => void;
  add: (input: {
    body: string;
    quote?: string;
    nth?: number;
    parent_id?: string;
    author?: "user" | "dara";
  }) => Promise<DbArtifactComment | null>;
  setResolved: (id: string, resolved: boolean) => Promise<void>;
  remove: (id: string) => Promise<void>;
}

const message = (e: unknown) =>
  e instanceof Error && e.message ? e.message : "Something went wrong with the comments";

export const useCommentsStore = create<CommentsState>((set, get) => ({
  docId: null,
  items: [],
  error: null,
  activeId: null,

  load: async (docId) => {
    set({ docId, items: [], error: null, activeId: null });
    try {
      const res = await storeApi.listArtifactComments(docId);
      if (get().docId === docId) set({ items: res.data });
    } catch (e) {
      if (get().docId === docId) set({ error: message(e) });
    }
  },

  reset: () => set({ docId: null, items: [], error: null, activeId: null }),

  setActive: (activeId) => set({ activeId }),

  add: async (input) => {
    const docId = get().docId;
    if (!docId) return null;
    try {
      const c = await storeApi.createArtifactComment(docId, { id: newId(), ...input });
      set((s) => (s.docId === docId ? { items: [...s.items, c], error: null } : s));
      return c;
    } catch (e) {
      set({ error: message(e) });
      return null;
    }
  },

  setResolved: async (id, resolved) => {
    const docId = get().docId;
    if (!docId) return;
    try {
      const c = await storeApi.patchArtifactComment(docId, id, { resolved });
      set((s) => ({ items: s.items.map((x) => (x.id === id ? c : x)) }));
    } catch (e) {
      set({ error: message(e) });
    }
  },

  remove: async (id) => {
    const docId = get().docId;
    if (!docId) return;
    try {
      await storeApi.deleteArtifactComment(docId, id);
      set((s) => ({
        items: s.items.filter((x) => x.id !== id && x.parent_id !== id),
        activeId: s.activeId === id ? null : s.activeId
      }));
    } catch (e) {
      set({ error: message(e) });
    }
  }
}));
