import { create } from "zustand";
import { newId, storeApi } from "../api/store-api.ts";
import { ApiError } from "../api/errors.ts";
import { unauthorizedMessage } from "../artifacts/errors.ts";
import type { ArtifactKind, DbArtifact, DbArtifactSummary } from "../api/types.ts";

// Artifacts (docs/ui/09-artifacts.md): the gallery list and the one open document. The database is the source of truth; the open
// document is edited in memory and written back a moment after the last keystroke (debounced PATCH), and always when it is left.

function userFacingError(err: unknown): { message: string } {
  if (err instanceof ApiError && err.status === 401) return { message: unauthorizedMessage() };
  return { message: err instanceof Error && err.message ? err.message : "Something went wrong" };
}

export type ArtifactSaveState = "idle" | "saving" | "saved" | "error";

interface ArtifactsState {
  items: DbArtifactSummary[];
  loading: boolean;
  /** why the list could not be loaded (engine not running, database unavailable, ...) */
  error: string | null;
  /** the open document, or null */
  doc: DbArtifact | null;
  docLoading: boolean;
  docError: string | null;
  saveState: ArtifactSaveState;

  load: () => Promise<void>;
  create: (
    kind?: ArtifactKind,
    init?: { title?: string; content?: string }
  ) => Promise<string | null>;
  open: (id: string) => Promise<void>;
  close: () => Promise<void>;
  setTitle: (title: string) => void;
  /** `snapshot` asks the server to keep the text as it was before this change as a labelled version. */
  setContent: (content: string, opts?: { snapshot?: string }) => void;
  /** Writes pending edits now (leaving the page, closing the window). */
  flush: () => Promise<void>;
  /** Puts back an earlier version (the server keeps the current text as "Before restore"). Returns the restored document. */
  restore: (rev: number) => Promise<DbArtifact | null>;
  remove: (id: string) => Promise<boolean>;
  duplicate: (id: string) => Promise<string | null>;
}

const SAVE_DELAY_MS = 700;
const SAVED_VISIBLE_MS = 2000;

let pending: { title?: string; content?: string; snapshot?: string } = {};
let timer: ReturnType<typeof setTimeout> | null = null;
let savedTimer: ReturnType<typeof setTimeout> | null = null;
let inFlight: Promise<void> = Promise.resolve();

function summaryOf(a: DbArtifact): DbArtifactSummary {
  const { content, ...rest } = a;
  return { ...rest, snippet: content.slice(0, 600) };
}

export const useArtifactsStore = create<ArtifactsState>((set, get) => {
  const writePending = async (): Promise<void> => {
    const doc = get().doc;
    if (!doc || (pending.title === undefined && pending.content === undefined)) return;
    const patch = pending;
    pending = {};
    set({ saveState: "saving" });
    try {
      const saved = await storeApi.patchArtifact(doc.id, patch);
      set((s) => ({
        saveState: "saved",
        // only the timestamp is taken from the answer: the text on screen may already be newer than what was just written
        doc: s.doc && s.doc.id === saved.id ? { ...s.doc, updated_at: saved.updated_at } : s.doc,
        items: s.items.map((i) => (i.id === saved.id ? summaryOf(saved) : i))
      }));
      if (savedTimer) clearTimeout(savedTimer);
      savedTimer = setTimeout(() => {
        if (get().saveState === "saved") set({ saveState: "idle" });
      }, SAVED_VISIBLE_MS);
    } catch (err) {
      pending = { ...patch, ...pending }; // keep it for the next try
      set({ saveState: "error", docError: userFacingError(err).message });
    }
  };

  const schedule = () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      inFlight = inFlight.then(writePending);
    }, SAVE_DELAY_MS);
  };

  return {
    items: [],
    loading: false,
    error: null,
    doc: null,
    docLoading: false,
    docError: null,
    saveState: "idle",

    load: async () => {
      set({ loading: true, error: null });
      try {
        const res = await storeApi.listArtifacts();
        set({ items: res.data, loading: false });
      } catch (err) {
        set({ loading: false, error: userFacingError(err).message });
      }
    },

    create: async (kind = "doc", init) => {
      try {
        const a = await storeApi.createArtifact({
          id: newId(),
          kind,
          title: init?.title ?? "",
          content: init?.content ?? ""
        });
        set((s) => ({ items: [summaryOf(a), ...s.items] }));
        return a.id;
      } catch (err) {
        set({ error: userFacingError(err).message });
        return null;
      }
    },

    open: async (id) => {
      await get().flush();
      set({ docLoading: true, docError: null, saveState: "idle", doc: null });
      try {
        const doc = await storeApi.getArtifact(id);
        set({ doc, docLoading: false });
      } catch (err) {
        set({ docLoading: false, docError: userFacingError(err).message });
      }
    },

    close: async () => {
      await get().flush();
      set({ doc: null, docError: null, saveState: "idle" });
    },

    setTitle: (title) => {
      set((s) => (s.doc ? { doc: { ...s.doc, title } } : s));
      pending = { ...pending, title };
      schedule();
    },

    setContent: (content, opts) => {
      set((s) => (s.doc ? { doc: { ...s.doc, content, chars: content.length } } : s));
      pending = { ...pending, content, ...(opts?.snapshot ? { snapshot: opts.snapshot } : {}) };
      if (opts?.snapshot) {
        // a labelled snapshot is written at once: the label is about the text as it is right now
        void get().flush();
      } else schedule();
    },

    flush: async () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      inFlight = inFlight.then(writePending);
      await inFlight;
    },

    restore: async (rev) => {
      const doc = get().doc;
      if (!doc) return null;
      try {
        await get().flush();
        const restored = await storeApi.restoreArtifact(doc.id, rev);
        set((s) => ({
          doc: s.doc && s.doc.id === restored.id ? restored : s.doc,
          items: s.items.map((i) => (i.id === restored.id ? summaryOf(restored) : i)),
          saveState: "idle"
        }));
        return restored;
      } catch (err) {
        set({ docError: userFacingError(err).message });
        return null;
      }
    },

    remove: async (id) => {
      try {
        if (get().doc?.id === id) {
          pending = {};
          if (timer) clearTimeout(timer);
          timer = null;
        }
        await storeApi.deleteArtifact(id);
        set((s) => ({
          items: s.items.filter((i) => i.id !== id),
          doc: s.doc?.id === id ? null : s.doc
        }));
        return true;
      } catch (err) {
        set({ error: userFacingError(err).message });
        return false;
      }
    },

    duplicate: async (id) => {
      try {
        await get().flush();
        // the server copies the document with its images and videos; an older server without that call gets a text-only copy
        try {
          const copy = await storeApi.duplicateArtifact(id, { id: newId() });
          set((s) => ({ items: [summaryOf(copy), ...s.items] }));
          return copy.id;
        } catch (err) {
          if (!(err instanceof ApiError) || err.status !== 404) throw err;
        }
        const src = await storeApi.getArtifact(id);
        return await get().create(src.kind, {
          title: src.title ? `${src.title} (copy)` : "",
          content: src.content
        });
      } catch (err) {
        set({ error: userFacingError(err).message });
        return null;
      }
    }
  };
});
