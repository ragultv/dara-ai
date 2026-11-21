import { create } from "zustand";
import { getBackend } from "../connectors/backend.ts";
import { validateNewConnector } from "../connectors/validate.ts";
import type {
  AddConnectorInput,
  AuditEntry,
  ConnectorEntry,
  ConnectorStatus,
  ConnectorTool,
  ToolCallResult,
  ToolPermission
} from "../connectors/types.ts";

// The connectors page's state. The desktop layer owns the truth (connections, permissions, audit); this follows it.

const message = (e: unknown) =>
  e instanceof Error && e.message ? e.message : "Something went wrong with the connector";

interface ConnectorsState {
  entries: ConnectorEntry[];
  loading: boolean;
  error: string | null;
  tools: Record<string, ConnectorTool[] | undefined>;
  audit: Record<string, AuditEntry[] | undefined>;
  /** connectors with a connect or disconnect under way */
  busy: Record<string, boolean | undefined>;

  load: () => Promise<void>;
  /** Returns an error message, or null when the connector was added. */
  add: (input: AddConnectorInput) => Promise<string | null>;
  remove: (id: string) => Promise<void>;
  setEnabled: (id: string, enabled: boolean) => Promise<void>;
  connect: (id: string) => Promise<void>;
  disconnect: (id: string) => Promise<void>;
  /** connectors waiting for the user to finish signing in, in the browser */
  signingIn: Record<string, boolean | undefined>;
  signIn: (id: string) => Promise<void>;
  cancelSignIn: (id: string) => Promise<void>;
  signOut: (id: string) => Promise<void>;
  loadTools: (id: string) => Promise<void>;
  setPermission: (id: string, tool: string, permission: ToolPermission) => Promise<void>;
  call: (
    id: string,
    tool: string,
    args: Record<string, unknown>,
    approved: boolean
  ) => Promise<ToolCallResult | { error: string }>;
  loadAudit: (id: string) => Promise<void>;
}

export const useConnectorsStore = create<ConnectorsState>((set, get) => {
  const setStatus = (id: string, status: ConnectorStatus) =>
    set((s) => ({ entries: s.entries.map((e) => (e.config.id === id ? { ...e, status } : e)) }));
  const withBusy = async (id: string, fn: () => Promise<void>) => {
    set((s) => ({ busy: { ...s.busy, [id]: true } }));
    try {
      await fn();
    } finally {
      set((s) => ({ busy: { ...s.busy, [id]: false } }));
    }
  };

  return {
    entries: [],
    loading: false,
    error: null,
    tools: {},
    audit: {},
    busy: {},
    signingIn: {},

    load: async () => {
      set({ loading: true, error: null });
      try {
        set({ entries: await getBackend().list(), loading: false });
      } catch (e) {
        set({ loading: false, error: message(e) });
      }
    },

    add: async (input) => {
      const problem = validateNewConnector(input, get().entries.length);
      if (problem) return problem;
      try {
        await getBackend().add(input);
        await get().load();
        return null;
      } catch (e) {
        return message(e);
      }
    },

    remove: async (id) => {
      try {
        await getBackend().remove(id);
        set((s) => ({
          entries: s.entries.filter((e) => e.config.id !== id),
          tools: { ...s.tools, [id]: undefined },
          audit: { ...s.audit, [id]: undefined }
        }));
      } catch (e) {
        set({ error: message(e) });
      }
    },

    setEnabled: async (id, enabled) => {
      try {
        await getBackend().update(id, { enabled });
        await get().load();
        if (!enabled) set((s) => ({ tools: { ...s.tools, [id]: undefined } }));
      } catch (e) {
        set({ error: message(e) });
      }
    },

    connect: (id) =>
      withBusy(id, async () => {
        setStatus(id, {
          ...(get().entries.find((e) => e.config.id === id)?.status ?? emptyStatus),
          state: "connecting",
          error: null
        });
        try {
          const status = await getBackend().connect(id);
          setStatus(id, status);
          if (status.state === "ready") await get().loadTools(id);
        } catch (e) {
          setStatus(id, { ...emptyStatus, state: "error", error: message(e) });
        }
      }),

    disconnect: (id) =>
      withBusy(id, async () => {
        try {
          await getBackend().disconnect(id);
          await get().load();
          set((s) => ({ tools: { ...s.tools, [id]: undefined } }));
        } catch (e) {
          set({ error: message(e) });
        }
      }),

    signIn: async (id) => {
      set((s) => ({ signingIn: { ...s.signingIn, [id]: true } }));
      try {
        const status = await getBackend().signIn(id);
        setStatus(id, status);
        if (status.state === "ready") await get().loadTools(id);
      } catch (e) {
        // the browser step was cancelled, timed out or failed: say so on the card
        const prior = get().entries.find((x) => x.config.id === id)?.status ?? emptyStatus;
        setStatus(id, { ...prior, state: "needs_auth", error: message(e) });
      } finally {
        set((s) => ({ signingIn: { ...s.signingIn, [id]: false } }));
      }
    },

    cancelSignIn: async (id) => {
      try {
        await getBackend().cancelSignIn(id);
      } catch (e) {
        set({ error: message(e) });
      }
    },

    signOut: async (id) => {
      try {
        setStatus(id, await getBackend().signOut(id));
        set((s) => ({ tools: { ...s.tools, [id]: undefined } }));
      } catch (e) {
        set({ error: message(e) });
      }
    },

    loadTools: async (id) => {
      try {
        const tools = await getBackend().tools(id);
        set((s) => ({ tools: { ...s.tools, [id]: tools } }));
      } catch (e) {
        set({ error: message(e) });
      }
    },

    setPermission: async (id, tool, permission) => {
      // the choice shows at once; the desktop layer's answer (re-read below) wins
      set((s) => ({
        tools: {
          ...s.tools,
          [id]: s.tools[id]?.map((t) =>
            t.name === tool ? { ...t, permission, changed: false } : t
          )
        }
      }));
      try {
        await getBackend().setPermission(id, tool, permission);
      } catch (e) {
        set({ error: message(e) });
      }
      await get().loadTools(id);
    },

    call: async (id, tool, args, approved) => {
      try {
        const result = await getBackend().call(id, tool, args, approved);
        void get().loadAudit(id);
        return result;
      } catch (e) {
        return { error: message(e) };
      }
    },

    loadAudit: async (id) => {
      try {
        const rows = await getBackend().audit(id, 20);
        set((s) => ({ audit: { ...s.audit, [id]: rows } }));
      } catch (e) {
        set({ error: message(e) });
      }
    }
  };
});

const emptyStatus: ConnectorStatus = {
  state: "disconnected",
  error: null,
  server: null,
  toolCount: 0,
  toolsChanged: false,
  resourceMetadata: null,
  signedIn: false,
  neededScope: null
};
