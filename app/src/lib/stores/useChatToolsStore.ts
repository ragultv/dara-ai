import { create } from "zustand";
import type { CallState } from "../chat/tools/types.ts";
import { fromToolParams, toToolParams, type ToolParams } from "../chat/tools/records.ts";

// Connectors in chat (docs/CHAT_TOOLS.md): which connectors each conversation uses, the live state of tool calls,
// and the approvals waiting for the user. The connector choice and the "always allow" list of a stored chat live in its
// conversation params (useChatStore writes them); only the choice made before a chat exists ("__new__") is kept here.

const STORAGE_KEY = "dara.chatConnectors";
/** the key for a chat that has not been created yet */
export const NEW_CHAT = "__new__";

const keyOf = (conversationId: string | null) => conversationId ?? NEW_CHAT;

/** "Always allow" is kept per real tool (connector id + tool name), never per chat tool name: chat names are rebuilt on
 *  every send and can move to another connector's tool when the set of connectors changes. */
export const allowKey = (connectorId: string, tool: string) => `${connectorId}\n${tool}`;

function readEnabled(): Record<string, string[]> {
  try {
    const raw = globalThis.localStorage?.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const out: Record<string, string[]> = {};
    for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
      if (Array.isArray(v)) out[k] = v.filter((x): x is string => typeof x === "string");
    }
    return out;
  } catch {
    return {};
  }
}

function writeEnabled(enabled: Record<string, string[]>): void {
  try {
    const pending = enabled[NEW_CHAT];
    globalThis.localStorage?.setItem(
      STORAGE_KEY,
      JSON.stringify(pending ? { [NEW_CHAT]: pending } : {})
    );
  } catch {
    // storage blocked or full: the choice still holds for this session
  }
}

/** resolvers for approvals waiting on the user; kept outside the state (functions are not state) */
const waiting = new Map<string, (allow: boolean) => void>();

interface ChatToolsState {
  enabled: Record<string, string[]>;
  calls: Record<string, CallState>;
  /** call ids waiting for the user, in order */
  approvals: string[];
  /** conversationId -> allowKey(connectorId, tool) of the tools the user allowed for the rest of that conversation */
  alwaysAllowed: Record<string, string[]>;
  /** conversationId -> short live line, undefined when idle */
  status: Record<string, string | undefined>;

  enabledFor: (conversationId: string | null) => string[];
  toggle: (conversationId: string | null, connectorId: string) => void;
  adoptNew: (conversationId: string) => void;
  setCall: (c: CallState) => void;
  patchCall: (id: string, patch: Partial<CallState>) => void;
  requestApproval: (callId: string, signal: AbortSignal) => Promise<boolean>;
  resolveApproval: (callId: string, allow: boolean, always?: boolean) => void;
  setStatus: (conversationId: string, s: string | undefined) => void;
  forgetConversation: (conversationId: string) => void;
  /** what the conversation's params store about connectors */
  paramsFor: (conversationId: string) => ToolParams;
  /** takes the connector choice and allowed tools from a stored conversation's params; older chats without them keep
   *  what this browser had. Returns true when the params had no choice but this browser did (it should be written back). */
  applyParams: (
    conversationId: string,
    params: Record<string, unknown> | null | undefined
  ) => boolean;
}

const EMPTY: string[] = [];

export const useChatToolsStore = create<ChatToolsState>((set, get) => ({
  enabled: readEnabled(),
  calls: {},
  approvals: [],
  alwaysAllowed: {},
  status: {},

  enabledFor: (conversationId) => get().enabled[keyOf(conversationId)] ?? EMPTY,

  toggle: (conversationId, connectorId) => {
    const key = keyOf(conversationId);
    const current = get().enabled[key] ?? [];
    const next = current.includes(connectorId)
      ? current.filter((x) => x !== connectorId)
      : [...current, connectorId];
    const enabled = { ...get().enabled };
    if (next.length) enabled[key] = next;
    else delete enabled[key];
    set({ enabled });
    writeEnabled(enabled);
  },

  adoptNew: (conversationId) => {
    const pending = get().enabled[NEW_CHAT];
    if (!pending) return;
    const enabled = { ...get().enabled };
    delete enabled[NEW_CHAT];
    enabled[conversationId] = pending;
    set({ enabled });
    writeEnabled(enabled);
  },

  setCall: (c) => set((s) => ({ calls: { ...s.calls, [c.id]: c } })),

  patchCall: (id, patch) =>
    set((s) => {
      const prev = s.calls[id];
      if (!prev) return s;
      return { calls: { ...s.calls, [id]: { ...prev, ...patch } } };
    }),

  requestApproval: (callId, signal) =>
    new Promise<boolean>((resolve) => {
      if (signal.aborted) {
        resolve(false);
        return;
      }
      const finish = (allow: boolean) => {
        if (!waiting.has(callId)) return;
        waiting.delete(callId);
        signal.removeEventListener("abort", onAbort);
        set((s) => ({ approvals: s.approvals.filter((x) => x !== callId) }));
        resolve(allow);
      };
      const onAbort = () => finish(false);
      // a second request for the same call replaces the first, which is refused
      waiting.get(callId)?.(false);
      waiting.set(callId, finish);
      signal.addEventListener("abort", onAbort, { once: true });
      set((s) => ({
        approvals: s.approvals.includes(callId) ? s.approvals : [...s.approvals, callId]
      }));
    }),

  resolveApproval: (callId, allow, always) => {
    if (allow && always) {
      const call = get().calls[callId];
      if (call?.connectorId) {
        const key = allowKey(call.connectorId, call.tool);
        const list = get().alwaysAllowed[call.conversationId] ?? [];
        if (!list.includes(key)) {
          set((s) => ({
            alwaysAllowed: { ...s.alwaysAllowed, [call.conversationId]: [...list, key] }
          }));
        }
      }
    }
    if (get().calls[callId])
      get().patchCall(callId, {
        decision: allow ? (always ? "approved_always" : "approved_once") : "denied_by_user"
      });
    const finish = waiting.get(callId);
    if (finish) finish(allow);
    else set((s) => ({ approvals: s.approvals.filter((x) => x !== callId) }));
  },

  setStatus: (conversationId, s) =>
    set((st) =>
      st.status[conversationId] === s ? st : { status: { ...st.status, [conversationId]: s } }
    ),

  forgetConversation: (conversationId) => {
    const s = get();
    const callIds = Object.values(s.calls)
      .filter((c) => c.conversationId === conversationId)
      .map((c) => c.id);
    for (const id of callIds) waiting.get(id)?.(false);
    const calls = { ...s.calls };
    for (const id of callIds) delete calls[id];
    const enabled = { ...s.enabled };
    const hadEnabled = conversationId in enabled;
    delete enabled[conversationId];
    const alwaysAllowed = { ...s.alwaysAllowed };
    delete alwaysAllowed[conversationId];
    const status = { ...s.status };
    delete status[conversationId];
    set({
      calls,
      enabled,
      alwaysAllowed,
      status,
      approvals: get().approvals.filter((x) => !callIds.includes(x))
    });
    if (hadEnabled) writeEnabled(enabled);
  },

  paramsFor: (conversationId) =>
    toToolParams(
      get().enabled[conversationId] ?? EMPTY,
      get().alwaysAllowed[conversationId] ?? EMPTY
    ),

  applyParams: (conversationId, params) => {
    const { enabled: fromDb, allowKeys } = fromToolParams(params);
    const s = get();
    const enabled = { ...s.enabled };
    const legacy = fromDb === null && (s.enabled[conversationId]?.length ?? 0) > 0;
    if (fromDb !== null) {
      if (fromDb.length) enabled[conversationId] = fromDb;
      else delete enabled[conversationId];
    }
    const alwaysAllowed = { ...s.alwaysAllowed };
    if (allowKeys.length) alwaysAllowed[conversationId] = allowKeys;
    else delete alwaysAllowed[conversationId];
    const same = (a: string[] | undefined, b: string[] | undefined) =>
      (a ?? EMPTY).join(" ") === (b ?? EMPTY).join(" ");
    if (
      !same(enabled[conversationId], s.enabled[conversationId]) ||
      !same(alwaysAllowed[conversationId], s.alwaysAllowed[conversationId])
    )
      set({ enabled, alwaysAllowed });
    return legacy;
  }
}));
