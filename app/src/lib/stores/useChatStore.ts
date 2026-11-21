import { useEffortStore } from "./useEffortStore.ts";
import { create } from "zustand";
import { apiClient } from "../api/client.ts";
import { ApiError } from "../api/errors.ts";
import { storeApi, newId } from "../api/store-api.ts";
import type {
  ChatMessage,
  ChatRequest,
  ContentPart,
  DbConversationDetail,
  DbConversationSummary,
  DbMessage
} from "../api/types.ts";
import {
  SUMMARY_MAX_TOKENS,
  buildSummaryMessages,
  currentSummary,
  currentWindowMessages,
  dropStaleWindows,
  estimateMessageTokens,
  estimateMessagesTokens,
  estimateTokens,
  pickRolloverStart,
  planMaxTokens,
  rolloverNeeded,
  statsFromRaw,
  summarySystemMessage,
  windowStartIndex
} from "../chat/context.ts";
import type { ContextView, ContextWindow, Conversation, UiMessage } from "../chat/types.ts";
import { toDataUri } from "../utils/image.ts";
import { toast } from "../ui/Toast.tsx";
import { useModelsStore } from "./useModelsStore.ts";
import { useServerStore } from "./useServerStore.ts";
import { useSettingsStore } from "./useSettingsStore.ts";
import { useChatToolsStore } from "./useChatToolsStore.ts";
import { loadChatTools, runCalls } from "../chat/tools/runner.ts";
import {
  formatProblems,
  formatResults,
  isToolResultsContent,
  parseResults,
  parseToolCalls,
  toolSystemPrompt
} from "../chat/tools/protocol.ts";
import { MAX_REPAIRS, MAX_STEPS, type ChatTool, type ToolResultItem } from "../chat/tools/types.ts";
import { MAX_NUDGES, nudgeText, promisesMore, unfinishedWork } from "../chat/tools/jobs.ts";
import {
  buildMessageStats,
  parseToolCallRecords,
  recordFromCall,
  type ToolCallRecord,
  type ToolParams
} from "../chat/tools/records.ts";

export type { Conversation, UiMessage, ContextWindow } from "../chat/types.ts";

export type ChatPhase = "idle" | "generating" | "summarizing";

interface ChatState {
  conversations: Conversation[];
  activeConversationId: string | null;
  isGenerating: boolean;
  phase: ChatPhase;
  streamingText: string;
  currentTps: number | undefined;
  /** false when the app database cannot be reached (old engine build, or it failed to open): history then lives in this browser only. */
  storeAvailable: boolean;

  loadConversations: () => Promise<void>;
  createConversation: (title?: string) => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => Promise<void>;
  renameConversation: (id: string, title: string) => void;
  setConversationSettings: (
    id: string,
    patch: Partial<Pick<Conversation, "temperature" | "maxTokens" | "flash" | "systemPrompt">>
  ) => void;
  sendMessage: (text: string, images?: string[], modelId?: string) => Promise<void>;
  stopGeneration: () => void;
  regenerate: (messageId?: string, modelId?: string) => void;
  editAndResend: (messageId: string, newText: string, modelId?: string) => void;
  continueGeneration: (modelId?: string) => void;
  /** Starts a new context window now (summarizes the earlier messages; they stay stored and visible). */
  rollOver: (id: string) => Promise<boolean>;
  getActiveConversation: () => Conversation | null;
}

const LOCAL_KEY = "dara_conversations_v3";
const LEGACY_KEY = "dara_conversations_v2";
const CHECKPOINT_MS = 4000;
const CONTINUE_TEXT = "Please continue from exactly where you stopped.";

let abortController: AbortController | null = null;
let tokenBuffer = "";
let rafId: number | null = null;

// ---- helpers --------------------------------------------------------------------------------------------------------------------------------------

const objectUrls = new Map<string, string>();

function revokeUrls(conv: Conversation): void {
  for (const m of conv.messages) {
    for (const id of m.attachmentIds ?? []) {
      const u = objectUrls.get(id);
      if (u) {
        URL.revokeObjectURL(u);
        objectUrls.delete(id);
      }
    }
  }
}

/** Serializes database writes per conversation so a reply is patched only after it was appended, and positions stay in order. */
const queues = new Map<string, Promise<unknown>>();
let lastStoreToast = 0;

function reportStoreError(err: unknown): void {
  const status = err instanceof ApiError ? err.status : 0;
  if (status === 404 || status === 503) useChatStore.setState({ storeAvailable: false });
  const now = Date.now();
  if (now - lastStoreToast < 20000) return;
  lastStoreToast = now;
  const msg = err instanceof Error ? err.message : "unknown error";
  toast.show(`Could not save to the chat history database: ${msg}`, "error");
}

function enqueue(convId: string, job: () => Promise<unknown>): Promise<void> {
  const prev = queues.get(convId) ?? Promise.resolve();
  const next = prev.then(job).catch(reportStoreError);
  queues.set(convId, next);
  return next.then(() => undefined);
}

function loadedModelInfo() {
  return useModelsStore.getState().getLoadedModel();
}

/** The loaded model's real context (it changes with the K/V format, Flash and vision), else what the last reply reported. */
export function contextCapacity(
  conv: Pick<Conversation, "context"> | null | undefined
): number | undefined {
  const fromModel = loadedModelInfo()?.dara.config?.context;
  if (typeof fromModel === "number" && fromModel > 0) return fromModel;
  return conv?.context?.capacity;
}

export function getContextView(conv: Conversation | null | undefined): {
  used: number;
  capacity: number | undefined;
  estimated: boolean;
} {
  return {
    used: conv?.context?.used ?? 0,
    capacity: contextCapacity(conv),
    estimated: conv?.context?.estimated === true
  };
}

function inferenceDefaults() {
  const s = useSettingsStore.getState().inference;
  return {
    temperature: s.defaultTemperature,
    maxTokens: s.defaultMaxTokens,
    systemPrompt: s.systemPrompt,
    flash: s.defaultFlash
  };
}

function blankConversation(title = "New Chat"): Conversation {
  const d = inferenceDefaults();
  const now = Date.now();
  return {
    id: newId(),
    title,
    createdAt: now,
    updatedAt: now,
    messages: [],
    temperature: d.temperature,
    maxTokens: 0,
    flash: d.flash !== "off",
    systemPrompt: d.systemPrompt,
    windows: [],
    persisted: false,
    loaded: true
  };
}

function paramsOf(c: Conversation): Record<string, unknown> {
  return {
    temperature: c.temperature ?? 0.7,
    max_tokens: c.maxTokens ?? 0,
    flash: c.flash !== false,
    windows: c.windows,
    // connectors on in this chat and the tools allowed for the rest of it (docs/CHAT_TOOLS.md)
    ...useChatToolsStore.getState().paramsFor(c.id)
  };
}

/** true while connector settings are being taken from the database (so they are not written straight back) */
let applyingToolParams = false;

/** Loads a stored chat's connector settings into useChatToolsStore; a choice only this browser had is written to the database. */
function applyToolParams(id: string, params: Record<string, unknown> | null | undefined): void {
  applyingToolParams = true;
  let legacy = false;
  try {
    legacy = useChatToolsStore.getState().applyParams(id, params);
  } finally {
    applyingToolParams = false;
  }
  if (legacy) {
    const c = findConv(id);
    if (c?.persisted)
      void enqueue(id, () => storeApi.patchConversation(id, { params: paramsOf(c) }));
  }
}

function summaryToConversation(s: DbConversationSummary): Conversation {
  const p = s.params ?? {};
  const windows = Array.isArray(p.windows) ? (p.windows as ContextWindow[]) : [];
  return {
    id: s.id,
    title: s.title,
    createdAt: s.created_at,
    updatedAt: s.updated_at,
    messages: [],
    temperature: typeof p.temperature === "number" ? p.temperature : 0.7,
    maxTokens: typeof p.max_tokens === "number" ? p.max_tokens : 0,
    flash: typeof p.flash === "boolean" ? p.flash : true,
    systemPrompt: s.system_prompt,
    windows,
    persisted: true,
    loaded: false
  };
}

function dbMessageToUi(m: DbMessage): UiMessage {
  const raw = m.stats ?? undefined;
  const finish = m.finish_reason ?? undefined;
  const toolCalls =
    m.role === "assistant"
      ? parseToolCallRecords((raw as { tool_calls?: unknown } | undefined)?.tool_calls)
      : undefined;
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    attachmentIds: m.attachments.map((a) => a.id),
    ...(finish ? { finishReason: finish } : {}),
    ...(finish === "interrupted" ? { interrupted: true } : {}),
    ...(m.error ? { error: m.error } : {}),
    ...(raw ? { raw, stats: statsFromRaw(raw) } : {}),
    ...(typeof (raw as { reasoning?: unknown } | undefined)?.reasoning === "string" &&
    (raw as { reasoning: string }).reasoning
      ? {
          reasoning: (raw as { reasoning: string }).reasoning,
          ...(raw?.timings?.thinking_ms ? { reasoningMs: raw.timings.thinking_ms } : {}),
          ...(raw?.timings?.thinking_cut ? { reasoningCut: true } : {})
        }
      : {}),
    ...(toolCalls ? { toolCalls } : {}),
    createdAt: m.created_at
  };
}

function updateConv(id: string, fn: (c: Conversation) => Conversation): void {
  useChatStore.setState((s) => ({
    conversations: s.conversations.map((c) => (c.id === id ? fn(c) : c))
  }));
}

function findConv(id: string): Conversation | undefined {
  return useChatStore.getState().conversations.find((c) => c.id === id);
}

function saveLocal(): void {
  if (useChatStore.getState().storeAvailable) return;
  try {
    const tools = useChatToolsStore.getState();
    const light = useChatStore.getState().conversations.map((c) => ({
      ...c,
      messages: c.messages.map((m) => ({ ...m, images: undefined })),
      toolParams: tools.paramsFor(c.id)
    }));
    localStorage.setItem(LOCAL_KEY, JSON.stringify(light));
  } catch {
    // storage full or blocked: nothing else to do
  }
}

function ensurePersisted(conv: Conversation): void {
  if (conv.persisted) return;
  updateConv(conv.id, (c) => ({ ...c, persisted: true }));
  const cur = findConv(conv.id) ?? conv;
  void enqueue(conv.id, () =>
    storeApi.createConversation({
      id: cur.id,
      title: cur.title,
      system_prompt: cur.systemPrompt ?? "",
      params: paramsOf(cur),
      ...(loadedModelInfo()?.id ? { model: loadedModelInfo()?.id as string } : {})
    })
  );
}

/** The context fill of a stored chat, from the engine's numbers saved with its last reply. Nothing is guessed: no stored number, no meter. */
function restoredContext(
  messages: UiMessage[],
  windows: ContextWindow[]
): { context: ContextView } | Record<string, never> {
  let at = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "assistant" && messages[i]?.raw?.context) {
      at = i;
      break;
    }
  }
  if (at < 0) return {};
  const last = windows[windows.length - 1];
  if (last) {
    const start = messages.findIndex((m) => m.id === last.startMessageId);
    if (start > at) return {}; // a newer window began after that reply, so its number no longer describes the chat
  }
  const c = messages[at]!.raw!.context!;
  return {
    context: {
      capacity: c.capacity,
      used: c.used,
      compacted: c.compacted,
      summarized_messages: c.summarized_messages,
      dropped_messages: c.dropped_messages,
      truncated_last_message: c.truncated_last_message ?? false,
      summary_cached: c.summary_cached ?? false
    }
  };
}

/** Fetches a stored conversation's messages and its images (shown through blob: URLs, because <img> cannot send the API key). */
async function hydrate(id: string): Promise<void> {
  const conv = findConv(id);
  if (!conv || conv.loaded || !conv.persisted) return;
  let detail: DbConversationDetail;
  try {
    detail = await storeApi.getConversation(id);
  } catch (err) {
    reportStoreError(err);
    return;
  }
  const messages = detail.messages.map(dbMessageToUi);
  const windows = Array.isArray(detail.params?.windows)
    ? (detail.params.windows as ContextWindow[])
    : conv.windows;
  const kept = dropStaleWindows(windows, messages);
  applyToolParams(id, detail.params);
  updateConv(id, (c) => ({
    ...c,
    messages,
    windows: kept,
    // the meter after a restart: what the engine reported for this chat's last reply (stored with it), unless a new window began since
    ...(c.context ? {} : restoredContext(messages, kept)),
    loaded: true
  }));
  for (const m of detail.messages) {
    if (m.attachments.length === 0) continue;
    const urls: string[] = [];
    for (const a of m.attachments) {
      let url = objectUrls.get(a.id);
      if (!url) {
        try {
          url = URL.createObjectURL(await storeApi.getAttachmentBlob(a.id));
          objectUrls.set(a.id, url);
        } catch {
          continue;
        }
      }
      urls.push(url);
    }
    if (urls.length > 0) {
      updateConv(id, (c) => ({
        ...c,
        messages: c.messages.map((x) => (x.id === m.id ? { ...x, images: urls } : x))
      }));
    }
  }
}

/** Chats the previous app version kept in localStorage (its database writes never succeeded): copied into the database once. */
async function importLegacy(): Promise<number> {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(LEGACY_KEY);
  } catch {
    return 0;
  }
  if (!raw) return 0;
  let old: Array<{
    title?: string;
    createdAt?: number;
    messages?: Array<{ role: string; content: string; images?: string[] }>;
    temperature?: number;
    systemPrompt?: string;
  }>;
  try {
    old = JSON.parse(raw) as typeof old;
  } catch {
    return 0;
  }
  let n = 0;
  for (const oc of Array.isArray(old) ? old.slice(0, 100) : []) {
    const msgs = (oc.messages ?? []).filter(
      (m) =>
        (m.role === "user" || m.role === "assistant") && (m.content?.trim() || m.images?.length)
    );
    if (msgs.length === 0) continue;
    const id = newId();
    await storeApi.createConversation({
      id,
      title: oc.title || "Imported chat",
      system_prompt: oc.systemPrompt ?? "",
      params: { temperature: oc.temperature ?? 0.7, max_tokens: 0, flash: true, windows: [] }
    });
    for (const m of msgs) {
      const mid = newId();
      await storeApi.appendMessage(id, {
        id: mid,
        role: m.role as "user" | "assistant",
        content: m.content ?? ""
      });
      for (const img of m.images ?? []) {
        if (img.startsWith("data:")) await storeApi.addAttachment(mid, img).catch(() => undefined);
      }
    }
    n++;
  }
  try {
    localStorage.removeItem(LEGACY_KEY);
  } catch {
    // ignore
  }
  return n;
}

function userFacingError(err: unknown): { message: string; code?: string | undefined } {
  if (err instanceof ApiError) {
    if (err.code === "vision_unavailable") {
      return { message: "Image support is not loaded for this model.", code: "vision_unavailable" };
    }
    return { message: err.message, code: err.code };
  }
  return { message: err instanceof Error ? err.message : "Connection failed" };
}

// ---- store ----------------------------------------------------------------------------------------------------------------------------------------

/** The stats blob stored with a message: the engine's numbers, the model's thinking and its tool calls (so they survive a restart). */
function withReasoning(m: UiMessage): Record<string, unknown> | null {
  return buildMessageStats(m.raw as Record<string, unknown> | undefined, m.reasoning, m.toolCalls);
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  activeConversationId: null,
  isGenerating: false,
  phase: "idle",
  streamingText: "",
  currentTps: undefined,
  storeAvailable: true,

  getActiveConversation: () => {
    const { conversations, activeConversationId } = get();
    return conversations.find((c) => c.id === activeConversationId) ?? null;
  },

  loadConversations: async () => {
    // 1. the server's database
    try {
      let list = await storeApi.listConversations(200, 0);
      set({ storeAvailable: true });
      const imported = await importLegacy().catch(() => 0);
      if (imported > 0) {
        list = await storeApi.listConversations(200, 0);
        toast.show(
          `Moved ${imported} earlier conversation${imported === 1 ? "" : "s"} into the database.`,
          "info"
        );
      }
      const convs = list.data.map(summaryToConversation);
      const blank = blankConversation();
      const keepBlank = get().conversations.find((c) => !c.persisted && c.messages.length === 0);
      const all = convs.length > 0 ? convs : [keepBlank ?? blank];
      const active = all[0]?.id ?? null;
      set({ conversations: all, activeConversationId: active });
      for (const s of list.data) applyToolParams(s.id, s.params);
      if (active) void hydrate(active);
      return;
    } catch {
      set({ storeAvailable: false });
    }
    // 2. this browser's copy (engine without the database)
    try {
      const stored = localStorage.getItem(LOCAL_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Array<Conversation & { toolParams?: ToolParams }>;
        if (Array.isArray(parsed) && parsed.length > 0) {
          const convs = parsed.map(({ toolParams: _t, ...c }) => ({
            ...c,
            windows: c.windows ?? [],
            persisted: false,
            loaded: true
          }));
          set({ conversations: convs, activeConversationId: convs[0]?.id ?? null });
          for (const c of parsed) {
            if (c.toolParams)
              applyToolParams(c.id, c.toolParams as unknown as Record<string, unknown>);
          }
          return;
        }
      }
    } catch {
      // ignore
    }
    const blank = blankConversation();
    set({ conversations: [blank], activeConversationId: blank.id });
  },

  createConversation: (title = "New Chat") => {
    const empty = get().conversations.find(
      (c) =>
        c.id === get().activeConversationId && c.messages.length === 0 && c.windows.length === 0
    );
    if (empty && title === "New Chat") return empty.id; // do not pile up empty chats
    const conv = blankConversation(title);
    set((s) => ({ conversations: [conv, ...s.conversations], activeConversationId: conv.id }));
    saveLocal();
    return conv.id;
  },

  selectConversation: (id) => {
    set({ activeConversationId: id });
    void hydrate(id);
  },

  deleteConversation: async (id) => {
    const conv = findConv(id);
    const rest = get().conversations.filter((c) => c.id !== id);
    const nextActive =
      get().activeConversationId === id ? (rest[0]?.id ?? null) : get().activeConversationId;
    if (conv) revokeUrls(conv);
    useChatToolsStore.getState().forgetConversation(id);
    set({ conversations: rest, activeConversationId: nextActive });
    if (rest.length === 0) {
      const blank = blankConversation();
      set({ conversations: [blank], activeConversationId: blank.id });
    } else if (nextActive) {
      void hydrate(nextActive);
    }
    saveLocal();
    if (conv?.persisted) void enqueue(id, () => storeApi.deleteConversation(id));
  },

  renameConversation: (id, title) => {
    const t = title.trim().slice(0, 120);
    if (!t) return;
    updateConv(id, (c) => ({ ...c, title: t }));
    const c = findConv(id);
    if (c?.persisted) void enqueue(id, () => storeApi.patchConversation(id, { title: t }));
    saveLocal();
  },

  setConversationSettings: (id, patch) => {
    updateConv(id, (c) => ({ ...c, ...patch }));
    const c = findConv(id);
    if (c?.persisted) {
      void enqueue(id, () =>
        storeApi.patchConversation(id, {
          params: paramsOf(findConv(id) ?? c),
          ...(patch.systemPrompt !== undefined ? { system_prompt: patch.systemPrompt } : {})
        })
      );
    }
    saveLocal();
  },

  rollOver: async (id) => {
    if (get().isGenerating) return false;
    const conv = findConv(id);
    if (!conv) return false;
    const capacity = contextCapacity(conv) ?? 4096;
    const winStart = windowStartIndex(conv.messages, conv.windows);
    const startIdx = pickRolloverStart(conv.messages, winStart);
    if (startIdx === null) return false;
    const startMsg = conv.messages[startIdx];
    if (!startMsg) return false;
    const toSummarize = conv.messages.slice(winStart, startIdx);
    const previous = currentSummary(conv.messages, conv.windows);

    set({ isGenerating: true, phase: "summarizing" });
    try {
      const target = loadedModelInfo()?.id ?? useServerStore.getState().loadedModel ?? undefined;
      const req: ChatRequest = {
        ...(target ? { model: target } : {}),
        messages: buildSummaryMessages({
          previousSummary: previous,
          messages: toSummarize,
          capacity
        }),
        temperature: 0.2,
        max_tokens: SUMMARY_MAX_TOKENS,
        flash: false,
        cache_write: false
      };
      const res = await apiClient.chat(req);
      const summary = (res.choices?.[0]?.message?.content ?? "").trim();
      if (!summary) throw new Error("The model returned an empty summary");
      const win: ContextWindow = {
        startMessageId: startMsg.id,
        summary,
        createdAt: Date.now(),
        summarizedMessages: toSummarize.length
      };
      const kept = conv.messages.slice(startIdx);
      const sysTokens = estimateTokens(conv.systemPrompt ?? "");
      const used = sysTokens + estimateTokens(summary) + estimateMessagesTokens(kept);
      updateConv(id, (c) => ({
        ...c,
        windows: [...c.windows, win],
        context: {
          capacity: res.context?.capacity ?? capacity,
          used,
          compacted: false,
          summarized_messages: toSummarize.length,
          dropped_messages: 0,
          truncated_last_message: false,
          summary_cached: false,
          estimated: true
        }
      }));
      const cur = findConv(id);
      if (cur?.persisted)
        void enqueue(id, () =>
          storeApi.patchConversation(id, { params: paramsOf(findConv(id) ?? cur) })
        );
      saveLocal();
      return true;
    } catch (err) {
      toast.show(
        `Could not start a new context window: ${err instanceof Error ? err.message : "unknown error"}`,
        "error"
      );
      return false;
    } finally {
      set({ isGenerating: false, phase: "idle" });
    }
  },

  sendMessage: async (text, images = [], modelId) => {
    if (!text.trim() && images.length === 0) return;
    if (get().isGenerating) return;

    let convId = get().activeConversationId;
    if (!convId || !findConv(convId)) {
      convId = get().createConversation(text.slice(0, 40).trim() || "New Chat");
      // connectors picked before the chat existed now belong to it
      if (useChatToolsStore.getState().enabledFor(convId).length === 0)
        useChatToolsStore.getState().adoptNew(convId);
    }
    let conv = findConv(convId);
    if (!conv) return;
    const id = convId;
    set({ isGenerating: true, phase: "generating", streamingText: "", currentTps: undefined });

    const loaded = loadedModelInfo();
    const targetModel =
      modelId ??
      loaded?.id ??
      useServerStore.getState().loadedModel ??
      useModelsStore.getState().models.find((m) => m.dara.status === "installed")?.id;
    if (!targetModel) {
      toast.show("Install and load a model first (Models page).", "error");
      set({ isGenerating: false, phase: "idle" });
      return;
    }

    // Make room first when this turn would not fit: older messages stay stored, a summary carries on in a new context window.
    const cap0 = contextCapacity(conv);
    if (cap0) {
      const pending = estimateMessageTokens({ content: text, images });
      const used = conv.context?.used ?? 0;
      if (rolloverNeeded({ used, capacity: cap0, pendingTokens: pending })) {
        set({ isGenerating: false, phase: "idle" });
        await get().rollOver(id);
        set({ isGenerating: true, phase: "generating" });
        conv = findConv(id) ?? conv;
      }
    }

    const now = Date.now();
    const userMsg: UiMessage = {
      id: newId(),
      role: "user",
      content: text,
      createdAt: now,
      ...(images.length > 0 ? { images } : {})
    };
    const assistantMsg: UiMessage = {
      id: newId(),
      role: "assistant",
      content: "",
      createdAt: now + 1
    };
    const isFirst = conv.messages.length === 0;
    const title = isFirst && text.trim() ? text.trim().slice(0, 40) : conv.title;
    updateConv(id, (c) => ({
      ...c,
      title,
      updatedAt: now,
      messages: [...c.messages, userMsg, assistantMsg]
    }));
    ensurePersisted(findConv(id) ?? conv);
    if (isFirst && title !== conv.title)
      void enqueue(id, () => storeApi.patchConversation(id, { title }));
    void enqueue(id, async () => {
      await storeApi.appendMessage(id, {
        id: userMsg.id,
        role: "user",
        content: userMsg.content,
        created_at: now
      });
      const ids: string[] = [];
      for (const img of images) {
        try {
          ids.push((await storeApi.addAttachment(userMsg.id, img)).id);
        } catch (err) {
          reportStoreError(err);
        }
      }
      if (ids.length > 0)
        updateConv(id, (c) => ({
          ...c,
          messages: c.messages.map((m) => (m.id === userMsg.id ? { ...m, attachmentIds: ids } : m))
        }));
      await storeApi.appendMessage(id, {
        id: assistantMsg.id,
        role: "assistant",
        content: "",
        created_at: now + 1
      });
    });
    saveLocal();

    // ---- connectors chosen for this chat (docs/CHAT_TOOLS.md); none chosen: nothing below changes the request ----
    abortController = new AbortController();
    const signal = abortController.signal;
    const chatTools = useChatToolsStore.getState();
    const enabledIds = chatTools.enabledFor(id);
    let tools: ChatTool[] = [];
    if (enabledIds.length > 0) {
      chatTools.setStatus(id, "Getting connectors ready");
      try {
        const r = await loadChatTools(enabledIds);
        tools = r.tools;
        if (r.problems.length > 0) toast.show(r.problems.join(". ") + ".", "info");
      } catch (err) {
        toast.show(
          `Could not get the connectors ready: ${err instanceof Error ? err.message : "unknown error"}`,
          "error"
        );
      } finally {
        chatTools.setStatus(id, undefined);
      }
    }
    const toolPrompt = tools.length > 0 ? toolSystemPrompt(tools) : "";

    // ---- build the request from the current window ----
    const latest = findConv(id) ?? conv;
    const windowMsgs = currentWindowMessages(latest.messages, latest.windows).filter(
      (m) => m.id !== assistantMsg.id && (m.role === "user" || m.content.trim() !== "")
    );
    const summary = currentSummary(latest.messages, latest.windows);
    const apiMessages: ChatMessage[] = [];
    const ownPrompt = latest.systemPrompt?.trim() ?? "";
    const systemText = toolPrompt
      ? ownPrompt
        ? `${ownPrompt}\n\n${toolPrompt}`
        : toolPrompt
      : ownPrompt;
    if (systemText) apiMessages.push({ role: "system", content: systemText });
    if (summary) apiMessages.push(summarySystemMessage(summary));
    try {
      for (const m of windowMsgs) {
        if (m.images && m.images.length > 0) {
          const parts: ContentPart[] = [];
          for (const img of m.images)
            parts.push({ type: "image_url", image_url: { url: await toDataUri(img) } });
          if (m.content) parts.push({ type: "text", text: m.content });
          apiMessages.push({ role: m.role, content: parts });
        } else {
          apiMessages.push({ role: m.role, content: m.content });
        }
      }
    } catch {
      toast.show("An attached image could not be read; sending the text only.", "error");
    }

    const capacity = contextCapacity(latest);
    let promptEstimate =
      estimateTokens(latest.systemPrompt ?? "") +
      (toolPrompt ? estimateTokens(toolPrompt) : 0) +
      estimateTokens(summary) +
      estimateMessagesTokens(windowMsgs);
    const wanted =
      latest.maxTokens && latest.maxTokens > 0
        ? latest.maxTokens
        : inferenceDefaults().maxTokens || 4096;
    const flashOn = loaded?.dara.flash?.active === true;
    const effort = useEffortStore.getState().effort;

    let finish: string | undefined;

    /** Streams one model reply into the assistant message `msgId`; returns how it ended. */
    const streamStep = async (msgId: string): Promise<string | undefined> => {
      const maxTokens = capacity
        ? planMaxTokens({ capacity, promptTokens: Math.max(promptEstimate, 0), wanted })
        : wanted;
      const chatReq: ChatRequest = {
        model: targetModel,
        messages: apiMessages,
        stream: true,
        temperature: latest.temperature ?? 0.7,
        max_tokens: maxTokens,
        ...(flashOn ? { flash: latest.flash !== false } : {}),
        ...(effort !== "none" ? { reasoning_effort: effort } : {})
      };

      tokenBuffer = "";
      let stepFinish: string | undefined;
      let raw: UiMessage["raw"];
      let lastCheckpoint = Date.now();
      let reasoningBuffer = "";
      let thinkStart: number | null = null;
      let thinkEnd: number | null = null;

      const patchAssistant = (patch: Partial<UiMessage>) =>
        updateConv(id, (c) => ({
          ...c,
          messages: c.messages.map((m) => (m.id === msgId ? { ...m, ...patch } : m))
        }));
      const flush = () => {
        set({ streamingText: tokenBuffer });
        patchAssistant({
          content: tokenBuffer,
          ...(reasoningBuffer ? { reasoning: reasoningBuffer } : {})
        });
      };
      const scheduleFlush = () => {
        if (rafId === null) {
          rafId = requestAnimationFrame(() => {
            rafId = null;
            flush();
          });
        }
      };

      // what to store about the thinking once the reply ends (the engine's own numbers win over the stopwatch here)
      const thinkingResult = (): Partial<UiMessage> => {
        if (!reasoningBuffer) return {};
        const measured = thinkStart !== null ? (thinkEnd ?? Date.now()) - thinkStart : undefined;
        const ms = raw?.timings?.thinking_ms || measured;
        return {
          reasoning: reasoningBuffer,
          ...(ms ? { reasoningMs: ms } : {}),
          ...(raw?.timings?.thinking_cut ? { reasoningCut: true } : {})
        };
      };

      try {
        const stream = await apiClient.streamChat(chatReq, signal);
        for await (const chunk of stream) {
          const serverError = (chunk as unknown as { error?: { message?: string } }).error;
          if (serverError) throw new Error(serverError.message || "The engine reported an error");

          const choice = chunk.choices?.[0];
          if (choice?.finish_reason) stepFinish = choice.finish_reason;
          const thought = choice?.delta?.reasoning_content;
          if (thought) {
            thinkStart ??= Date.now();
            reasoningBuffer += thought;
            scheduleFlush();
          }
          const delta = choice?.delta?.content;
          if (delta) {
            if (thinkStart !== null) thinkEnd ??= Date.now();
            tokenBuffer += delta;
            scheduleFlush();
            if (Date.now() - lastCheckpoint > CHECKPOINT_MS) {
              lastCheckpoint = Date.now();
              const snapshot = tokenBuffer;
              void enqueue(id, () => storeApi.patchMessage(msgId, { content: snapshot }));
            }
          }

          if (chunk.timings || chunk.usage) {
            raw = {
              ...(chunk.usage ? { usage: chunk.usage } : {}),
              ...(chunk.timings ? { timings: chunk.timings } : {}),
              ...(chunk.context ? { context: chunk.context } : {}),
              ...(chunk.speculative ? { speculative: chunk.speculative } : {})
            } as UiMessage["raw"];
            const tps = chunk.timings?.decode_tokens_per_second;
            if (tps) set({ currentTps: tps });
          }
          if (chunk.context) {
            const ctx: ContextView = {
              capacity: chunk.context.capacity,
              used: chunk.context.used,
              compacted: chunk.context.compacted,
              summarized_messages: chunk.context.summarized_messages,
              dropped_messages: chunk.context.dropped_messages,
              truncated_last_message: chunk.context.truncated_last_message ?? false,
              summary_cached: chunk.context.summary_cached ?? false
            };
            updateConv(id, (c) => ({ ...c, context: ctx }));
          }
        }
        if (rafId !== null) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
        flush();
        patchAssistant({
          finishReason: stepFinish ?? "stop",
          raw,
          stats: statsFromRaw(raw),
          ...thinkingResult()
        });
      } catch (err: unknown) {
        if (rafId !== null) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
        if (signal.aborted) {
          stepFinish = "interrupted";
          patchAssistant({
            content: tokenBuffer,
            ...thinkingResult(),
            interrupted: true,
            finishReason: "interrupted",
            raw,
            stats: statsFromRaw(raw)
          });
        } else {
          stepFinish = "error";
          const e = userFacingError(err);
          patchAssistant({
            content: tokenBuffer,
            ...thinkingResult(),
            error: e.message,
            errorCode: e.code,
            finishReason: "error"
          });
        }
      } finally {
        const final = findConv(id)?.messages.find((m) => m.id === msgId);
        if (final) {
          void enqueue(id, () =>
            storeApi.patchMessage(msgId, {
              content: final.content,
              finish_reason: final.finishReason ?? stepFinish ?? "stop",
              error: final.error ?? null,
              stats: withReasoning(final)
            })
          );
        }
        saveLocal();
      }
      return stepFinish;
    };

    /** Appends the tool results (a user message) and the next empty assistant reply, stored like any other messages. */
    const appendStep = (resultsText: string, nextAssistant: UiMessage | null): void => {
      const t = Date.now();
      const resultsMsg: UiMessage = {
        id: newId(),
        role: "user",
        content: resultsText,
        createdAt: t
      };
      updateConv(id, (c) => ({
        ...c,
        updatedAt: t,
        messages: [...c.messages, resultsMsg, ...(nextAssistant ? [nextAssistant] : [])]
      }));
      void enqueue(id, async () => {
        await storeApi.appendMessage(id, {
          id: resultsMsg.id,
          role: "user",
          content: resultsText,
          created_at: t
        });
        if (nextAssistant)
          await storeApi.appendMessage(id, {
            id: nextAssistant.id,
            role: "assistant",
            content: "",
            created_at: nextAssistant.createdAt ?? t + 1
          });
      });
      saveLocal();
    };

    /** Stores what happened to a step's calls with the assistant message that made them (after its own final patch). */
    const saveToolCalls = (msgId: string, callIds: string[]): void => {
      const live = useChatToolsStore.getState().calls;
      const records: ToolCallRecord[] = [];
      for (const cid of callIds) {
        const c = live[cid];
        if (c) records.push(recordFromCall(c));
      }
      if (records.length === 0) return;
      updateConv(id, (c) => ({
        ...c,
        messages: c.messages.map((m) => (m.id === msgId ? { ...m, toolCalls: records } : m))
      }));
      const msg = findConv(id)?.messages.find((m) => m.id === msgId);
      if (msg) void enqueue(id, () => storeApi.patchMessage(msgId, { stats: withReasoning(msg) }));
      saveLocal();
    };

    /** Adds a short note to the end of an assistant reply (shown and stored). */
    const noteOn = (msgId: string, note: string): void => {
      const cur = findConv(id)?.messages.find((m) => m.id === msgId);
      if (!cur) return;
      const content = `${cur.content.trimEnd()}\n\n${note}`;
      updateConv(id, (c) => ({
        ...c,
        messages: c.messages.map((m) => (m.id === msgId ? { ...m, content } : m))
      }));
      void enqueue(id, () => storeApi.patchMessage(msgId, { content }));
      saveLocal();
    };

    try {
      let stepMsgId = assistantMsg.id;
      let repairs = 0;
      let nudges = 0;
      // the results of the previous step, to tell unfinished work from a finished answer
      let lastItems: ToolResultItem[] | null = null;
      for (let step = 1; ; step++) {
        set({ streamingText: "", currentTps: undefined });
        finish = await streamStep(stepMsgId);
        if (tools.length === 0 || finish === "error" || finish === "interrupted" || signal.aborted)
          break;

        const reply = findConv(id)?.messages.find((m) => m.id === stepMsgId)?.content ?? "";
        const parsed = parseToolCalls(reply, tools, stepMsgId);
        let resultsText: string;
        if (parsed.calls.length > 0) {
          if (step >= MAX_STEPS) {
            noteOn(stepMsgId, `Stopped after ${MAX_STEPS} steps.`);
            break;
          }
          const items = await runCalls(id, parsed.calls, tools, signal);
          lastItems = items;
          saveToolCalls(
            stepMsgId,
            parsed.calls.map((c) => c.id)
          );
          const problemItems =
            parsed.problems.length > 0 ? parseResults(formatProblems(parsed.problems, tools)) : [];
          resultsText = formatResults([...items, ...problemItems]);
        } else if (parsed.problems.length > 0 && repairs < MAX_REPAIRS) {
          if (step >= MAX_STEPS) {
            noteOn(stepMsgId, `Stopped after ${MAX_STEPS} steps.`);
            break;
          }
          repairs++;
          lastItems = null;
          resultsText = formatProblems(parsed.problems, tools);
        } else {
          // A reply with no call is the answer, unless work a tool started is not finished and the model is giving up on it
          // ("I will check back shortly"): send it back to wait and check, a few times at most. Any connector, nothing specific.
          const unfinished = lastItems ? unfinishedWork(lastItems) : null;
          if (
            lastItems === null ||
            nudges >= MAX_NUDGES ||
            step >= MAX_STEPS ||
            !(unfinished || promisesMore(parsed.text))
          )
            break;
          nudges++;
          lastItems = null;
          resultsText =
            formatResults([
              {
                id: `dara:nudge:${nudges}`,
                name: "dara",
                status: "ok",
                text: "The work is not finished yet."
              }
            ]) +
            `
${nudgeText(unfinished)}`;
        }

        if (signal.aborted) {
          // keep the history whole: the calls that were made, and that the rest was stopped
          appendStep(resultsText, null);
          finish = "interrupted";
          break;
        }
        const next: UiMessage = {
          id: newId(),
          role: "assistant",
          content: "",
          createdAt: Date.now() + 1
        };
        appendStep(resultsText, next);
        apiMessages.push({ role: "assistant", content: reply });
        apiMessages.push({ role: "user", content: resultsText });
        promptEstimate += estimateTokens(reply) + estimateTokens(resultsText);
        stepMsgId = next.id;
      }
    } finally {
      abortController = null;
      if (tools.length > 0) useChatToolsStore.getState().setStatus(id, undefined);
      set({ isGenerating: false, phase: "idle", currentTps: undefined });
    }

    // A reply that filled the window: start the next context window now, while the user reads.
    const after = findConv(id);
    const ctx = after?.context;
    if (
      after &&
      ctx &&
      finish !== "error" &&
      rolloverNeeded({ used: ctx.used, capacity: contextCapacity(after) ?? ctx.capacity })
    ) {
      void get().rollOver(id);
    }
  },

  stopGeneration: () => {
    abortController?.abort();
  },

  regenerate: (messageId, modelId) => {
    const conv = get().getActiveConversation();
    if (!conv || conv.messages.length === 0 || get().isGenerating) return;
    let target = conv.messages.length - 1;
    if (messageId) {
      const found = conv.messages.findIndex((m) => m.id === messageId);
      if (found !== -1) target = found;
    }
    let userIdx = -1;
    for (let i = target; i >= 0; i--) {
      const m = conv.messages[i];
      // tool results are user messages too; a regenerate goes back to what the person wrote and drops the tool steps
      if (m?.role === "user" && !isToolResultsContent(m.content)) {
        userIdx = i;
        break;
      }
    }
    const userMsg = conv.messages[userIdx];
    if (!userMsg) return;
    const kept = conv.messages.slice(0, userIdx);
    updateConv(conv.id, (c) => ({
      ...c,
      messages: kept,
      windows: dropStaleWindows(c.windows, kept)
    }));
    if (conv.persisted) void enqueue(conv.id, () => storeApi.deleteMessagesFrom(conv.id, userIdx));
    void get().sendMessage(userMsg.content, userMsg.images ?? [], modelId);
  },

  editAndResend: (messageId, newText, modelId) => {
    const conv = get().getActiveConversation();
    if (!conv || get().isGenerating) return;
    const idx = conv.messages.findIndex((m) => m.id === messageId);
    const original = conv.messages[idx];
    if (idx === -1 || !original || isToolResultsContent(original.content)) return;
    const kept = conv.messages.slice(0, idx);
    updateConv(conv.id, (c) => ({
      ...c,
      messages: kept,
      windows: dropStaleWindows(c.windows, kept)
    }));
    if (conv.persisted) void enqueue(conv.id, () => storeApi.deleteMessagesFrom(conv.id, idx));
    void get().sendMessage(newText, original.images ?? [], modelId);
  },

  continueGeneration: (modelId) => {
    if (get().isGenerating) return;
    void get().sendMessage(CONTINUE_TEXT, [], modelId);
  }
}));

// A chat's connector choice and "always allow" list are stored in its params: write them whenever they change.
useChatToolsStore.subscribe((next, prev) => {
  if (applyingToolParams) return;
  if (next.enabled === prev.enabled && next.alwaysAllowed === prev.alwaysAllowed) return;
  const changed = new Set<string>();
  for (const k of new Set([...Object.keys(next.enabled), ...Object.keys(prev.enabled)]))
    if (next.enabled[k] !== prev.enabled[k]) changed.add(k);
  for (const k of new Set([...Object.keys(next.alwaysAllowed), ...Object.keys(prev.alwaysAllowed)]))
    if (next.alwaysAllowed[k] !== prev.alwaysAllowed[k]) changed.add(k);
  let any = false;
  for (const id of changed) {
    const c = findConv(id);
    if (!c) continue;
    any = true;
    if (c.persisted && useChatStore.getState().storeAvailable)
      void enqueue(id, () =>
        storeApi.patchConversation(id, { params: paramsOf(findConv(id) ?? c) })
      );
  }
  if (any) saveLocal();
});
