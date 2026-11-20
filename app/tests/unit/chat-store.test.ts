import { describe, it, expect, beforeEach, vi } from "vitest";

const h = vi.hoisted(() => ({ calls: [] as Array<{ name: string; args: unknown[] }> }));

vi.mock("../../src/lib/api/store-api.ts", () => {
  const rec = (name: string, ret: unknown = {}) =>
    vi.fn(async (...args: unknown[]) => {
      h.calls.push({ name, args });
      return ret;
    });
  let n = 0;
  return {
    newId: () => `00000000-0000-4000-8000-${String(++n).padStart(12, "0")}`,
    isUuid: () => true,
    storeApi: {
      createConversation: rec("createConversation"),
      patchConversation: rec("patchConversation"),
      deleteConversation: rec("deleteConversation"),
      appendMessage: rec("appendMessage"),
      patchMessage: rec("patchMessage"),
      deleteMessagesFrom: rec("deleteMessagesFrom"),
      addAttachment: rec("addAttachment", { id: "att-1" }),
      listConversations: rec("listConversations", { data: [], total: 0 }),
      getConversation: rec("getConversation"),
      getAttachmentBlob: rec("getAttachmentBlob")
    }
  };
});

import { apiClient } from "../../src/lib/api/client.ts";
import { ApiError } from "../../src/lib/api/errors.ts";
import { useChatStore } from "../../src/lib/stores/useChatStore.ts";
import { useModelsStore } from "../../src/lib/stores/useModelsStore.ts";
import type { ChatChunk, ChatRequest, ModelInfo } from "../../src/lib/api/types.ts";
import type { Conversation, UiMessage } from "../../src/lib/chat/types.ts";

const model = (over: { flashActive?: boolean; context?: number } = {}): ModelInfo =>
  ({
    id: "dara-nano",
    object: "model",
    owned_by: "dara",
    created: 1,
    dara: {
      name: "Dara Nano",
      base_model: "x",
      description: "",
      family: "gemma4",
      quant: "Q4_K_M",
      status: "installed",
      loaded: true,
      fits_gpu: "full",
      min_vram_mb: 3000,
      full_vram_mb: 4096,
      context: 131072,
      download_size_mb: 1,
      download: null,
      files: [],
      flash: { available: true, installed: true, active: over.flashActive ?? true },
      vision: { available: true, installed: true, active: true },
      config: {
        kv: "q8",
        flash: true,
        context: over.context ?? 4096,
        layers: 35,
        layers_on_gpu: 35,
        head: "gpu",
        vision: true,
        source: "recommended",
        reason: ""
      }
    }
  }) as ModelInfo;

const ctxInfo = (used: number, capacity = 4096) => ({
  capacity,
  used,
  compacted: false,
  summarized_messages: 0,
  dropped_messages: 0,
  truncated_last_message: false,
  summary_cached: false
});

function chunks(text: string, finish: string, used: number): ChatChunk[] {
  const base = {
    id: "c",
    object: "chat.completion.chunk",
    created: 1,
    model: "dara-nano"
  } as const;
  return [
    {
      ...base,
      choices: [{ index: 0, finish_reason: null, delta: { role: "assistant", content: "" } }]
    },
    { ...base, choices: [{ index: 0, finish_reason: null, delta: { content: text } }] },
    {
      ...base,
      choices: [{ index: 0, finish_reason: finish, delta: {} }],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
        prompt_tokens_details: { cached_tokens: 3, image_tokens: 0 }
      },
      timings: { decode_tokens_per_second: 50, ttft_ms: 40, decode_tokens: 5 } as never,
      context: ctxInfo(used)
    }
  ];
}

let sent: ChatRequest[] = [];
let next: ChatChunk[] = [];

async function* gen(list: ChatChunk[]) {
  for (const c of list) yield c;
}

function seedConversation(messages: UiMessage[], over: Partial<Conversation> = {}): string {
  const conv: Conversation = {
    id: "11111111-1111-4111-8111-111111111111",
    title: "Seed",
    createdAt: 1,
    updatedAt: 1,
    messages,
    temperature: 0.7,
    maxTokens: 0,
    flash: true,
    systemPrompt: "",
    windows: [],
    persisted: true,
    loaded: true,
    ...over
  };
  useChatStore.setState({
    conversations: [conv],
    activeConversationId: conv.id,
    isGenerating: false,
    phase: "idle"
  });
  return conv.id;
}

const m = (id: string, role: "user" | "assistant", content: string): UiMessage => ({
  id,
  role,
  content
});

beforeEach(() => {
  localStorage.clear();
  h.calls.length = 0;
  sent = [];
  next = [];
  useChatStore.setState({
    conversations: [],
    activeConversationId: null,
    isGenerating: false,
    phase: "idle",
    storeAvailable: true
  });
  useModelsStore.setState({ models: [model()] });
  vi.restoreAllMocks();
  vi.spyOn(apiClient, "streamChat").mockImplementation(async (req: ChatRequest) => {
    sent.push(req);
    return gen(next) as never;
  });
});

describe("ChatStore basics", () => {
  it("creates and selects a new conversation, and does not pile up empty chats", () => {
    const id = useChatStore.getState().createConversation("Test Conversation");
    expect(useChatStore.getState().getActiveConversation()?.title).toBe("Test Conversation");
    expect(useChatStore.getState().activeConversationId).toBe(id);
    // "New Chat" while the open chat is still empty reuses it instead of adding another empty one
    expect(useChatStore.getState().createConversation()).toBe(id);
    expect(useChatStore.getState().conversations.length).toBe(1);
    useChatStore.setState((st) => ({
      conversations: st.conversations.map((c) => ({
        ...c,
        messages: [{ id: "x", role: "user" as const, content: "hi" }]
      }))
    }));
    const second = useChatStore.getState().createConversation();
    expect(second).not.toBe(id);
    expect(useChatStore.getState().conversations.length).toBe(2);
  });

  it("deletes a conversation and moves the selection; the last one is replaced by a blank chat", async () => {
    const id1 = useChatStore.getState().createConversation("First");
    const id2 = useChatStore.getState().createConversation("Second");
    await useChatStore.getState().deleteConversation(id2);
    expect(useChatStore.getState().activeConversationId).toBe(id1);
    await useChatStore.getState().deleteConversation(id1);
    expect(useChatStore.getState().conversations.length).toBe(1);
    expect(useChatStore.getState().getActiveConversation()?.messages.length).toBe(0);
  });
});

describe("sending a message", () => {
  it("saves the user message before the reply, keeps the finish reason and the engine's numbers", async () => {
    next = chunks("Hello", "length", 100);
    useChatStore.getState().createConversation();
    await useChatStore.getState().sendMessage("Hi there");
    await vi.waitFor(() => expect(h.calls.some((c) => c.name === "patchMessage")).toBe(true));

    const names = h.calls.map((c) => c.name);
    expect(names[0]).toBe("createConversation");
    const roles = h.calls
      .filter((c) => c.name === "appendMessage")
      .map((c) => (c.args[1] as { role: string }).role);
    expect(roles).toEqual(["user", "assistant"]);
    const final = h.calls.filter((c) => c.name === "patchMessage").pop()?.args[1] as {
      content: string;
      finish_reason: string;
      stats: { timings: { decode_tokens_per_second: number } };
    };
    expect(final.content).toBe("Hello");
    expect(final.finish_reason).toBe("length");
    expect(final.stats.timings.decode_tokens_per_second).toBe(50);

    const conv = useChatStore.getState().getActiveConversation();
    const reply = conv?.messages[1];
    expect(reply?.finishReason).toBe("length");
    expect(reply?.stats?.decodeTokensPerSecond).toBe(50);
    expect(reply?.stats?.cachedTokens).toBe(3);
    expect(conv?.context?.used).toBe(100);
    expect(conv?.title).toBe("Hi there");
  });

  it("asks for a reply that fits the loaded context and only sends flash when the drafter is loaded", async () => {
    next = chunks("ok", "stop", 50);
    useChatStore.getState().createConversation();
    await useChatStore.getState().sendMessage("Hi");
    expect(sent[0]?.max_tokens).toBeLessThanOrEqual(4096 - 384);
    expect(sent[0]?.max_tokens).toBeGreaterThanOrEqual(64);
    expect(sent[0]?.flash).toBe(true);

    useModelsStore.setState({ models: [model({ flashActive: false })] });
    useChatStore.getState().createConversation("again");
    await useChatStore.getState().sendMessage("Hi again");
    expect(sent[1] && "flash" in sent[1]).toBe(false);
  });

  it("shows an engine error on the reply and marks a missing image encoder so the UI can fix it", async () => {
    vi.spyOn(apiClient, "streamChat").mockRejectedValue(
      new ApiError(
        400,
        "this server has no image encoder (load the model with vision on)",
        "invalid_request_error",
        "vision_unavailable"
      )
    );
    useChatStore.getState().createConversation();
    await useChatStore.getState().sendMessage("what is this?", ["data:image/png;base64,AAAA"]);
    const reply = useChatStore.getState().getActiveConversation()?.messages[1];
    expect(reply?.errorCode).toBe("vision_unavailable");
    expect(reply?.finishReason).toBe("error");
    expect(useChatStore.getState().isGenerating).toBe(false);
  });

  it("uploads attached images as attachments of the user message", async () => {
    next = chunks("a cat", "stop", 600);
    useChatStore.getState().createConversation();
    await useChatStore.getState().sendMessage("describe", ["data:image/png;base64,AAAA"]);
    await vi.waitFor(() => expect(h.calls.some((c) => c.name === "addAttachment")).toBe(true));
    const att = h.calls.find((c) => c.name === "addAttachment");
    const userAppend = h.calls.find(
      (c) => c.name === "appendMessage" && (c.args[1] as { role: string }).role === "user"
    );
    expect(att?.args[0]).toBe((userAppend?.args[1] as { id: string }).id);
    const parts = sent[0]?.messages.find((x) => x.role === "user")?.content;
    expect(Array.isArray(parts) && parts.some((p) => p.type === "image_url")).toBe(true);
  });
});

describe("context windows", () => {
  const six = () => [
    m("a1", "user", "first question ".repeat(40)),
    m("a2", "assistant", "first answer ".repeat(40)),
    m("a3", "user", "second question ".repeat(40)),
    m("a4", "assistant", "second answer ".repeat(40)),
    m("a5", "user", "third question"),
    m("a6", "assistant", "third answer")
  ];

  it("starts a new window before a turn that would not fit, keeps the old messages, and sends the summary instead of them", async () => {
    const id = seedConversation(six(), { context: ctxInfo(3500) });
    vi.spyOn(apiClient, "chat").mockResolvedValue({
      choices: [
        { index: 0, finish_reason: "stop", message: { role: "assistant", content: "SUMMARY TEXT" } }
      ],
      context: ctxInfo(900)
    } as never);
    next = chunks("answer", "stop", 300);

    await useChatStore.getState().sendMessage("and now?");

    const conv = useChatStore.getState().conversations.find((c) => c.id === id);
    expect(conv?.windows.length).toBe(1);
    expect(conv?.windows[0]?.startMessageId).toBe("a5"); // the last exchange stays verbatim
    expect(conv?.windows[0]?.summary).toBe("SUMMARY TEXT");
    expect(conv?.messages.length).toBe(8); // nothing was deleted
    const req =
      sent[0]?.messages.map((x) => (typeof x.content === "string" ? x.content : "")) ?? [];
    expect(req.some((c) => c.includes("SUMMARY TEXT"))).toBe(true);
    expect(req.some((c) => c.includes("first question"))).toBe(false);
    expect(req.some((c) => c.includes("third question"))).toBe(true);
    await vi.waitFor(() =>
      expect(
        h.calls.some(
          (c) =>
            c.name === "patchConversation" && JSON.stringify(c.args[1]).includes("SUMMARY TEXT")
        )
      ).toBe(true)
    );
  });

  it("does not start a window while there is room", async () => {
    seedConversation(six(), { context: ctxInfo(800) });
    const chat = vi.spyOn(apiClient, "chat");
    next = chunks("fine", "stop", 900);
    await useChatStore.getState().sendMessage("hello again");
    expect(chat).not.toHaveBeenCalled();
    expect(useChatStore.getState().getActiveConversation()?.windows.length).toBe(0);
  });

  it("starts the next window right after a reply that filled the context", async () => {
    const id = seedConversation(six(), { context: ctxInfo(1000) });
    const chat = vi.spyOn(apiClient, "chat").mockResolvedValue({
      choices: [
        {
          index: 0,
          finish_reason: "stop",
          message: { role: "assistant", content: "LATER SUMMARY" }
        }
      ]
    } as never);
    next = chunks("long reply", "stop", 3600); // 3600 of 4096 held after this reply
    await useChatStore.getState().sendMessage("one more");
    await vi.waitFor(() =>
      expect(useChatStore.getState().conversations.find((c) => c.id === id)?.windows.length).toBe(1)
    );
    expect(chat).toHaveBeenCalledTimes(1);
  });

  it("drops a window when regenerate removes its first message", async () => {
    const msgs = six();
    const id = seedConversation(msgs, {
      windows: [{ startMessageId: "a5", summary: "S", createdAt: 1, summarizedMessages: 4 }]
    });
    next = chunks("again", "stop", 100);
    useChatStore.getState().regenerate("a5");
    const conv = useChatStore.getState().conversations.find((c) => c.id === id);
    expect(conv?.windows.length).toBe(0);
    return vi.waitFor(() =>
      expect(h.calls.find((c) => c.name === "deleteMessagesFrom")?.args).toEqual([id, 4])
    );
  });
});
