import { apiClient } from "../api/client.ts";
import type { ChatRequest, ChatMessage, ContentPart } from "../api/types.ts";

export interface UiMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  images?: string[] | undefined;
  interrupted?: boolean | undefined;
  stats?:
    | {
        decodeTokensPerSecond?: number | undefined;
        ttftMs?: number | undefined;
        cachedTokens?: number | undefined;
        totalTokens?: number | undefined;
        speculative?:
          | {
              mode: string;
              acceptance: number;
            }
          | undefined;
      }
    | undefined;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: UiMessage[];
  systemPrompt?: string | undefined;
  temperature?: number | undefined;
  maxTokens?: number | undefined;
  flash?: boolean | undefined;
  context?:
    | {
        capacity: number;
        used: number;
        compacted: boolean;
        summarizedMessages: number;
        droppedMessages: number;
      }
    | undefined;
}

const STORAGE_KEY = "dara-conversations";

class ChatStore {
  conversations = $state<Conversation[]>([]);
  activeConversationId = $state<string | null>(null);
  isGenerating = $state<boolean>(false);
  streamingText = $state<string>("");
  currentTps = $state<number | undefined>(undefined);

  private abortController: AbortController | null = null;
  private tokenBuffer = "";
  private rafId: number | null = null;

  get activeConversation(): Conversation | null {
    if (!this.activeConversationId) return null;
    return this.conversations.find((c) => c.id === this.activeConversationId) ?? null;
  }

  constructor() {
    this.loadFromStorage();
  }

  private loadFromStorage(): void {
    if (typeof window === "undefined") return;
    try {
      const data = localStorage.getItem(STORAGE_KEY);
      if (data) {
        const parsed = JSON.parse(data) as Conversation[];
        if (Array.isArray(parsed)) {
          this.conversations = parsed;
          if (parsed[0]) {
            this.activeConversationId = parsed[0].id;
          }
        }
      }
    } catch {
      // ignore
    }
  }

  private saveToStorage(): void {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.conversations));
    } catch {
      // ignore
    }
  }

  createConversation(title = "New Chat"): string {
    const id = `conv-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
    const newConv: Conversation = {
      id,
      title,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
      temperature: 0.7,
      maxTokens: 2048,
      flash: true
    };
    this.conversations = [newConv, ...this.conversations];
    this.activeConversationId = id;
    this.saveToStorage();
    return id;
  }

  selectConversation(id: string): void {
    this.activeConversationId = id;
  }

  deleteConversation(id: string): void {
    this.conversations = this.conversations.filter((c) => c.id !== id);
    if (this.activeConversationId === id) {
      this.activeConversationId = this.conversations[0]?.id ?? null;
    }
    this.saveToStorage();
  }

  clearAllConversations(): void {
    this.conversations = [];
    this.activeConversationId = null;
    this.saveToStorage();
  }

  renameConversation(id: string, newTitle: string): void {
    const conv = this.conversations.find((c) => c.id === id);
    if (conv) {
      conv.title = newTitle;
      conv.updatedAt = Date.now();
      this.conversations = [...this.conversations];
      this.saveToStorage();
    }
  }

  async sendMessage(text: string, images: string[] = [], modelId?: string): Promise<void> {
    if (!text.trim() && images.length === 0) return;
    if (this.isGenerating) return;

    // Ensure active conversation exists
    let convId = this.activeConversationId;
    if (!convId || !this.conversations.find((c) => c.id === convId)) {
      // Use first user message as title truncated to 30 chars
      const title = text.slice(0, 30).trim() || "New Chat";
      convId = this.createConversation(title);
    }

    const currentConv = this.conversations.find((c) => c.id === convId);
    if (!currentConv) return;

    // If first message in thread, auto-title
    if (currentConv.messages.length === 0 && text.trim()) {
      currentConv.title = text.slice(0, 30).trim();
    }

    // Add User Message
    const userMsg: UiMessage = {
      id: `msg-${Date.now()}-user`,
      role: "user",
      content: text,
      images: images.length > 0 ? images : undefined
    };

    currentConv.messages = [...currentConv.messages, userMsg];
    currentConv.updatedAt = Date.now();
    this.conversations = [...this.conversations];
    this.saveToStorage();

    // Prepare Assistant Message
    const assistantMsgId = `msg-${Date.now()}-assistant`;
    const assistantMsg: UiMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: ""
    };

    currentConv.messages = [...currentConv.messages, assistantMsg];
    this.conversations = [...this.conversations];

    // Build API ChatMessages history
    const apiMessages: ChatMessage[] = [];
    if (currentConv.systemPrompt?.trim()) {
      apiMessages.push({ role: "system", content: currentConv.systemPrompt.trim() });
    }

    for (const msg of currentConv.messages) {
      if (msg.id === assistantMsgId) continue; // Skip unfinished assistant response
      if (msg.role === "assistant" && msg.interrupted) continue; // Skip interrupted

      if (msg.images && msg.images.length > 0) {
        const parts: ContentPart[] = [];
        for (const img of msg.images) {
          parts.push({ type: "image_url", image_url: { url: img } });
        }
        if (msg.content) {
          parts.push({ type: "text", text: msg.content });
        }
        apiMessages.push({ role: msg.role, content: parts });
      } else {
        apiMessages.push({ role: msg.role, content: msg.content });
      }
    }

    const chatReq: ChatRequest = {
      messages: apiMessages,
      stream: true,
      temperature: currentConv.temperature ?? 0.7,
      max_tokens: currentConv.maxTokens ?? 2048,
      flash: currentConv.flash ?? true
    };
    if (modelId) {
      chatReq.model = modelId;
    }

    this.isGenerating = true;
    this.streamingText = "";
    this.tokenBuffer = "";
    this.currentTps = undefined;
    this.abortController = new AbortController();

    try {
      const stream = await apiClient.streamChat(chatReq, this.abortController.signal);

      for await (const chunk of stream) {
        const delta = chunk.choices?.[0]?.delta?.content;
        if (delta) {
          this.tokenBuffer += delta;
          this.scheduleBufferFlush(convId, assistantMsgId);
        }

        // Final chunk telemetry
        if (chunk.timings) {
          if (chunk.timings.decode_tokens_per_second) {
            this.currentTps = chunk.timings.decode_tokens_per_second;
          }
          assistantMsg.stats = {
            decodeTokensPerSecond: chunk.timings.decode_tokens_per_second,
            ttftMs: chunk.timings.ttft_ms,
            cachedTokens: chunk.usage?.prompt_tokens_details?.cached_tokens,
            totalTokens: chunk.usage?.total_tokens,
            speculative: chunk.speculative
              ? {
                  mode: chunk.speculative.mode,
                  acceptance: chunk.speculative.acceptance
                }
              : undefined
          };
        }

        if (chunk.context) {
          currentConv.context = {
            capacity: chunk.context.capacity,
            used: chunk.context.used,
            compacted: chunk.context.compacted,
            summarizedMessages: chunk.context.summarized_messages,
            droppedMessages: chunk.context.dropped_messages
          };
        }
      }

      this.flushBufferImmediate(convId, assistantMsgId);
    } catch (err: unknown) {
      if (this.abortController?.signal.aborted) {
        assistantMsg.interrupted = true;
      } else {
        assistantMsg.content += `\n\n[Error: ${err instanceof Error ? err.message : "Connection failed"}]`;
      }
      this.flushBufferImmediate(convId, assistantMsgId);
    } finally {
      this.isGenerating = false;
      this.abortController = null;
      this.currentTps = undefined;
      this.conversations = [...this.conversations];
      this.saveToStorage();
    }
  }

  private scheduleBufferFlush(convId: string, msgId: string): void {
    if (this.rafId !== null) return;
    this.rafId = requestAnimationFrame(() => {
      this.rafId = null;
      this.flushBufferImmediate(convId, msgId);
    });
  }

  private flushBufferImmediate(convId: string, msgId: string): void {
    this.streamingText = this.tokenBuffer;
    const conv = this.conversations.find((c) => c.id === convId);
    if (conv) {
      const msg = conv.messages.find((m) => m.id === msgId);
      if (msg) {
        msg.content = this.tokenBuffer;
      }
    }
  }

  stopGeneration(): void {
    if (this.abortController) {
      this.abortController.abort();
    }
  }

  regenerate(modelId?: string): void {
    const conv = this.activeConversation;
    if (!conv || conv.messages.length === 0 || this.isGenerating) return;

    // Remove last assistant message
    const last = conv.messages[conv.messages.length - 1];
    if (last && last.role === "assistant") {
      conv.messages.pop();
    }

    // Find previous user message
    const prevUser = [...conv.messages].reverse().find((m) => m.role === "user");
    if (!prevUser) return;

    // Resend
    const text = prevUser.content;
    const images = prevUser.images ?? [];
    conv.messages.pop(); // Remove user message to re-trigger
    void this.sendMessage(text, images, modelId);
  }

  editAndResend(messageId: string, newText: string, modelId?: string): void {
    const conv = this.activeConversation;
    if (!conv || this.isGenerating) return;

    const idx = conv.messages.findIndex((m) => m.id === messageId);
    if (idx === -1) return;

    // Truncate messages after this index
    conv.messages = conv.messages.slice(0, idx);
    void this.sendMessage(newText, [], modelId);
  }
}

export const chatStore = new ChatStore();
