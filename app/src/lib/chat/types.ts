import type { ChatStats, ContextInfo, SpeculativeInfo } from "../api/types.ts";

/** What the UI shows about one finished reply. Every number comes from the server's final chunk. */
export interface MessageStats {
  decodeTokensPerSecond?: number | undefined;
  prefillTokensPerSecond?: number | undefined;
  ttftMs?: number | undefined;
  prefillMs?: number | undefined;
  decodeMs?: number | undefined;
  totalMs?: number | undefined;
  decodeTokens?: number | undefined;
  prefillTokens?: number | undefined;
  totalTokens?: number | undefined;
  cachedTokens?: number | undefined;
  speculative?: SpeculativeInfo | undefined;
}

export type FinishReason = "stop" | "length" | "interrupted" | "error" | (string & {});

export interface UiMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  /** URLs to display (data: for just-attached images, blob: for images loaded from the database). */
  images?: string[] | undefined;
  /** Attachment ids in the database, parallel to `images` once they are stored. */
  attachmentIds?: string[] | undefined;
  interrupted?: boolean | undefined;
  finishReason?: FinishReason | undefined;
  error?: string | undefined;
  errorCode?: string | undefined;
  stats?: MessageStats | undefined;
  /** The server's numbers exactly as received; stored in the database. */
  raw?: Partial<ChatStats> | undefined;
  createdAt?: number | undefined;
}

/**
 * A context window: when the conversation fills the model's context, the earlier messages stay stored (and visible), a summary
 * of them is written, and the model continues from `startMessageId` with that summary in front.
 */
export interface ContextWindow {
  startMessageId: string;
  summary: string;
  createdAt: number;
  /** How many messages the summary replaces (for the divider text). */
  summarizedMessages: number;
}

export interface ContextView extends ContextInfo {
  /** true when `used` is a local estimate (right after a rollover, before the next reply measures it). */
  estimated?: boolean | undefined;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: UiMessage[];
  temperature?: number | undefined;
  /** 0 = use the default from Settings */
  maxTokens?: number | undefined;
  /** Per-request Flash switch (only has an effect while the loaded model has its drafter). */
  flash?: boolean | undefined;
  systemPrompt?: string | undefined;
  context?: ContextView | undefined;
  windows: ContextWindow[];
  /** The conversation exists in the database. New empty chats are kept local until the first message. */
  persisted: boolean;
  /** Messages have been fetched from the database. */
  loaded: boolean;
}
