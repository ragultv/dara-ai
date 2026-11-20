// Pure helpers for the context budget and for context windows (rollover). No network, no store: unit-tested in tests/unit/chat-context.test.ts.
import type { ChatMessage } from "../api/types.ts";
import type { ContextWindow, MessageStats, UiMessage } from "./types.ts";
import type { ChatStats } from "../api/types.ts";

/** Roll over when the held tokens plus what is about to be added reach this share of the loaded context. */
export const ROLLOVER_FRACTION = 0.8;
/** Tokens always left free: chat template, the server's speculation snapshot, rounding. */
export const SAFETY_TOKENS = 384;
/** Budgeting only: the real count comes from the server's `context.used`. */
export const CHARS_PER_TOKEN = 3.5;
export const IMAGE_TOKEN_ESTIMATE = 420;
/** The summary request asks for at most this many tokens. */
export const SUMMARY_MAX_TOKENS = 450;

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

export function estimateMessageTokens(m: Pick<UiMessage, "content" | "images">): number {
  return estimateTokens(m.content) + (m.images?.length ?? 0) * IMAGE_TOKEN_ESTIMATE + 8;
}

export function estimateMessagesTokens(
  msgs: ReadonlyArray<Pick<UiMessage, "content" | "images">>
): number {
  let n = 0;
  for (const m of msgs) n += estimateMessageTokens(m);
  return n;
}

/** Index of the first message of the current window (0 when there is none, or its start message no longer exists). */
export function windowStartIndex(
  messages: ReadonlyArray<Pick<UiMessage, "id">>,
  windows: ReadonlyArray<ContextWindow>
): number {
  for (let w = windows.length - 1; w >= 0; w--) {
    const win = windows[w];
    if (!win) continue;
    const idx = messages.findIndex((m) => m.id === win.startMessageId);
    if (idx >= 0) return idx;
  }
  return 0;
}

export function currentWindowMessages<T extends Pick<UiMessage, "id">>(
  messages: ReadonlyArray<T>,
  windows: ReadonlyArray<ContextWindow>
): T[] {
  return messages.slice(windowStartIndex(messages, windows));
}

/** The summary that goes in front of the current window, or "" when there is none. */
export function currentSummary(
  messages: ReadonlyArray<Pick<UiMessage, "id">>,
  windows: ReadonlyArray<ContextWindow>
): string {
  for (let w = windows.length - 1; w >= 0; w--) {
    const win = windows[w];
    if (win && messages.some((m) => m.id === win.startMessageId)) return win.summary;
  }
  return "";
}

/** Windows whose first message was removed (regenerate / edit and resend) are dropped. */
export function dropStaleWindows(
  windows: ReadonlyArray<ContextWindow>,
  messages: ReadonlyArray<Pick<UiMessage, "id">>
): ContextWindow[] {
  const ids = new Set(messages.map((m) => m.id));
  return windows.filter((w) => ids.has(w.startMessageId));
}

/**
 * Largest reply that still fits: the room left after the prompt (and a safety margin), never more than the user's setting.
 * Asking for more than the room makes the server shrink the prompt (summarize or drop history) to make space.
 */
export function planMaxTokens(args: {
  capacity: number;
  promptTokens: number;
  wanted: number;
}): number {
  const room = args.capacity - args.promptTokens - SAFETY_TOKENS;
  return Math.max(64, Math.min(args.wanted, room));
}

/** True when the next turn would not fit comfortably: time to start a new context window. */
export function rolloverNeeded(args: {
  used: number;
  capacity: number;
  pendingTokens?: number;
}): boolean {
  if (args.capacity <= 0) return false;
  return args.used + (args.pendingTokens ?? 0) >= args.capacity * ROLLOVER_FRACTION;
}

/**
 * Where the new window starts: the user message of the last exchange, so the model keeps the latest question and answer verbatim.
 * Returns the index into `messages`, or null when there is nothing worth summarizing (fewer than two exchanges in the window).
 */
export function pickRolloverStart(
  messages: ReadonlyArray<Pick<UiMessage, "role" | "content">>,
  windowStart: number
): number | null {
  let lastUser = -1;
  for (let i = messages.length - 1; i >= windowStart; i--) {
    if (messages[i]?.role === "user") {
      lastUser = i;
      break;
    }
  }
  // need something before the last exchange inside this window
  if (lastUser <= windowStart) return null;
  const before = messages.slice(windowStart, lastUser).filter((m) => m.content.trim().length > 0);
  return before.length >= 2 ? lastUser : null;
}

function transcriptOf(msgs: ReadonlyArray<Pick<UiMessage, "role" | "content" | "images">>): string {
  const lines: string[] = [];
  for (const m of msgs) {
    if (m.role === "system") continue;
    const who = m.role === "user" ? "User" : "Assistant";
    const imgs =
      m.images && m.images.length > 0
        ? ` [${m.images.length} image${m.images.length > 1 ? "s" : ""} attached]`
        : "";
    const text = m.content.trim();
    if (!text && !imgs) continue;
    lines.push(`${who}:${imgs} ${text}`.trim());
  }
  return lines.join("\n\n");
}

/** Keeps the most recent part when the transcript is longer than what the model can read in one go. */
export function clipTranscript(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  return "[...earlier part omitted...]\n" + text.slice(text.length - maxChars);
}

/** The request that writes the summary. `capacity` limits how much transcript is sent. */
export function buildSummaryMessages(args: {
  previousSummary: string;
  messages: ReadonlyArray<Pick<UiMessage, "role" | "content" | "images">>;
  capacity: number;
}): ChatMessage[] {
  const maxChars = Math.max(
    2000,
    Math.floor((args.capacity - SUMMARY_MAX_TOKENS - 600) * CHARS_PER_TOKEN * 0.9)
  );
  const parts: string[] = [];
  if (args.previousSummary.trim())
    parts.push("Summary of the conversation before this part:\n" + args.previousSummary.trim());
  parts.push(transcriptOf(args.messages));
  const transcript = clipTranscript(parts.join("\n\n---\n\n"), maxChars);
  return [
    {
      role: "system",
      content:
        "You write compact, faithful summaries of conversations so that an assistant can continue them later. Never add new information."
    },
    {
      role: "user",
      content:
        "Summarize the conversation below in at most 250 words. Keep the user's goals, names, numbers, decisions, constraints, code identifiers " +
        "and anything still unresolved. Write plain text, no preamble.\n\n" +
        transcript
    }
  ];
}

/** The system message that carries an earlier window's summary into the next window. */
export function summarySystemMessage(summary: string): ChatMessage {
  return {
    role: "system",
    content:
      "Summary of the earlier part of this conversation (older messages were moved out of the context window to make room):\n" +
      summary
  };
}

/** Maps the server's final-chunk numbers to what the message footer shows. Only what the server sent; nothing is invented. */
export function statsFromRaw(raw: Partial<ChatStats> | null | undefined): MessageStats | undefined {
  if (!raw) return undefined;
  const t = raw.timings;
  const u = raw.usage;
  if (!t && !u) return undefined;
  const out: MessageStats = {
    decodeTokensPerSecond: t?.decode_tokens_per_second,
    prefillTokensPerSecond: t?.prefill_tokens_per_second,
    ttftMs: t?.ttft_ms,
    prefillMs: t?.prefill_ms,
    decodeMs: t?.decode_ms,
    totalMs: t?.total_ms,
    decodeTokens: t?.decode_tokens ?? u?.completion_tokens,
    prefillTokens: t?.prefill_tokens ?? u?.prompt_tokens,
    totalTokens: u?.total_tokens,
    cachedTokens: u?.prompt_tokens_details?.cached_tokens,
    speculative: raw.speculative
  };
  return out;
}

/**
 * Token counts for tight spaces: 131072 -> "131k", 8192 -> "8.2k", 900 -> "900". Always thousands of 1000, so it reads the same as the
 * exact number shown in the context popup (131,072). The popup also says "128K" is the same size when the number is a power of two.
 */
export function formatTokensShort(n: number): string {
  if (n < 1000) return String(n);
  const k = n / 1000;
  return `${k >= 100 ? Math.round(k) : Math.round(k * 10) / 10}k`;
}
