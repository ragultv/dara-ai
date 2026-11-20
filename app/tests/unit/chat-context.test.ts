import { describe, it, expect } from "vitest";
import {
  ROLLOVER_FRACTION,
  SAFETY_TOKENS,
  buildSummaryMessages,
  clipTranscript,
  currentSummary,
  currentWindowMessages,
  dropStaleWindows,
  estimateTokens,
  pickRolloverStart,
  planMaxTokens,
  rolloverNeeded,
  statsFromRaw,
  windowStartIndex
} from "../../src/lib/chat/context.ts";
import type { ContextWindow } from "../../src/lib/chat/types.ts";

const msg = (id: string, role: "user" | "assistant", content = "x") => ({ id, role, content });
const win = (startMessageId: string, summary = "S"): ContextWindow => ({
  startMessageId,
  summary,
  createdAt: 1,
  summarizedMessages: 2
});

describe("context budget", () => {
  it("estimates tokens from characters (budgeting only)", () => {
    expect(estimateTokens("")).toBe(0);
    expect(estimateTokens("a".repeat(35))).toBe(10);
  });

  it("never plans a reply larger than the room left after the prompt", () => {
    expect(planMaxTokens({ capacity: 17408, promptTokens: 1000, wanted: 4096 })).toBe(4096);
    expect(planMaxTokens({ capacity: 4096, promptTokens: 3000, wanted: 4096 })).toBe(
      4096 - 3000 - SAFETY_TOKENS
    );
    expect(planMaxTokens({ capacity: 2048, promptTokens: 2000, wanted: 4096 })).toBe(64); // floor
  });

  it("rolls over at the configured share of the loaded context", () => {
    const cap = 10000;
    expect(rolloverNeeded({ used: cap * ROLLOVER_FRACTION - 1, capacity: cap })).toBe(false);
    expect(rolloverNeeded({ used: cap * ROLLOVER_FRACTION, capacity: cap })).toBe(true);
    expect(rolloverNeeded({ used: 5000, capacity: cap, pendingTokens: 3500 })).toBe(true);
    expect(rolloverNeeded({ used: 9999, capacity: 0 })).toBe(false);
  });
});

describe("context windows", () => {
  const msgs = [
    msg("a", "user"),
    msg("b", "assistant"),
    msg("c", "user"),
    msg("d", "assistant"),
    msg("e", "user"),
    msg("f", "assistant")
  ];

  it("uses the whole conversation until a window exists", () => {
    expect(windowStartIndex(msgs, [])).toBe(0);
    expect(currentWindowMessages(msgs, []).length).toBe(6);
    expect(currentSummary(msgs, [])).toBe("");
  });

  it("starts at the latest window's first message and carries its summary", () => {
    const windows = [win("c", "first"), win("e", "second")];
    expect(windowStartIndex(msgs, windows)).toBe(4);
    expect(currentWindowMessages(msgs, windows).map((m) => m.id)).toEqual(["e", "f"]);
    expect(currentSummary(msgs, windows)).toBe("second");
  });

  it("falls back to an earlier window when the latest start message was removed", () => {
    const windows = [win("c", "first"), win("zzz", "gone")];
    expect(windowStartIndex(msgs, windows)).toBe(2);
    expect(currentSummary(msgs, windows)).toBe("first");
  });

  it("drops windows whose start message no longer exists (regenerate / edit)", () => {
    const kept = msgs.slice(0, 3); // a b c
    expect(dropStaleWindows([win("c"), win("e")], kept).map((w) => w.startMessageId)).toEqual([
      "c"
    ]);
  });

  it("keeps the last exchange verbatim and needs two earlier messages to summarize", () => {
    expect(pickRolloverStart(msgs, 0)).toBe(4); // new window starts at the last user message
    expect(pickRolloverStart(msgs.slice(0, 2), 0)).toBeNull();
    expect(pickRolloverStart(msgs.slice(0, 4), 0)).toBe(2);
    expect(pickRolloverStart(msgs, 4)).toBeNull(); // nothing before the last exchange inside this window
  });
});

describe("summary request", () => {
  it("includes the previous summary and the transcript, with images noted but not sent", () => {
    const m = buildSummaryMessages({
      previousSummary: "User is building a game.",
      messages: [
        { role: "user", content: "Add jumping", images: ["data:image/png;base64,AAAA"] },
        { role: "assistant", content: "Done: press space." }
      ],
      capacity: 8192
    });
    expect(m[0]?.role).toBe("system");
    const user = m[1]?.content as string;
    expect(user).toContain("User is building a game.");
    expect(user).toContain("User: [1 image attached] Add jumping");
    expect(user).toContain("Assistant: Done: press space.");
    expect(user).not.toContain("base64");
  });

  it("keeps the most recent part of an oversized transcript", () => {
    const text = "OLD".repeat(100) + "NEWEST";
    const clipped = clipTranscript(text, 50);
    expect(clipped.endsWith("NEWEST")).toBe(true);
    expect(clipped.length).toBeLessThan(120);
  });
});

describe("message stats", () => {
  it("maps only what the engine reported", () => {
    expect(statsFromRaw(null)).toBeUndefined();
    expect(statsFromRaw({})).toBeUndefined();
    const s = statsFromRaw({
      usage: {
        prompt_tokens: 14,
        completion_tokens: 19,
        total_tokens: 33,
        prompt_tokens_details: { cached_tokens: 5, image_tokens: 0 }
      },
      timings: { decode_tokens_per_second: 56.7, ttft_ms: 77.6, decode_tokens: 19 } as never
    });
    expect(s?.decodeTokensPerSecond).toBe(56.7);
    expect(s?.cachedTokens).toBe(5);
    expect(s?.totalTokens).toBe(33);
    expect(s?.prefillMs).toBeUndefined(); // not invented
  });
});
