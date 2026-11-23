import { describe, expect, it } from "vitest";
import { stepThinking, type TurnStep } from "../../src/lib/components/assistant-ui/ToolCalls.tsx";
import type { UiMessage } from "../../src/lib/chat/types.ts";

function step(id: string, extra: Partial<UiMessage> = {}): TurnStep {
  return { message: { id, role: "assistant", content: "", createdAt: 1, ...extra }, results: null };
}

// The model's thinking before a tool call and between calls was lost once the answer finished: every step keeps its own thinking, but only
// the last message of a reply printed it. The tool fold now prints each step's thinking, in order.
describe("the thinking of a tool step", () => {
  it("is what the step stored, with how long it took", () => {
    expect(stepThinking(step("m1", { reasoning: "Look for the invoice first.", reasoningMs: 3000 }))).toEqual({
      text: "Look for the invoice first.",
      durationMs: 3000,
      cut: undefined
    });
  });

  it("carries the limit-reached flag", () => {
    expect(stepThinking(step("m1", { reasoning: "x", reasoningCut: true }))?.cut).toBe(true);
  });

  it("is nothing for a step that did not think", () => {
    expect(stepThinking(step("m1"))).toBeNull();
    expect(stepThinking(step("m1", { reasoning: "   " }))).toBeNull();
  });

  it("is left out for the message the reply prints below the fold, so it is not shown twice", () => {
    const s = step("m2", { reasoning: "shown elsewhere" });
    expect(stepThinking(s, "m2")).toBeNull();
    expect(stepThinking(s, "m3")?.text).toBe("shown elsewhere");
  });
});
