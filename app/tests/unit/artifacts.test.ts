import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildEditMessages,
  cleanEditResult,
  editMaxTokens,
  EDIT_SYSTEM_PROMPT
} from "../../src/lib/artifacts/askDara.ts";
import { exportFileName, relativeTime } from "../../src/lib/artifacts/format.ts";

describe("cleanEditResult (what the model returns for 'Ask Dara')", () => {
  it("leaves a plain document alone", () => {
    expect(cleanEditResult("# Title\n\n- one\n- two")).toBe("# Title\n\n- one\n- two");
  });
  it("strips one fence around the whole answer, keeps fences inside", () => {
    const inner = "# Doc\n\n```js\nlet a = 1;\n```\n\ntext";
    expect(cleanEditResult("```markdown\n" + inner + "\n```")).toBe(inner);
    expect(cleanEditResult("```md\n" + inner + "\n```\n")).toBe(inner);
  });
  it("works on a partial answer while it streams (fence opened, not closed yet)", () => {
    expect(cleanEditResult("```markdown\n# Doc\n\nhalf a sen")).toBe("# Doc\n\nhalf a sen");
  });
  it("drops a one-line lead-in and the <document> tags", () => {
    expect(cleanEditResult("Here is the updated document:\n\n# Doc\n\ntext")).toBe("# Doc\n\ntext");
    expect(cleanEditResult("<document>\n# Doc\n</document>")).toBe("# Doc");
  });
  it("does not remove a sentence that merely starts with 'here'", () => {
    expect(cleanEditResult("Here we go again with the plan.")).toBe(
      "Here we go again with the plan."
    );
  });
});

describe("buildEditMessages", () => {
  it("sends the system prompt, the document and the instruction", () => {
    const m = buildEditMessages("# A", "make it shorter");
    expect(m[0]).toEqual({ role: "system", content: EDIT_SYSTEM_PROMPT });
    expect(m[1]!.role).toBe("user");
    expect(m[1]!.content).toContain("<document>\n# A\n</document>");
    expect(m[1]!.content).toContain("Instruction: make it shorter");
  });
  it("says so when the document is empty", () => {
    expect(String(buildEditMessages("  \n", "write a poem")[1]!.content)).toContain(
      "(the document is empty)"
    );
  });
});

describe("editMaxTokens", () => {
  it("leaves room to write the document back, within bounds", () => {
    expect(editMaxTokens("")).toBe(1024 > 768 ? 1024 : 768);
    expect(editMaxTokens("x".repeat(3000))).toBeGreaterThan(3000 / 3);
    expect(editMaxTokens("x".repeat(10_000_000))).toBe(8192);
  });
});

describe("relativeTime", () => {
  const now = new Date(2026, 10, 20, 12, 0, 0).getTime();
  it("reads like the gallery cards", () => {
    expect(relativeTime(now - 20_000, now)).toBe("now");
    expect(relativeTime(now - 5 * 60_000, now)).toBe("5 minutes ago");
    expect(relativeTime(now - 60_000, now)).toBe("1 minute ago");
    expect(relativeTime(now - 3 * 3_600_000, now)).toBe("3 hours ago");
    expect(relativeTime(new Date(2026, 10, 19, 8, 0).getTime(), now)).toBe("Yesterday");
    expect(relativeTime(new Date(2026, 10, 17, 8, 0).getTime(), now)).toBe("Nov 17");
    expect(relativeTime(new Date(2025, 10, 17, 8, 0).getTime(), now)).toBe("Nov 17, 2025");
  });
});

describe("exportFileName", () => {
  it("removes what Windows forbids and falls back to Untitled", () => {
    expect(exportFileName('Plan: v2 / "final"?')).toBe("Plan v2 final.md");
    expect(exportFileName("   ")).toBe("Untitled.md");
    expect(exportFileName("a".repeat(200)).length).toBe(83);
  });
});

describe("router: #/artifacts/<id>", () => {
  afterEach(() => {
    window.location.hash = "";
    vi.resetModules();
  });
  it("opens one artifact from a UUID, ignores anything else", async () => {
    window.location.hash = "#/artifacts/0F0F0F0F-1234-4abc-8def-0123456789ab";
    vi.resetModules();
    let { useRouter } = await import("../../src/lib/router/useRouter.ts");
    expect(useRouter.getState().current).toBe("artifact");
    expect(useRouter.getState().artifactId).toBe("0f0f0f0f-1234-4abc-8def-0123456789ab");

    window.location.hash = "#/artifacts/not-a-uuid";
    vi.resetModules();
    ({ useRouter } = await import("../../src/lib/router/useRouter.ts"));
    expect(useRouter.getState().current).toBe("chat");

    window.location.hash = "#/artifacts";
    vi.resetModules();
    ({ useRouter } = await import("../../src/lib/router/useRouter.ts"));
    expect(useRouter.getState().current).toBe("artifacts");
    expect(useRouter.getState().artifactId).toBeNull();
  });
});
