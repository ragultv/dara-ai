import { describe, expect, it } from "vitest";
import { formatTokensShort } from "../../src/lib/chat/context.ts";

describe("formatTokensShort", () => {
  it("reads the same as the exact number in the popup (131,072 -> 131k, not 128k)", () => {
    expect(formatTokensShort(131072)).toBe("131k");
  });
  it("keeps one decimal below 100k and plain numbers below 1000", () => {
    expect(formatTokensShort(8192)).toBe("8.2k");
    expect(formatTokensShort(32768)).toBe("32.8k");
    expect(formatTokensShort(900)).toBe("900");
  });
});
