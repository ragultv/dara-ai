import { describe, expect, it } from "vitest";
import {
  COMPACT_SHARE,
  describeTool,
  formatProblems,
  parseToolCalls,
  shouldUseCompactList,
  toolSignature,
  toolSystemPrompt,
  waitTool
} from "../../src/lib/chat/tools/protocol.ts";
import { estimateTokens } from "../../src/lib/chat/context.ts";
import { DESCRIBE_TOOL, type ChatTool } from "../../src/lib/chat/tools/types.ts";

// A long tool list is shortened to one line per tool, with a lookup tool for the arguments. These are the pure parts of that.

const LONG =
  "Searches the user's mail threads with the same query language the search box uses and returns their ids, subjects and snippets. " +
  "Use it before reading a thread; results are paged and the newest come first, so ask for the next page when you need more.";

function tool(i: number, extra: Partial<ChatTool> = {}): ChatTool {
  return {
    name: `mail__tool_${i}`,
    connectorId: "m",
    connectorName: "Mail",
    tool: `tool_${i}`,
    title: null,
    description: LONG,
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "What to look for, in the search box's query language" },
        limit: { type: "integer", description: "How many results, 1 to 50" },
        label: { type: "string", description: "Only threads with this label" },
        order: { type: "string", enum: ["newest", "oldest"], description: "Sort order" }
      },
      required: ["query"]
    },
    permission: "ask",
    ...extra
  };
}

const many = (n: number) => Array.from({ length: n }, (_, i) => tool(i));
const NOW = new Date("2026-01-01T00:00:00");

describe("the compact tool list", () => {
  it("names each tool in one line with its required arguments, and says how to look one up", () => {
    const p = toolSystemPrompt([...many(2), describeTool()], NOW, true);
    expect(p).toContain("- mail__tool_0(query, ...): Searches the user's mail threads");
    expect(p).toContain(`- ${DESCRIBE_TOOL}(name): Shows every argument of one tool`);
    expect(p).toContain(`look it up with a ${DESCRIBE_TOOL} block`);
    // the example is the lookup, not a call that guesses its arguments
    expect(p).toContain(`{"name": "${DESCRIBE_TOOL}", "arguments": {"name": "mail__tool_0"}}`);
    // no full descriptions, no argument lists
    expect(p).not.toContain("### ");
    expect(p).not.toContain("Arguments:");
    expect(p).not.toContain("How many results");
  });

  it("is far smaller than the full list for a long set of tools", () => {
    const tools = many(40);
    const full = estimateTokens(toolSystemPrompt(tools, NOW));
    const compact = estimateTokens(toolSystemPrompt([...tools, describeTool()], NOW, true));
    expect(compact).toBeLessThan(full / 3);
  });

  it("leaves the full list exactly as it was", () => {
    const p = toolSystemPrompt(many(2), NOW);
    expect(p).toContain("### mail__tool_0");
    expect(p).toContain("- limit (integer, optional): How many results, 1 to 50");
    expect(p).toContain("- order (string, optional) one of \"newest\", \"oldest\": Sort order");
    expect(p).not.toContain(DESCRIBE_TOOL);
  });

  it("a tool with only optional arguments shows just the dots, one with none shows empty brackets", () => {
    const onlyOptional = tool(1, {
      name: "x__a",
      inputSchema: { type: "object", properties: { q: { type: "string" } } }
    });
    const none = tool(2, { name: "x__b", inputSchema: {} });
    const p = toolSystemPrompt([onlyOptional, none], NOW, true);
    expect(p).toContain("- x__a(...):");
    expect(p).toContain("- x__b():");
  });
});

describe("when the list is shortened", () => {
  it("never for a short list, whatever the context", () => {
    expect(shouldUseCompactList(many(2))).toBe(false);
    expect(shouldUseCompactList(many(2), 65536)).toBe(false);
  });

  it("only as a last resort: when the full list takes a quarter of the context, or 6,000 tokens when the context is unknown", () => {
    expect(shouldUseCompactList(many(40))).toBe(true);
    expect(shouldUseCompactList(many(10))).toBe(false); // ~3,500 tokens, under 6,000
    const tokens = estimateTokens(toolSystemPrompt(many(12)));
    expect(shouldUseCompactList(many(12), Math.floor(tokens / COMPACT_SHARE) + 1000)).toBe(false);
    expect(shouldUseCompactList(many(12), Math.floor(tokens / COMPACT_SHARE) - 1000)).toBe(true);
    // a roomy context keeps the full list even for a long set of tools (Dara Nano runs 131,072 tokens on a 4 GB card)
    expect(shouldUseCompactList(many(40), 131072)).toBe(false);
    // a small one cannot hold it at all
    expect(shouldUseCompactList(many(40), 16384)).toBe(true);
  });
});

describe("looking a tool up", () => {
  const tools = [...many(3), waitTool(), describeTool()];

  it("is a call like any other: it resolves, and needs the name", () => {
    const ok = parseToolCalls(
      `<tool_call>{"name": "${DESCRIBE_TOOL}", "arguments": {"name": "mail__tool_1"}}</tool_call>`,
      tools,
      "m1"
    );
    expect(ok.problems).toEqual([]);
    expect(ok.calls).toEqual([
      { id: "m1:0", name: DESCRIBE_TOOL, arguments: { name: "mail__tool_1" } }
    ]);
    const missing = parseToolCalls(
      `<tool_call>{"name": "${DESCRIBE_TOOL}", "arguments": {}}</tool_call>`,
      tools,
      "m1"
    );
    expect(missing.problems[0]?.reason).toContain("missing required argument 'name'");
  });

  it("shows the tool in full with an example call", () => {
    const s = toolSignature(tool(1));
    expect(s).toContain("### mail__tool_1");
    expect(s).toContain("- query (string, required)");
    expect(s).toContain("- order (string, optional) one of");
    expect(s).toContain('Call it like this: <tool_call>{"name":"mail__tool_1","arguments":{"query":"..."}}</tool_call>');
  });

  it("a call with wrong arguments comes back with the tool's arguments, once per tool", () => {
    const text =
      '<tool_call>{"name": "mail__tool_1", "arguments": {}}</tool_call>' +
      '<tool_call>{"name": "mail__tool_1", "arguments": {"limit": "x"}}</tool_call>';
    const parsed = parseToolCalls(text, tools, "m1");
    expect(parsed.problems.map((p) => p.tool)).toEqual(["mail__tool_1", "mail__tool_1"]);
    const fix = formatProblems(parsed.problems, tools);
    expect(fix.match(/### mail__tool_1/g)).toHaveLength(1);
    expect(fix).toContain("- query (string, required)");
    // a block that was not about a known tool shows no signature
    const unknown = parseToolCalls('<tool_call>{"name": "nope"}</tool_call>', tools, "m1");
    expect(formatProblems(unknown.problems, tools)).not.toContain("### ");
  });
});
