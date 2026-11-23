import { readFileSync } from "node:fs";
import { describe, it } from "vitest";
import {
  describeTool,
  shouldUseCompactList,
  toolSystemPrompt
} from "../../src/lib/chat/tools/protocol.ts";
import { estimateTokens } from "../../src/lib/chat/context.ts";
import type { ChatTool } from "../../src/lib/chat/tools/types.ts";

// A measurement, not a regression test: how many tokens do the tools of real servers cost in the system prompt of every request?
// It runs only when DARA_TOOL_JSON names a file written by the crate's example:
//   cargo run -p dara-mcp --example probe_connect -- --json tools.json <server urls...>
//   DARA_TOOL_JSON=tools.json npx vitest run tests/unit/tool-prompt-size.test.ts
const file = process.env["DARA_TOOL_JSON"];

interface Dumped {
  url: string;
  server: string;
  tools: Array<{
    name: string;
    title: string | null;
    description: string;
    inputSchema: Record<string, unknown>;
  }>;
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

function toChatTools(d: Dumped): ChatTool[] {
  const s = slug(d.server.split(" ")[0] ?? "server");
  return d.tools.map((t) => ({
    name: `${s}__${slug(t.name)}`,
    connectorId: s,
    connectorName: d.server,
    tool: t.name,
    title: t.title,
    description: t.description,
    inputSchema: t.inputSchema,
    permission: "ask" as const
  }));
}

describe.skipIf(!file)("tool prompt size (measurement)", () => {
  it("prints what each server's tools cost per request", () => {
    const dumped = JSON.parse(readFileSync(file as string, "utf8")) as Dumped[];
    const now = new Date();
    const fixed = estimateTokens(toolSystemPrompt([], now));
    const rows: string[] = [];
    const all: ChatTool[] = [];
    for (const d of dumped) {
      const tools = toChatTools(d);
      all.push(...tools);
      const full = estimateTokens(toolSystemPrompt(tools, now)) - fixed;
      const compact =
        estimateTokens(toolSystemPrompt([...tools, describeTool()], now, true)) - fixed;
      rows.push(
        `${d.server.padEnd(26)} ${String(tools.length).padStart(3)} tools  full ${String(full).padStart(6)}  compact ${String(compact).padStart(5)}`
      );
    }
    const total = estimateTokens(toolSystemPrompt(all, now));
    const compactTotal = estimateTokens(toolSystemPrompt([...all, describeTool()], now, true));
    const lines = [
      "",
      `instructions with no tools: ${fixed} tokens`,
      ...rows,
      `ALL TOGETHER: ${all.length} tools -> full ${total} tokens, compact ${compactTotal} tokens in the system prompt of every request`,
      ...[8192, 16384, 32768, 65536].map(
        (ctx) =>
          `  ${String(ctx / 1024).padStart(2)}K context: full ${((100 * total) / ctx).toFixed(1)}%, compact ${((100 * compactTotal) / ctx).toFixed(1)}%  -> the app uses ${shouldUseCompactList(all, ctx) ? "compact" : "full"}`
      )
    ];
    console.log(lines.join("\n"));
  });
});
