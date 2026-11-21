import { beforeEach, describe, expect, it } from "vitest";
import {
  MAX_ARGS_CHARS,
  buildMessageStats,
  capArguments,
  fromToolParams,
  parseToolCallRecords,
  recordFromCall,
  toToolParams,
  type ToolCallRecord
} from "../../src/lib/chat/tools/records.ts";
import { allowKey, useChatToolsStore } from "../../src/lib/stores/useChatToolsStore.ts";
import type { CallState } from "../../src/lib/chat/tools/types.ts";

const call = (i: number, args: Record<string, unknown> = { range: "today" }): CallState => ({
  id: `m1:${i}`,
  conversationId: "c1",
  name: "work_calendar__list_events",
  connectorId: "conn-1",
  connectorName: "Work calendar",
  tool: "list_events",
  arguments: args,
  status: "ok",
  decision: "approved_once",
  output: "2 events",
  startedAt: 1000,
  durationMs: 42
});

const size = (v: unknown) => new TextEncoder().encode(JSON.stringify(v)).length;

describe("tool call records", () => {
  it("keeps short arguments and replaces long ones with a marked preview", () => {
    expect(capArguments({ a: 1 })).toEqual({ a: 1 });
    const long = capArguments({ body: "x".repeat(5000) });
    expect(long["_truncated"]).toBe(true);
    expect(typeof long["preview"]).toBe("string");
    expect(JSON.stringify(long).length).toBeLessThanOrEqual(MAX_ARGS_CHARS + 100);
  });

  it("records a call without its output, with its decision and timing", () => {
    const r = recordFromCall(call(0));
    expect(r).toEqual({
      id: "m1:0",
      name: "work_calendar__list_events",
      connectorId: "conn-1",
      connectorName: "Work calendar",
      tool: "list_events",
      arguments: { range: "today" },
      status: "ok",
      decision: "approved_once",
      startedAt: 1000,
      durationMs: 42
    });
    expect("output" in r).toBe(false);
  });

  it("builds the stats with the engine's numbers, the thinking and the calls, and reads them back", () => {
    const records = [recordFromCall(call(0)), recordFromCall(call(1))];
    const stats = buildMessageStats(
      { timings: { decode_tokens_per_second: 50 }, reasoning: "old", tool_calls: [] },
      "thought",
      records
    );
    expect(stats?.["timings"]).toEqual({ decode_tokens_per_second: 50 });
    expect(stats?.["reasoning"]).toBe("thought");
    expect(parseToolCallRecords(JSON.parse(JSON.stringify(stats))["tool_calls"])).toEqual(records);
  });

  it("has no stats when there is nothing to store", () => {
    expect(buildMessageStats(null, undefined, undefined)).toBeNull();
    expect(buildMessageStats({}, undefined, [])).toBeNull();
  });

  it("stays under the limit: arguments go first, then the last calls", () => {
    const many: ToolCallRecord[] = Array.from({ length: 40 }, (_, i) =>
      recordFromCall(call(i, { body: "y".repeat(4000) }))
    );
    const limit = 4 * 1024;
    const s = buildMessageStats({ usage: { total_tokens: 1 } }, undefined, many, limit);
    expect(size(s)).toBeLessThanOrEqual(limit);
    const kept = s?.["tool_calls"] as ToolCallRecord[];
    expect(kept.length).toBeGreaterThan(0);
    expect(kept.length).toBeLessThan(40);
    expect(kept.every((r) => r.arguments === undefined)).toBe(true);
    expect(kept[0]?.id).toBe("m1:0");

    // a few calls with long arguments fit once each one is capped
    const few = [recordFromCall(call(0, { body: "z".repeat(50_000) }))];
    const s2 = buildMessageStats(null, undefined, few);
    const r2 = (s2?.["tool_calls"] as ToolCallRecord[])[0];
    expect(r2?.arguments?.["_truncated"]).toBe(true);
    expect(size(s2)).toBeLessThan(64 * 1024);

    // the default limit is the engine's 64 KB minus a margin
    const huge = Array.from({ length: 200 }, (_, i) =>
      recordFromCall(call(i, { b: "q".repeat(1900) }))
    );
    expect(size(buildMessageStats(null, "t".repeat(1000), huge))).toBeLessThanOrEqual(64 * 1024);
  });

  it("skips malformed stored entries", () => {
    expect(parseToolCallRecords(undefined)).toBeUndefined();
    expect(parseToolCallRecords([{ id: "a" }, null, "x"])).toBeUndefined();
    const ok = parseToolCallRecords([
      { id: "a:0", status: "denied", decision: "nonsense", durationMs: "5", tool: "t" }
    ]);
    expect(ok).toEqual([
      { id: "a:0", name: "", connectorId: null, connectorName: "", tool: "t", status: "denied" }
    ]);
  });
});

describe("conversation params", () => {
  beforeEach(() => {
    localStorage.clear();
    useChatToolsStore.setState({
      calls: {},
      approvals: [],
      alwaysAllowed: {},
      status: {},
      enabled: {}
    });
  });

  it("round-trips the connector choice and the allowed tools", () => {
    const keys = [allowKey("conn-1", "create_event"), allowKey("conn-2", "send")];
    const p = toToolParams(["conn-1", "conn-2"], keys);
    expect(p).toEqual({
      connectors: ["conn-1", "conn-2"],
      allowed_tools: [
        ["conn-1", "create_event"],
        ["conn-2", "send"]
      ]
    });
    const back = fromToolParams(JSON.parse(JSON.stringify({ temperature: 0.7, ...p })));
    expect(back).toEqual({ enabled: ["conn-1", "conn-2"], allowKeys: keys });
    expect(fromToolParams({ temperature: 1 })).toEqual({ enabled: null, allowKeys: [] });
  });

  it("the store takes the params of a stored chat and gives them back", () => {
    const st = useChatToolsStore.getState();
    const legacy = st.applyParams("c1", {
      connectors: ["conn-1"],
      allowed_tools: [["conn-1", "create_event"], ["bad"]]
    });
    expect(legacy).toBe(false);
    expect(useChatToolsStore.getState().enabledFor("c1")).toEqual(["conn-1"]);
    expect(useChatToolsStore.getState().alwaysAllowed["c1"]).toEqual([
      allowKey("conn-1", "create_event")
    ]);
    expect(useChatToolsStore.getState().paramsFor("c1")).toEqual({
      connectors: ["conn-1"],
      allowed_tools: [["conn-1", "create_event"]]
    });

    // an older chat without a stored choice keeps this browser's, and says it should be written back
    useChatToolsStore.getState().toggle("c2", "conn-9");
    expect(useChatToolsStore.getState().applyParams("c2", { temperature: 0.7 })).toBe(true);
    expect(useChatToolsStore.getState().enabledFor("c2")).toEqual(["conn-9"]);
    // a stored empty choice wins
    useChatToolsStore.getState().applyParams("c2", { connectors: [] });
    expect(useChatToolsStore.getState().enabledFor("c2")).toEqual([]);
  });

  it("only the choice for a chat not created yet stays in this browser", () => {
    const st = useChatToolsStore.getState();
    st.toggle("c1", "conn-1");
    st.toggle(null, "conn-2");
    expect(JSON.parse(localStorage.getItem("dara.chatConnectors") ?? "{}")).toEqual({
      __new__: ["conn-2"]
    });
  });
});
