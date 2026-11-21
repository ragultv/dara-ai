import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../src/lib/chat/tools/types.ts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../src/lib/chat/tools/types.ts")>()),
  CALL_TIMEOUT_MS: 150
}));

import { createDemoBackend, setBackend } from "../../src/lib/connectors/backend.ts";
import type { ConnectorsBackend } from "../../src/lib/connectors/types.ts";
import { useConnectorsStore } from "../../src/lib/stores/useConnectorsStore.ts";
import { allowKey, useChatToolsStore } from "../../src/lib/stores/useChatToolsStore.ts";
import { loadChatTools, runCalls } from "../../src/lib/chat/tools/runner.ts";
import type { ChatTool, ParsedCall } from "../../src/lib/chat/tools/types.ts";

let backend: ConnectorsBackend;
let delay: (tool: string, args: Record<string, unknown>) => number;
let connectorId: string;

const tool = (name: "list_events" | "create_event"): ChatTool => ({
  name: `demo__${name}`,
  connectorId,
  connectorName: "Demo",
  tool: name,
  title: null,
  description: "",
  inputSchema: {},
  permission: "ask"
});
let tools: ChatTool[];

const call = (i: number, name: string, args: Record<string, unknown> = {}): ParsedCall => ({
  id: `m1:${i}`,
  name,
  arguments: args
});

const resetChatStore = () =>
  useChatToolsStore.setState({
    calls: {},
    approvals: [],
    alwaysAllowed: {},
    status: {},
    enabled: {}
  });

/** waits until a call is waiting for approval */
async function untilWaiting(id: string) {
  for (let i = 0; i < 200; i++) {
    if (useChatToolsStore.getState().approvals.includes(id)) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error(`${id} never asked for approval`);
}

beforeEach(async () => {
  const demo = createDemoBackend();
  delay = () => 0;
  backend = {
    ...demo,
    call: async (id, t, args, approved) => {
      const ms = delay(t, args);
      if (ms) await new Promise((r) => setTimeout(r, ms));
      return demo.call(id, t, args, approved);
    }
  };
  setBackend(backend);
  useConnectorsStore.setState({ entries: [], tools: {}, audit: {}, busy: {}, error: null });
  resetChatStore();
  const c = await backend.add({ name: "Demo", kind: "stdio", command: "demo" });
  connectorId = c.id;
  await backend.setPermission(connectorId, "list_events", "allow");
  await backend.setPermission(connectorId, "create_event", "ask");
  tools = [tool("list_events"), tool("create_event")];
});

afterEach(() => setBackend(null));

describe("runCalls", () => {
  it("runs calls in parallel and keeps their order", async () => {
    delay = (_t, a) => (a["range"] === "slow" ? 120 : 80);
    const started = Date.now();
    const out = await runCalls(
      "c1",
      [
        call(0, "demo__list_events", { range: "slow" }),
        call(1, "demo__list_events", { range: "fast" })
      ],
      tools,
      new AbortController().signal
    );
    const took = Date.now() - started;
    expect(took).toBeLessThan(190); // ~max (120), not sum (200)
    expect(out.map((r) => r.id)).toEqual(["m1:0", "m1:1"]);
    expect(out.every((r) => r.status === "ok")).toBe(true);
    expect(out[0]!.text).toContain("slow");
    expect(out[1]!.text).toContain("fast");
    const st = useChatToolsStore.getState();
    expect(st.calls["m1:0"]?.status).toBe("ok");
    expect(st.calls["m1:0"]?.durationMs).toBeGreaterThanOrEqual(0);
    expect(st.status["c1"]).toBeUndefined();
  });

  it("waits for approval, then runs once allowed", async () => {
    const p = runCalls(
      "c1",
      [call(0, "demo__create_event", { title: "Lunch", start: "12:00" })],
      tools,
      new AbortController().signal
    );
    await untilWaiting("m1:0");
    expect(useChatToolsStore.getState().calls["m1:0"]?.status).toBe("waiting_approval");
    expect(useChatToolsStore.getState().status["c1"]).toBe("Waiting for your approval");
    useChatToolsStore.getState().resolveApproval("m1:0", true);
    const [r] = await p;
    expect(r!.status).toBe("ok");
    expect(r!.text).toContain("Lunch");
    expect(useChatToolsStore.getState().approvals).toEqual([]);
    expect(useChatToolsStore.getState().calls["m1:0"]?.decision).toBe("approved_once");
  });

  it("always allow skips the next prompt in the same chat only", async () => {
    const p1 = runCalls(
      "c1",
      [call(0, "demo__create_event", { title: "A", start: "1" })],
      tools,
      new AbortController().signal
    );
    await untilWaiting("m1:0");
    useChatToolsStore.getState().resolveApproval("m1:0", true, true);
    expect((await p1)[0]!.status).toBe("ok");
    expect(useChatToolsStore.getState().calls["m1:0"]?.decision).toBe("approved_always");
    expect(useChatToolsStore.getState().alwaysAllowed["c1"]).toEqual([
      allowKey(connectorId, "create_event")
    ]);

    // the same chat name now pointing at another tool (names are rebuilt each send) must still ask
    await backend.setPermission(connectorId, "list_events", "ask");
    const moved: ChatTool = { ...tool("list_events"), name: "demo__create_event" };
    const pm = runCalls(
      "c1",
      [{ id: "m4:0", name: "demo__create_event", arguments: { range: "x" } }],
      [moved],
      new AbortController().signal
    );
    await untilWaiting("m4:0");
    useChatToolsStore.getState().resolveApproval("m4:0", false);
    expect((await pm)[0]!.status).toBe("denied");
    expect(useChatToolsStore.getState().calls["m4:0"]?.decision).toBe("denied_by_user");

    const second = await runCalls(
      "c1",
      [{ id: "m2:0", name: "demo__create_event", arguments: { title: "B", start: "2" } }],
      tools,
      new AbortController().signal
    );
    expect(second[0]!.status).toBe("ok");
    expect(useChatToolsStore.getState().calls["m2:0"]?.decision).toBe("auto");

    const p3 = runCalls(
      "c2",
      [{ id: "m3:0", name: "demo__create_event", arguments: { title: "C", start: "3" } }],
      tools,
      new AbortController().signal
    );
    await untilWaiting("m3:0");
    useChatToolsStore.getState().resolveApproval("m3:0", false);
    expect((await p3)[0]!.status).toBe("denied");
  });

  it("a blocked tool comes back denied", async () => {
    await backend.setPermission(connectorId, "list_events", "deny");
    const [r] = await runCalls(
      "c1",
      [call(0, "demo__list_events", { range: "x" })],
      tools,
      new AbortController().signal
    );
    expect(r!.status).toBe("denied");
    expect(useChatToolsStore.getState().calls["m1:0"]?.decision).toBe("blocked");
  });

  it("stopping skips running calls and refuses waiting approvals", async () => {
    delay = (t) => (t === "list_events" ? 1000 : 0);
    const ac = new AbortController();
    const p = runCalls(
      "c1",
      [
        call(0, "demo__list_events", { range: "x" }),
        call(1, "demo__create_event", { title: "A", start: "1" })
      ],
      tools,
      ac.signal
    );
    await untilWaiting("m1:1");
    ac.abort();
    const out = await p;
    expect(out.map((r) => r.status)).toEqual(["skipped", "skipped"]);
    // list_events was allowed by its rule before it was stopped; create_event never got an answer
    expect(useChatToolsStore.getState().calls["m1:0"]?.decision).toBe("skipped");
    expect(useChatToolsStore.getState().calls["m1:1"]?.decision).toBe("skipped");
    expect(useChatToolsStore.getState().approvals).toEqual([]);
    expect(useChatToolsStore.getState().status["c1"]).toBeUndefined();
  });

  it("a call already stopped is skipped without running", async () => {
    const ac = new AbortController();
    ac.abort();
    const [r] = await runCalls(
      "c1",
      [call(0, "demo__list_events", { range: "x" })],
      tools,
      ac.signal
    );
    expect(r!.status).toBe("skipped");
  });

  it("a slow tool times out as an error", async () => {
    delay = () => 1000;
    const [r] = await runCalls(
      "c1",
      [call(0, "demo__list_events", { range: "x" })],
      tools,
      new AbortController().signal
    );
    expect(r!.status).toBe("error");
    expect(r!.text).toMatch(/did not answer/);
  });

  it("an unknown tool is an error", async () => {
    const [r] = await runCalls("c1", [call(0, "nope__x")], tools, new AbortController().signal);
    expect(r!.status).toBe("error");
    expect(r!.text).toMatch(/Unknown tool nope__x/);
    expect(useChatToolsStore.getState().calls["m1:0"]?.status).toBe("error");
  });

  it("a backend failure is an error with its message", async () => {
    backend.call = async () => {
      throw new Error("server went away");
    };
    const [r] = await runCalls(
      "c1",
      [call(0, "demo__list_events", { range: "x" })],
      tools,
      new AbortController().signal
    );
    expect(r!.status).toBe("error");
    expect(r!.text).toContain("server went away");
  });
});

describe("loadChatTools", () => {
  it("connects the chosen connectors and lists their tools", async () => {
    const { tools: list, problems } = await loadChatTools([connectorId]);
    expect(problems).toEqual([]);
    expect(list.map((t) => t.tool).sort()).toEqual(["create_event", "list_events"]);
    expect(list.every((t) => t.connectorId === connectorId)).toBe(true);
  });

  it("says which connector is not signed in", async () => {
    const c = await backend.add({
      name: "Mail",
      kind: "http",
      url: "https://auth.example.com/mcp"
    });
    const { problems } = await loadChatTools([c.id]);
    expect(problems).toEqual(["Mail is not signed in"]);
  });

  it("says when a connector is gone", async () => {
    const { tools: list, problems } = await loadChatTools(["missing"]);
    expect(list).toEqual([]);
    expect(problems).toHaveLength(1);
  });
});

describe("the store", () => {
  it("keeps the new chat's choice and hands it to the created chat", () => {
    const s = useChatToolsStore.getState();
    s.toggle(null, "a");
    s.toggle(null, "b");
    s.toggle(null, "a");
    expect(useChatToolsStore.getState().enabledFor(null)).toEqual(["b"]);
    useChatToolsStore.getState().adoptNew("c9");
    expect(useChatToolsStore.getState().enabledFor("c9")).toEqual(["b"]);
    expect(useChatToolsStore.getState().enabledFor(null)).toEqual([]);
  });
});
