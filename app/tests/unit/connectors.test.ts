import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it } from "vitest";
import {
  checkServerUrl,
  parseArgs,
  parseEnv,
  skeletonArgs,
  validateNewConnector
} from "../../src/lib/connectors/validate.ts";
import { createDemoBackend, setBackend } from "../../src/lib/connectors/backend.ts";
import { resultText } from "../../src/lib/connectors/ToolTestDialog.tsx";
import { useConnectorsStore } from "../../src/lib/stores/useConnectorsStore.ts";

describe("server addresses", () => {
  it("accepts https and this computer, nothing else", () => {
    expect(checkServerUrl("https://example.com/mcp")).toBeNull();
    expect(checkServerUrl("http://127.0.0.1:8080/mcp")).toBeNull();
    expect(checkServerUrl("http://localhost:3000")).toBeNull();
    expect(checkServerUrl("http://example.com/mcp")).toMatch(/https/);
    expect(checkServerUrl("ftp://example.com")).toMatch(/https/);
    expect(checkServerUrl("not a url")).toMatch(/web address/);
  });
});

describe("the add form", () => {
  it("reads one argument per line without splitting on spaces", () => {
    expect(parseArgs("-y\n  @scope/server \n\nC:\\Program Files\\x\n")).toEqual([
      "-y",
      "@scope/server",
      "C:\\Program Files\\x"
    ]);
  });
  it("reads NAME=value lines and names the bad one", () => {
    expect(parseEnv("A=1\nTOKEN=a=b=c\n")).toEqual({ env: { A: "1", TOKEN: "a=b=c" }, bad: null });
    expect(parseEnv("A=1\noops").bad).toBe("oops");
    expect(parseEnv("1BAD=x").bad).toBe("1BAD");
  });
  it("checks a new connector like the desktop layer does", () => {
    expect(validateNewConnector({ name: " ", kind: "stdio", command: "npx" }, 0)).toMatch(/name/);
    expect(validateNewConnector({ name: "x", kind: "stdio", command: "" }, 0)).toMatch(/program/);
    expect(validateNewConnector({ name: "x", kind: "stdio", command: "a\nb" }, 0)).toMatch(
      /line break/
    );
    expect(
      validateNewConnector({ name: "x", kind: "http", url: "http://evil.example" }, 0)
    ).toMatch(/https/);
    expect(validateNewConnector({ name: "x", kind: "stdio", command: "npx" }, 32)).toMatch(
      /At most/
    );
    expect(validateNewConnector({ name: "x", kind: "stdio", command: "npx" }, 0)).toBeNull();
  });
  it("starts a tool's arguments from its required properties", () => {
    expect(
      skeletonArgs({
        properties: {
          q: { type: "string" },
          n: { type: "integer" },
          f: { type: "boolean" },
          o: { type: "string" }
        },
        required: ["q", "n", "f"]
      })
    ).toEqual({ q: "", n: 0, f: false });
  });
});

describe("tool results are shown as plain text", () => {
  it("joins text blocks and names the rest", () => {
    const base = {
      outcome: "ok" as const,
      structuredContent: null,
      isError: false,
      truncated: false
    };
    expect(
      resultText({
        ...base,
        content: [
          { type: "text", text: "<img src=x onerror=alert(1)>" },
          { type: "image", data: "AAAA" },
          { type: "resource_link", uri: "https://x.io/a" }
        ]
      })
    ).toBe("<img src=x onerror=alert(1)>\n[an image]\n[link] https://x.io/a");
    expect(resultText({ ...base, content: [], structuredContent: { a: 1 } })).toBe(
      '{\n  "a": 1\n}'
    );
  });
});

describe("connectors store (demo backend)", () => {
  beforeEach(() => {
    setBackend(createDemoBackend());
    useConnectorsStore.setState({ entries: [], tools: {}, audit: {}, busy: {}, error: null });
  });

  it("adds, connects and lists tools that all start on Ask", async () => {
    const s = useConnectorsStore.getState();
    expect(await s.add({ name: "Demo", kind: "stdio", command: "demo" })).toBeNull();
    const id = useConnectorsStore.getState().entries[0]!.config.id;
    await useConnectorsStore.getState().connect(id);
    expect(useConnectorsStore.getState().entries[0]!.status.state).toBe("ready");
    const tools = useConnectorsStore.getState().tools[id]!;
    expect(tools.map((t) => t.name)).toEqual(["list_events", "create_event"]);
    expect(tools.every((t) => t.permission === "ask")).toBe(true);
    // a server's read-only claim does not change the permission
    expect(tools[0]!.readOnlyHint).toBe(true);
    expect(tools[0]!.permission).toBe("ask");
  });

  it("refuses a bad connector without calling the backend", async () => {
    expect(
      await useConnectorsStore.getState().add({ name: "", kind: "http", url: "https://x.io" })
    ).toMatch(/name/);
    expect(useConnectorsStore.getState().entries).toHaveLength(0);
  });

  it("gates a call on the permission and keeps an audit trail", async () => {
    await useConnectorsStore
      .getState()
      .add({ name: "Demo", kind: "http", url: "https://x.io/mcp" });
    const id = useConnectorsStore.getState().entries[0]!.config.id;
    await useConnectorsStore.getState().connect(id);

    const first = await useConnectorsStore
      .getState()
      .call(id, "list_events", { range: "today" }, false);
    expect("outcome" in first && first.outcome).toBe("needs_approval");

    const once = await useConnectorsStore
      .getState()
      .call(id, "list_events", { range: "today" }, true);
    expect("outcome" in once && once.outcome).toBe("ok");

    await useConnectorsStore.getState().setPermission(id, "list_events", "allow");
    const allowed = await useConnectorsStore
      .getState()
      .call(id, "list_events", { range: "today" }, false);
    expect("outcome" in allowed && allowed.outcome).toBe("ok");

    await useConnectorsStore.getState().setPermission(id, "create_event", "deny");
    const denied = await useConnectorsStore
      .getState()
      .call(id, "create_event", { title: "x", start: "now" }, true);
    expect("outcome" in denied && denied.outcome).toBe("denied");

    await useConnectorsStore.getState().loadAudit(id);
    expect(useConnectorsStore.getState().audit[id]!.map((a) => a.outcome)).toEqual([
      "denied",
      "ok",
      "ok",
      "needs_approval"
    ]);
  });

  it("removes a connector and what it kept", async () => {
    await useConnectorsStore.getState().add({ name: "Demo", kind: "stdio", command: "demo" });
    const id = useConnectorsStore.getState().entries[0]!.config.id;
    await useConnectorsStore.getState().connect(id);
    await useConnectorsStore.getState().remove(id);
    expect(useConnectorsStore.getState().entries).toHaveLength(0);
    expect(useConnectorsStore.getState().tools[id]).toBeUndefined();
  });
});

// The desktop layer's real output is recorded in docs/ui/samples/connectors.json by the Rust contract test. The demo backend (what a browser
// tab and these tests run on) must have the same fields, so the page cannot work in the demo and break in the app.
function shape(v: unknown): unknown {
  if (v === null) return "null";
  if (Array.isArray(v)) return v.length ? [shape(v[0])] : [];
  if (typeof v === "object")
    return Object.fromEntries(Object.entries(v as object).map(([k, x]) => [k, shape(x)]));
  return typeof v;
}
// free-form by nature: a tool's JSON Schema and the content blocks a server returns
const FREE_FORM = [".inputSchema", ".content", ".structuredContent"];
function differences(a: unknown, b: unknown, path = "$"): string[] {
  if (a === "null" || b === "null") return [];
  if (FREE_FORM.some((f) => path.endsWith(f))) return [];
  if (Array.isArray(a) && Array.isArray(b))
    return a.length && b.length ? differences(a[0], b[0], `${path}[]`) : [];
  if (a && b && typeof a === "object" && typeof b === "object") {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    return [...keys].flatMap((k) => {
      const x = (a as Record<string, unknown>)[k];
      const y = (b as Record<string, unknown>)[k];
      return x === undefined || y === undefined
        ? [`${path}.${k} is on one side only`]
        : differences(x, y, `${path}.${k}`);
    });
  }
  return a === b ? [] : [`${path}: ${String(a)} vs ${String(b)}`];
}

describe("the demo backend has the same shapes as the desktop layer", () => {
  it("matches docs/ui/samples/connectors.json", async () => {
    const recorded = JSON.parse(
      readFileSync(resolve(process.cwd(), "../docs/ui/samples/connectors.json"), "utf-8")
    ) as Record<string, unknown>;
    const b = createDemoBackend();
    const cfg = await b.add({ name: "Mock", kind: "stdio", command: "mock" });
    await b.connect(cfg.id);
    const tools = await b.tools(cfg.id);
    const ask = await b.call(cfg.id, "list_events", { range: "today" }, false);
    const ok = await b.call(cfg.id, "list_events", { range: "today" }, true);
    const actual = {
      connectors: await b.list(),
      tools,
      callNeedsApproval: ask,
      callOk: ok,
      audit: await b.audit(cfg.id, 20)
    };
    expect(differences(shape(actual), shape(recorded))).toEqual([]);
  });
});
