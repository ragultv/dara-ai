import type {
  AddConnectorInput,
  AuditEntry,
  ConnectorConfig,
  ConnectorEntry,
  ConnectorPatch,
  ConnectorsBackend,
  ConnectorStatus,
  ConnectorTool,
  ToolCallResult,
  ToolPermission
} from "./types.ts";

// The desktop app talks to its Rust layer with Tauri commands. In a browser tab (development, tests) there is no Rust layer, so a demo with
// sample data stands in: it behaves like a connector that is already set up, and nothing leaves the page.

const isDesktop = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
  try {
    return await tauriInvoke<T>(cmd, args);
  } catch (e) {
    // the desktop layer answers with a plain string
    throw new Error(
      typeof e === "string" ? e : e instanceof Error ? e.message : "The connector call failed"
    );
  }
}

export const tauriBackend: ConnectorsBackend = {
  isDemo: false,
  list: () => invoke("connectors_list"),
  add: (input) => invoke("connector_add", { input }),
  update: (id, patch) => invoke("connector_update", { id, patch }),
  remove: (id) => invoke("connector_remove", { id }),
  connect: (id) => invoke("connector_connect", { id }),
  disconnect: (id) => invoke("connector_disconnect", { id }),
  signIn: (id) => invoke("connector_sign_in", { id }),
  cancelSignIn: (id) => invoke("connector_cancel_sign_in", { id }),
  signOut: (id) => invoke("connector_sign_out", { id }),
  tools: (id) => invoke("connector_tools", { id }),
  setPermission: (id, tool, permission) =>
    invoke("connector_set_permission", { id, tool, permission }),
  call: (id, tool, args, approved) => invoke("connector_call", { id, tool, args, approved }),
  audit: (id, limit) => invoke("connector_audit", { id, limit }),
  stderr: (id) => invoke("connector_stderr", { id })
};

// ---- the demo ----

interface DemoTool {
  name: string;
  title: string;
  description: string;
  inputSchema: Record<string, unknown>;
  readOnlyHint: boolean;
  run: (args: Record<string, unknown>) => string;
}

const DEMO_TOOLS: DemoTool[] = [
  {
    name: "list_events",
    title: "List events",
    description: "Lists the events on the calendar between two dates.",
    inputSchema: {
      type: "object",
      properties: { range: { type: "string", description: "today, tomorrow or a date" } },
      required: ["range"]
    },
    readOnlyHint: true,
    run: (a) => `2 events (${String(a["range"] ?? "today")}): 09:30 Stand-up, 14:00 Design review`
  },
  {
    name: "create_event",
    title: "Create event",
    description: "Adds an event to the calendar.",
    inputSchema: {
      type: "object",
      properties: { title: { type: "string" }, start: { type: "string" } },
      required: ["title", "start"]
    },
    readOnlyHint: false,
    run: (a) => `Created "${String(a["title"] ?? "")}" at ${String(a["start"] ?? "")}`
  }
];

export function createDemoBackend(): ConnectorsBackend {
  let nextId = 1;
  let nextAudit = 1;
  const configs = new Map<string, ConnectorConfig>();
  const status = new Map<string, ConnectorStatus>();
  const perms = new Map<string, ToolPermission>(); // `${id}/${tool}`
  const signedIn = new Set<string>();
  const audit: AuditEntry[] = [];

  const disconnected = (): ConnectorStatus => ({
    state: "disconnected",
    error: null,
    server: null,
    toolCount: 0,
    toolsChanged: false,
    resourceMetadata: null,
    signedIn: false,
    neededScope: null
  });
  const permOf = (id: string, tool: string) => perms.get(`${id}/${tool}`) ?? "ask";
  const need = (id: string) => {
    const c = configs.get(id);
    if (!c) throw new Error("There is no such connector.");
    return c;
  };

  return {
    isDemo: true,
    async list(): Promise<ConnectorEntry[]> {
      return [...configs.values()].map((config) => ({
        config,
        status: status.get(config.id) ?? disconnected()
      }));
    },
    async add(input: AddConnectorInput): Promise<ConnectorConfig> {
      const id = `demo-${nextId++}`;
      const config: ConnectorConfig = {
        id,
        name: input.name.trim(),
        kind: input.kind,
        command: input.kind === "stdio" ? (input.command ?? "").trim() : null,
        args: input.args ?? [],
        envNames: Object.keys(input.env ?? {}),
        url: input.kind === "http" ? (input.url ?? "").trim() : null,
        enabled: true,
        createdAt: Date.now(),
        oauth: {
          clientId: input.oauth?.clientId?.trim() || null,
          scope: input.oauth?.scope?.trim() || null,
          hasSecret: Boolean(input.oauth?.clientSecret)
        },
        apiKey: input.apiKey?.value?.trim()
          ? { hasKey: true, header: input.apiKey.header?.trim() || null }
          : { hasKey: false, header: null }
      };
      configs.set(id, config);
      status.set(id, disconnected());
      return config;
    },
    async update(id: string, patch: ConnectorPatch) {
      const c = need(id);
      configs.set(id, {
        ...c,
        ...(patch.name !== undefined ? { name: patch.name.trim() } : {}),
        ...(patch.enabled !== undefined ? { enabled: patch.enabled } : {}),
        ...(patch.args !== undefined ? { args: patch.args } : {}),
        ...(patch.env !== undefined ? { envNames: Object.keys(patch.env) } : {}),
        ...(patch.oauth !== undefined
          ? {
              oauth: {
                clientId:
                  patch.oauth.clientId === undefined
                    ? c.oauth.clientId
                    : patch.oauth.clientId.trim() || null,
                scope:
                  patch.oauth.scope === undefined
                    ? c.oauth.scope
                    : patch.oauth.scope.trim() || null,
                // an empty secret keeps the stored one
                hasSecret: patch.oauth.clientSecret ? true : c.oauth.hasSecret
              }
            }
          : {}),
        ...(patch.apiKey !== undefined
          ? {
              apiKey: patch.apiKey.clear
                ? { hasKey: false, header: null }
                : {
                    hasKey: patch.apiKey.value?.trim() ? true : c.apiKey.hasKey,
                    header:
                      patch.apiKey.header === undefined
                        ? c.apiKey.header
                        : patch.apiKey.header.trim() || null
                  }
            }
          : {})
      });
      if (patch.enabled === false) status.set(id, disconnected());
    },
    async remove(id: string) {
      need(id);
      configs.delete(id);
      status.delete(id);
    },
    async connect(id: string) {
      const c = need(id);
      if (!c.enabled) throw new Error("This connector is turned off.");
      const s: ConnectorStatus = {
        state: "ready",
        error: null,
        server: { name: "Demo calendar", version: "1.0.0", protocol: "2025-06-18" },
        toolCount: DEMO_TOOLS.length,
        toolsChanged: false,
        resourceMetadata: null,
        signedIn: signedIn.has(id),
        neededScope: null
      };
      // a server on a host that starts with "auth." asks for sign-in first, so the demo can show that state too
      if (c.kind === "http" && /^https?:\/\/auth\./.test(c.url ?? "") && !signedIn.has(id)) {
        const needs: ConnectorStatus = {
          ...disconnected(),
          state: "needs_auth",
          resourceMetadata: `${c.url}/.well-known/oauth-protected-resource`
        };
        status.set(id, needs);
        return needs;
      }
      status.set(id, s);
      return s;
    },
    async disconnect(id: string) {
      need(id);
      status.set(id, { ...disconnected(), signedIn: signedIn.has(id) });
    },
    async signIn(id: string) {
      need(id);
      signedIn.add(id);
      return this.connect(id);
    },
    async cancelSignIn() {},
    async signOut(id: string) {
      need(id);
      signedIn.delete(id);
      const s = { ...disconnected(), signedIn: false };
      status.set(id, s);
      return s;
    },
    async tools(id: string): Promise<ConnectorTool[]> {
      need(id);
      if (status.get(id)?.state !== "ready") return [];
      return DEMO_TOOLS.map((t) => ({
        name: t.name,
        title: t.title,
        description: t.description,
        inputSchema: t.inputSchema,
        readOnlyHint: t.readOnlyHint,
        permission: permOf(id, t.name),
        changed: false,
        available: true
      }));
    },
    async setPermission(id: string, tool: string, permission: ToolPermission) {
      need(id);
      perms.set(`${id}/${tool}`, permission);
    },
    async call(id, tool, args, approved): Promise<ToolCallResult> {
      need(id);
      const t = DEMO_TOOLS.find((x) => x.name === tool);
      if (!t) throw new Error(`The connector has no tool called ${tool}.`);
      const record = (outcome: ToolCallResult["outcome"], chars: number) =>
        audit.unshift({
          id: nextAudit++,
          connectorId: id,
          tool,
          argsJson: JSON.stringify(args),
          outcome,
          durationMs: 3,
          resultChars: chars,
          createdAt: Date.now()
        });
      const none = { content: [], structuredContent: null, isError: false, truncated: false };
      const p = permOf(id, tool);
      if (p === "deny") {
        record("denied", 0);
        return { outcome: "denied", ...none };
      }
      if (p === "ask" && !approved) {
        record("needs_approval", 0);
        return { outcome: "needs_approval", ...none };
      }
      const text = t.run(args);
      record("ok", text.length);
      return { outcome: "ok", ...none, content: [{ type: "text", text }] };
    },
    async audit(id: string | null, limit: number): Promise<AuditEntry[]> {
      return audit.filter((a) => id === null || a.connectorId === id).slice(0, limit);
    },
    async stderr() {
      return "";
    }
  };
}

let backend: ConnectorsBackend | null = null;

/** The desktop layer inside the app, the demo in a browser tab. */
export function getBackend(): ConnectorsBackend {
  backend ??= isDesktop() ? tauriBackend : createDemoBackend();
  return backend;
}

/** For tests: replace the backend. */
export function setBackend(b: ConnectorsBackend | null): void {
  backend = b;
}
