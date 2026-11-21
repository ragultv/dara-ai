// What is stored about tool calls (docs/CHAT_TOOLS.md): per call details in the assistant message's `stats.tool_calls`, and the
// chat's connector choice and "always allow" list in the conversation's `params`. The engine keeps both as opaque JSON
// (at most 64 KB each), so everything here is shaped to stay well under that.

import type { CallDecision, CallState, CallStatus } from "./types.ts";

/** One finished call as stored with the assistant message that made it. */
export interface ToolCallRecord {
  id: string;
  name: string;
  connectorId: string | null;
  connectorName: string;
  tool: string;
  /** left out when the stats would not fit otherwise; `{_truncated: true, preview}` when the arguments were too long */
  arguments?: Record<string, unknown> | undefined;
  status: CallStatus;
  decision?: CallDecision | undefined;
  startedAt?: number | undefined;
  durationMs?: number | undefined;
}

/** the engine refuses a stats or params object above 64 KB; keep a margin */
export const STATS_LIMIT_BYTES = 63 * 1024;
export const MAX_ARGS_CHARS = 2000;

const DECISIONS: readonly CallDecision[] = [
  "auto",
  "approved_once",
  "approved_always",
  "denied_by_user",
  "blocked",
  "skipped"
];
const STATUSES: readonly CallStatus[] = [
  "waiting_approval",
  "running",
  "ok",
  "error",
  "denied",
  "skipped"
];

const bytes = (v: unknown): number => new TextEncoder().encode(JSON.stringify(v)).length;

/** The arguments as stored: unchanged when short, else a marked preview of their JSON. */
export function capArguments(args: Record<string, unknown>): Record<string, unknown> {
  let s: string;
  try {
    s = JSON.stringify(args);
  } catch {
    return { _truncated: true, preview: "" };
  }
  if (s.length <= MAX_ARGS_CHARS) return args;
  return { _truncated: true, preview: s.slice(0, MAX_ARGS_CHARS - 40) };
}

export function recordFromCall(c: CallState): ToolCallRecord {
  return {
    id: c.id,
    name: c.name,
    connectorId: c.connectorId,
    connectorName: c.connectorName,
    tool: c.tool,
    arguments: capArguments(c.arguments),
    status: c.status,
    ...(c.decision ? { decision: c.decision } : {}),
    ...(c.startedAt !== undefined ? { startedAt: c.startedAt } : {}),
    ...(c.durationMs !== undefined ? { durationMs: c.durationMs } : {})
  };
}

/**
 * The stats blob of a message: the engine's numbers, the thinking and the tool calls, kept under the engine's limit.
 * When it does not fit, the calls' arguments go first, then the last calls.
 */
export function buildMessageStats(
  base: Record<string, unknown> | null | undefined,
  reasoning: string | undefined,
  records: ToolCallRecord[] | undefined,
  limit = STATS_LIMIT_BYTES
): Record<string, unknown> | null {
  const rest: Record<string, unknown> = { ...(base ?? {}) };
  delete rest.reasoning;
  delete rest.tool_calls;
  if (reasoning) rest.reasoning = reasoning;
  if (!records || records.length === 0) return Object.keys(rest).length ? rest : null;

  const full = {
    ...rest,
    tool_calls: records.map((r) => ({ ...r, arguments: r.arguments && capArguments(r.arguments) }))
  };
  if (bytes(full) <= limit) return full;

  const bare = records.map(({ arguments: _dropped, ...r }) => r);
  for (let n = bare.length; n > 0; n--) {
    const s = { ...rest, tool_calls: bare.slice(0, n) };
    if (bytes(s) <= limit) return s;
  }
  return Object.keys(rest).length ? rest : null;
}

const str = (v: unknown): string | undefined => (typeof v === "string" ? v : undefined);
const num = (v: unknown): number | undefined =>
  typeof v === "number" && Number.isFinite(v) ? v : undefined;

/** Reads `stats.tool_calls` back from the database; anything malformed is skipped. */
export function parseToolCallRecords(raw: unknown): ToolCallRecord[] | undefined {
  if (!Array.isArray(raw)) return undefined;
  const out: ToolCallRecord[] = [];
  for (const x of raw) {
    if (!x || typeof x !== "object") continue;
    const o = x as Record<string, unknown>;
    const id = str(o.id);
    const status = STATUSES.find((s) => s === o.status);
    if (!id || !status) continue;
    const decision = DECISIONS.find((d) => d === o.decision);
    const args =
      o.arguments && typeof o.arguments === "object" && !Array.isArray(o.arguments)
        ? (o.arguments as Record<string, unknown>)
        : undefined;
    const startedAt = num(o.startedAt);
    const durationMs = num(o.durationMs);
    out.push({
      id,
      name: str(o.name) ?? "",
      connectorId: str(o.connectorId) ?? null,
      connectorName: str(o.connectorName) ?? "",
      tool: str(o.tool) ?? str(o.name) ?? "",
      ...(args ? { arguments: args } : {}),
      status,
      ...(decision ? { decision } : {}),
      ...(startedAt !== undefined ? { startedAt } : {}),
      ...(durationMs !== undefined ? { durationMs } : {})
    });
  }
  return out.length ? out : undefined;
}

// ---- conversation params ---------------------------------------------------------------------------------------------------------------------------

/** What a conversation stores about connectors: the ones on in it, and the tools allowed for the rest of it. */
export interface ToolParams {
  connectors: string[];
  /** [connectorId, tool] pairs */
  allowed_tools: Array<[string, string]>;
}

const SEP = "\n"; // allowKey's separator (useChatToolsStore)

export function toToolParams(enabled: readonly string[], allowKeys: readonly string[]): ToolParams {
  const allowed_tools: Array<[string, string]> = [];
  for (const k of allowKeys) {
    const i = k.indexOf(SEP);
    if (i > 0) allowed_tools.push([k.slice(0, i), k.slice(i + 1)]);
  }
  return { connectors: [...enabled], allowed_tools };
}

/** Reads the connector choice back from params. `enabled` is null when the params never had one (an older chat). */
export function fromToolParams(params: Record<string, unknown> | null | undefined): {
  enabled: string[] | null;
  allowKeys: string[];
} {
  const p = params ?? {};
  const enabled = Array.isArray(p.connectors)
    ? p.connectors.filter((x): x is string => typeof x === "string")
    : null;
  const allowKeys: string[] = [];
  if (Array.isArray(p.allowed_tools)) {
    for (const pair of p.allowed_tools) {
      if (
        Array.isArray(pair) &&
        pair.length === 2 &&
        typeof pair[0] === "string" &&
        typeof pair[1] === "string" &&
        pair[0]
      )
        allowKeys.push(`${pair[0]}${SEP}${pair[1]}`);
    }
  }
  return { enabled, allowKeys };
}
