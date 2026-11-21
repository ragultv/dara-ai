import type { AddConnectorInput } from "./types.ts";

// The same rules the desktop layer enforces, so a form can say what is wrong before anything is sent. The desktop layer checks again.

export const MAX_CONNECTORS = 32;

/** A server address is acceptable when it is https, or when it points at this computer. */
export function checkServerUrl(raw: string): string | null {
  let u: URL;
  try {
    u = new URL(raw.trim());
  } catch {
    return "That is not a web address. It looks like https://example.com/mcp";
  }
  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(u.hostname);
  if (u.protocol === "https:") return null;
  if (u.protocol === "http:" && loopback) return null;
  return u.protocol === "http:"
    ? "Only https addresses are accepted (plain http is allowed only for this computer)."
    : "Only https addresses are accepted.";
}

/** One argument per line; blank lines are dropped. A program's arguments are never split on spaces, so a path with spaces stays whole. */
export function parseArgs(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
}

/** NAME=value per line. Returns the pairs and the first line that could not be read. */
export function parseEnv(text: string): { env: Record<string, string>; bad: string | null } {
  const env: Record<string, string> = {};
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const eq = line.indexOf("=");
    const name = eq > 0 ? line.slice(0, eq).trim() : "";
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) return { env, bad: line.split("=")[0] || line };
    env[name] = line.slice(eq + 1);
  }
  return { env, bad: null };
}

/** An error message for a new connector, or null when it can be added. */
export function validateNewConnector(input: AddConnectorInput, existing: number): string | null {
  const name = input.name.trim();
  if (!name) return "Give the connector a name.";
  if (name.length > 60) return "The name is longer than 60 characters.";
  if (existing >= MAX_CONNECTORS) return `At most ${MAX_CONNECTORS} connectors can be added.`;
  if (input.kind === "stdio") {
    const c = (input.command ?? "").trim();
    if (!c) return "Enter the program to run.";
    if (/[\u0000\r\n]/.test(c)) return "The program name has a line break in it.";
    return null;
  }
  const bad = checkServerUrl(input.url ?? "");
  if (bad) return bad;
  const o = input.oauth;
  if (o?.clientSecret && !o.clientId?.trim()) return "A client secret needs a client ID.";
  if (o?.scope && /[\r\n\u0000]/.test(o.scope)) return "The scope must be on one line.";
  return null;
}

/** A starting set of arguments for a tool, from its input schema: every required property, empty. */
export function skeletonArgs(schema: Record<string, unknown>): Record<string, unknown> {
  const props = (schema["properties"] ?? {}) as Record<
    string,
    { type?: unknown; default?: unknown }
  >;
  const required = Array.isArray(schema["required"]) ? (schema["required"] as string[]) : [];
  const out: Record<string, unknown> = {};
  for (const key of required) {
    const p = props[key] ?? {};
    out[key] =
      p.default !== undefined
        ? p.default
        : p.type === "number" || p.type === "integer"
          ? 0
          : p.type === "boolean"
            ? false
            : p.type === "array"
              ? []
              : p.type === "object"
                ? {}
                : "";
  }
  return out;
}
