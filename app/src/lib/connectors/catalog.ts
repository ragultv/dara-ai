// The connectors Dara suggests, shown as apps in Settings > Connectors. Every address here was checked against the live server (it answers an
// MCP request and, for the ones that sign in, publishes its OAuth metadata); the "registers" flag says what the server's metadata offered:
//   "automatic": the server registers Dara by itself when you sign in (Dynamic Client Registration);
//   "own-client": the server has no registration, so you create a client ID in its console (Google does this for every app);
//   "api-key": the server has no sign-in at all: you paste an API key (kept in the Windows credential store, sent with every request);
//   "none": the server is open: no sign-in and no key (public documentation or data).
// A connector added from here is an ordinary connector: it appears in the list under its name and can be removed.

import data from "./catalog.json";

export type CatalogCategory =
  | "Email & messages"
  | "Events & tickets"
  | "Files & documents"
  | "Project management"
  | "Design"
  | "Web & research"
  | "Developer tools";

export interface AccessLevel {
  id: "read" | "write";
  label: string;
  /** what this level lets the connector do, in a sentence */
  description: string;
  /** the OAuth scope sent when signing in at this level */
  scope: string;
}

export interface CatalogEntry {
  id: string;
  name: string;
  publisher: string;
  /** the logo image (bundled from src/assets/connectors) */
  logo?: string;
  /** one line on the card */
  tagline: string;
  /** a paragraph on the detail page */
  about: string;
  url: string;
  categories: CatalogCategory[];
  registers: "automatic" | "own-client" | "api-key" | "none";
  /** how to get the client ID, for "own-client" servers */
  clientHelp?: string[];
  /** for "api-key" servers (no sign-in: the user pastes a key): the steps to get one, and the page where the key is made */
  keyHelp?: string[];
  keyUrl?: string;
  /** the header the key goes in; left out, it is sent as `Authorization: Bearer <key>` */
  keyHeader?: string;
  /** servers that need a scope choose it here; the first level is the default */
  access?: AccessLevel[];
  status: "available" | "soon";
}

// The connector texts live in catalog.json and the logos in src/assets/connectors/, so they can be edited without touching code.
interface CatalogData {
  googleClientHelp: string[];
  entries: Array<
    Omit<CatalogEntry, "clientHelp" | "logo"> & { logo?: string; clientHelp?: "google" | string[] }
  >;
}

const LOGOS = import.meta.glob("../../assets/connectors/*.{svg,png}", {
  eager: true,
  query: "?url",
  import: "default"
}) as Record<string, string>;

export const CATALOG: CatalogEntry[] = (data as CatalogData).entries.map(
  ({ logo, clientHelp, ...rest }) => {
    const url = logo ? LOGOS[`../../assets/connectors/${logo}`] : undefined;
    const help = clientHelp === "google" ? (data as CatalogData).googleClientHelp : clientHelp;
    return {
      ...rest,
      ...(url ? { logo: url } : {}),
      ...(help ? { clientHelp: help } : {})
    } as CatalogEntry;
  }
);

/** The logo of the catalog connector with this name, if it has one. */
export function logoForName(name: string | null | undefined): string | undefined {
  const n = (name ?? "").trim().toLowerCase();
  return n ? CATALOG.find((e) => e.name.toLowerCase() === n)?.logo : undefined;
}

/**
 * Dara's own Google sign-in client, if this build has one. Google gives each APP a client ID, not each user: the developer creates one
 * (Desktop app type) and every user signs in with their own Google account through it, so nobody has to create a client of their own.
 * A desktop app cannot keep a secret, and Google's rules for installed apps say the secret there is not confidential (PKCE protects the
 * sign-in), so it is read from the build (VITE_DARA_GOOGLE_CLIENT_ID and VITE_DARA_GOOGLE_CLIENT_SECRET) and never committed.
 */
export function builtinClient(entry: Pick<CatalogEntry, "registers" | "url">): {
  clientId: string;
  clientSecret?: string;
} | null {
  if (
    entry.registers !== "own-client" ||
    !/^https:\/\/[a-z]+mcp\.googleapis\.com\//.test(entry.url)
  )
    return null;
  const env = import.meta.env as Record<string, string | undefined>;
  const clientId = env["VITE_DARA_GOOGLE_CLIENT_ID"]?.trim();
  if (!clientId) return null;
  const clientSecret = env["VITE_DARA_GOOGLE_CLIENT_SECRET"]?.trim();
  return { clientId, ...(clientSecret ? { clientSecret } : {}) };
}

export function normalizeUrl(u: string | null | undefined): string {
  return (u ?? "").trim().replace(/\/+$/, "").toLowerCase();
}

/** The catalog entry a connector was added from, found by its server address. */
export function entryForUrl(url: string | null | undefined): CatalogEntry | undefined {
  const n = normalizeUrl(url);
  return n ? CATALOG.find((e) => e.url && normalizeUrl(e.url) === n) : undefined;
}

/** "outlook_calendar_search" becomes "Outlook calendar search". */
export function humanizeToolName(name: string): string {
  const s = name
    .replace(/[_-]+/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .trim();
  return s ? s[0]!.toUpperCase() + s.slice(1) : name;
}

/** The host a connector sends data to, for "Can send data to ...". */
export function hostOf(url: string | null | undefined): string {
  try {
    return new URL(url ?? "").host;
  } catch {
    return "";
  }
}
