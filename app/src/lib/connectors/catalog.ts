// The connectors Dara suggests, shown as apps in Settings > Connectors. Every address here was checked against the live server (it answers an
// MCP request and, for the ones that sign in, publishes its OAuth metadata); the "registers" flag says what the server's metadata offered:
//   "automatic": the server registers Dara by itself when you sign in (Dynamic Client Registration);
//   "own-client": the server has no registration, so you create a client ID in its console (Google does this for every app).
// A connector added from here is an ordinary connector: it appears in the list under its name and can be removed.

export type CatalogCategory =
  "Email & messages" | "Events & tickets" | "Files & documents" | "Project management" | "Design";

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
  /** one line on the card */
  tagline: string;
  /** a paragraph on the detail page */
  about: string;
  url: string;
  categories: CatalogCategory[];
  registers: "automatic" | "own-client";
  /** how to get the client ID, for "own-client" servers */
  clientHelp?: string[];
  /** servers that need a scope choose it here; the first level is the default */
  access?: AccessLevel[];
  status: "available" | "soon";
}

const GOOGLE_CLIENT_HELP = [
  "In the Google Cloud console, create a project (or pick one) and enable this API for it.",
  'Under APIs & Services > Credentials, create an OAuth client ID of type "Desktop app".',
  "Paste its client ID (and secret, if shown) below. Google treats a desktop app's secret as not confidential.",
  "While your Google app is in testing mode only the test users you list can sign in, and its sign-in expires after about 7 days."
];

export const CATALOG: CatalogEntry[] = [
  {
    id: "google-calendar",
    name: "Google Calendar",
    publisher: "Google",
    tagline: "Read your calendars and events, and change them when you allow it",
    about:
      "Lists your calendars and the events in a date range, searches events, finds free time, and (only at the Read and write level) creates, updates and deletes events. Dara asks before using any tool unless you allow it.",
    url: "https://calendarmcp.googleapis.com/mcp/v1",
    categories: ["Events & tickets"],
    registers: "own-client",
    clientHelp: GOOGLE_CLIENT_HELP,
    access: [
      {
        id: "read",
        label: "Read only",
        description: "See your calendars and events. Nothing can be changed.",
        scope:
          "https://www.googleapis.com/auth/calendar.readonly https://www.googleapis.com/auth/calendar.calendarlist.readonly"
      },
      {
        id: "write",
        label: "Read and write",
        description: "Also create, update, delete and answer events.",
        scope: "https://www.googleapis.com/auth/calendar"
      }
    ],
    status: "available"
  },
  {
    id: "gmail",
    name: "Gmail",
    publisher: "Google",
    tagline: "Search and read your mail, and write drafts when you allow it",
    about:
      "Searches threads, reads messages and drafts, lists labels, and (at the Read and organize level) writes drafts, labels, trashes and marks spam. It cannot send mail. Gmail's read scopes are restricted by Google, which matters for a public release of Dara.",
    url: "https://gmailmcp.googleapis.com/mcp/v1",
    categories: ["Email & messages"],
    registers: "own-client",
    clientHelp: GOOGLE_CLIENT_HELP,
    access: [
      {
        id: "read",
        label: "Read only",
        description: "Search and read mail. Nothing can be changed.",
        scope: "https://www.googleapis.com/auth/gmail.readonly"
      },
      {
        id: "write",
        label: "Read and organize",
        description: "Also write drafts, label, trash and mark spam. It cannot send.",
        scope:
          "https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/gmail.compose"
      }
    ],
    status: "available"
  },
  {
    id: "canva",
    name: "Canva",
    publisher: "Canva",
    tagline: "Search, create, autofill and export Canva designs",
    about:
      "Works with your Canva designs, folders and brand templates. The server registers Dara itself when you sign in, so there is nothing to set up.",
    url: "https://mcp.canva.com/mcp",
    categories: ["Design", "Files & documents"],
    registers: "automatic",
    status: "available"
  },
  {
    id: "notion",
    name: "Notion",
    publisher: "Notion",
    tagline: "Search, read and update your Notion workspace",
    about:
      "Searches and reads pages and databases and can change them. The server registers Dara itself when you sign in.",
    url: "https://mcp.notion.com/mcp",
    categories: ["Files & documents"],
    registers: "automatic",
    status: "available"
  },
  {
    id: "linear",
    name: "Linear",
    publisher: "Linear",
    tagline: "Manage issues, projects and team workflows in Linear",
    about:
      "Finds and updates issues, projects and cycles. The server registers Dara itself when you sign in.",
    url: "https://mcp.linear.app/mcp",
    categories: ["Project management"],
    registers: "automatic",
    status: "available"
  },
  {
    id: "outlook",
    name: "Outlook mail and calendar",
    publisher: "Dara (built in)",
    tagline: "Your Outlook mail and calendar through Microsoft Graph",
    about:
      "Microsoft does not offer a hosted MCP server for Outlook that Dara can rely on, so this will be a built-in connector that reads through Microsoft Graph with read-only permissions. Not available yet.",
    url: "",
    categories: ["Email & messages", "Events & tickets"],
    registers: "automatic",
    status: "soon"
  }
];

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
