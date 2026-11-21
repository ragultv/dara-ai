import { CATALOG, entryForUrl, hostOf, type CatalogEntry } from "../../connectors/catalog.ts";
import type { ConnectorEntry } from "../../connectors/types.ts";

/** One tile in Settings > Connectors: a catalog connector (added or not) or one the user added by hand. */
export interface ConnectorItem {
  /** the catalog id, or `c:<connector id>` for a connector added by hand */
  key: string;
  name: string;
  publisher: string;
  tagline: string;
  catalog?: CatalogEntry;
  connector?: ConnectorEntry;
  soon: boolean;
}

export function buildItems(entries: readonly ConnectorEntry[]): ConnectorItem[] {
  const byCatalog = new Map<string, ConnectorEntry>();
  const custom: ConnectorEntry[] = [];
  for (const e of entries) {
    const hit = entryForUrl(e.config.url);
    if (hit && !byCatalog.has(hit.id)) byCatalog.set(hit.id, e);
    else custom.push(e);
  }
  const fromCatalog = CATALOG.map<ConnectorItem>((c) => {
    const connector = byCatalog.get(c.id);
    return {
      key: c.id,
      name: c.name,
      publisher: c.publisher,
      tagline: c.tagline,
      catalog: c,
      ...(connector ? { connector } : {}),
      soon: c.status === "soon"
    };
  });
  const byHand = custom.map<ConnectorItem>((e) => ({
    key: `c:${e.config.id}`,
    name: e.config.name,
    publisher:
      e.config.kind === "stdio"
        ? "A program on this computer"
        : hostOf(e.config.url) || "Your server",
    tagline:
      e.config.kind === "stdio"
        ? [e.config.command, ...e.config.args].join(" ")
        : (e.config.url ?? ""),
    connector: e,
    soon: false
  }));
  return [...byHand, ...fromCatalog];
}

/** A short status for a card or header. */
export function statusText(e: ConnectorEntry | undefined): string {
  if (!e) return "";
  const s = e.status;
  switch (s.state) {
    case "ready":
      return `Connected${s.toolCount ? ` · ${s.toolCount} ${s.toolCount === 1 ? "tool" : "tools"}` : ""}`;
    case "connecting":
      return "Connecting...";
    case "needs_auth":
      return s.neededScope ? "Needs more access" : "Needs sign-in";
    case "error":
      return "Could not connect";
    default:
      return e.config.enabled ? "Not connected" : "Turned off";
  }
}
