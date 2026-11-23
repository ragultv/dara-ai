import { builtinClient, normalizeUrl, type CatalogEntry } from "../../connectors/catalog.ts";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";

/**
 * Adds a catalog connector with the first (least access) level as its scope. A server that wants an API key needs one here: it is
 * saved with the connector. Returns an error message, or null when it was added.
 */
export async function addFromCatalog(entry: CatalogEntry, apiKey?: string): Promise<string | null> {
  if (entry.status !== "available" || !entry.url) return "This connector is not available yet.";
  if (entry.registers === "api-key") {
    if (!apiKey?.trim()) return "Paste the API key first.";
    return useConnectorsStore.getState().add({
      name: entry.name,
      kind: "http",
      url: entry.url,
      apiKey: { value: apiKey, ...(entry.keyHeader ? { header: entry.keyHeader } : {}) }
    });
  }
  const scope = entry.access?.[0]?.scope;
  const client = builtinClient(entry);
  const problem = await useConnectorsStore.getState().add({
    name: entry.name,
    kind: "http",
    url: entry.url,
    ...(scope || client ? { oauth: { ...(scope ? { scope } : {}), ...(client ?? {}) } } : {})
  });
  // an open server has nothing to set up, so it connects as soon as it is added
  if (!problem && entry.registers === "none") {
    const added = useConnectorsStore
      .getState()
      .entries.find((e) => normalizeUrl(e.config.url) === normalizeUrl(entry.url));
    if (added) void useConnectorsStore.getState().connect(added.config.id);
  }
  return problem;
}
