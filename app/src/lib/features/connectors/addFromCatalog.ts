import type { CatalogEntry } from "../../connectors/catalog.ts";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";

/** Adds a catalog connector with the first (least access) level as its scope. Returns an error message, or null when it was added. */
export async function addFromCatalog(entry: CatalogEntry): Promise<string | null> {
  if (entry.status !== "available" || !entry.url) return "This connector is not available yet.";
  const scope = entry.access?.[0]?.scope;
  return useConnectorsStore.getState().add({
    name: entry.name,
    kind: "http",
    url: entry.url,
    ...(scope ? { oauth: { scope } } : {})
  });
}
