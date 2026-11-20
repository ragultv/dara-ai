import { useCallback, useEffect, useState } from "react";
import { apiClient } from "../../api/client.ts";
import { storeApi } from "../../api/store-api.ts";
import { ApiError } from "../../api/errors.ts";
import type { DbStoreInfo, Stats, VersionInfo } from "../../api/types.ts";

export interface ServerFacts {
  version: VersionInfo | null;
  stats: Stats | null;
  store: DbStoreInfo | null;
  /** Why the database could not be read (503 or 404 mean the server runs without one). */
  storeError: string | null;
  isLoading: boolean;
  refresh: () => void;
}

function describe(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 503 || err.status === 404) {
      return `The database is unavailable on this engine (${err.message}).`;
    }
    return err.message;
  }
  return err instanceof Error ? err.message : "Unknown error.";
}

/** One-shot reads of version, stats and database info, repeated on demand. No polling. */
export function useServerFacts(): ServerFacts {
  const [version, setVersion] = useState<VersionInfo | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [store, setStore] = useState<DbStoreInfo | null>(null);
  const [storeError, setStoreError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    setIsLoading(true);
    void Promise.allSettled([
      apiClient.getVersion(signal),
      apiClient.getStats(signal),
      storeApi.getStoreInfo(signal)
    ]).then(([v, s, d]) => {
      if (signal.aborted) return;
      setVersion(v.status === "fulfilled" ? v.value : null);
      setStats(s.status === "fulfilled" ? s.value : null);
      if (d.status === "fulfilled") {
        setStore(d.value);
        setStoreError(null);
      } else {
        setStore(null);
        setStoreError(describe(d.reason));
      }
      setIsLoading(false);
    });
    return () => controller.abort();
  }, [tick]);

  const refresh = useCallback(() => setTick((n) => n + 1), []);
  return { version, stats, store, storeError, isLoading, refresh };
}
