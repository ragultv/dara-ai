import { create } from "zustand";
import { apiClient } from "../api/client.ts";
import type { Stats, VersionInfo } from "../api/types.ts";

/** One reading per poll. `null` means "the server did not report it" (drawn as a gap, never as zero). */
export interface MonitorSample {
  t: number;
  /** Live decode speed; only while the engine is generating. */
  tps: number | null;
  vramUsedMb: number | null;
  gpuUtilPercent: number | null;
}

export const MAX_SAMPLES = 120;
export const POLL_INTERVAL_MS = 1000;

interface MonitorState {
  stats: Stats | null;
  version: VersionInfo | null;
  samples: MonitorSample[];
  /** Message of the last failed poll; cleared by the next success. */
  error: string | null;
  isPolling: boolean;
  lastUpdated: number | null;
  startPolling: () => void;
  stopPolling: () => void;
  refresh: () => Promise<void>;
  reset: () => void;
}

/** Pure: turns a stats response into one history sample. Exported for tests. */
export function sampleFromStats(stats: Stats, now: number): MonitorSample {
  const engine = stats.engine;
  const live =
    engine.busy && engine.live_tokens_per_second > 0 ? engine.live_tokens_per_second : null;
  const gpu = stats.gpu;
  const hasGpu = gpu !== undefined && gpu.nvml;
  return {
    t: now,
    tps: live,
    vramUsedMb: hasGpu ? gpu.vram_used_mb : null,
    gpuUtilPercent:
      hasGpu && typeof gpu.utilization_percent === "number" ? gpu.utilization_percent : null
  };
}

/** Pure ring buffer: keeps the newest `max` samples. */
export function pushSample(
  samples: MonitorSample[],
  sample: MonitorSample,
  max = MAX_SAMPLES
): MonitorSample[] {
  const next = samples.length >= max ? samples.slice(samples.length - max + 1) : samples.slice();
  next.push(sample);
  return next;
}

let timer: ReturnType<typeof setInterval> | null = null;
let controller: AbortController | null = null;
let inFlight = false;
let visibilityHandler: (() => void) | null = null;

function pageIsVisible(): boolean {
  return typeof document === "undefined" || document.visibilityState !== "hidden";
}

function clearTimer(): void {
  if (timer !== null) {
    clearInterval(timer);
    timer = null;
  }
}

export const useMonitorStore = create<MonitorState>((set, get) => ({
  stats: null,
  version: null,
  samples: [],
  error: null,
  isPolling: false,
  lastUpdated: null,

  refresh: async () => {
    if (inFlight) return;
    inFlight = true;
    const ctl = new AbortController();
    controller = ctl;
    try {
      const stats = await apiClient.getStats(ctl.signal);
      if (ctl.signal.aborted) return;
      const now = Date.now();
      set((s) => ({
        stats,
        error: null,
        lastUpdated: now,
        samples: pushSample(s.samples, sampleFromStats(stats, now))
      }));
      if (get().version === null) {
        void apiClient
          .getVersion(ctl.signal)
          .then((version) => {
            if (!ctl.signal.aborted) set({ version });
          })
          .catch(() => undefined);
      }
    } catch (err: unknown) {
      if (ctl.signal.aborted) return;
      set({
        error: err instanceof Error ? err.message : "Could not read the engine's statistics."
      });
    } finally {
      inFlight = false;
      if (controller === ctl) controller = null;
    }
  },

  startPolling: () => {
    if (get().isPolling) return;
    set({ isPolling: true });

    const begin = (): void => {
      clearTimer();
      void get().refresh();
      timer = setInterval(() => void get().refresh(), POLL_INTERVAL_MS);
    };

    // No timers and no requests while the window is hidden or minimized.
    visibilityHandler = () => {
      if (pageIsVisible()) begin();
      else {
        clearTimer();
        controller?.abort();
      }
    };
    if (typeof document !== "undefined")
      document.addEventListener("visibilitychange", visibilityHandler);
    if (pageIsVisible()) begin();
  },

  stopPolling: () => {
    clearTimer();
    controller?.abort();
    controller = null;
    inFlight = false;
    if (visibilityHandler && typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", visibilityHandler);
    }
    visibilityHandler = null;
    set({ isPolling: false });
  },

  reset: () => set({ stats: null, samples: [], error: null, lastUpdated: null })
}));
