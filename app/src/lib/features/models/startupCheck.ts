import type { Stats } from "../../api/types.ts";

// First-launch check: can Dara run here at all? The engine is built for NVIDIA GPUs from Turing (GTX 16xx / RTX 20xx) on, which
// is compute capability 7.5. These turn what the engine and the stats report into one plain message, or null when all is well.

export const MIN_COMPUTE_CAPABILITY = 7.5;
export const DRIVER_URL = "https://www.nvidia.com/Download/index.aspx";

export type StartupProblemKind = "engine" | "no_gpu" | "old_gpu";

export interface StartupProblem {
  kind: StartupProblemKind;
  title: string;
  body: string;
  /** a page that helps, opened in the browser */
  helpUrl?: string;
}

/** "8.6" -> 8.6; null when the engine did not say. */
export function parseCapability(cc: string | undefined): number | null {
  if (!cc) return null;
  const n = Number(cc.trim());
  return Number.isFinite(n) && n > 0 ? n : null;
}

/**
 * `engineUp`: the engine answered. `stats`: its stats (null when they could not be read). Returns the one thing to tell the user.
 * An engine that is not up yet is not a problem until `waitedMs` shows it had time to start.
 */
export function startupProblem(input: {
  engineUp: boolean;
  stats: Stats | null;
  waitedMs: number;
}): StartupProblem | null {
  const { engineUp, stats, waitedMs } = input;
  if (!engineUp) {
    if (waitedMs < 15_000) return null;
    return {
      kind: "engine",
      title: "Dara's engine did not start",
      body:
        "The part of Dara that runs models is not answering. Close Dara and open it again. If this stays, " +
        "an antivirus program may have removed engine\\dara.exe from the Dara folder: restore it, or reinstall Dara."
    };
  }
  if (!stats) return null;
  const name = stats.gpu?.name?.trim();
  if (!name) {
    return {
      kind: "no_gpu",
      title: "No NVIDIA graphics card found",
      body:
        "Dara runs models on an NVIDIA graphics card (GTX 16xx, RTX 20xx or newer) with an up-to-date driver. " +
        "If you have one, install the latest driver and open Dara again.",
      helpUrl: DRIVER_URL
    };
  }
  const cc = parseCapability(stats.gpu?.compute_capability);
  if (cc !== null && cc < MIN_COMPUTE_CAPABILITY) {
    return {
      kind: "old_gpu",
      title: `${name} is too old for Dara`,
      body: "Dara needs a graphics card from the GTX 16xx or RTX 20xx generation or newer."
    };
  }
  return null;
}
