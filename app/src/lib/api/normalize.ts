import type { OptionsResponse, Stats } from "./types.ts";

// The engine and the app are built separately, and the engine leaves fields out or sends null when it has nothing to report (no model loaded,
// no GPU sensors, no downloads). These make the answers safe to read before the screens see them.

/** `GET /dara/stats` with nulls where the app expects "missing", and lists where it expects a list. */
export function normalizeStats(raw: Stats): Stats {
  const s: Stats = { ...raw };
  if (s.gpu) {
    // a sensor the driver cannot read may arrive as null: the panels treat only `undefined` as "not reported"
    const g = { ...s.gpu } as Record<string, unknown>;
    for (const k of Object.keys(g)) if (g[k] === null) delete g[k];
    s.gpu = g as unknown as NonNullable<Stats["gpu"]>;
  }
  if (!Array.isArray(s.downloads)) s.downloads = [];
  if (
    s.context &&
    (typeof s.context.capacity !== "number" || typeof s.context.held_tokens !== "number")
  )
    s.context = null;
  if (s.disk && typeof s.disk !== "object") s.disk = null;
  return s;
}

/**
 * Why an options answer cannot be shown, or null when it can. A missing piece means the engine and the app are different versions; saying so
 * beats a page that fails to draw.
 */
export function optionsProblem(r: unknown): string | null {
  const o = r as Partial<OptionsResponse> | null;
  if (!o || typeof o !== "object") return "The engine's answer about options could not be read.";
  const missing: string[] = [];
  if (!o.machine || typeof o.machine !== "object") missing.push("machine");
  if (!Array.isArray(o.formats)) missing.push("formats");
  if (!Array.isArray(o.options)) missing.push("options");
  if (!o.recommended || typeof o.recommended !== "object") missing.push("recommended");
  if (!o.calibration || typeof o.calibration !== "object") missing.push("calibration");
  if (!o.vision || typeof o.vision !== "object") missing.push("vision");
  if (missing.length === 0) return null;
  return `The engine's answer about options is missing ${missing.join(", ")}. The app and the engine are probably different versions: update both, then reload.`;
}

/** An options answer with the optional list present. */
export function normalizeOptions(o: OptionsResponse): OptionsResponse {
  return Array.isArray(o.notes) ? o : { ...o, notes: [] };
}
