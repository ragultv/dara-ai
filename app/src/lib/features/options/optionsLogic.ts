// Pure helpers for the Options screen. They pick and re-word what /dara/models/{id}/options returned;
// they never compute speeds, context sizes or quality.
import type {
  FormatInfo,
  KvFormat,
  ModelInfo,
  OptionEntry,
  OptionsResponse,
  Placement,
  QualityLabel,
  Speed
} from "../../api/types.ts";
import { formatContext, formatMb, formatTps } from "../../utils/format.ts";

export type BadgeVariant = "primary" | "success" | "warning" | "danger" | "neutral";

/** The option entry for a K/V format and Flash choice, or undefined when the server did not offer it. */
export function findOption(
  options: OptionEntry[],
  kv: KvFormat,
  flash: boolean
): OptionEntry | undefined {
  return options.find((o) => o.kv === kv && o.flash === flash);
}

/** Quality label to badge color, icon and word. Color is never the only signal. */
export function qualityPresentation(label: QualityLabel): {
  variant: BadgeVariant;
  icon: string;
  word: string;
} {
  switch (label) {
    case "high":
      return { variant: "success", icon: "check", word: "High" };
    case "medium":
      return { variant: "primary", icon: "info", word: "Medium" };
    case "experimental":
      return { variant: "warning", icon: "alert-triangle", word: "Experimental" };
    default:
      return { variant: "neutral", icon: "chevron-down", word: "Lower" };
  }
}

export interface FeatureAvailability {
  /** The toggle can be switched on right now. */
  canEnable: boolean;
  /** Why it cannot, in plain words (null when it can). */
  reason: string | null;
  /** The file can be downloaded to enable it. */
  canInstall: boolean;
}

/** Reads the top-level `flash` object of the options response, which the types leave open. */
export function readFlashInfo(
  value: unknown
): { available: boolean; installed: boolean; note: string } | null {
  if (typeof value !== "object" || value === null) return null;
  const v = value as Record<string, unknown>;
  if (typeof v["available"] !== "boolean") return null;
  return {
    available: v["available"],
    installed: v["installed"] === true,
    note: typeof v["note"] === "string" ? v["note"] : ""
  };
}

export function flashAvailability(
  options: OptionsResponse | null,
  model: ModelInfo | undefined
): FeatureAvailability {
  const fromOptions = options ? readFlashInfo(options.flash) : null;
  const available = fromOptions ? fromOptions.available : (model?.dara.flash.available ?? false);
  const installed = model ? model.dara.flash.installed : (fromOptions?.installed ?? false);
  if (!available) {
    const note = fromOptions?.note ?? model?.dara.flash.note ?? "";
    return {
      canEnable: false,
      canInstall: false,
      reason: note || "Flash is not available for this model."
    };
  }
  if (!installed) {
    return {
      canEnable: false,
      canInstall: true,
      reason: "The Flash drafter is not installed. Install it to turn Flash on."
    };
  }
  return { canEnable: true, canInstall: false, reason: null };
}

export function visionAvailability(model: ModelInfo | undefined): FeatureAvailability {
  if (!model)
    return { canEnable: false, canInstall: false, reason: "Model information is not available." };
  if (!model.dara.vision.available) {
    return {
      canEnable: false,
      canInstall: false,
      reason: model.dara.vision.note || "Image support is not available for this model."
    };
  }
  if (!model.dara.vision.installed) {
    return {
      canEnable: false,
      canInstall: true,
      reason: "The image support files are not installed. Install them to turn it on."
    };
  }
  return { canEnable: true, canInstall: false, reason: null };
}

export interface SpeedText {
  main: string;
  detail: string | null;
  isMeasured: boolean;
}

/**
 * Measured speeds as the server gave them, or "Not measured yet". A setup that does not fit this GPU comes with no speed at all (the
 * server answers `{ ok: false, error }` for it), which shows as a dash.
 */
export function speedText(speed: Speed | undefined): SpeedText {
  if (!speed) return { main: "—", detail: null, isMeasured: false };
  if (speed.source !== "measured")
    return { main: "Not measured yet", detail: null, isMeasured: false };
  const parts: string[] = [];
  if (speed.decode_tps_full_estimate !== undefined) {
    parts.push(`about ${formatTps(speed.decode_tps_full_estimate)} when the context is full`);
  }
  if (speed.power === "battery") parts.push("measured on battery");
  return {
    main: formatTps(speed.decode_tps_start),
    detail: parts.length > 0 ? parts.join(", ") : null,
    isMeasured: true
  };
}

/** Where the model runs for one placement; every number comes from the placement itself. */
export function placementText(p: Placement): string {
  if (!p.ok) return "Does not fit";
  if (p.fully_on_gpu) return "Fully on the GPU";
  const layers = `${p.layers_gpu} of ${p.layers} layers on the GPU`;
  return p.head === "cpu" ? `${layers}, output head on the CPU` : layers;
}

/** "Fastest setup: 128K tokens of context, 31.2 tok/s" (speed only when measured). */
export function fastestSetupText(fast: Placement): string {
  const context = `${formatContext(fast.context)} tokens of context`;
  if (fast.speed.source === "measured") {
    return `Fastest setup: ${context}, ${formatTps(fast.speed.decode_tps_start)}`;
  }
  return `Fastest setup: ${context}, speed not measured yet`;
}

export function kvMemoryText(p: Placement): string {
  return formatMb(p.kv_cache_mb);
}

export function isBenchmarkFinished(status: string): boolean {
  return status === "done" || status === "failed" || status === "cancelled";
}

/** First selection: what is loaded now for this model, else the server's recommendation, else the first format. */
export function initialSelection(
  options: OptionsResponse,
  model: ModelInfo | undefined
): { kv: KvFormat; flash: boolean; vision: boolean } {
  const loaded = model?.dara.loaded ? model.dara.config : undefined;
  if (loaded) return { kv: loaded.kv, flash: loaded.flash, vision: loaded.vision };
  const first: FormatInfo | undefined = options.formats[0];
  return {
    kv: options.recommended?.kv ?? first?.id ?? "q8",
    flash: options.recommended?.flash ?? false,
    vision: model?.dara.vision.installed ?? false
  };
}
