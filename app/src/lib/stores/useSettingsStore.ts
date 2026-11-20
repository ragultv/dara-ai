// App settings. They live in the server's database as FLAT keys (GET/PATCH /dara/settings, see
// docs/ui/08-database.md); the nested shape below is only the in-memory view the UI and the chat
// code read. Engine settings (port, models folder, LAN, VRAM reserve) are mirrored here but the
// engine itself reads them from the Tauri side (see useEngineStore).
import { create } from "zustand";
import { storeApi } from "../api/store-api.ts";
import type { KvChoice } from "../api/types.ts";
import { toast } from "../ui/Toast.tsx";
import { registerThemePersister, useThemeStore, type ThemeChoice } from "./useThemeStore.ts";

export interface GeneralSettings {
  startAtLogin: boolean;
  minimizeToTray: boolean;
}

export interface ServerSettings {
  port: number;
  modelsDir: string;
  allowNetworkAccess: boolean;
}

export interface InferenceSettings {
  defaultKv: KvChoice;
  defaultFlash: "auto" | "on" | "off";
  vramReserveMb: number;
  defaultMaxTokens: number;
  defaultTemperature: number;
  visionEnabled: boolean;
  systemPrompt: string;
}

export type SaveState = "idle" | "saving" | "saved" | "error";

interface SettingsState {
  general: GeneralSettings;
  server: ServerSettings;
  inference: InferenceSettings;
  isLoading: boolean;
  error: string | null;
  saveState: SaveState;

  init: () => Promise<void>;
  updateGeneral: (patch: Partial<GeneralSettings>) => void;
  updateServer: (patch: Partial<ServerSettings>) => void;
  updateInference: (patch: Partial<InferenceSettings>) => void;
  /** Writes pending changes now (page unmount, window hidden). */
  flush: () => Promise<void>;
  /** Restores the general and generation defaults. Engine settings are not touched. */
  reset: () => void;
}

export const LEGACY_LOCAL_STORAGE_KEY = "dara_settings_v2";
const LEGACY_DB_KEYS = ["general", "server", "inference"] as const;
// Values the old (fake) settings page wrote by default; they were never real choices.
const LEGACY_FAKE_MODELS_DIR = "D:\\darap-ai\\models";
const LEGACY_FAKE_SYSTEM_PROMPT = "You are Dara, a helpful and precise local AI assistant.";

export const DEBOUNCE_MS = 400;
const SAVED_VISIBLE_MS = 2500;

export const DEFAULT_GENERAL: GeneralSettings = { startAtLogin: false, minimizeToTray: true };
export const DEFAULT_SERVER: ServerSettings = {
  port: 11435,
  modelsDir: "",
  allowNetworkAccess: false
};
export const DEFAULT_INFERENCE: InferenceSettings = {
  defaultKv: "auto",
  defaultFlash: "auto",
  vramReserveMb: 64,
  defaultMaxTokens: 4096,
  defaultTemperature: 0.7,
  visionEnabled: true,
  systemPrompt: ""
};

const KV_VALUES: readonly string[] = ["auto", "q8", "k8v4", "q4", "tq4", "k16v8"];

// ---- coercion helpers: every value read from the database is untrusted JSON ----

function asBool(v: unknown, fallback: boolean): boolean {
  return typeof v === "boolean" ? v : fallback;
}
function asString(v: unknown, fallback: string): string {
  return typeof v === "string" ? v : fallback;
}
function asInt(v: unknown, fallback: number, min: number, max: number): number {
  if (typeof v !== "number" || !Number.isFinite(v)) return fallback;
  const n = Math.round(v);
  return n >= min && n <= max ? n : fallback;
}
function asTemperature(v: unknown, fallback: number): number {
  if (typeof v !== "number" || !Number.isFinite(v)) return fallback;
  return Math.min(2, Math.max(0, v));
}
function asKv(v: unknown, fallback: KvChoice): KvChoice {
  return typeof v === "string" && KV_VALUES.includes(v) ? (v as KvChoice) : fallback;
}
function asFlash(
  v: unknown,
  fallback: InferenceSettings["defaultFlash"]
): InferenceSettings["defaultFlash"] {
  if (v === "auto" || v === "on" || v === "off") return v;
  if (v === true) return "on";
  if (v === false) return "off";
  return fallback;
}
function asTheme(v: unknown): ThemeChoice | undefined {
  return v === "system" || v === "light" || v === "dark" ? v : undefined;
}

export interface SettingsView {
  general: GeneralSettings;
  server: ServerSettings;
  inference: InferenceSettings;
}

/** Pure: flat database keys to the nested view. Unknown or invalid values fall back to `base`. */
export function fromFlat(raw: Record<string, unknown>, base: SettingsView): SettingsView {
  const { general: g, server: s, inference: i } = base;
  return {
    general: {
      startAtLogin: asBool(raw["app.start_at_login"], g.startAtLogin),
      minimizeToTray: asBool(raw["app.minimize_to_tray"], g.minimizeToTray)
    },
    server: {
      port: asInt(raw["server.port"], s.port, 1024, 65535),
      modelsDir:
        raw["server.models_dir"] === null ? "" : asString(raw["server.models_dir"], s.modelsDir),
      allowNetworkAccess: asBool(raw["server.allow_lan"], s.allowNetworkAccess)
    },
    inference: {
      defaultKv: asKv(raw["defaults.kv"], i.defaultKv),
      defaultFlash: asFlash(raw["defaults.flash"], i.defaultFlash),
      vramReserveMb: asInt(raw["defaults.vram_reserve_mb"], i.vramReserveMb, 0, 8192),
      defaultMaxTokens: asInt(raw["chat.max_tokens"], i.defaultMaxTokens, 1, 1_000_000),
      defaultTemperature: asTemperature(raw["chat.temperature"], i.defaultTemperature),
      visionEnabled: asBool(raw["defaults.vision"], i.visionEnabled),
      systemPrompt: asString(raw["chat.system_prompt"], i.systemPrompt)
    }
  };
}

/** Pure: the flat keys for the parts of the view that changed. Empty models folder is stored as null. */
export function toFlat(patch: {
  general?: Partial<GeneralSettings>;
  server?: Partial<ServerSettings>;
  inference?: Partial<InferenceSettings>;
}): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const g = patch.general;
  if (g?.startAtLogin !== undefined) out["app.start_at_login"] = g.startAtLogin;
  if (g?.minimizeToTray !== undefined) out["app.minimize_to_tray"] = g.minimizeToTray;
  const s = patch.server;
  if (s?.port !== undefined) out["server.port"] = s.port;
  if (s?.modelsDir !== undefined)
    out["server.models_dir"] = s.modelsDir === "" ? null : s.modelsDir;
  if (s?.allowNetworkAccess !== undefined) out["server.allow_lan"] = s.allowNetworkAccess;
  const i = patch.inference;
  if (i?.defaultKv !== undefined) out["defaults.kv"] = i.defaultKv;
  if (i?.defaultFlash !== undefined) out["defaults.flash"] = i.defaultFlash;
  if (i?.vramReserveMb !== undefined) out["defaults.vram_reserve_mb"] = i.vramReserveMb;
  if (i?.defaultMaxTokens !== undefined) out["chat.max_tokens"] = i.defaultMaxTokens;
  if (i?.defaultTemperature !== undefined) out["chat.temperature"] = i.defaultTemperature;
  if (i?.visionEnabled !== undefined) out["defaults.vision"] = i.visionEnabled;
  if (i?.systemPrompt !== undefined) out["chat.system_prompt"] = i.systemPrompt;
  return out;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * Pure: turns the old nested settings (database keys "general"/"server"/"inference" and the old
 * localStorage entry "dara_settings_v2") into flat keys to write. Database values win over
 * localStorage. The old fake API key is never migrated, nor are the old hard-coded defaults.
 */
export function migrateLegacy(
  db: Record<string, unknown>,
  local: unknown
): { flat: Record<string, unknown>; hasLegacy: boolean } {
  const pick = (name: (typeof LEGACY_DB_KEYS)[number]): Record<string, unknown> => {
    const fromLocal = isRecord(local) && isRecord(local[name]) ? local[name] : {};
    const fromDb = isRecord(db[name]) ? db[name] : {};
    return { ...fromLocal, ...fromDb };
  };
  const hasLegacy = LEGACY_DB_KEYS.some((k) => k in db) || isRecord(local);
  if (!hasLegacy) return { flat: {}, hasLegacy: false };

  const g = pick("general");
  const s = pick("server");
  const i = pick("inference");
  const patch = {
    general: {
      ...(typeof g["startAtLogin"] === "boolean" ? { startAtLogin: g["startAtLogin"] } : {}),
      ...(typeof g["minimizeToTray"] === "boolean" ? { minimizeToTray: g["minimizeToTray"] } : {})
    },
    server: {
      ...(typeof s["port"] === "number" ? { port: s["port"] } : {}),
      ...(typeof s["modelsDir"] === "string" && s["modelsDir"] !== LEGACY_FAKE_MODELS_DIR
        ? { modelsDir: s["modelsDir"] }
        : {}),
      ...(typeof s["allowNetworkAccess"] === "boolean"
        ? { allowNetworkAccess: s["allowNetworkAccess"] }
        : {})
    },
    inference: {
      ...(typeof i["defaultKv"] === "string" ? { defaultKv: asKv(i["defaultKv"], "auto") } : {}),
      ...(i["defaultFlash"] !== undefined
        ? { defaultFlash: asFlash(i["defaultFlash"], "auto") }
        : {}),
      ...(typeof i["vramReserveMb"] === "number" ? { vramReserveMb: i["vramReserveMb"] } : {}),
      ...(typeof i["defaultMaxTokens"] === "number"
        ? { defaultMaxTokens: i["defaultMaxTokens"] }
        : {}),
      ...(typeof i["defaultTemperature"] === "number"
        ? { defaultTemperature: i["defaultTemperature"] }
        : {}),
      ...(typeof i["visionEnabled"] === "boolean" ? { visionEnabled: i["visionEnabled"] } : {}),
      ...(typeof i["systemPrompt"] === "string" && i["systemPrompt"] !== LEGACY_FAKE_SYSTEM_PROMPT
        ? { systemPrompt: i["systemPrompt"] }
        : {})
    }
  };
  // Validate through the same coercion as a normal read, so bad old values are dropped.
  const requested = toFlat(patch);
  const checked = toFlat(
    fromFlat(requested, {
      general: DEFAULT_GENERAL,
      server: DEFAULT_SERVER,
      inference: DEFAULT_INFERENCE
    })
  );
  const flat: Record<string, unknown> = {};
  for (const key of Object.keys(requested)) flat[key] = checked[key];
  return { flat, hasLegacy };
}

// ---- debounced write queue (module state: one queue for the whole app) ----

let pending: Record<string, unknown> = {};
let debounceTimer: ReturnType<typeof setTimeout> | null = null;
let savedTimer: ReturnType<typeof setTimeout> | null = null;
let writing: Promise<void> | null = null;

function clearTimers(): void {
  if (debounceTimer !== null) clearTimeout(debounceTimer);
  if (savedTimer !== null) clearTimeout(savedTimer);
  debounceTimer = null;
  savedTimer = null;
}

/** Test helper: forgets queued writes and timers. */
export function resetSettingsQueueForTests(): void {
  pending = {};
  clearTimers();
  writing = null;
}

export const useSettingsStore = create<SettingsState>((set, get) => {
  async function writePending(): Promise<void> {
    if (writing) await writing;
    if (Object.keys(pending).length === 0) return;
    const batch = pending;
    pending = {};
    set({ saveState: "saving" });
    writing = (async () => {
      try {
        await storeApi.patchSettings(batch);
        set({ saveState: "saved" });
        if (savedTimer !== null) clearTimeout(savedTimer);
        savedTimer = setTimeout(() => {
          if (get().saveState === "saved") set({ saveState: "idle" });
        }, SAVED_VISIBLE_MS);
      } catch (err: unknown) {
        // Keep the user's values; newer edits to the same key win over the failed ones.
        pending = { ...batch, ...pending };
        set({ saveState: "error" });
        const reason = err instanceof Error ? err.message : "unknown error";
        toast.show(
          `Could not save your settings (${reason}). They stay on screen and will be saved with the next change.`,
          "error"
        );
      }
    })();
    try {
      await writing;
    } finally {
      writing = null;
    }
  }

  function queue(flat: Record<string, unknown>): void {
    if (Object.keys(flat).length === 0) return;
    pending = { ...pending, ...flat };
    set({ saveState: "saving" });
    if (debounceTimer !== null) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      debounceTimer = null;
      void writePending();
    }, DEBOUNCE_MS);
  }

  registerThemePersister((choice) => queue({ theme: choice }));

  return {
    general: { ...DEFAULT_GENERAL },
    server: { ...DEFAULT_SERVER },
    inference: { ...DEFAULT_INFERENCE },
    isLoading: false,
    error: null,
    saveState: "idle",

    init: async () => {
      set({ isLoading: true });
      try {
        const db = await storeApi.getSettings();

        // One-time migration of the old nested keys and the old localStorage entry.
        let local: unknown;
        try {
          const text = localStorage.getItem(LEGACY_LOCAL_STORAGE_KEY);
          local = text ? JSON.parse(text) : undefined;
        } catch {
          local = undefined;
        }
        const legacy = migrateLegacy(db, local);
        let flat: Record<string, unknown> = { ...db };
        if (legacy.hasLegacy) {
          try {
            if (Object.keys(legacy.flat).length > 0) await storeApi.patchSettings(legacy.flat);
            flat = { ...db, ...legacy.flat };
            for (const key of LEGACY_DB_KEYS) {
              if (key in db) await storeApi.deleteSetting(key);
            }
            try {
              localStorage.removeItem(LEGACY_LOCAL_STORAGE_KEY);
            } catch {
              // ignore
            }
          } catch {
            // Leave the legacy data in place; the next start tries again.
          }
        }

        const view = fromFlat(flat, {
          general: get().general,
          server: get().server,
          inference: get().inference
        });
        set({ ...view, isLoading: false, error: null });
        const theme = asTheme(flat["theme"]);
        if (theme) useThemeStore.getState().hydrate(theme);
      } catch (err: unknown) {
        set({
          isLoading: false,
          error: err instanceof Error ? err.message : "Could not load settings from the database."
        });
      }
    },

    updateGeneral: (patch) => {
      set({ general: { ...get().general, ...patch } });
      queue(toFlat({ general: patch }));
    },

    updateServer: (patch) => {
      set({ server: { ...get().server, ...patch } });
      queue(toFlat({ server: patch }));
    },

    updateInference: (patch) => {
      set({ inference: { ...get().inference, ...patch } });
      queue(toFlat({ inference: patch }));
    },

    flush: async () => {
      if (debounceTimer !== null) {
        clearTimeout(debounceTimer);
        debounceTimer = null;
      }
      await writePending();
    },

    reset: () => {
      const general = { ...DEFAULT_GENERAL };
      const inference = { ...DEFAULT_INFERENCE, vramReserveMb: get().inference.vramReserveMb };
      set({ general, inference });
      queue(toFlat({ general, inference }));
    }
  };
});

// Do not lose a pending edit when the window is hidden or closed.
if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") void useSettingsStore.getState().flush();
  });
}
