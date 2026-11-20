// Engine settings that only the Tauri side can apply: port, models folder, LAN access, VRAM reserve,
// plus restarting the engine. Outside Tauri (browser dev or mock server) none of this is available.
import { create } from "zustand";
import { apiClient } from "../api/client.ts";
import { isTauri } from "../utils/app-version.ts";
import { useServerStore } from "./useServerStore.ts";

export interface EngineSettingsValues {
  port: number;
  /** Empty string means the default folder. */
  models_dir: string;
  allow_lan: boolean;
  vram_reserve_mb: number;
}

export interface EngineSettingsInfo {
  saved: EngineSettingsValues;
  running: EngineSettingsValues;
  default_models_dir: string;
  restart_needed: boolean;
}

interface ServerConnection {
  baseUrl: string;
  apiKey: string;
}

interface EngineState {
  isAvailable: boolean;
  info: EngineSettingsInfo | null;
  isRestarting: boolean;
  error: string | null;
  load: () => Promise<void>;
  save: (values: EngineSettingsValues) => Promise<boolean>;
  restart: (regenerateKey: boolean) => Promise<boolean>;
}

const HEALTH_WAIT_MS = 90_000;

function messageOf(err: unknown): string {
  if (typeof err === "string") return err;
  if (err instanceof Error) return err.message;
  return "Something went wrong.";
}

async function invokeCommand<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

/** Polls /health until the restarted engine answers. */
async function waitForEngine(): Promise<boolean> {
  const deadline = Date.now() + HEALTH_WAIT_MS;
  while (Date.now() < deadline) {
    try {
      await apiClient.getHealth(undefined, 1500);
      return true;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  return false;
}

export const useEngineStore = create<EngineState>((set, get) => ({
  isAvailable: isTauri(),
  info: null,
  isRestarting: false,
  error: null,

  load: async () => {
    if (!isTauri()) return;
    try {
      const info = await invokeCommand<EngineSettingsInfo>("get_engine_settings");
      set({ info, isAvailable: true, error: null });
    } catch (err: unknown) {
      set({ error: messageOf(err) });
    }
  },

  save: async (values) => {
    try {
      const info = await invokeCommand<EngineSettingsInfo>("set_engine_settings", {
        settings: values
      });
      set({ info, error: null });
      return true;
    } catch (err: unknown) {
      set({ error: messageOf(err) });
      return false;
    }
  },

  restart: async (regenerateKey) => {
    if (get().isRestarting) return false;
    set({ isRestarting: true, error: null });
    try {
      const connection = await invokeCommand<ServerConnection>("restart_engine", { regenerateKey });
      // Reach the new engine first; it may have a new port and key.
      apiClient.setConnection(connection.baseUrl, connection.apiKey);
      const healthy = await waitForEngine();
      useServerStore.getState().setConnection(connection.baseUrl, connection.apiKey);
      await get().load();
      if (!healthy) {
        set({
          error: "The engine did not answer after restarting. Check the Monitor page or try again."
        });
        return false;
      }
      return true;
    } catch (err: unknown) {
      set({ error: messageOf(err) });
      return false;
    } finally {
      set({ isRestarting: false });
    }
  }
}));
