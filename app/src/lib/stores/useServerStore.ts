import { create } from "zustand";
import { apiClient } from "../api/client.ts";
import type { Health } from "../api/types.ts";
import { useModelsStore } from "./useModelsStore.ts";
import { useSettingsStore } from "./useSettingsStore.ts";
import { useChatStore } from "./useChatStore.ts";

export type ConnectionStatus = "starting" | "ready" | "offline" | "error";

interface ServerConnection {
  baseUrl: string;
  apiKey: string;
}

interface ServerState {
  status: ConnectionStatus;
  isReady: boolean;
  health: Health | null;
  loadedModel: string | null;
  baseUrl: string;
  apiKey: string | null;
  errorMessage: string | null;
  init: () => Promise<void>;
  checkHealth: () => Promise<boolean>;
  setConnection: (baseUrl: string, apiKey?: string) => void;
}

async function syncDependentStores() {
  try {
    await Promise.allSettled([
      useModelsStore.getState().fetchModels(),
      useSettingsStore.getState().init(),
      useChatStore.getState().loadConversations()
    ]);
  } catch {
    // ignore
  }
}

export const useServerStore = create<ServerState>((set, get) => ({
  status: "starting",
  isReady: false,
  health: null,
  loadedModel: null,
  baseUrl: "http://127.0.0.1:11435",
  apiKey: null,
  errorMessage: null,

  init: async () => {
    set({ status: "starting", errorMessage: null });
    const isTauriApp = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

    if (isTauriApp) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const connection = await invoke<ServerConnection>("get_server_connection");
        set({ baseUrl: connection.baseUrl, apiKey: connection.apiKey });
        apiClient.setConnection(connection.baseUrl, connection.apiKey);
      } catch (err: unknown) {
        set({
          status: "error",
          errorMessage:
            err instanceof Error ? err.message : "Failed to obtain server connection from Tauri"
        });
        return;
      }
    } else {
      apiClient.setConnection(get().baseUrl);
    }

    const healthy = await get().checkHealth();
    if (healthy) {
      void syncDependentStores();
    }
  },

  checkHealth: async () => {
    try {
      const health = await apiClient.getHealth();
      set({
        status: "ready",
        isReady: true,
        health,
        loadedModel: health.model ?? null,
        errorMessage: null
      });
      return true;
    } catch {
      set({ status: "offline", isReady: false, health: null, loadedModel: null });
      return false;
    }
  },

  setConnection: (baseUrl: string, apiKey?: string) => {
    set({ baseUrl, apiKey: apiKey ?? null });
    apiClient.setConnection(baseUrl, apiKey);
    void get()
      .checkHealth()
      .then((healthy) => {
        if (healthy) void syncDependentStores();
      });
  }
}));
