import { apiClient } from "../api/client.ts";
import type { Health } from "../api/types.ts";

export type ConnectionStatus = "starting" | "ready" | "offline" | "error";

interface ServerConnection {
  baseUrl: string;
  apiKey: string;
}

class ServerStore {
  status = $state<ConnectionStatus>("starting");
  health = $state<Health | null>(null);
  baseUrl = $state<string>("http://127.0.0.1:11435");
  apiKey = $state<string | null>(null);
  errorMessage = $state<string | null>(null);

  get isStarting(): boolean {
    return this.status === "starting";
  }
  get isReady(): boolean {
    return this.status === "ready";
  }
  get loadedModel(): string | null {
    return this.health?.model ?? null;
  }

  private pollInterval: ReturnType<typeof setInterval> | null = null;
  private abortController: AbortController | null = null;
  private isTauriApp = false;

  constructor() {
    this.isTauriApp = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
  }

  async init(): Promise<void> {
    this.status = "starting";
    this.errorMessage = null;

    if (this.isTauriApp) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const connection = await invoke<ServerConnection>("get_server_connection");
        this.baseUrl = connection.baseUrl;
        this.apiKey = connection.apiKey;
        apiClient.setConnection(connection.baseUrl, connection.apiKey);
      } catch (err: unknown) {
        this.status = "error";
        this.errorMessage =
          err instanceof Error ? err.message : "Failed to obtain server connection from Tauri";
        return;
      }
    } else {
      // In web dev or test environment
      const envUrl =
        typeof import.meta !== "undefined" &&
        import.meta.env &&
        typeof import.meta.env["VITE_DARA_BASE_URL"] === "string"
          ? (import.meta.env["VITE_DARA_BASE_URL"] as string)
          : undefined;

      if (envUrl && envUrl.length > 0) {
        this.baseUrl = envUrl;
      }
      apiClient.setConnection(this.baseUrl, this.apiKey ?? undefined);
    }

    this.startHealthPolling();
  }

  startHealthPolling(): void {
    this.stopHealthPolling();

    const checkHealth = async () => {
      this.abortController = new AbortController();
      try {
        const healthData = await apiClient.getHealth(this.abortController.signal, 2000);
        this.health = healthData;
        if (healthData.status === "ok") {
          this.status = "ready";
          this.errorMessage = null;
          this.stopHealthPolling();
        } else if (healthData.status === "loading") {
          this.status = "starting";
        } else {
          this.status = "error";
          this.errorMessage = "Engine reported error status";
        }
      } catch (err: unknown) {
        // Keep starting while server boots up
        if (this.status === "starting") {
          // Continue polling
        } else {
          this.status = "offline";
          this.errorMessage = err instanceof Error ? err.message : "Connection failed";
        }
      }
    };

    // Initial immediate check
    void checkHealth();

    // Poll every 300 ms per docs/ui/04-backend-api.md
    this.pollInterval = setInterval(() => {
      void checkHealth();
    }, 300);
  }

  stopHealthPolling(): void {
    if (this.pollInterval !== null) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    if (this.abortController !== null) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  retry(): void {
    void this.init();
  }
}

export const serverStore = new ServerStore();
