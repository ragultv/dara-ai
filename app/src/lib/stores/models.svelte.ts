import { apiClient } from "../api/client.ts";
import type { ModelInfo, DownloadJob, LoadStatus, PullRequest, LoadRequest } from "../api/types.ts";
import { serverStore } from "./server.svelte.ts";

class ModelsStore {
  models = $state<ModelInfo[]>([]);
  isLoading = $state<boolean>(false);
  error = $state<string | null>(null);

  // Active downloads map: modelId -> DownloadJob
  downloads = $state<Record<string, DownloadJob>>({});

  // Loading state into VRAM
  loadingModelId = $state<string | null>(null);
  loadProgress = $state<LoadStatus | null>(null);

  // Free disk space in GB
  diskFreeGb = $state<number | null>(null);

  private pollIntervals = new Map<string, ReturnType<typeof setInterval>>();
  private loadInterval: ReturnType<typeof setInterval> | null = null;

  get hasInstalledModels(): boolean {
    return this.models.some((m) => m.dara.status === "installed" || m.dara.loaded);
  }

  get loadedModel(): ModelInfo | null {
    return this.models.find((m) => m.dara.loaded) ?? null;
  }

  async fetchModels(): Promise<void> {
    this.isLoading = true;
    this.error = null;
    try {
      const list = await apiClient.getModels();
      this.models = list.data;

      // Sync active downloads from backend model status
      for (const m of this.models) {
        if (m.dara.download && !/done|failed|cancelled/i.test(m.dara.download.status)) {
          this.downloads = { ...this.downloads, [m.id]: m.dara.download };
          if (!this.pollIntervals.has(m.id)) {
            this.startDownloadPolling(m.id, m.dara.download.id);
          }
        }
      }

      // Update disk stats
      try {
        const stats = await apiClient.getStats();
        if (stats.disk) {
          this.diskFreeGb = stats.disk.volume_free_gb;
        }
      } catch {
        // Disk stats optional
      }
    } catch (err: unknown) {
      this.error = err instanceof Error ? err.message : "Failed to load models list";
    } finally {
      this.isLoading = false;
    }
  }

  async downloadModel(
    modelId: string,
    files: Array<"model" | "mmproj" | "flash"> = ["model"]
  ): Promise<void> {
    const model = this.models.find((m) => m.id === modelId);
    if (!model) {
      this.error = `Model ${modelId} not found`;
      return;
    }

    // Disk space check per docs/ui/01-product-and-pages.md:
    // "Block a download if free space is lower than the size, with the numbers."
    // Only block if we have a valid, measured positive free space.
    const requiredGb = model.dara.download_size_mb / 1024;
    if (this.diskFreeGb !== null && this.diskFreeGb > 0 && this.diskFreeGb < requiredGb) {
      this.error = `Not enough disk space: needs ${requiredGb.toFixed(1)} GB, ${this.diskFreeGb.toFixed(1)} GB free.`;
      return;
    }

    this.error = null;
    try {
      const req: PullRequest = { model: modelId, files };
      const job = await apiClient.pullModel(req);
      this.downloads = { ...this.downloads, [modelId]: job };

      // Poll progress every 500 ms
      this.startDownloadPolling(modelId, job.id);
    } catch (err: unknown) {
      this.error = err instanceof Error ? err.message : "Failed to start download";
    }
  }

  private startDownloadPolling(modelId: string, jobId: string): void {
    this.stopDownloadPolling(modelId);

    const check = async () => {
      try {
        const progress = await apiClient.getPullProgress(jobId);
        this.downloads = { ...this.downloads, [modelId]: progress };

        if (progress.status === "done" || progress.status === "failed") {
          this.stopDownloadPolling(modelId);
          await this.fetchModels();
        }
      } catch {
        this.stopDownloadPolling(modelId);
      }
    };

    this.pollIntervals.set(
      modelId,
      setInterval(() => {
        void check();
      }, 500)
    );
  }

  stopDownloadPolling(modelId: string): void {
    const timer = this.pollIntervals.get(modelId);
    if (timer) {
      clearInterval(timer);
      this.pollIntervals.delete(modelId);
    }
  }

  async cancelDownload(modelId: string): Promise<void> {
    const job = this.downloads[modelId];
    if (!job) return;

    this.stopDownloadPolling(modelId);
    try {
      await apiClient.cancelPull(job.id);
    } catch {
      // ignore
    } finally {
      const current = { ...this.downloads };
      Reflect.deleteProperty(current, modelId);
      this.downloads = current;
      await this.fetchModels();
    }
  }

  async loadModel(modelId: string, options: Partial<LoadRequest> = {}): Promise<void> {
    this.loadingModelId = modelId;
    this.loadProgress = { state: "loading", percent: 0, model: modelId };
    this.error = null;

    try {
      await apiClient.loadModel({ model: modelId, ...options });
      this.startLoadPolling();
    } catch (err: unknown) {
      this.error = err instanceof Error ? err.message : "Failed to initiate model loading";
      this.loadingModelId = null;
      this.loadProgress = null;
    }
  }

  private startLoadPolling(): void {
    this.stopLoadPolling();

    const check = async () => {
      try {
        const status = await apiClient.getLoadStatus();
        this.loadProgress = status;

        if (status.state === "ready" || status.state === "failed") {
          this.stopLoadPolling();
          this.loadingModelId = null;
          await this.fetchModels();
          // Update engine connection state
          serverStore.startHealthPolling();
        }
      } catch {
        this.stopLoadPolling();
        this.loadingModelId = null;
      }
    };

    // Poll every 300 ms per docs/ui/04-backend-api.md
    this.loadInterval = setInterval(() => {
      void check();
    }, 300);
  }

  stopLoadPolling(): void {
    if (this.loadInterval !== null) {
      clearInterval(this.loadInterval);
      this.loadInterval = null;
    }
  }

  async unloadModel(): Promise<void> {
    this.error = null;
    try {
      await apiClient.unloadModel();
      await this.fetchModels();
      serverStore.startHealthPolling();
    } catch (err: unknown) {
      this.error = err instanceof Error ? err.message : "Failed to unload model";
    }
  }

  async deleteModel(modelId: string): Promise<void> {
    this.error = null;
    try {
      await apiClient.deleteModel(modelId);
      await this.fetchModels();
    } catch (err: unknown) {
      this.error = err instanceof Error ? err.message : "Failed to delete model";
    }
  }
}

export const modelsStore = new ModelsStore();
