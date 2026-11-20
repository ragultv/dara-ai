import { create } from "zustand";
import { apiClient } from "../api/client.ts";
import { ApiError } from "../api/errors.ts";
import type { ModelInfo, DownloadJob, LoadStatus, KvChoice } from "../api/types.ts";

export type AddonKind = "mmproj" | "flash";
export type PullFile = "model" | AddonKind;

export interface LoadOptions {
  kv?: KvChoice | undefined;
  flash?: "auto" | boolean | undefined;
  vision?: boolean | undefined;
  ctx?: number | undefined;
}

interface ModelsState {
  models: ModelInfo[];
  isLoading: boolean;
  error: string | null;
  /** A gentle, non-fatal message (for example: the Flash drafter cannot be downloaded yet). */
  notice: string | null;
  downloads: Record<string, DownloadJob>;
  loadingModelId: string | null;
  loadProgress: LoadStatus | null;
  diskFreeGb: number | null;
  /** First-run setup in progress (download, then load). Lets the shell keep the onboarding screen up. */
  setupModelId: string | null;
  setupError: string | null;

  /** Download what is missing (model, image support, Flash best effort), then load with the server default. Rejects with the reason on failure. */
  setupModel: (modelId: string) => Promise<void>;
  fetchModels: () => Promise<void>;
  downloadModel: (modelId: string, files?: PullFile[]) => Promise<void>;
  downloadAddon: (modelId: string, kind: AddonKind) => Promise<void>;
  /** Resolves when the download started by downloadModel/downloadAddon (and the best-effort Flash step) ends; rejects if the model files failed or were cancelled. */
  waitForDownload: (modelId: string) => Promise<void>;
  cancelDownload: (modelId: string) => Promise<void>;
  loadModel: (modelId: string, options?: LoadOptions) => Promise<void>;
  unloadModel: () => Promise<void>;
  deleteModel: (modelId: string) => Promise<void>;
  getLoadedModel: () => ModelInfo | null;
  clearError: () => void;
  clearNotice: () => void;
}

const DOWNLOAD_POLL_MS = 1000;
const LOAD_POLL_MS = 300;
const MAX_POLL_FAILURES = 5;
const MAX_IDLE_POLLS = 20;
const TERMINAL_RE = /done|failed|cancelled|complete|installed/i;

export function isDownloadFinished(status: string): boolean {
  return TERMINAL_RE.test(status);
}

export function isDownloadActive(job: DownloadJob | undefined | null): boolean {
  return Boolean(job && job.id && !isDownloadFinished(job.status));
}

function errorText(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---- download followers: one poller per model, removed as soon as its job ends ----

interface Follower {
  timer: ReturnType<typeof setTimeout> | null;
  stop: (job: DownloadJob | null) => void;
}

const followers = new Map<string, Follower>();
const chains = new Map<string, Promise<void>>();

function stopFollowing(modelId: string): void {
  followers.get(modelId)?.stop(null);
}

/** Polls a pull job until it ends. Resolves with the last known job; never rejects. */
function follow(modelId: string, first: DownloadJob): Promise<DownloadJob> {
  stopFollowing(modelId);
  return new Promise<DownloadJob>((resolve) => {
    let last = first;
    let failures = 0;
    const self: Follower = {
      timer: null,
      stop: (job) => {
        if (self.timer) clearTimeout(self.timer);
        self.timer = null;
        if (followers.get(modelId) === self) followers.delete(modelId);
        resolve(job ?? last);
      }
    };
    followers.set(modelId, self);

    const tick = async (): Promise<void> => {
      self.timer = null;
      try {
        const status = await apiClient.getPullProgress(first.id);
        if (followers.get(modelId) !== self) return;
        last = status;
        failures = 0;
        useModelsStore.setState((s) => ({ downloads: { ...s.downloads, [modelId]: status } }));
        if (isDownloadFinished(status.status)) {
          self.stop(status);
          return;
        }
      } catch (err: unknown) {
        if (followers.get(modelId) !== self) return;
        failures += 1;
        if (err instanceof ApiError && err.status === 404) {
          // The server forgot the job: the model list tells what is on disk now.
          self.stop({ ...last, status: "cancelled" });
          return;
        }
        if (failures >= MAX_POLL_FAILURES) {
          self.stop({
            ...last,
            status: "failed",
            error: errorText(err, "Lost contact with the engine.")
          });
          return;
        }
      }
      self.timer = setTimeout(() => void tick(), DOWNLOAD_POLL_MS);
    };
    self.timer = setTimeout(() => void tick(), DOWNLOAD_POLL_MS);
  });
}

function dropDownload(modelId: string): void {
  useModelsStore.setState((s) => {
    const next = { ...s.downloads };
    delete next[modelId];
    return { downloads: next };
  });
}

function flashNotice(serverText: string): string {
  const base = "Flash drafter download isn't available yet.";
  return serverText ? `${base} The server said: ${serverText}` : base;
}

/** Flash is optional: a failure here never fails the model download. */
async function fetchFlashBestEffort(modelId: string): Promise<void> {
  const model = useModelsStore.getState().models.find((m) => m.id === modelId);
  if (model && (!model.dara.flash.available || model.dara.flash.installed)) return;
  try {
    const job = await apiClient.pullModel({ model: modelId, files: ["flash"] });
    if (!job.id) return;
    useModelsStore.setState((s) => ({ downloads: { ...s.downloads, [modelId]: job } }));
    const final = await follow(modelId, job);
    if (/failed/i.test(final.status)) {
      useModelsStore.setState({ notice: flashNotice(final.error ?? "") });
    }
  } catch (err: unknown) {
    useModelsStore.setState({ notice: flashNotice(errorText(err, "")) });
  } finally {
    dropDownload(modelId);
    await useModelsStore.getState().fetchModels();
  }
}

function pickDefaultFiles(model: ModelInfo | undefined): PullFile[] {
  const wanted: PullFile[] = ["model", "mmproj"];
  const kinds = model?.dara.files.map((f) => f.kind) ?? [];
  if (kinds.length === 0) return wanted;
  return wanted.filter((k) => kinds.includes(k));
}

export const useModelsStore = create<ModelsState>((set, get) => {
  /** Shared by downloadModel and downloadAddon. */
  const startDownload = async (
    modelId: string,
    files: PullFile[],
    afterFiles: (() => Promise<void>) | null,
    failureLabel: string
  ): Promise<void> => {
    set({ error: null });
    let job: DownloadJob;
    try {
      job = await apiClient.pullModel({ model: modelId, files });
    } catch (err: unknown) {
      set({ error: errorText(err, "Download failed") });
      throw err;
    }

    const hasJob = typeof job.id === "string" && job.id.length > 0;
    if (hasJob) set((s) => ({ downloads: { ...s.downloads, [modelId]: job } }));

    const chain = (async (): Promise<void> => {
      if (hasJob) {
        const final = await follow(modelId, job);
        if (/failed/i.test(final.status)) {
          const message = final.error ? `${failureLabel}: ${final.error}` : failureLabel;
          set({ error: message });
          throw new Error(message);
        }
        if (/cancelled/i.test(final.status) || !isDownloadFinished(final.status)) {
          dropDownload(modelId);
          await get().fetchModels();
          throw new Error("Download cancelled");
        }
        dropDownload(modelId);
      }
      await get().fetchModels();
      if (afterFiles) await afterFiles();
    })();
    chains.set(modelId, chain);
    void chain
      .catch(() => undefined)
      .finally(() => {
        if (chains.get(modelId) === chain) chains.delete(modelId);
      });
  };

  return {
    models: [],
    isLoading: false,
    error: null,
    notice: null,
    downloads: {},
    loadingModelId: null,
    loadProgress: null,
    diskFreeGb: null,
    setupModelId: null,
    setupError: null,

    setupModel: async (modelId) => {
      set({ setupModelId: modelId, setupError: null });
      try {
        const model = get().models.find((m) => m.id === modelId);
        const missing = model
          ? model.dara.files.some(
              (f) => !f.installed && !(f.kind === "flash" && !model.dara.flash.available)
            )
          : true;
        if (missing) {
          await get().downloadModel(modelId);
          await get().waitForDownload(modelId);
        }
        await get().loadModel(modelId);
      } catch (err: unknown) {
        set({ setupError: errorText(err, "Setup failed") });
        throw err;
      } finally {
        set({ setupModelId: null });
      }
    },

    clearError: () => set({ error: null }),
    clearNotice: () => set({ notice: null }),

    getLoadedModel: () => get().models.find((m) => m.dara.loaded) ?? null,

    fetchModels: async () => {
      set({ isLoading: true, error: null });
      try {
        const list = await apiClient.getModels();
        const downloads: Record<string, DownloadJob> = { ...get().downloads };
        for (const m of list.data) {
          const job = m.dara.download;
          if (job && isDownloadActive(job)) {
            downloads[m.id] = job;
            if (!followers.has(m.id)) {
              // A download started earlier (or by another window): keep following it.
              void follow(m.id, job).then(async () => {
                dropDownload(m.id);
                await get().fetchModels();
              });
            }
          }
        }

        let diskFree: number | null = null;
        try {
          const stats = await apiClient.getStats();
          diskFree = stats.disk.volume_free_gb;
        } catch {
          // Disk space is a convenience; the list is still useful without it.
        }

        set({ models: list.data, downloads, diskFreeGb: diskFree, isLoading: false });
      } catch (err: unknown) {
        set({ error: errorText(err, "Failed to load models list"), isLoading: false });
      }
    },

    downloadModel: async (modelId, files) => {
      const explicit = files !== undefined;
      const wanted = files ?? pickDefaultFiles(get().models.find((m) => m.id === modelId));
      // Model and image support are one job (a failed file fails the whole job); Flash is a separate best-effort job.
      await startDownload(
        modelId,
        wanted,
        explicit ? null : () => fetchFlashBestEffort(modelId),
        "The download failed"
      );
    },

    downloadAddon: async (modelId, kind) => {
      if (kind === "flash") {
        set({ error: null });
        await fetchFlashBestEffort(modelId);
        return;
      }
      await startDownload(modelId, [kind], null, "The image support download failed");
    },

    waitForDownload: (modelId) => chains.get(modelId) ?? Promise.resolve(),

    cancelDownload: async (modelId) => {
      const job = get().downloads[modelId];
      if (!job || !job.id) return;
      try {
        await apiClient.cancelPull(job.id);
        followers.get(modelId)?.stop({ ...job, status: "cancelled" });
        dropDownload(modelId);
        await get().fetchModels();
      } catch (err: unknown) {
        set({ error: errorText(err, "Cancel download failed") });
      }
    },

    loadModel: async (modelId, options) => {
      set({ loadingModelId: modelId, loadProgress: null, error: null });
      try {
        let status: LoadStatus;
        try {
          status = await apiClient.loadModel({
            model: modelId,
            ...(options?.kv !== undefined ? { kv: options.kv } : {}),
            ...(options?.flash !== undefined ? { flash: options.flash } : {}),
            ...(options?.vision !== undefined ? { vision: options.vision } : {}),
            ...(options?.ctx !== undefined ? { ctx: options.ctx } : {})
          });
        } catch (err: unknown) {
          // 409: a load is already running; follow that job instead of failing.
          if (err instanceof ApiError && err.status === 409) {
            status = await apiClient.getLoadStatus();
          } else {
            throw err;
          }
        }

        let idlePolls = 0;
        let failures = 0;
        for (;;) {
          set({ loadProgress: status });
          if (status.state === "ready") break;
          if (status.state === "failed") {
            throw new Error(
              status.error || "The model could not be loaded. The previous model was restored."
            );
          }
          idlePolls = status.state === "idle" ? idlePolls + 1 : 0;
          if (idlePolls >= MAX_IDLE_POLLS) {
            throw new Error("The engine did not start loading the model.");
          }
          await sleep(LOAD_POLL_MS);
          try {
            status = await apiClient.getLoadStatus();
            failures = 0;
          } catch (err: unknown) {
            failures += 1;
            if (failures >= MAX_POLL_FAILURES) throw err;
          }
        }

        set({ loadingModelId: null, loadProgress: null });
        await get().fetchModels();
      } catch (err: unknown) {
        set({
          loadingModelId: null,
          loadProgress: null,
          error: errorText(err, "Failed to load model")
        });
        throw err;
      }
    },

    unloadModel: async () => {
      try {
        await apiClient.unloadModel();
        await get().fetchModels();
      } catch (err: unknown) {
        set({ error: errorText(err, "Unload model failed") });
      }
    },

    deleteModel: async (modelId) => {
      try {
        await apiClient.deleteModel(modelId);
        await get().fetchModels();
      } catch (err: unknown) {
        set({ error: errorText(err, "Delete model failed") });
        throw err;
      }
    }
  };
});
