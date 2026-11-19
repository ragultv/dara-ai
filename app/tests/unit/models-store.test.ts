import { describe, it, expect, vi, beforeEach } from "vitest";
import { modelsStore } from "../../src/lib/stores/models.svelte.ts";
import { apiClient } from "../../src/lib/api/client.ts";
import type {
  ModelInfo,
  ModelList,
  DownloadJob,
  LoadStatus,
  Stats
} from "../../src/lib/api/types.ts";

describe("ModelsStore", () => {
  const mockNanoModel: ModelInfo = {
    id: "dara-nano",
    object: "model",
    owned_by: "dara",
    created: 1700000000,
    dara: {
      name: "Dara Nano",
      base_model: "gemma-2-2b",
      description: "Ultra-compact model",
      family: "gemma4",
      quant: "Q4_K_M",
      status: "not_installed",
      loaded: false,
      fits_gpu: "full",
      min_vram_mb: 1400,
      full_vram_mb: 2100,
      context: 32768,
      download_size_mb: 800,
      download: null,
      files: [],
      flash: { available: true, installed: false, active: false },
      vision: { available: false, installed: false, active: false }
    }
  };

  const mockModelList: ModelList = {
    object: "list",
    data: [mockNanoModel],
    models_dir: "C:\\models"
  };

  const mockStats: Stats = {
    uptime_seconds: 100,
    pid: 1234,
    queued: 0,
    context: { capacity: 32768, held_tokens: 0 },
    engine: {
      requests: 1,
      errors: 0,
      busy: false,
      live_tokens: 0,
      live_tokens_per_second: 0,
      avg_decode_tokens_per_second: 25,
      avg_prefill_tokens_per_second: 150,
      compactions: 0,
      prompt_tokens_total: 10,
      completion_tokens_total: 20,
      cached_tokens_total: 5
    },
    disk: {
      models_dir: "C:\\models",
      volume_free_gb: 50.0,
      volume_total_gb: 500.0,
      installed_models_mb: 0,
      model_file_mb: 0,
      mmproj_file_mb: 0,
      flash_file_mb: 0
    },
    downloads: [],
    last_request: null
  };

  beforeEach(() => {
    vi.restoreAllMocks();
    modelsStore.models = [];
    modelsStore.error = null;
    modelsStore.diskFreeGb = null;
  });

  it("fetches models list and updates disk free space", async () => {
    vi.spyOn(apiClient, "getModels").mockResolvedValue(mockModelList);
    vi.spyOn(apiClient, "getStats").mockResolvedValue(mockStats);

    await modelsStore.fetchModels();

    expect(modelsStore.models.length).toBe(1);
    expect(modelsStore.models[0]?.id).toBe("dara-nano");
    expect(modelsStore.diskFreeGb).toBe(50.0);
    expect(modelsStore.hasInstalledModels).toBe(false);
  });

  it("blocks download when disk free space is lower than model size", async () => {
    modelsStore.models = [mockNanoModel];
    modelsStore.diskFreeGb = 0.5; // 0.5 GB free < 800 MB (0.78 GB) ? Wait: 800 MB = 0.78125 GB > 0.5 GB

    await modelsStore.downloadModel("dara-nano");

    expect(modelsStore.error).toContain("Not enough disk space");
  });

  it("initiates download and tracks download progress job", async () => {
    modelsStore.models = [mockNanoModel];
    modelsStore.diskFreeGb = 20.0;

    const mockJob: DownloadJob = {
      id: "job-1",
      model: "dara-nano",
      status: "downloading",
      file: "model",
      files_done: 0,
      files_total: 1,
      completed_bytes: 400000000,
      total_bytes: 800000000,
      percent: 50,
      mb_per_second: 30,
      eta_seconds: 13,
      seconds: 13
    };

    vi.spyOn(apiClient, "pullModel").mockResolvedValue(mockJob);

    await modelsStore.downloadModel("dara-nano");

    expect(modelsStore.downloads["dara-nano"]).toEqual(mockJob);
    modelsStore.stopDownloadPolling("dara-nano");
  });

  it("initiates model loading and tracks status", async () => {
    const mockStatus: LoadStatus = {
      state: "loading",
      percent: 30,
      model: "dara-nano"
    };

    vi.spyOn(apiClient, "loadModel").mockResolvedValue(mockStatus);

    await modelsStore.loadModel("dara-nano");

    expect(modelsStore.loadingModelId).toBe("dara-nano");
    expect(modelsStore.loadProgress?.state).toBe("loading");
    modelsStore.stopLoadPolling();
  });
});
