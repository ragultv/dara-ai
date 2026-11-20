import { describe, it, expect, vi, beforeEach } from "vitest";
import { useModelsStore } from "../../src/lib/stores/useModelsStore.ts";
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
    useModelsStore.setState({ models: [], error: null, diskFreeGb: null, downloads: {} });
  });

  it("fetches models list and updates disk free space", async () => {
    vi.spyOn(apiClient, "getModels").mockResolvedValue(mockModelList);
    vi.spyOn(apiClient, "getStats").mockResolvedValue(mockStats);

    await useModelsStore.getState().fetchModels();

    const state = useModelsStore.getState();
    expect(state.models.length).toBe(1);
    expect(state.models[0]?.id).toBe("dara-nano");
    expect(state.diskFreeGb).toBe(50.0);
  });

  it("initiates download and tracks download progress job", async () => {
    useModelsStore.setState({ models: [mockNanoModel], diskFreeGb: 20.0 });

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

    vi.useFakeTimers();
    const pull = vi.spyOn(apiClient, "pullModel").mockResolvedValue(mockJob);

    await useModelsStore.getState().downloadModel("dara-nano");

    expect(useModelsStore.getState().downloads["dara-nano"]).toEqual(mockJob);
    expect(pull).toHaveBeenCalledWith({ model: "dara-nano", files: ["model", "mmproj"] });
    vi.useRealTimers();
  });
});
