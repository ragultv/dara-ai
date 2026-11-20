import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useModelsStore } from "../../src/lib/stores/useModelsStore.ts";
import { apiClient } from "../../src/lib/api/client.ts";
import { ApiError } from "../../src/lib/api/errors.ts";
import type {
  DownloadJob,
  LoadStatus,
  ModelInfo,
  ModelList,
  Stats
} from "../../src/lib/api/types.ts";

const nano: ModelInfo = {
  id: "dara-nano",
  object: "model",
  owned_by: "dara",
  created: 1,
  dara: {
    name: "Dara Nano",
    base_model: "base",
    description: "d",
    family: "gemma4",
    quant: "Q4_K_M",
    status: "not_installed",
    loaded: false,
    fits_gpu: "full",
    min_vram_mb: 1,
    full_vram_mb: 2,
    context: 32768,
    download_size_mb: 800,
    download: null,
    files: [],
    flash: { available: true, installed: false, active: false },
    vision: { available: true, installed: false, active: false }
  }
};

const list: ModelList = { object: "list", data: [nano], models_dir: "models" };
const stats = { disk: { volume_free_gb: 50 } } as unknown as Stats;

function job(over: Partial<DownloadJob> = {}): DownloadJob {
  return {
    id: "job-1",
    model: "dara-nano",
    status: "running",
    file: "model",
    files_done: 0,
    files_total: 2,
    completed_bytes: 0,
    total_bytes: 100,
    percent: 0,
    mb_per_second: 0,
    eta_seconds: 0,
    seconds: 0,
    ...over
  };
}

describe("models store: downloads", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(apiClient, "getModels").mockResolvedValue(list);
    vi.spyOn(apiClient, "getStats").mockResolvedValue(stats);
    useModelsStore.setState({ models: [nano], downloads: {}, error: null, notice: null });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("downloads model and image support as one job, then Flash as a separate job", async () => {
    const pull = vi
      .spyOn(apiClient, "pullModel")
      .mockResolvedValueOnce(job({ id: "main" }))
      .mockResolvedValueOnce(job({ id: "flash", file: "flash" }));
    vi.spyOn(apiClient, "getPullProgress").mockImplementation((id) =>
      Promise.resolve(job({ id, status: "done", percent: 100 }))
    );

    await useModelsStore.getState().downloadModel("dara-nano");
    const waiting = useModelsStore.getState().waitForDownload("dara-nano");
    await vi.advanceTimersByTimeAsync(5000);
    await waiting;

    expect(pull).toHaveBeenNthCalledWith(1, { model: "dara-nano", files: ["model", "mmproj"] });
    expect(pull).toHaveBeenNthCalledWith(2, { model: "dara-nano", files: ["flash"] });
    expect(useModelsStore.getState().downloads["dara-nano"]).toBeUndefined();
  });

  it("keeps the download successful when the Flash drafter cannot be fetched", async () => {
    vi.spyOn(apiClient, "pullModel")
      .mockResolvedValueOnce(job({ id: "main" }))
      .mockRejectedValueOnce(new ApiError(502, "drafter returned 404", "server_error"));
    vi.spyOn(apiClient, "getPullProgress").mockResolvedValue(job({ status: "done", percent: 100 }));

    await useModelsStore.getState().downloadModel("dara-nano");
    const waiting = useModelsStore.getState().waitForDownload("dara-nano");
    await vi.advanceTimersByTimeAsync(5000);
    await expect(waiting).resolves.toBeUndefined();

    const state = useModelsStore.getState();
    expect(state.notice).toContain("Flash drafter download isn't available yet");
    expect(state.notice).toContain("drafter returned 404");
    expect(state.error).toBeNull();
  });

  it("does not poll when the pull answers installed without an id", async () => {
    vi.spyOn(apiClient, "pullModel").mockResolvedValue({
      status: "installed"
    } as unknown as DownloadJob);
    const progress = vi.spyOn(apiClient, "getPullProgress");

    await useModelsStore.getState().downloadModel("dara-nano", ["model"]);
    await useModelsStore.getState().waitForDownload("dara-nano");
    await vi.advanceTimersByTimeAsync(5000);

    expect(progress).not.toHaveBeenCalled();
    expect(useModelsStore.getState().downloads["dara-nano"]).toBeUndefined();
  });

  it("requests a single add-on file and reports a failure", async () => {
    const pull = vi
      .spyOn(apiClient, "pullModel")
      .mockResolvedValue(job({ id: "mm", files_total: 1 }));
    vi.spyOn(apiClient, "getPullProgress").mockResolvedValue(
      job({ id: "mm", status: "failed", error: "disk full" })
    );

    await useModelsStore.getState().downloadAddon("dara-nano", "mmproj");
    const assertion = expect(
      useModelsStore.getState().waitForDownload("dara-nano")
    ).rejects.toThrow("disk full");
    await vi.advanceTimersByTimeAsync(2000);
    await assertion;

    expect(pull).toHaveBeenCalledWith({ model: "dara-nano", files: ["mmproj"] });
    expect(useModelsStore.getState().error).toContain("disk full");
  });

  it("stops polling once a download is cancelled", async () => {
    vi.spyOn(apiClient, "pullModel").mockResolvedValue(job({ id: "main" }));
    const progress = vi.spyOn(apiClient, "getPullProgress").mockResolvedValue(job({ percent: 10 }));
    vi.spyOn(apiClient, "cancelPull").mockResolvedValue(undefined);

    await useModelsStore.getState().downloadModel("dara-nano", ["model"]);
    const assertion = expect(
      useModelsStore.getState().waitForDownload("dara-nano")
    ).rejects.toThrow("cancelled");
    await useModelsStore.getState().cancelDownload("dara-nano");
    await assertion;
    const calls = progress.mock.calls.length;
    await vi.advanceTimersByTimeAsync(5000);

    expect(progress.mock.calls.length).toBe(calls);
  });
});

describe("models store: loading", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(apiClient, "getModels").mockResolvedValue(list);
    vi.spyOn(apiClient, "getStats").mockResolvedValue(stats);
    useModelsStore.setState({
      models: [nano],
      loadingModelId: null,
      loadProgress: null,
      error: null
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("passes the options through and resolves only when the server reports ready", async () => {
    const load = vi
      .spyOn(apiClient, "loadModel")
      .mockResolvedValue({ state: "loading", percent: 5 });
    const statuses: LoadStatus[] = [
      { state: "loading", percent: 50, stage: "weights" },
      { state: "ready", percent: 100 }
    ];
    vi.spyOn(apiClient, "getLoadStatus").mockImplementation(() =>
      Promise.resolve(statuses.shift() ?? { state: "ready" })
    );

    let settled = false;
    const promise = useModelsStore
      .getState()
      .loadModel("dara-nano", { kv: "q8", flash: true, vision: false, ctx: 8192 })
      .then(() => {
        settled = true;
      });

    await vi.advanceTimersByTimeAsync(300);
    expect(settled).toBe(false);
    expect(useModelsStore.getState().loadingModelId).toBe("dara-nano");
    expect(useModelsStore.getState().loadProgress?.percent).toBe(50);

    await vi.advanceTimersByTimeAsync(300);
    await promise;

    expect(load).toHaveBeenCalledWith({
      model: "dara-nano",
      kv: "q8",
      flash: true,
      vision: false,
      ctx: 8192
    });
    expect(useModelsStore.getState().loadingModelId).toBeNull();
  });

  it("rejects with the server's message when the load fails", async () => {
    vi.spyOn(apiClient, "loadModel").mockResolvedValue({ state: "queued" });
    vi.spyOn(apiClient, "getLoadStatus").mockResolvedValue({
      state: "failed",
      error: "out of memory"
    });

    const assertion = expect(useModelsStore.getState().loadModel("dara-nano")).rejects.toThrow(
      "out of memory"
    );
    await vi.advanceTimersByTimeAsync(300);
    await assertion;

    const state = useModelsStore.getState();
    expect(state.loadingModelId).toBeNull();
    expect(state.error).toBe("out of memory");
  });

  it("follows the running job when the server answers 409", async () => {
    vi.spyOn(apiClient, "loadModel").mockRejectedValue(
      new ApiError(409, "a load is running", "conflict")
    );
    const status = vi.spyOn(apiClient, "getLoadStatus").mockResolvedValue({ state: "ready" });

    await useModelsStore.getState().loadModel("dara-nano");

    expect(status).toHaveBeenCalled();
    expect(useModelsStore.getState().loadingModelId).toBeNull();
    expect(useModelsStore.getState().error).toBeNull();
  });
});
