import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from "vitest";
import {
  redactUserPaths,
  formatDiagnostics,
  pathToFileName,
  scrubText,
  sanitizeValue
} from "../../src/lib/utils/redact.ts";
import { apiClient, DaraApiClient } from "../../src/lib/api/client.ts";
import { storeApi } from "../../src/lib/api/store-api.ts";
import type { Stats } from "../../src/lib/api/types.ts";
import {
  useSettingsStore,
  fromFlat,
  toFlat,
  migrateLegacy,
  resetSettingsQueueForTests,
  DEFAULT_GENERAL,
  DEFAULT_SERVER,
  DEFAULT_INFERENCE,
  DEBOUNCE_MS
} from "../../src/lib/stores/useSettingsStore.ts";
import { useThemeStore } from "../../src/lib/stores/useThemeStore.ts";
import {
  useMonitorStore,
  pushSample,
  sampleFromStats,
  MAX_SAMPLES,
  type MonitorSample
} from "../../src/lib/stores/useMonitorStore.ts";
import { buildRuns } from "../../src/lib/ui/Chart.tsx";
import { validateDraft } from "../../src/lib/features/settings/EngineSettingsForm.tsx";
import { createMockServer } from "../../mocks/server.ts";
import statsSample from "../../../docs/ui/samples/stats.json";

const WIN_USER_PATH = "C:\\Users\\someone\\AppData\\Local\\Dara\\models\\gemma-4.gguf";

describe("Diagnostics redaction and privacy", () => {
  it("masks the user folder in Windows paths", () => {
    const sanitized = redactUserPaths(
      "Error loading model from C:\\Users\\tragu\\AppData\\Local\\dara\\models"
    );
    expect(sanitized).toBe(
      "Error loading model from C:\\Users\\<redacted>\\AppData\\Local\\dara\\models"
    );
    expect(sanitized).not.toContain("tragu");
  });

  it("masks JSON-escaped double-backslash paths and Unix home paths", () => {
    expect(redactUserPaths('{"p": "D:\\\\Users\\\\JohnDoe\\\\m.gguf"}')).not.toContain("JohnDoe");
    expect(redactUserPaths("/home/developer/.config/x")).toBe("/home/<redacted>/.config/x");
    expect(redactUserPaths("/Users/alice/Library/x")).toBe("/Users/<redacted>/Library/x");
  });

  it("reduces a path to its file name, with either slash style", () => {
    expect(pathToFileName(WIN_USER_PATH)).toBe("gemma-4.gguf");
    expect(pathToFileName("C:/Users/someone/AppData/x.json")).toBe("x.json");
    expect(pathToFileName("D:\\models\\")).toBe("models");
  });

  it("reduces paths inside a sentence", () => {
    const text = scrubText(`failed to open ${WIN_USER_PATH} (code 5)`);
    expect(text).toBe("failed to open gemma-4.gguf (code 5)");
    expect(text).not.toContain("someone");
  });

  it("drops secret-looking keys", () => {
    const out = sanitizeValue({
      api_key: "abc",
      Authorization: "Bearer x",
      prompt_tokens_total: 4,
      ok: 1
    });
    expect(out).toEqual({ prompt_tokens_total: 4, ok: 1 });
  });

  it("never leaks the user name or full paths from real stats", () => {
    const stats = {
      ...statsSample,
      disk: { ...statsSample.disk, models_dir: "C:\\Users\\someone\\AppData\\Local\\Dara\\models" },
      model: { ...statsSample.model, path: WIN_USER_PATH, mmproj: "C:\\Users\\someone\\mm.gguf" }
    };
    const version = {
      version: "0.1.0-dev",
      api: 1,
      built: "x",
      calibration_file: "C:\\Users\\someone\\calib.json"
    };
    const text = formatDiagnostics(stats, version, {
      appVersion: "0.1.0",
      errors: [`could not read ${WIN_USER_PATH}`]
    });
    expect(text).not.toContain("someone");
    expect(text).not.toContain("AppData");
    expect(text).toContain("gemma-4.gguf");
    expect(text).toContain("NVIDIA GeForce GTX 1650");
    expect(text).toContain("diagnostics_created_at");
    expect(text).toContain("0.1.0");
  });
});

describe("Settings mapping (flat database keys)", () => {
  const base = { general: DEFAULT_GENERAL, server: DEFAULT_SERVER, inference: DEFAULT_INFERENCE };

  it("reads the pre-populated server sample", () => {
    const view = fromFlat(
      {
        "app.minimize_to_tray": true,
        "app.start_at_login": false,
        "chat.max_tokens": 4096,
        "chat.system_prompt": "",
        "chat.temperature": 0.7,
        "defaults.flash": "auto",
        "defaults.kv": "auto",
        "defaults.vision": true,
        "defaults.vram_reserve_mb": 64,
        "server.allow_lan": false,
        "server.models_dir": null,
        "server.port": 11435,
        theme: "system"
      },
      base
    );
    expect(view.server.modelsDir).toBe("");
    expect(view.inference.defaultMaxTokens).toBe(4096);
    expect(view.inference.defaultFlash).toBe("auto");
  });

  it("ignores wrongly typed or out-of-range values", () => {
    const view = fromFlat(
      {
        "chat.temperature": "hot",
        "server.port": 80,
        "defaults.kv": "bogus",
        "chat.max_tokens": 0
      },
      base
    );
    expect(view.inference.defaultTemperature).toBe(0.7);
    expect(view.server.port).toBe(11435);
    expect(view.inference.defaultKv).toBe("auto");
    expect(view.inference.defaultMaxTokens).toBe(4096);
  });

  it("writes flat keys and stores an empty models folder as null", () => {
    expect(toFlat({ server: { modelsDir: "" }, inference: { defaultTemperature: 1.2 } })).toEqual({
      "server.models_dir": null,
      "chat.temperature": 1.2
    });
  });

  it("migrates legacy nested settings without the fake key or fake defaults", () => {
    const { flat, hasLegacy } = migrateLegacy(
      {
        server: {
          port: 12000,
          modelsDir: "D:\\darap-ai\\models",
          allowNetworkAccess: false,
          apiKey: "dara_sk_live_99f8d1c742"
        },
        inference: {
          defaultTemperature: 1.1,
          systemPrompt: "You are Dara, a helpful and precise local AI assistant."
        }
      },
      { general: { startAtLogin: true } }
    );
    expect(hasLegacy).toBe(true);
    expect(flat).toEqual({
      "app.start_at_login": true,
      "server.port": 12000,
      "server.allow_lan": false,
      "chat.temperature": 1.1
    });
    expect(JSON.stringify(flat)).not.toContain("dara_sk_live");
  });

  it("reports no legacy data on a clean database", () => {
    expect(migrateLegacy({ theme: "dark" }, undefined)).toEqual({ flat: {}, hasLegacy: false });
  });
});

describe("Settings store persistence", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    resetSettingsQueueForTests();
    localStorage.clear();
    useSettingsStore.setState({
      general: { ...DEFAULT_GENERAL },
      server: { ...DEFAULT_SERVER },
      inference: { ...DEFAULT_INFERENCE },
      saveState: "idle",
      error: null
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("debounces writes and sends one merged flat patch", async () => {
    const patch = vi.spyOn(storeApi, "patchSettings").mockResolvedValue({});
    useSettingsStore.getState().updateInference({ defaultTemperature: 0.9 });
    useSettingsStore
      .getState()
      .updateInference({ defaultTemperature: 1.0, defaultMaxTokens: 2048 });
    expect(useSettingsStore.getState().inference.defaultTemperature).toBe(1.0);
    expect(patch).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(DEBOUNCE_MS + 10);
    expect(patch).toHaveBeenCalledTimes(1);
    expect(patch).toHaveBeenCalledWith({ "chat.temperature": 1.0, "chat.max_tokens": 2048 });
    expect(useSettingsStore.getState().saveState).toBe("saved");
  });

  it("flush writes immediately", async () => {
    const patch = vi.spyOn(storeApi, "patchSettings").mockResolvedValue({});
    useSettingsStore.getState().updateGeneral({ startAtLogin: true });
    await useSettingsStore.getState().flush();
    expect(patch).toHaveBeenCalledWith({ "app.start_at_login": true });
  });

  it("keeps the value and reports an error when the write fails, then retries with the next change", async () => {
    const patch = vi
      .spyOn(storeApi, "patchSettings")
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue({});
    useSettingsStore.getState().updateInference({ defaultMaxTokens: 100 });
    await vi.advanceTimersByTimeAsync(DEBOUNCE_MS + 10);
    expect(useSettingsStore.getState().saveState).toBe("error");
    expect(useSettingsStore.getState().inference.defaultMaxTokens).toBe(100);

    useSettingsStore.getState().updateInference({ systemPrompt: "hi" });
    await vi.advanceTimersByTimeAsync(DEBOUNCE_MS + 10);
    expect(patch).toHaveBeenLastCalledWith({ "chat.max_tokens": 100, "chat.system_prompt": "hi" });
  });

  it("theme changes go through the same flat key", async () => {
    const patch = vi.spyOn(storeApi, "patchSettings").mockResolvedValue({});
    useThemeStore.getState().setTheme("dark");
    expect(localStorage.getItem("dara_theme_choice")).toBe("dark");
    await vi.advanceTimersByTimeAsync(DEBOUNCE_MS + 10);
    expect(patch).toHaveBeenCalledWith({ theme: "dark" });
  });

  it("init loads flat keys, migrates legacy keys once and applies the saved theme", async () => {
    vi.spyOn(storeApi, "getSettings").mockResolvedValue({
      "chat.temperature": 0.3,
      "server.port": 11435,
      theme: "light",
      inference: { defaultMaxTokens: 777 }
    });
    const patch = vi.spyOn(storeApi, "patchSettings").mockResolvedValue({});
    const del = vi.spyOn(storeApi, "deleteSetting").mockResolvedValue({ deleted: true });
    localStorage.setItem(
      "dara_settings_v2",
      JSON.stringify({ server: { apiKey: "dara_sk_live_x" } })
    );

    await useSettingsStore.getState().init();

    const s = useSettingsStore.getState();
    expect(s.inference.defaultTemperature).toBe(0.3);
    expect(s.inference.defaultMaxTokens).toBe(777);
    expect(useThemeStore.getState().theme).toBe("light");
    expect(patch).toHaveBeenCalledWith({ "chat.max_tokens": 777 });
    expect(del).toHaveBeenCalledWith("inference");
    expect(localStorage.getItem("dara_settings_v2")).toBeNull();
    expect(JSON.stringify(patch.mock.calls)).not.toContain("dara_sk_live");
  });

  it("init reports an error and keeps defaults when the database is unreachable", async () => {
    vi.spyOn(storeApi, "getSettings").mockRejectedValue(new Error("down"));
    await useSettingsStore.getState().init();
    expect(useSettingsStore.getState().error).toBe("down");
    expect(useSettingsStore.getState().inference.defaultTemperature).toBe(0.7);
  });

  it("keeps the public fields the chat code reads", () => {
    const s = useSettingsStore.getState();
    expect(Object.keys(s.inference).sort()).toEqual(
      [
        "defaultFlash",
        "defaultKv",
        "defaultMaxTokens",
        "defaultTemperature",
        "systemPrompt",
        "visionEnabled",
        "vramReserveMb"
      ].sort()
    );
    expect(Object.keys(s.server).sort()).toEqual(["allowNetworkAccess", "modelsDir", "port"]);
    expect("apiKey" in s.server).toBe(false);
  });
});

describe("Engine settings form validation", () => {
  const ok = { port: "11435", modelsDir: "", allowLan: false, vramReserve: "64" };
  it("accepts defaults", () => {
    expect(validateDraft(ok)).toEqual({});
  });
  it("rejects bad port, reserve and relative folders", () => {
    expect(validateDraft({ ...ok, port: "80" }).port).toBeDefined();
    expect(validateDraft({ ...ok, port: "abc" }).port).toBeDefined();
    expect(validateDraft({ ...ok, vramReserve: "-1" }).vram).toBeDefined();
    expect(validateDraft({ ...ok, modelsDir: "models" }).folder).toBeDefined();
    expect(validateDraft({ ...ok, modelsDir: "D:\\models" })).toEqual({});
  });
});

describe("Monitor history", () => {
  const stats = statsSample as unknown as Stats;

  it("starts empty and keeps a ring of the last 120 samples", () => {
    expect(useMonitorStore.getState().samples).toEqual([]);
    let ring: MonitorSample[] = [];
    for (let i = 0; i < 150; i++) {
      ring = pushSample(ring, { t: i, tps: i, vramUsedMb: null, gpuUtilPercent: null });
    }
    expect(ring).toHaveLength(MAX_SAMPLES);
    expect(ring[0]?.t).toBe(30);
    expect(ring[ring.length - 1]?.t).toBe(149);
  });

  it("records a gap (null), never zero, when the engine is idle", () => {
    expect(sampleFromStats(stats, 1).tps).toBeNull();
    const busy = {
      ...stats,
      engine: { ...stats.engine, busy: true, live_tokens_per_second: 42.5 }
    };
    expect(sampleFromStats(busy, 2).tps).toBe(42.5);
  });

  it("records no GPU values without NVML", () => {
    const noGpu = { ...stats, gpu: { ...(stats.gpu as NonNullable<Stats["gpu"]>), nvml: false } };
    const s = sampleFromStats(noGpu, 3);
    expect(s.vramUsedMb).toBeNull();
    expect(s.gpuUtilPercent).toBeNull();
    const withoutGpu: Stats = { ...stats };
    delete withoutGpu.gpu;
    expect(sampleFromStats(withoutGpu, 4).vramUsedMb).toBeNull();
    expect(sampleFromStats(stats, 5).vramUsedMb).toBeCloseTo(3844.94, 1);
  });

  it("splits the chart line at gaps", () => {
    const runs = buildRuns([1, 2, null, null, 3], 10, 0, 4);
    expect(runs).toHaveLength(2);
    expect(runs[0]?.xs).toHaveLength(2);
    expect(runs[1]?.xs).toHaveLength(1);
    // newest point sits at the right edge
    expect(runs[1]?.xs[0]).toBeGreaterThan(runs[0]?.xs[1] ?? 0);
  });
});

describe("Monitor polling", () => {
  let visibility: "visible" | "hidden" = "visible";

  beforeEach(() => {
    vi.useFakeTimers();
    visibility = "visible";
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: () => visibility
    });
    useMonitorStore.getState().stopPolling();
    useMonitorStore.getState().reset();
  });

  afterEach(() => {
    useMonitorStore.getState().stopPolling();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("polls every second while visible and stops when asked", async () => {
    const getStats = vi
      .spyOn(apiClient, "getStats")
      .mockResolvedValue(statsSample as unknown as Stats);
    vi.spyOn(apiClient, "getVersion").mockResolvedValue({
      version: "x",
      api: 1,
      built: "",
      calibration_file: ""
    });
    useMonitorStore.getState().startPolling();
    await vi.advanceTimersByTimeAsync(3100);
    expect(getStats.mock.calls.length).toBeGreaterThanOrEqual(3);
    expect(useMonitorStore.getState().samples.length).toBeGreaterThanOrEqual(3);

    useMonitorStore.getState().stopPolling();
    const calls = getStats.mock.calls.length;
    await vi.advanceTimersByTimeAsync(5000);
    expect(getStats.mock.calls.length).toBe(calls);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("makes no requests while the window is hidden and resumes when it returns", async () => {
    const getStats = vi
      .spyOn(apiClient, "getStats")
      .mockResolvedValue(statsSample as unknown as Stats);
    vi.spyOn(apiClient, "getVersion").mockResolvedValue({
      version: "x",
      api: 1,
      built: "",
      calibration_file: ""
    });
    visibility = "hidden";
    useMonitorStore.getState().startPolling();
    await vi.advanceTimersByTimeAsync(3000);
    expect(getStats).not.toHaveBeenCalled();

    visibility = "visible";
    document.dispatchEvent(new Event("visibilitychange"));
    await vi.advanceTimersByTimeAsync(1100);
    expect(getStats).toHaveBeenCalled();

    visibility = "hidden";
    document.dispatchEvent(new Event("visibilitychange"));
    const calls = getStats.mock.calls.length;
    await vi.advanceTimersByTimeAsync(3000);
    expect(getStats.mock.calls.length).toBe(calls);
  });

  it("keeps the last reading and reports the error when the engine goes away", async () => {
    vi.spyOn(apiClient, "getVersion").mockResolvedValue({
      version: "x",
      api: 1,
      built: "",
      calibration_file: ""
    });
    vi.spyOn(apiClient, "getStats").mockResolvedValueOnce(statsSample as unknown as Stats);
    await useMonitorStore.getState().refresh();
    expect(useMonitorStore.getState().stats).not.toBeNull();

    vi.spyOn(apiClient, "getStats").mockRejectedValue(new Error("connection refused"));
    await useMonitorStore.getState().refresh();
    expect(useMonitorStore.getState().error).toBe("connection refused");
    expect(useMonitorStore.getState().stats).not.toBeNull();
  });
});

describe("Version API Client Endpoint", () => {
  let mockServer: ReturnType<typeof createMockServer>;
  let mockPort: number;
  let client: DaraApiClient;

  beforeAll(async () => {
    mockServer = createMockServer();
    mockPort = await mockServer.start();
    client = new DaraApiClient({ baseUrl: `http://127.0.0.1:${mockPort}` });
  });

  afterAll(async () => {
    await mockServer.stop();
  });

  it("fetches engine version from mock server", async () => {
    const ver = await client.getVersion();
    expect(ver.version).toBe("0.1.0-dev");
    expect(ver.api).toBe(1);
    expect(ver.built).toBeDefined();
  });
});
