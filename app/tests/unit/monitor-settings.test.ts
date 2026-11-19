import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { redactUserPaths, formatDiagnostics } from "../../src/lib/utils/redact.ts";
import { settingsStore } from "../../src/lib/stores/settings.svelte.ts";
import { DaraApiClient } from "../../src/lib/api/client.ts";
import { createMockServer } from "../../mocks/server.ts";

describe("Diagnostics Redaction & Privacy", () => {
  it("redacts Windows single-backslash user paths", () => {
    const raw = "Error loading model from C:\\Users\\tragu\\AppData\\Local\\dara\\models";
    const sanitized = redactUserPaths(raw);
    expect(sanitized).toBe(
      "Error loading model from C:\\Users\\<redacted>\\AppData\\Local\\dara\\models"
    );
    expect(sanitized).not.toContain("tragu");
  });

  it("redacts Windows JSON-escaped double-backslash user paths", () => {
    const raw = '{"path": "D:\\\\Users\\\\JohnDoe\\\\models\\\\nano.gguf"}';
    const sanitized = redactUserPaths(raw);
    expect(sanitized).toBe('{"path": "D:\\\\Users\\\\<redacted>\\\\models\\\\nano.gguf"}');
    expect(sanitized).not.toContain("JohnDoe");
  });

  it("redacts Unix home and user paths", () => {
    const linux = "Log at /home/developer/.config/dara/server.log";
    expect(redactUserPaths(linux)).toBe("Log at /home/<redacted>/.config/dara/server.log");

    const mac = "Model at /Users/alice/Library/Application Support/dara";
    expect(redactUserPaths(mac)).toBe(
      "Model at /Users/<redacted>/Library/Application Support/dara"
    );
  });

  it("formats diagnostics JSON without exposing sensitive user directories", () => {
    const fakeStats = {
      model: {
        path: "C:\\Users\\alice\\AppData\\Local\\dara\\models\\nano.gguf"
      },
      disk: {
        models_dir: "C:\\Users\\alice\\AppData\\Local\\dara\\models"
      }
    };
    const fakeVersion = {
      version: "0.1.0-dev",
      api: 1,
      built: "Nov 19 2026",
      calibration_file: "C:\\Users\\alice\\AppData\\Local\\dara\\calib.json"
    };

    const diagnostics = formatDiagnostics(fakeStats, fakeVersion);
    expect(diagnostics).not.toContain("alice");
    expect(diagnostics).toContain("<redacted>");
    expect(diagnostics).toContain("diagnostics_created_at");
  });
});

describe("Settings Store", () => {
  it("updates and persists settings changes", () => {
    settingsStore.updateGeneral({ startAtLogin: true });
    expect(settingsStore.general.startAtLogin).toBe(true);

    settingsStore.updateServer({ port: 12345 });
    expect(settingsStore.server.port).toBe(12345);

    settingsStore.updateInference({ defaultKv: "k8v4", vramReserveMb: 128 });
    expect(settingsStore.inference.defaultKv).toBe("k8v4");
    expect(settingsStore.inference.vramReserveMb).toBe(128);
  });

  it("regenerates API keys", () => {
    const key1 = settingsStore.server.apiKey;
    const key2 = settingsStore.regenerateApiKey();
    expect(key2).not.toBe(key1);
    expect(key2.startsWith("dara-key-")).toBe(true);
  });

  it("resets settings to defaults", () => {
    settingsStore.reset();
    expect(settingsStore.general.startAtLogin).toBe(false);
    expect(settingsStore.server.port).toBe(11435);
    expect(settingsStore.inference.defaultKv).toBe("auto");
    expect(settingsStore.inference.vramReserveMb).toBe(64);
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
