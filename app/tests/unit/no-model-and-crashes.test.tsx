import React from "react";
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { sampleFromStats } from "../../src/lib/stores/useMonitorStore.ts";
import { EnginePanel } from "../../src/lib/features/monitor/EnginePanel.tsx";
import { SummaryTiles } from "../../src/lib/features/monitor/SummaryTiles.tsx";
import { PageErrorBoundary } from "../../src/lib/ui/PageErrorBoundary.tsx";
import type { Stats } from "../../src/lib/api/types.ts";

// What `GET /dara/stats` really answers while no model is loaded: engine, model, context and last_request are null. The Monitor page used to
// read `engine.busy` and crash.
const NO_MODEL = {
  uptime_seconds: 12,
  pid: 1,
  queued: 0,
  context: null,
  engine: null,
  model: null,
  last_request: null,
  downloads: [],
  gpu: {
    name: "GPU",
    nvml: true,
    compute_capability: "7.5",
    vram_total_mb: 4096,
    vram_used_mb: 300,
    vram_free_mb: 3796,
    vram_used_percent: 7,
    vram_weights_mb: 0
  },
  disk: {
    models_dir: "x",
    volume_free_gb: 1,
    volume_total_gb: 2,
    installed_models_mb: 0,
    model_file_mb: 0,
    mmproj_file_mb: 0,
    flash_file_mb: 0
  }
} as unknown as Stats;

describe("the Monitor with no model loaded", () => {
  it("makes a history sample without throwing", () => {
    expect(sampleFromStats(NO_MODEL, 5)).toEqual({
      t: 5,
      tps: null,
      vramUsedMb: 300,
      gpuUtilPercent: null
    });
    expect(
      sampleFromStats({ ...NO_MODEL, gpu: null } as unknown as Stats, 6).vramUsedMb
    ).toBeNull();
  });
  it("draws the engine panel and the summary tiles", () => {
    render(
      <>
        <EnginePanel stats={NO_MODEL} />
        <SummaryTiles stats={NO_MODEL} />
      </>
    );
    expect(screen.getByText(/No model is loaded/)).toBeTruthy();
  });
});

describe("a page that throws", () => {
  it("shows which page and why, instead of leaving the window blank", () => {
    const Bomb: React.FC = () => {
      throw new Error("Cannot read properties of null (reading 'busy')");
    };
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <PageErrorBoundary page="monitor">
        <Bomb />
      </PageErrorBoundary>
    );
    spy.mockRestore();
    expect(screen.getByText("The monitor page ran into a problem")).toBeTruthy();
    expect(screen.getByText(/reading 'busy'/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Try Again" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Copy details" })).toBeTruthy();
  });
});

describe("answers the engine leaves incomplete", () => {
  it("normalizes stats: null sensors are missing, a missing downloads list is empty, a half context is none", async () => {
    const { normalizeStats } = await import("../../src/lib/api/normalize.ts");
    const s = normalizeStats({
      ...NO_MODEL,
      downloads: null,
      context: { capacity: 4096 },
      gpu: { ...NO_MODEL.gpu, power_w: null, temperature_c: 50 }
    } as unknown as Stats);
    expect(s.downloads).toEqual([]);
    expect(s.context).toBeNull();
    expect(s.gpu && "power_w" in s.gpu).toBe(false);
    expect(s.gpu?.temperature_c).toBe(50);
  });
  it("says why an options answer cannot be shown (different engine and app versions)", async () => {
    const { optionsProblem } = await import("../../src/lib/api/normalize.ts");
    expect(optionsProblem(null)).toMatch(/could not be read/);
    expect(
      optionsProblem({
        machine: {},
        formats: [],
        options: [],
        recommended: {},
        calibration: {},
        vision: {}
      })
    ).toBeNull();
    const bad = optionsProblem({ machine: {}, formats: null, options: [], vision: {} });
    expect(bad).toMatch(/missing formats, recommended, calibration/);
    expect(bad).toMatch(/different versions/);
  });
});
