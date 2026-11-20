import { describe, it, expect } from "vitest";
import {
  fastestSetupText,
  findOption,
  flashAvailability,
  initialSelection,
  placementText,
  qualityPresentation,
  speedText,
  visionAvailability
} from "../../src/lib/features/options/optionsLogic.ts";
import {
  diskCheck,
  fileRows,
  fitsText,
  recommendTier
} from "../../src/lib/features/models/modelHelpers.ts";
import { formatContext, formatEta, formatMb } from "../../src/lib/utils/format.ts";
import type { ModelInfo, OptionsResponse, Placement } from "../../src/lib/api/types.ts";

function placement(over: Partial<Placement> = {}): Placement {
  return {
    context: 131072,
    layers: 35,
    layers_gpu: 35,
    head: "gpu",
    fully_on_gpu: true,
    all_layers_on_gpu: true,
    flash_active: false,
    kv_cache_mb: 1296,
    ok: true,
    speed: { source: "not_measured" },
    ...over
  };
}

function model(over: Partial<ModelInfo["dara"]> = {}, id = "dara-nano"): ModelInfo {
  return {
    id,
    object: "model",
    owned_by: "dara",
    created: 1,
    dara: {
      name: id,
      base_model: "b",
      description: "d",
      family: "gemma4",
      quant: "Q4",
      status: "installed",
      loaded: false,
      fits_gpu: "full",
      min_vram_mb: 1,
      full_vram_mb: 2,
      context: 4096,
      download_size_mb: 1000,
      download: null,
      files: [
        { kind: "model", name: "m", installed: true, size_mb: 900, url: "" },
        { kind: "mmproj", name: "p", installed: false, size_mb: 90, url: "" },
        { kind: "flash", name: "f", installed: false, size_mb: 10, url: "" }
      ],
      flash: { available: true, installed: false, active: false },
      vision: { available: true, installed: false, active: false },
      ...over
    }
  };
}

describe("options logic", () => {
  it("finds the entry for a kv and flash combination", () => {
    const options = [
      { kv: "q8", flash: false, fast: placement(), points: [] },
      { kv: "q8", flash: true, fast: placement({ context: 1 }), points: [] }
    ] as OptionsResponse["options"];
    expect(findOption(options, "q8", true)?.fast.context).toBe(1);
    expect(findOption(options, "q4", false)).toBeUndefined();
  });

  it("never shows a speed that was not measured", () => {
    expect(speedText({ source: "not_measured" })).toEqual({
      main: "Not measured yet",
      detail: null,
      isMeasured: false
    });
    const measured = speedText({
      source: "measured",
      decode_tps_start: 31.24,
      decode_tps_full_estimate: 20.05,
      power: "battery"
    });
    expect(measured.main).toBe("31.2 tok/s");
    expect(measured.detail).toBe("about 20.1 tok/s when the context is full, measured on battery");
  });

  it("describes placement from the server's own numbers", () => {
    expect(placementText(placement())).toBe("Fully on the GPU");
    expect(placementText(placement({ fully_on_gpu: false, layers_gpu: 30, head: "cpu" }))).toBe(
      "30 of 35 layers on the GPU, output head on the CPU"
    );
    expect(placementText(placement({ ok: false }))).toBe("Does not fit");
  });

  it("words the fastest setup with or without a measurement", () => {
    expect(fastestSetupText(placement())).toBe(
      "Fastest setup: 128K tokens of context, speed not measured yet"
    );
    expect(
      fastestSetupText(placement({ speed: { source: "measured", decode_tps_start: 12.34 } }))
    ).toBe("Fastest setup: 128K tokens of context, 12.3 tok/s");
  });

  it("pairs every quality label with an icon and a word", () => {
    for (const label of ["high", "medium", "experimental", "lower"] as const) {
      const q = qualityPresentation(label);
      expect(q.icon.length).toBeGreaterThan(0);
      expect(q.word.length).toBeGreaterThan(0);
    }
  });

  it("explains why Flash and image support are disabled", () => {
    expect(
      flashAvailability(
        null,
        model({ flash: { available: false, installed: false, active: false } })
      )
    ).toMatchObject({
      canEnable: false,
      canInstall: false
    });
    expect(flashAvailability(null, model())).toMatchObject({ canEnable: false, canInstall: true });
    expect(
      flashAvailability(null, model({ flash: { available: true, installed: true, active: false } }))
    ).toMatchObject({ canEnable: true, reason: null });
    expect(visionAvailability(model())).toMatchObject({ canEnable: false, canInstall: true });
    expect(visionAvailability(undefined).canInstall).toBe(false);
  });

  it("starts from the loaded configuration, else the recommendation", () => {
    const data = {
      formats: [{ id: "q8" }],
      recommended: { kv: "k8v4", flash: true, context: 1, reason: "", rule: "" }
    } as unknown as OptionsResponse;
    expect(initialSelection(data, undefined)).toEqual({ kv: "k8v4", flash: true, vision: false });
    const loaded = model({
      loaded: true,
      config: {
        kv: "q4",
        flash: false,
        context: 1,
        layers: 1,
        layers_on_gpu: 1,
        head: "gpu",
        vision: true,
        source: "user",
        reason: ""
      }
    });
    expect(initialSelection(data, loaded)).toEqual({ kv: "q4", flash: false, vision: true });
  });
});

describe("models helpers", () => {
  it("lists model, image support and Flash states", () => {
    const rows = fileRows(model());
    expect(rows.map((r) => [r.kind, r.state])).toEqual([
      ["model", "installed"],
      ["mmproj", "available"],
      ["flash", "available"]
    ]);
    const noFlash = fileRows(
      model({ flash: { available: false, installed: false, active: false } })
    );
    expect(noFlash[2]?.state).toBe("unavailable");
  });

  it("blocks a download with both numbers when the disk is too full", () => {
    expect(diskCheck(2900, 1.1).message).toBe("Not enough disk space: needs 2.8 GB, 1.1 GB free.");
    expect(diskCheck(500, 10).ok).toBe(true);
    expect(diskCheck(500, null).ok).toBe(true);
  });

  it("recommends the largest tier that fully fits, else the smallest", () => {
    const a = model({ download_size_mb: 4000 }, "a");
    const b = model({ download_size_mb: 9000 }, "b");
    const c = model({ download_size_mb: 12000, fits_gpu: "partial" }, "c");
    expect(recommendTier([a, b, c])).toBe("b");
    expect(recommendTier([model({ fits_gpu: "no", download_size_mb: 5 }, "x"), c])).toBe("x");
    expect(recommendTier([])).toBeNull();
  });

  it("explains fits_gpu in plain words", () => {
    expect(fitsText("full").label).toBe("Fits your GPU");
    expect(fitsText("partial").detail).toContain("CPU");
  });
});

describe("format helpers", () => {
  it("formats sizes, context and time", () => {
    expect(formatMb(940)).toBe("940 MB");
    expect(formatMb(2962.8)).toBe("2.9 GB");
    expect(formatContext(131072)).toBe("128K");
    expect(formatContext(1000)).toBe("1,000");
    expect(formatEta(45)).toBe("45 s");
    expect(formatEta(185)).toBe("3 min");
  });
});
