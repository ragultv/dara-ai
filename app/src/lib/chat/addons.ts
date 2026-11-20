// Image support (the model's image encoder, "mmproj") and the Flash drafter are separate files on disk that the engine loads together with the model.
// These flows make them usable from the chat: install the file if needed, reload the model with it, and report the REAL context that results
// (Flash and vision use memory, so the context the engine can offer changes; it is read back from the engine, never guessed).
import { create } from "zustand";
import type { LoadedConfig, ModelInfo } from "../api/types.ts";
import { toast } from "../ui/Toast.tsx";
import { useModelsStore } from "../stores/useModelsStore.ts";

export type AddonKind = "vision" | "flash";
export type AddonState =
  | "active" // loaded and working
  | "reload" // installed, but the loaded model was started without it
  | "install" // available for this model, not on disk yet
  | "unavailable"; // this model has no such file (or no model is loaded)

export function addonState(model: ModelInfo | null | undefined, kind: AddonKind): AddonState {
  const f = model?.dara[kind];
  if (!model || !f || !f.available) return "unavailable";
  if (f.active) return "active";
  return f.installed ? "reload" : "install";
}

/** Size in MB of the file an add-on needs, from the catalog the engine reports. */
export function addonSizeMb(
  model: ModelInfo | null | undefined,
  kind: AddonKind
): number | undefined {
  return model?.dara.files.find((f) => f.kind === (kind === "vision" ? "mmproj" : "flash"))
    ?.size_mb;
}

export interface AddonProgress {
  busy: boolean;
  kind: AddonKind | null;
  step: "idle" | "downloading" | "loading";
  percent: number;
  message: string;
}

export const useAddonProgress = create<AddonProgress>(() => ({
  busy: false,
  kind: null,
  step: "idle",
  percent: 0,
  message: ""
}));

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

async function waitForInstall(modelId: string, kind: AddonKind): Promise<void> {
  const deadline = Date.now() + 60 * 60 * 1000;
  for (;;) {
    await useModelsStore.getState().fetchModels();
    const m = useModelsStore.getState().models.find((x) => x.id === modelId);
    if (m?.dara[kind].installed) return;
    const job = useModelsStore.getState().downloads[modelId];
    if (job) {
      useAddonProgress.setState({
        percent: Math.round(job.percent ?? 0),
        message: `${job.file || "Downloading"} (${Math.round(job.percent ?? 0)}%)`
      });
      if (/failed/i.test(job.status)) throw new Error(job.error || "The download failed");
      if (/cancelled/i.test(job.status)) throw new Error("The download was cancelled");
    } else if (m && !m.dara.download) {
      // no running job and still not installed: the download ended without the file
      const again = useModelsStore.getState().downloads[modelId];
      if (!again) throw new Error("The download ended without installing the file");
    }
    if (Date.now() > deadline) throw new Error("The download is taking too long");
    await sleep(1000);
  }
}

/**
 * Makes `kind` work for `modelId`: downloads it when missing and reloads the model with it, keeping the K/V format and the other add-on.
 * Resolves with the configuration the engine actually loaded. Throws a readable Error on failure (the engine restores the previous model).
 */
export async function enableAddon(
  modelId: string,
  kind: AddonKind
): Promise<LoadedConfig | undefined> {
  if (useAddonProgress.getState().busy) throw new Error("Another setup step is already running");
  const store = useModelsStore.getState();
  const model = store.models.find((m) => m.id === modelId);
  if (!model) throw new Error("Unknown model");
  const state = addonState(model, kind);
  if (state === "unavailable")
    throw new Error(
      kind === "vision" ? "This model has no image encoder." : "This model has no Flash drafter."
    );
  useAddonProgress.setState({ busy: true, kind, step: "idle", percent: 0, message: "" });
  try {
    if (state === "install") {
      useAddonProgress.setState({ step: "downloading", message: "Starting the download..." });
      await store.downloadAddon(modelId, kind === "vision" ? "mmproj" : "flash");
      await waitForInstall(modelId, kind);
    }
    useAddonProgress.setState({ step: "loading", percent: 0, message: "Reloading the model..." });
    const cfg = useModelsStore.getState().models.find((m) => m.id === modelId)?.dara.config;
    const timer = setInterval(() => {
      const p = useModelsStore.getState().loadProgress;
      if (p)
        useAddonProgress.setState({
          percent: p.percent ?? 0,
          message: p.stage ? `${p.stage}` : "Reloading the model..."
        });
    }, 300);
    try {
      await useModelsStore.getState().loadModel(modelId, {
        ...(cfg ? { kv: cfg.kv } : {}),
        flash: kind === "flash" ? true : cfg ? cfg.flash : "auto",
        vision: kind === "vision" ? true : cfg ? cfg.vision : true
      });
    } finally {
      clearInterval(timer);
    }
    await useModelsStore.getState().fetchModels();
    const after = useModelsStore.getState().models.find((m) => m.id === modelId);
    if (after && !after.dara[kind].active) {
      const why = after.dara.config?.reason ?? "";
      toast.show(
        `${kind === "vision" ? "Image support" : "Flash"} could not be turned on with the memory available. ${why}`.trim(),
        "error"
      );
    }
    return after?.dara.config;
  } finally {
    useAddonProgress.setState({ busy: false, kind: null, step: "idle", percent: 0, message: "" });
  }
}
