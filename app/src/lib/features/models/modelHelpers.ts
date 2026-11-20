// Pure helpers for the Models and Onboarding screens. They only re-word what the server reported.
import type { ModelInfo, ModelFile } from "../../api/types.ts";
import { formatGb, formatMb } from "../../utils/format.ts";

export type FileKind = ModelFile["kind"];
export type FileState = "installed" | "available" | "unavailable";

export interface FileRow {
  kind: FileKind;
  label: string;
  state: FileState;
  sizeMb: number | null;
}

const KIND_LABEL: Record<FileKind, string> = {
  model: "Model",
  mmproj: "Image support",
  flash: "Flash drafter"
};

const KINDS: FileKind[] = ["model", "mmproj", "flash"];

/** One row per file kind: installed, available to download, or not offered for this model. */
export function fileRows(model: ModelInfo): FileRow[] {
  return KINDS.map((kind) => {
    const file = model.dara.files.find((f) => f.kind === kind);
    let state: FileState = "unavailable";
    if (file?.installed) state = "installed";
    else if (file)
      state = kind === "flash" && !model.dara.flash.available ? "unavailable" : "available";
    return { kind, label: KIND_LABEL[kind], state, sizeMb: file ? file.size_mb : null };
  });
}

export function fileStateText(state: FileState): string {
  if (state === "installed") return "Installed";
  if (state === "available") return "Not installed";
  return "Not available";
}

/** Plain-words reading of `fits_gpu`. */
export function fitsText(fits: string): { label: string; detail: string } {
  if (fits === "full") {
    return {
      label: "Fits your GPU",
      detail: "The whole model fits in your graphics card memory, so it runs at full speed."
    };
  }
  if (fits === "partial") {
    return {
      label: "Partly on your GPU",
      detail:
        "Some of the model runs on the CPU because the graphics card memory is too small, so it is slower."
    };
  }
  return {
    label: "Too big for your GPU",
    detail: "This model does not fit your graphics card memory and would run mostly on the CPU."
  };
}

export function statusText(status: ModelInfo["dara"]["status"] | string): string {
  switch (status) {
    case "installed":
      return "Installed";
    case "partial":
      return "Some files are missing";
    case "downloading":
      return "Downloading";
    default:
      return "Not installed";
  }
}

export interface DiskCheck {
  ok: boolean;
  message: string | null;
}

/** Blocks a download when the free space is lower than the size, naming both numbers. */
export function diskCheck(neededMb: number, freeGb: number | null): DiskCheck {
  if (freeGb === null || neededMb <= 0) return { ok: true, message: null };
  const freeMb = freeGb * 1024;
  if (freeMb >= neededMb) return { ok: true, message: null };
  return {
    ok: false,
    message: `Not enough disk space: needs ${formatMb(neededMb)}, ${formatGb(freeGb)} free.`
  };
}

/** Size of the files that a plain "Download" fetches: model plus image support that are still missing. */
export function missingDownloadMb(model: ModelInfo, kinds: FileKind[]): number {
  return model.dara.files
    .filter((f) => kinds.includes(f.kind) && !f.installed)
    .reduce((sum, f) => sum + f.size_mb, 0);
}

/** Files that exist on disk for this model (named in the delete confirmation). */
export function installedFiles(model: ModelInfo): ModelFile[] {
  return model.dara.files.filter((f) => f.installed);
}

export function installedSizeMb(model: ModelInfo): number {
  return installedFiles(model).reduce((sum, f) => sum + f.size_mb, 0);
}

/** First-run pick: the largest tier that fully fits the GPU; else the smallest download that partly fits; else the smallest. */
export function recommendTier(models: ModelInfo[]): string | null {
  if (models.length === 0) return null;
  const bySize = [...models].sort((a, b) => a.dara.download_size_mb - b.dara.download_size_mb);
  const full = bySize.filter((m) => m.dara.fits_gpu === "full");
  const pick = full.length > 0 ? full[full.length - 1] : bySize[0];
  return pick ? pick.id : null;
}
